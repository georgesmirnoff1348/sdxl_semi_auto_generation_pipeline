"""Фейковые legacy-обёртки и сборка временных YAML-задач для юнит-тестов.

Ничего не скачивает и не конструирует модели: фейки пишут 1x1 PNG через Pillow
и ведут журнал вызовов, чтобы доказать делегирование существующим обёрткам.
"""

from __future__ import annotations

import io
import tempfile
import unittest
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import yaml
from PIL import Image

from datasetgen.config import load_yaml
from datasetgen.engines import Dependencies
from datasetgen.manifest import Manifest
from datasetgen.schema import InferenceSpec

MODEL_ID = "test/model-id"
BACKGROUND_MODEL_ID = "test/background-model-id"

STAGE_SINGLE = "single"
STAGE_SUBJECT = "subject"
STAGE_CUT = "cut"
STAGE_INPAINT = "inpaint"


class FailingStage:
    """Маркер шага, на котором фейк имитирует ошибку."""

    SINGLE = STAGE_SINGLE
    SUBJECT = STAGE_SUBJECT
    CUT = STAGE_CUT
    INPAINT = STAGE_INPAINT

    @staticmethod
    def error(stage: str) -> RuntimeError:
        return RuntimeError(f"injected failure at stage '{stage}'")


def write_png(path: Path, seed: int | None = None) -> Path:
    """Написать 1x1 PNG; цвет зависит от seed, чтобы сравнивать байты."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    value = 0 if seed is None else int(seed)
    color = ((value % 251), ((value // 251) % 251), ((value // 251**2) % 251))
    Image.new("RGB", (1, 1), color).save(path, format="PNG")
    return path


def image_bytes(path: Path) -> bytes:
    return Path(path).read_bytes()


class _FakeBackend:
    kind = "backend"

    def __init__(self, log: list[dict], *, fail_at: str | None = None, save: bool = True, **identity: Any) -> None:
        self.log = log
        self.fail_at = fail_at
        self.save = save
        self.identity = identity
        for key, value in identity.items():
            setattr(self, key, value)
        log.append({"kind": f"make_{self.kind}", **identity})

    def _fail_if_needed(self, stage: str) -> None:
        if self.fail_at == stage:
            raise FailingStage.error(stage)

    def __enter__(self):
        self.log.append({"kind": f"enter_{self.kind}", **self.identity})
        return self

    def __exit__(self, exc_type, exc, tb):
        self.log.append({"kind": f"exit_{self.kind}", **self.identity})
        self.close()

    def close(self) -> None:  # pragma: no cover - переопределяется наследниками
        self.log.append({"kind": f"close_{self.kind}", **self.identity})


class FakeGenerator(_FakeBackend):
    kind = "generator"

    def generate_image(self, prompts: Any, config: Any, save_path: Any = None) -> Any:
        stage = STAGE_SUBJECT if "_intermediate" in str(save_path) else STAGE_SINGLE
        self.log.append(
            {
                "kind": "generate_image",
                "stage": stage,
                "prompts": prompts,
                "config": config,
                "save_path": None if save_path is None else Path(save_path),
                **self.identity,
            }
        )
        self._fail_if_needed(stage)
        if self.save and save_path is not None:
            write_png(Path(save_path), getattr(config, "seed", None))
        return Image.new("RGB", (1, 1))

    def unload(self) -> None:
        self.log.append({"kind": "unload", **self.identity})


class FakeCutter(_FakeBackend):
    kind = "cutter"

    def remove_background(self, image: Any, save_path: Any = None) -> Any:
        self.log.append(
            {
                "kind": "remove_background",
                "image": image,
                "save_path": None if save_path is None else Path(save_path),
                **self.identity,
            }
        )
        self._fail_if_needed(STAGE_CUT)
        if self.save and save_path is not None:
            write_png(Path(save_path))
        return image.convert("RGBA") if image is not None else None

    def close(self) -> None:
        self.log.append({"kind": "close", **self.identity})


class FakeInpainter(_FakeBackend):
    kind = "inpainter"

    def inpaint_image(self, prompts: Any, config: Any, image: Any, mask: Any, save_path: Any = None) -> Any:
        self.log.append(
            {
                "kind": "inpaint_image",
                "prompts": prompts,
                "config": config,
                "image": image,
                "mask": mask,
                "save_path": None if save_path is None else Path(save_path),
                **self.identity,
            }
        )
        self._fail_if_needed(STAGE_INPAINT)
        if self.save and save_path is not None:
            write_png(Path(save_path), getattr(config, "seed", None))
        return Image.new("RGB", (1, 1))

    def unload(self) -> None:
        self.log.append({"kind": "unload", **self.identity})


class RecordingBackends:
    """Считает создание обёрток и ведёт общий журнал вызовов."""

    def __init__(self, *, fail_at: str | None = None, save: bool = True) -> None:
        self.log: list[dict] = []
        self.fail_at = fail_at
        self.save = save
        self.counters = {"make_generator": 0, "make_cutter": 0, "make_inpainter": 0}
        self.instances: list[Any] = []

    def _make(self, factory, kind: str, **identity: Any):
        self.counters[f"make_{kind}"] += 1
        instance = factory(self.log, fail_at=self.fail_at, save=self.save, **identity)
        self.instances.append(instance)
        return instance

    def make_generator(self, model_id: str) -> FakeGenerator:
        return self._make(FakeGenerator, "generator", model_id=model_id)

    def make_cutter(self, model_name: str) -> FakeCutter:
        return self._make(FakeCutter, "cutter", model_name=model_name)

    def make_inpainter(self, model_id: str) -> FakeInpainter:
        return self._make(FakeInpainter, "inpainter", model_id=model_id)

    # ------------------------------------------------------------------ helpers
    def kinds(self) -> list[str]:
        return [entry["kind"] for entry in self.log]

    def calls(self, kind: str) -> list[dict]:
        return [entry for entry in self.log if entry["kind"] == kind]

    def raise_if_called(self) -> None:
        """TC-30: после ``--dry-run`` ни одна обёртка не должна быть создана."""

        if any(self.counters.values()):
            raise AssertionError(f"backends were used during a dry run: {self.counters}")


@dataclass
class FakePrompts:
    """Дублирует публичные атрибуты ``configs.FactorPrompts`` (без импорта torch)."""

    prompt: str = ""
    negative_prompt: str = ""
    prompt_2: Optional[str] = None
    negative_prompt_2: Optional[str] = None

    def _as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class FakeParams:
    """Дублирует публичные атрибуты ``configs.FactorInferenceParameters``."""

    num_inference_steps: int = 20
    guidance_scale: float = 7.0
    seed: Optional[int] = None
    height: int = 1024
    width: int = 1024
    extra: dict = field(default_factory=dict)

    def _as_dict(self) -> dict:
        data = {
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "height": self.height,
            "width": self.width,
            "seed": self.seed,
        }
        data.update(self.extra)
        return data


class FakeLegacyTypes:
    def __init__(self, *, fail_at: str | None = None) -> None:
        self.fail_at = fail_at
        self.prompts_created: list[FakePrompts] = []
        self.params_created: list[FakeParams] = []

    def prompts(self, prompt: str, negative_prompt: str) -> FakePrompts:
        if self.fail_at in (STAGE_SINGLE, STAGE_SUBJECT):
            raise FailingStage.error(self.fail_at)
        created = FakePrompts(prompt=prompt, negative_prompt=negative_prompt)
        self.prompts_created.append(created)
        return created

    def params(self, seed: int, spec: InferenceSpec) -> FakeParams:
        created = FakeParams(
            num_inference_steps=spec.num_inference_steps,
            guidance_scale=spec.guidance_scale,
            height=spec.height,
            width=spec.width,
            seed=seed,
            extra=dict(spec.extra),
        )
        self.params_created.append(created)
        return created


class FixedClock:
    """Детерминированные метки времени для манифеста."""

    def __init__(self, start: datetime | None = None) -> None:
        self.start = start or datetime(2026, 9, 26, 13, 59, 42, tzinfo=timezone.utc)
        self.ticks = 0

    def __call__(self) -> datetime:
        moment = self.start + timedelta(seconds=self.ticks)
        self.ticks += 1
        return moment


def make_dependencies(
    *, fail_at: str | None = None, save: bool = True, clock: Any = None
) -> Dependencies:
    return Dependencies(
        backends=RecordingBackends(fail_at=fail_at, save=save),
        legacy_types=FakeLegacyTypes(fail_at=fail_at),
        clock=clock or FixedClock(),
    )


def fake_find_spec(missing: frozenset[str]):
    """Подмена ``importlib.util.find_spec``: перечисленные имена «не установлены».

    Используется вместо реального поиска модуля, чтобы preflight-зависимостей
    проверялся детерминированно, без установки и без импорта тяжёлых пакетов.
    """

    def finder(name: str, package: str | None = None):
        return None if name in missing else object()

    return finder


# --------------------------------------------------------------------- YAML
def class_entry(name: str, count: int, template: str | None = None, variables=None, negative: str = "") -> dict:
    return {
        "name": name,
        "count": count,
        "template": template or "a {worker} in {place}, studio photography",
        "negative_prompt": negative,
        "variables": variables
        if variables is not None
        else {"worker": ["factory worker", "engineer"], "place": ["a hall", "a lab"]},
    }


def job_dict(
    *,
    task: str = "classification",
    count: int = 2,
    seed: int = 20260926,
    output_dir: str | None = None,
    mode: str = "single_layer",
    description: dict | None = None,
    classes: list[dict] | None = None,
    generation: dict | None = None,
    inference: dict | None = None,
    version: int = 1,
) -> dict:
    if classes is None:
        classes = (
            [class_entry("class_a", count), class_entry("class_b", count)]
            if task == "classification"
            else [class_entry("normal", count)]
        )
    data: dict[str, Any] = {
        "version": version,
        "task": task,
        "output_dir": output_dir if output_dir is not None else "./out",
        "seed": seed,
    }
    if description is not None:
        data["dataset_description"] = description
    if generation is not None:
        data["generation"] = generation
    else:
        data["generation"] = {"mode": mode, "cutter": "u2net", "cutter_model_name": None}
    if inference is not None:
        data["inference"] = inference
    data["classes"] = classes
    return data


def two_layer_generation(cutter: str = "u2net", cutter_model_name: str | None = None, extra: dict | None = None) -> dict:
    return {
        "mode": "two_layer",
        "cutter": cutter,
        "cutter_model_name": cutter_model_name,
        "background": {
            "model_id": BACKGROUND_MODEL_ID,
            "template": "background photography of {place}, 1980s",
            "negative_prompt": "people, text",
            "variables": {"place": ["hospital room", "industrial hall"]},
            "inference": {
                "num_inference_steps": 18,
                "guidance_scale": 6.5,
                "width": 1024,
                "height": 1024,
                "extra": extra if extra is not None else {"inner_pad": 20, "strength": 0.9},
            },
        },
    }


def write_job(root: Path, data: dict, name: str = "job.yaml") -> Path:
    """Записать YAML-задачу в tmp-каталог и вернуть путь."""

    path = Path(root) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data.get("output_dir"), str) and data["output_dir"].startswith("./out"):
        data = dict(data)
        data["output_dir"] = str(Path(root) / "out")
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def tmp_job(root: Path, name: str = "job.yaml", **kwargs: Any) -> Path:
    return write_job(root, job_dict(**kwargs), name=name)


# --------------------------------------------------------------- базовый кейс
class TmpCase(unittest.TestCase):
    """Общая обвязка: tmp-каталог, tmp-задача, буферы вывода, чтение манифеста."""

    def setUp(self) -> None:
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        self.root = Path(holder.name).resolve()
        self.out = self.root / "out"
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self.stdout = io.StringIO()

    # ------------------------------------------------------------------ задачи
    def job_data(self, **kwargs: Any) -> dict:
        kwargs.setdefault("output_dir", str(self.out))
        return job_dict(**kwargs)

    def job_path(self, name: str = "job.yaml", **kwargs: Any) -> Path:
        return write_job(self.root, self.job_data(**kwargs), name=name)

    def job(self, name: str = "job.yaml", **kwargs: Any):
        return load_yaml(self.job_path(name=name, **kwargs))

    # ------------------------------------------------------------------ запуск
    def make_deps(self, **kwargs: Any) -> Dependencies:
        return make_dependencies(**kwargs)

    def execute(self, job, deps, **kwargs: Any):
        from datasetgen import runner

        return runner.run_job(
            job,
            kwargs.pop("model_id", MODEL_ID),
            dry_run=kwargs.pop("dry_run", False),
            deps=deps,
            out=self.stdout,
            warn=self.warnings.append,
            error=self.errors.append,
            **kwargs,
        )

    def execute_regenerate(self, job, deps, selection, **kwargs: Any):
        from datasetgen import runner

        return runner.regenerate(
            job,
            kwargs.pop("model_id", MODEL_ID),
            selection,
            dry_run=kwargs.pop("dry_run", False),
            deps=deps,
            out=self.stdout,
            warn=self.warnings.append,
            error=self.errors.append,
            **kwargs,
        )

    # --------------------------------------------------------------- манифест
    def manifest_path(self) -> Path:
        return self.out / "manifest.jsonl"

    def records(self) -> list[dict]:
        return Manifest.load(self.manifest_path()).records

    def image_records(self) -> list[dict]:
        return [record for record in self.records() if record["type"] == "image"]

    def run_records(self) -> list[dict]:
        return [record for record in self.records() if record["type"] == "run"]

    def statuses(self, class_name: str, index: int) -> list[str]:
        return [
            record["status"]
            for record in self.image_records()
            if record["class"] == class_name and record["index"] == index
        ]

    # ------------------------------------------------- ручное наполнение журнала
    def make_image_record(
        self,
        job,
        class_name: str,
        index: int,
        attempt: int = 1,
        status: str = "succeeded",
        **overrides: Any,
    ) -> dict:
        from datasetgen.planner import settings_for

        record = {
            "schema_version": 1,
            "type": "image",
            "run_id": "seeded",
            "class": class_name,
            "index": index,
            "attempt": attempt,
            "status": status,
            "output": f"{class_name}/{class_name}_{index}.png",
            "prompt": f"seeded prompt for {class_name} {index} attempt {attempt}",
            "negative_prompt": "",
            "seed": 1000 + index,
            "model_id": MODEL_ID,
            "settings": settings_for(job.inference),
            "generation_mode": job.generation.mode,
            "intermediate": [],
            "timestamp": "2026-09-26T13:59:43Z",
            "error": None,
            "bytes": 104 if status == "succeeded" else None,
            "duration_sec": 0.5 if status == "succeeded" else None,
        }
        record.update(overrides)
        return record

    def seed_manifest(self, job, images: list[dict] | None = None, run: bool = True) -> None:
        from datasetgen.planner import settings_for

        manifest = Manifest.load(self.out / "manifest.jsonl")
        if run:
            manifest.append(
                {
                    "schema_version": 1,
                    "type": "run",
                    "run_id": "seeded",
                    "command": "run",
                    "started_at": "2026-09-26T13:59:42Z",
                    "job_file": str(job.source_path),
                    "job_sha256": job.raw_sha256,
                    "output_dir": str(self.out),
                    "task": job.task,
                    "generation_mode": job.generation.mode,
                    "model_id": MODEL_ID,
                    "job_seed": job.seed,
                    "classes": [{"name": spec.name, "count": spec.count} for spec in job.classes],
                    "inference": settings_for(job.inference),
                    "two_layer": None,
                }
            )
        for record in images or []:
            manifest.append(record)
