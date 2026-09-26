"""Frozen dataclasses схемы YAML и плана выходов (без логики и без I/O)."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

#: Идентификатор формулы детерминизма. Менять только с новой версией схемы.
SALT = "datasetgen/v1"

TASK_CLASSIFICATION = "classification"
TASK_ANOMALY = "anomaly_detection"
TASKS = (TASK_CLASSIFICATION, TASK_ANOMALY)

MODE_SINGLE = "single_layer"
MODE_TWO = "two_layer"
MODES = (MODE_SINGLE, MODE_TWO)

CUTTER_U2NET = "u2net"
CUTTER_BIREFNET = "birefnet"
CUTTER_CUSTOM = "custom"
CUTTERS = (CUTTER_U2NET, CUTTER_BIREFNET, CUTTER_CUSTOM)

#: Имя rembg-модели для каждого встроенного значения ``generation.cutter``.
CUTTER_MODEL_NAMES: Mapping[str, str] = {
    CUTTER_U2NET: "u2net",
    CUTTER_BIREFNET: "birefnet-general",
}

RESERVED_NAMES = frozenset(
    {
        "_rejected",
        "_intermediate",
        "manifest.jsonl",
        "dataset_description.md",
        "dataset_description_request.md",
    }
)

MANIFEST_NAME = "manifest.jsonl"
REJECTED_DIR_NAME = "_rejected"
INTERMEDIATE_DIR_NAME = "_intermediate"
DESCRIPTION_NAME = "dataset_description.md"
DESCRIPTION_REQUEST_NAME = "dataset_description_request.md"

STATUS_PLANNED = "planned"
STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"
STATUS_REJECTED = "rejected"

#: Статусы отдельной попытки в манифесте.
IMAGE_STATUSES = (STATUS_PLANNED, STATUS_SUCCEEDED, STATUS_FAILED, STATUS_REJECTED)

CLASS_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
CLASS_NAME_PATTERN_TEXT = "^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$"
VARIABLE_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")


@dataclass(frozen=True)
class PromptSpec:
    """Чистые данные промпта. НЕ является объектом ``configs.FactorPrompts``."""

    template: str
    negative_prompt: str
    variables: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class InferenceSpec:
    num_inference_steps: int = 20
    guidance_scale: float = 7.0
    width: int = 1024
    height: int = 1024
    extra: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class BackgroundSpec:
    """Блок ``generation.background``; валиден только при ``mode == two_layer``."""

    model_id: str
    prompt: PromptSpec
    inference: InferenceSpec


@dataclass(frozen=True)
class GenerationSpec:
    mode: str
    background: BackgroundSpec | None = None
    cutter: str = CUTTER_U2NET
    cutter_model_name: str | None = None


@dataclass(frozen=True)
class ClassSpec:
    name: str
    count: int
    prompt: PromptSpec


@dataclass(frozen=True)
class DescriptionSpec:
    enabled: bool = False
    request_file: str | None = None
    max_prompt_rows: int = 50


@dataclass(frozen=True)
class JobConfig:
    version: int
    task: str
    output_dir: Path
    seed: int
    generation: GenerationSpec
    inference: InferenceSpec
    classes: tuple[ClassSpec, ...]
    description: DescriptionSpec
    source_path: Path
    raw_sha256: str

    def class_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.classes)

    def class_counts(self) -> dict[str, int]:
        return {spec.name: spec.count for spec in self.classes}

    def class_spec(self, name: str) -> ClassSpec:
        for spec in self.classes:
            if spec.name == name:
                return spec
        raise KeyError(name)

    def all_prompt_specs(self) -> tuple[tuple[str, PromptSpec], ...]:
        pairs = [(spec.name, spec.prompt) for spec in self.classes]
        background = self.generation.background
        if background is not None:
            pairs.append(("background", background.prompt))
        return tuple(pairs)


@dataclass(frozen=True)
class PlannedOutput:
    class_name: str
    index: int  # 1-based, стабильный
    attempt: int  # 1-based
    filename: str  # "<class>_<index>.png"
    prompt: str  # полностью разрешённый
    negative_prompt: str
    seed: int
    settings: dict  # ровно то, что уйдёт в FactorInferenceParameters (без seed)
    status: str  # "pending" | "complete"
    material: str  # для тестов/отладки

    @property
    def key(self) -> tuple[str, int]:
        return (self.class_name, self.index)


@dataclass(frozen=True)
class Plan:
    job: JobConfig
    model_id: str
    command: str  # "run" | "regenerate"
    outputs: tuple[PlannedOutput, ...]  # только те, что нужно выполнить
    complete: tuple[PlannedOutput, ...]  # уже готовые (для отчёта)


@dataclass(frozen=True)
class RunResult:
    command: str
    planned: int
    succeeded: int
    failed: int
    skipped_complete: int
    rejected_moved: int
    written: bool  # False для dry-run
    exit_code: int
