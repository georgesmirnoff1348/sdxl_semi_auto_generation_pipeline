"""Единственный потребитель ``yaml``: чтение файла задачи и строгая валидация.

Неизвестные ключи — ошибка ``CONFIG`` (ловит опечатки). Все сообщения об
ошибках имеют вид ``<путь-поля>: <проблема>``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

import yaml

from .errors import ConfigError, JobNotFoundError
from .paths import validate_class_name
from .prompts import validate_template, validate_variables
from .schema import (
    CLASS_NAME_PATTERN_TEXT,
    CUTTER_CUSTOM,
    CUTTER_U2NET,
    CUTTERS,
    MODE_SINGLE,
    MODE_TWO,
    MODES,
    TASK_ANOMALY,
    TASK_CLASSIFICATION,
    TASKS,
    VARIABLE_NAME_RE,
    BackgroundSpec,
    ClassSpec,
    DescriptionSpec,
    GenerationSpec,
    InferenceSpec,
    JobConfig,
    PromptSpec,
)

ROOT_KEYS = ("version", "task", "output_dir", "seed", "dataset_description", "generation", "inference", "classes")
DESCRIPTION_KEYS = ("enabled", "request_file", "max_prompt_rows")
GENERATION_KEYS = ("mode", "cutter", "cutter_model_name", "background")
BACKGROUND_KEYS = ("model_id", "template", "negative_prompt", "variables", "inference")
INFERENCE_KEYS = ("num_inference_steps", "guidance_scale", "width", "height", "extra")
PROMPT_KEYS = ("template", "negative_prompt", "variables")
CLASS_KEYS = ("name", "count", "template", "negative_prompt", "variables")

WARN_UNUSED_TWO_LAYER = "unused two_layer block"


def _type_name(value: object) -> str:
    return {
        dict: "a mapping",
        list: "a list",
        str: "a string",
        bool: "a boolean",
        int: "an integer",
        float: "a float",
        type(None): "null",
    }.get(type(value), type(value).__name__)


def _require_mapping(value: object, label: str) -> dict:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{label}: expected a mapping, got {_type_name(value)}")
    return dict(value)


def _check_keys(mapping: Mapping, allowed: tuple[str, ...], label: str) -> None:
    unknown = [key for key in mapping if key not in allowed]
    if unknown:
        first = sorted(unknown, key=lambda key: str(key))[0]
        raise ConfigError(
            f"unknown key in {label}: '{first}' (allowed: {', '.join(sorted(allowed))})"
        )


def _require_int(value: object, label: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{label}: expected an integer, got {_type_name(value)}")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{label}: expected an integer >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{label}: expected an integer <= {maximum}, got {value}")
    return value


def _require_float(value: object, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{label}: expected a number, got {_type_name(value)}")
    number = float(value)
    if minimum is not None and number < minimum:
        raise ConfigError(f"{label}: expected a number >= {minimum}, got {number}")
    return number


def _require_str(value: object, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{label}: expected a string, got {_type_name(value)}")
    if not allow_empty and not value.strip():
        raise ConfigError(f"{label}: expected a non-empty string")
    return value


def _require_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{label}: expected a boolean, got {_type_name(value)}")
    return value


def _parse_inference(data: object, label: str) -> InferenceSpec:
    if data is None:
        return InferenceSpec()
    mapping = _require_mapping(data, label)
    _check_keys(mapping, INFERENCE_KEYS, label)
    steps = _require_int(mapping.get("num_inference_steps", 20), f"{label}.num_inference_steps", minimum=1)
    guidance = _require_float(mapping.get("guidance_scale", 7.0), f"{label}.guidance_scale", minimum=0.0)
    width = _require_int(mapping.get("width", 1024), f"{label}.width", minimum=64, maximum=2048)
    if width % 8:
        raise ConfigError(f"{label}.width: expected a multiple of 8, got {width}")
    height = _require_int(mapping.get("height", 1024), f"{label}.height", minimum=64, maximum=2048)
    if height % 8:
        raise ConfigError(f"{label}.height: expected a multiple of 8, got {height}")
    extra = mapping.get("extra")
    extra_mapping = {} if extra is None else _require_mapping(extra, f"{label}.extra")
    return InferenceSpec(steps, guidance, width, height, dict(extra_mapping))


def _parse_prompt(data: object, label: str, *, check_keys: bool = True) -> PromptSpec:
    mapping = _require_mapping(data, label)
    if check_keys:
        _check_keys(mapping, PROMPT_KEYS, label)
    if "template" not in mapping:
        raise ConfigError(f"{label}.template: required key is missing")
    template = _require_str(mapping["template"], f"{label}.template")
    negative = _require_str(
        mapping.get("negative_prompt", ""), f"{label}.negative_prompt", allow_empty=True
    )
    if "variables" not in mapping:
        raise ConfigError(f"{label}.variables: required key is missing")
    raw_variables = _require_mapping(mapping["variables"], f"{label}.variables")
    if not raw_variables:
        raise ConfigError(f"{label}.variables: must not be empty")
    for name in raw_variables:
        if not VARIABLE_NAME_RE.match(name):
            raise ConfigError(
                f"{label}.variables: invalid variable name '{name}' (expected ^[a-z][a-z0-9_]{{0,31}}$)"
            )
    variables = validate_variables(raw_variables, label)
    validate_template(template, variables, label)
    return PromptSpec(template, negative, variables)


def _parse_class(data: object, position: int, seen: set[str]) -> ClassSpec:
    label = f"classes[{position}]"
    mapping = _require_mapping(data, label)
    _check_keys(mapping, CLASS_KEYS, label)
    if "name" not in mapping:
        raise ConfigError(f"{label}.name: required key is missing")
    name = mapping["name"]
    validate_class_name(name, f"{label}.name")
    if name in seen:
        raise ConfigError(f"{label}.name: duplicate class name '{name}'")
    seen.add(name)
    if "count" not in mapping:
        raise ConfigError(f"{label}.count: required key is missing")
    count = _require_int(mapping["count"], f"{label}.count", minimum=1)
    prompt = _parse_prompt(mapping, label, check_keys=False)
    return ClassSpec(name, count, prompt)


def _parse_background(data: object) -> BackgroundSpec:
    label = "generation.background"
    mapping = _require_mapping(data, label)
    _check_keys(mapping, BACKGROUND_KEYS, label)
    if "model_id" not in mapping:
        raise ConfigError(f"{label}.model_id: required key is missing")
    model_id = _require_str(mapping["model_id"], f"{label}.model_id")
    prompt = _parse_prompt(mapping, label, check_keys=False)
    inference = _parse_inference(mapping.get("inference"), f"{label}.inference")
    return BackgroundSpec(model_id, prompt, inference)


def _parse_generation(data: object, warnings: list[str]) -> GenerationSpec:
    label = "generation"
    mapping = _require_mapping(data, label)
    _check_keys(mapping, GENERATION_KEYS, label)
    if "mode" not in mapping:
        raise ConfigError(f"{label}.mode: required key is missing")
    mode = mapping["mode"]
    if not isinstance(mode, str) or mode not in MODES:
        raise ConfigError(f"{label}.mode: expected one of {', '.join(MODES)}, got {mode!r}")
    cutter = mapping.get("cutter", CUTTER_U2NET)
    if not isinstance(cutter, str) or cutter not in CUTTERS:
        raise ConfigError(f"{label}.cutter: expected one of {', '.join(CUTTERS)}, got {cutter!r}")
    cutter_model_name = mapping.get("cutter_model_name")
    if cutter_model_name is not None:
        cutter_model_name = _require_str(cutter_model_name, f"{label}.cutter_model_name")

    background = None
    if mapping.get("background") is not None:
        background = _parse_background(mapping["background"])

    if mode == MODE_TWO:
        if background is None:
            raise ConfigError(
                "generation.background: required when generation.mode is 'two_layer'"
            )
        if cutter == CUTTER_CUSTOM and not cutter_model_name:
            raise ConfigError(
                f"{label}.cutter: '{CUTTER_CUSTOM}' requires generation.cutter_model_name"
            )
    elif (
        background is not None
        or cutter != CUTTER_U2NET
        or cutter_model_name is not None
    ):
        warnings.append(
            f"{WARN_UNUSED_TWO_LAYER}: generation.background, generation.cutter and "
            "generation.cutter_model_name are ignored when generation.mode is 'single_layer'"
        )
    return GenerationSpec(mode, background, cutter, cutter_model_name)


def _parse_description(data: object, warnings: list[str]) -> DescriptionSpec:
    label = "dataset_description"
    if data is None:
        return DescriptionSpec()
    mapping = _require_mapping(data, label)
    _check_keys(mapping, DESCRIPTION_KEYS, label)
    enabled = _require_bool(mapping.get("enabled", False), f"{label}.enabled")
    request_file = mapping.get("request_file")
    if request_file is not None:
        request_file = _require_str(request_file, f"{label}.request_file")
    max_prompt_rows = _require_int(
        mapping.get("max_prompt_rows", 50), f"{label}.max_prompt_rows", minimum=1
    )
    return DescriptionSpec(enabled, request_file, max_prompt_rows)


def _read_source(path: Path) -> tuple[str, str]:
    if not path.exists():
        raise JobNotFoundError(f"job file not found: {path}")
    if not path.is_file():
        raise JobNotFoundError(f"job file is not a regular file: {path}")
    payload = path.read_bytes()
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ConfigError(f"job file: not valid UTF-8 ({exc.reason})") from exc
    return text, hashlib.sha256(payload).hexdigest()


def load_yaml(path: str | Path, *, warnings: list[str] | None = None, cwd: Path | None = None) -> JobConfig:
    """Прочитать YAML-задачу, провалидировать и вернуть :class:`JobConfig`."""

    collected: list[str] = [] if warnings is None else warnings
    source = Path(path).expanduser()
    source = source if source.is_absolute() else (Path(cwd) / source if cwd else Path.cwd() / source)
    source = source.resolve()
    text, digest = _read_source(source)

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"job file: invalid YAML ({exc})") from exc
    if data is None:
        raise ConfigError("root: expected a mapping, got null")
    root = _require_mapping(data, "root")
    _check_keys(root, ROOT_KEYS, "root")

    if "version" not in root:
        raise ConfigError("version: required key is missing")
    version = _require_int(root["version"], "version")
    if version != 1:
        raise ConfigError(f"version: expected 1, got {version}")

    if "task" not in root:
        raise ConfigError("task: required key is missing")
    task = root["task"]
    if not isinstance(task, str) or task not in TASKS:
        raise ConfigError(f"task: expected one of {', '.join(TASKS)}, got {task!r}")

    if "output_dir" not in root:
        raise ConfigError("output_dir: required key is missing")
    raw_output_dir = _require_str(root["output_dir"], "output_dir")
    base = Path(cwd) if cwd else Path.cwd()
    output_dir = Path(raw_output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = base / output_dir

    if "seed" not in root:
        raise ConfigError("seed: required key is missing")
    seed = _require_int(root["seed"], "seed", minimum=0)

    description = _parse_description(root.get("dataset_description"), collected)
    if description.request_file is not None and not Path(description.request_file).is_file():
        collected.append(
            f"dataset_description.request_file: '{description.request_file}' is not a readable file"
        )

    if "generation" not in root:
        raise ConfigError("generation: required key is missing")
    generation = _parse_generation(root["generation"], collected)

    inference = _parse_inference(root.get("inference"), "inference")

    if "classes" not in root:
        raise ConfigError("classes: required key is missing")
    raw_classes = root["classes"]
    if not isinstance(raw_classes, list):
        raise ConfigError(f"classes: expected a list, got {_type_name(raw_classes)}")

    seen: set[str] = set()
    classes = tuple(
        _parse_class(item, position, seen) for position, item in enumerate(raw_classes)
    )
    if task == TASK_CLASSIFICATION and len(classes) != 2:
        raise ConfigError(
            f"task '{task}' requires exactly 2 classes, got {len(classes)}"
        )
    if task == TASK_ANOMALY and (len(classes) != 1 or classes[0].name != "normal"):
        raise ConfigError(
            f"task '{task}' requires exactly 1 class named 'normal', got {len(classes)}"
        )

    return JobConfig(
        version=version,
        task=task,
        output_dir=output_dir,
        seed=seed,
        generation=generation,
        inference=inference,
        classes=classes,
        description=description,
        source_path=source,
        raw_sha256=digest,
    )


__all__ = [
    "CLASS_NAME_PATTERN_TEXT",
    "MODE_SINGLE",
    "MODE_TWO",
    "WARN_UNUSED_TWO_LAYER",
    "load_yaml",
]
