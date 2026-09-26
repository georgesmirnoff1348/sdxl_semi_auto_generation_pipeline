"""Построение плана выходов. Ни I/O, ни моделей, ни тяжёлых импортов.

Один и тот же код строит план для ``--dry-run`` и для реального запуска.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .determinism import (
    BACKGROUND_SUFFIX,
    INPAINT_SUFFIX,
    derive_seed,
    material,
    pick_variable,
)
from .errors import ConfigError
from .layout import filename_for
from .prompts import render
from .schema import (
    CUTTER_CUSTOM,
    CUTTER_MODEL_NAMES,
    GenerationSpec,
    InferenceSpec,
    JobConfig,
    Plan,
    PlannedOutput,
    PromptSpec,
)

STATUS_PENDING = "pending"
STATUS_COMPLETE = "complete"


def resolve_prompt(spec: PromptSpec, material_text: str) -> str:
    """Разрешить шаблон: переменные обходятся в порядке сортировки имён."""

    values = {
        name: pick_variable(material_text, name, spec.variables[name])
        for name in sorted(spec.variables)
    }
    return render(spec.template, values)


def settings_for(spec: InferenceSpec) -> dict:
    """Ровно те kwargs, которые уйдут в ``FactorInferenceParameters`` (без seed)."""

    return {
        "num_inference_steps": spec.num_inference_steps,
        "guidance_scale": spec.guidance_scale,
        "width": spec.width,
        "height": spec.height,
        "extra": dict(spec.extra),
    }


def make_output(
    job: JobConfig,
    class_name: str,
    index: int,
    attempt: int,
    *,
    status: str = STATUS_PENDING,
) -> PlannedOutput:
    class_spec = job.class_spec(class_name)
    material_text = material(job.seed, job.task, class_name, index, attempt)
    return PlannedOutput(
        class_name=class_name,
        index=index,
        attempt=attempt,
        filename=filename_for(class_name, index),
        prompt=resolve_prompt(class_spec.prompt, material_text),
        negative_prompt=class_spec.prompt.negative_prompt,
        seed=derive_seed(material_text),
        settings=settings_for(job.inference),
        status=status,
        material=material_text,
    )


def all_keys(job: JobConfig) -> tuple[tuple[str, int], ...]:
    """Все пары (class, index) в порядке YAML, затем по возрастанию индекса."""

    counts = job.class_counts()
    return tuple(
        (name, index)
        for name in job.class_names()
        for index in range(1, counts[name] + 1)
    )


def build_plan(
    job: JobConfig,
    model_id: str,
    command: str,
    *,
    max_attempts: Mapping[tuple[str, int], int] | None = None,
    complete_flags: Mapping[tuple[str, int], bool] | None = None,
    selection: Sequence[tuple[str, int]] | None = None,
    skip: Sequence[tuple[str, int]] | None = None,
    planned_attempts: Mapping[tuple[str, int], int] | None = None,
) -> Plan:
    """План выходов.

    ``selection is None`` — режим ``run``: берутся все пары (class, index), кроме
    перечисленных в ``skip`` (отклонённые индексы ``run`` не перегенерирует);
    готовые (``complete_flags``) уходят в :attr:`Plan.complete`.
    ``selection`` задан — режим ``regenerate``: только перечисленные пары.
    ``planned_attempts`` — явные номера попыток (resume незавершённой ``planned``).
    """

    attempts = dict(max_attempts or {})
    flags = dict(complete_flags or {})
    explicit = dict(planned_attempts or {})

    if selection is None:
        skipped = set(skip or ())
        keys: Sequence[tuple[str, int]] = [key for key in all_keys(job) if key not in skipped]
    else:
        keys = tuple(selection)

    outputs: list[PlannedOutput] = []
    complete: list[PlannedOutput] = []
    for class_name, index in keys:
        known = attempts.get((class_name, index), 0)
        if flags.get((class_name, index), False):
            complete.append(
                make_output(job, class_name, index, max(1, known), status=STATUS_COMPLETE)
            )
        elif (class_name, index) in explicit:
            outputs.append(make_output(job, class_name, index, explicit[(class_name, index)]))
        else:
            outputs.append(make_output(job, class_name, index, known + 1))
    return Plan(
        job=job,
        model_id=model_id,
        command=command,
        outputs=tuple(outputs),
        complete=tuple(complete),
    )


def background_material(material_text: str) -> str:
    return material_text + "|" + BACKGROUND_SUFFIX


def inpaint_seed(material_text: str) -> int:
    """Зерно слоя инпейнтинга (``two_layer``)."""

    return derive_seed(material_text + "|" + INPAINT_SUFFIX)


def background_prompt(job: JobConfig, material_text: str) -> str | None:
    """Разрешённый промпт фона для двухслойного режима (переменные — свои)."""

    background = job.generation.background
    if background is None:
        return None
    return resolve_prompt(background.prompt, background_material(material_text))


def cutter_model_name(generation: GenerationSpec) -> str:
    """Имя rembg-модели для ``generation.cutter``."""

    if generation.cutter == CUTTER_CUSTOM:
        if not generation.cutter_model_name:
            raise ConfigError(
                f"generation.cutter: '{CUTTER_CUSTOM}' requires generation.cutter_model_name"
            )
        return generation.cutter_model_name
    return CUTTER_MODEL_NAMES[generation.cutter]
