"""Граница делегирования к существующим legacy-обёрткам репозитория.

Это единственный модуль, который знает про ``cuda_mps_gens`` / ``cutter`` /
``configs``. ВСЕ импорты выполняются ВНУТРИ функций, чтобы ``--dry-run`` и
юнит-тесты не тянули torch/diffusers/rembg.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

from .bootstrap import ensure_legacy_importable
from .errors import DependencyMissingError
from .schema import MODE_SINGLE, MODE_TWO, InferenceSpec


@runtime_checkable
class GeneratorLike(Protocol):
    def generate_image(self, prompts: Any, config: Any, save_path: Any = None) -> Any: ...

    def unload(self) -> None: ...

    def __enter__(self) -> "GeneratorLike": ...

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None: ...


@runtime_checkable
class CutterLike(Protocol):
    def remove_background(self, image: Any, save_path: Any = None) -> Any: ...

    def close(self) -> None: ...

    def __enter__(self) -> "CutterLike": ...

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None: ...


@runtime_checkable
class InpainterLike(Protocol):
    def inpaint_image(
        self, prompts: Any, config: Any, image: Any, mask: Any, save_path: Any = None
    ) -> Any: ...

    def unload(self) -> None: ...

    def __enter__(self) -> "InpainterLike": ...

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None: ...


@runtime_checkable
class Backends(Protocol):
    def make_generator(self, model_id: str) -> GeneratorLike: ...

    def make_cutter(self, model_name: str) -> CutterLike: ...

    def make_inpainter(self, model_id: str) -> InpainterLike: ...


@runtime_checkable
class LegacyTypes(Protocol):
    def prompts(self, prompt: str, negative_prompt: str) -> Any:
        """-> configs.FactorPrompts"""

    def params(self, seed: int, spec: InferenceSpec) -> Any:
        """-> configs.FactorInferenceParameters"""


@dataclass(frozen=True)
class Dependencies:
    backends: Backends
    legacy_types: LegacyTypes
    clock: Callable[[], datetime]


class _LegacyBackends:
    """Фабрики существующих обёрток. Никакого собственного конструирования пайплайнов."""

    def make_generator(self, model_id: str) -> GeneratorLike:
        ensure_legacy_importable()
        from cuda_mps_gens import OrdinaryGen  # noqa: PLC0415 - ленивый импорт по контракту

        return OrdinaryGen(model=model_id)

    def make_cutter(self, model_name: str) -> CutterLike:
        ensure_legacy_importable()
        from cutter import Cutter  # noqa: PLC0415 - ленивый импорт по контракту

        return Cutter(model_name=model_name)

    def make_inpainter(self, model_id: str) -> InpainterLike:
        ensure_legacy_importable()
        from cuda_mps_gens import BackInpainter  # noqa: PLC0415 - ленивый импорт

        return BackInpainter(model_id=model_id)


class _LegacyTypes:
    """Адаптер к dataclass-конфигам репозитория (ленивый импорт ``configs``)."""

    def prompts(self, prompt: str, negative_prompt: str) -> Any:
        from configs import FactorPrompts  # noqa: PLC0415 - ленивый импорт по контракту

        return FactorPrompts(prompt=prompt, negative_prompt=negative_prompt)

    def params(self, seed: int, spec: InferenceSpec) -> Any:
        from configs import FactorInferenceParameters  # noqa: PLC0415 - ленивый импорт

        return FactorInferenceParameters(
            num_inference_steps=spec.num_inference_steps,
            guidance_scale=spec.guidance_scale,
            height=spec.height,
            width=spec.width,
            seed=seed,
            extra=dict(spec.extra),
        )


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def default_dependencies() -> Dependencies:
    return Dependencies(backends=_LegacyBackends(), legacy_types=_LegacyTypes(), clock=utc_now)


#: Реальные зависимости режимов генерации (импортные имена, не имена дистрибутивов).
#:
#: ``single_layer`` — ровно то, что тянут legacy-обёртки: ``cuda_mps_gens.py``
#: импортирует ``torch``, ``diffusers``, ``PIL``, ``numpy``, ``cv2``, а
#: ``diffusers`` дополнительно требует ``transformers`` и ``safetensors`` при
#: импорте. ``two_layer`` добавляет ``rembg`` (``cutter.py``), который в свою
#: очередь работает поверх ``onnxruntime``. Списки совпадают с extras
#: ``single-layer`` / ``two-layer`` из ``pyproject.toml``, поэтому подсказка
#: «install the extra» всегда точна.
RUNTIME_MODULES: dict[str, tuple[str, ...]] = {
    MODE_SINGLE: (
        "PIL",
        "accelerate",
        "cv2",
        "diffusers",
        "numpy",
        "safetensors",
        "torch",
        "transformers",
    ),
    MODE_TWO: (
        "PIL",
        "accelerate",
        "cv2",
        "diffusers",
        "numpy",
        "onnxruntime",
        "rembg",
        "safetensors",
        "torch",
        "transformers",
    ),
}

#: Имя extra, устанавливающего зависимости режима.
EXTRA_FOR_MODE: dict[str, str] = {
    MODE_SINGLE: "single-layer",
    MODE_TWO: "two-layer",
}


def missing_runtime_modules(mode: str) -> tuple[str, ...]:
    """Отсортированные имена top-level модулей режима, которых нет в окружении.

    Используется ``importlib.util.find_spec``: модуль только НАХОДИТСЯ, но не
    импортируется, поэтому preflight не тянет ни тяжёлые зависимости, ни
    legacy-модули. Пустой кортеж — значит, окружение готово к генерации.
    """

    names = RUNTIME_MODULES.get(mode, ())
    return tuple(sorted(name for name in names if importlib.util.find_spec(name) is None))


def ensure_runtime_available(mode: str) -> None:
    """Preflight зависимостей режима до любых записей на диск.

    Бросает :class:`~datasetgen.errors.DependencyMissingError` (exit 9) с
    перечнем отсутствующих модулей и подсказкой, какой extra доустановить.
    """

    missing = missing_runtime_modules(mode)
    if not missing:
        return
    extra = EXTRA_FOR_MODE.get(mode, "single-layer")
    raise DependencyMissingError(
        f"generation.mode: {mode} requires modules that are not installed: "
        f"{', '.join(missing)} (install the extra: pip install 'ai-basic-alina[{extra}]')"
    )
