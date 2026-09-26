"""Безопасные пути и проверки файловой системы (правила R1..R6, R10).

Модуль работает в двух режимах (см. :mod:`datasetgen.bootstrap`):

* **source checkout** — ``REPO_ROOT`` это корень репозитория, рядом есть
  ``archive/``, ``gen/``, ``old sh/``, ``examples/`` и сам каталог пакета;
* **установленный wheel** — ``REPO_ROOT`` это каталог установленного пакета в
  ``site-packages``, рядом нет ничего, кроме установленного кода.
"""

from __future__ import annotations

from pathlib import Path

from .bootstrap import IS_SOURCE_CHECKOUT, LEGACY_ROOT, examples_dir, package_dir
from .errors import ConfigError, UnsafePathError
from .schema import CLASS_NAME_PATTERN_TEXT, CLASS_NAME_RE, MANIFEST_NAME, RESERVED_NAMES

__all__ = [
    "PROTECTED_RELATIVE_PATHS",
    "REPO_ROOT",
    "examples_dir",
    "protected_paths",
    "repo_root",
    "safe_join",
    "validate_class_name",
    "validate_output_dir",
]

#: Корень source checkout либо каталог установленного пакета (родитель ``datasetgen``).
REPO_ROOT = LEGACY_ROOT

#: Каталоги репозитория, в которые запрещено писать (R2). Существуют только в checkout.
PROTECTED_RELATIVE_PATHS = ("archive", "gen", "old sh", "examples")


def repo_root() -> Path:
    return REPO_ROOT


def protected_paths() -> tuple[Path, ...]:
    """Абсолютные пути каталогов, в которые запрещено писать (R2).

    В source checkout это четыре каталога репозитория ПЛЮС каталог самого пакета
    ``datasetgen`` — строгое усиление защиты, раньше пакет не был защищён.
    В установленном режиме каталогов репозитория рядом нет, а писать в
    установленный код нельзя никогда, поэтому защищён только каталог пакета.
    """

    if not IS_SOURCE_CHECKOUT:
        return (package_dir(),)
    repository_dirs = tuple((REPO_ROOT / name).resolve() for name in PROTECTED_RELATIVE_PATHS)
    return (*repository_dirs, package_dir())


def _is_ancestor(candidate: Path, other: Path) -> bool:
    return candidate != other and other.is_relative_to(candidate)


def safe_join(root: Path, *parts: str) -> Path:
    """R1: единственная функция построения пути для записи.

    Результат обязан находиться внутри ``root`` после ``resolve()``,
    иначе — ``UnsafePathError`` (exit 6).
    """

    base = Path(root).resolve()
    candidate = base.joinpath(*parts).resolve()
    if candidate != base and not candidate.is_relative_to(base):
        raise UnsafePathError(
            f"refusing to write outside of output_dir: {candidate} is not inside {base}"
        )
    return candidate


def validate_class_name(name: str, label: str = "classes[i].name") -> None:
    """R10 + зарезервированные имена: валидация имени класса."""

    if not isinstance(name, str):
        raise ConfigError(f"{label}: expected a string, got {type(name).__name__}")
    if name in RESERVED_NAMES:
        raise ConfigError(f"{label}: reserved name '{name}'")
    if not CLASS_NAME_RE.match(name):
        raise ConfigError(
            f"{label}: invalid class name '{name}' (expected {CLASS_NAME_PATTERN_TEXT})"
        )


def validate_output_dir(output_dir: Path, job_file: Path) -> Path:
    """R2/R3/R4: проверка ``output_dir`` до любых записей. Возвращает абсолютный путь."""

    resolved = Path(output_dir).expanduser().resolve()
    repo = REPO_ROOT.resolve()

    if resolved.exists() and not resolved.is_dir():
        raise UnsafePathError(f"output_dir is an existing file: {resolved}")
    if resolved == repo:
        if IS_SOURCE_CHECKOUT:
            raise UnsafePathError(f"output_dir must not be the repository root: {resolved}")
        raise UnsafePathError(f"output_dir must not be the installation directory: {resolved}")
    for protected in protected_paths():
        if resolved == protected:
            raise UnsafePathError(f"output_dir must not be a protected path: {resolved}")

    job_resolved = Path(job_file).expanduser().resolve()
    if _is_ancestor(resolved, job_resolved):
        raise UnsafePathError(f"output_dir must not contain the job file: {resolved}")
    if IS_SOURCE_CHECKOUT and _is_ancestor(resolved, repo):
        # Только в checkout: в установленном режиме `repo` — это site-packages, и
        # правило отвергало бы ~/.local, /usr/local, $HOME, / и самый частый случай —
        # каталог проекта, внутри которого создан venv, хотя примеры кладут вывод
        # ровно в ./out/... внутри такого каталога. Роль проверки там компенсирует
        # правило «каталог непустой и без manifest.jsonl» ниже.
        raise UnsafePathError(f"output_dir must not be an ancestor of the repository root: {resolved}")
    for protected in protected_paths():
        if _is_ancestor(resolved, protected):
            raise UnsafePathError(f"output_dir must not contain a protected path: {resolved}")

    if resolved.is_dir() and not (resolved / MANIFEST_NAME).exists():
        if any(resolved.iterdir()):
            raise UnsafePathError(
                f"output_dir is not empty and has no {MANIFEST_NAME}: {resolved}"
            )
    return resolved


