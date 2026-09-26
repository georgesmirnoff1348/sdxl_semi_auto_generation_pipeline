"""Раскладка файлов внутри ``output_dir``.

Все пути строятся через :func:`datasetgen.paths.safe_join` (R1).
"""

from __future__ import annotations

from pathlib import Path

from .paths import safe_join
from .schema import (
    DESCRIPTION_NAME,
    DESCRIPTION_REQUEST_NAME,
    INTERMEDIATE_DIR_NAME,
    MANIFEST_NAME,
    REJECTED_DIR_NAME,
)

SUBJECT_SUFFIX = "_subject.png"
CUT_SUFFIX = "_cut.png"


def filename_for(class_name: str, index: int) -> str:
    """R10: имя файла выхода — из валидированного имени класса и целого индекса."""

    return f"{class_name}_{index}.png"


def class_dir(root: Path, class_name: str) -> Path:
    return safe_join(root, class_name)


def final_image_path(root: Path, class_name: str, index: int) -> Path:
    return safe_join(root, class_name, filename_for(class_name, index))


def rejected_dir(root: Path) -> Path:
    return safe_join(root, REJECTED_DIR_NAME)


def rejected_path(root: Path, class_name: str, index: int, attempt: int) -> Path:
    return safe_join(root, REJECTED_DIR_NAME, f"{class_name}_{index}_{attempt}.png")


def intermediate_dir(root: Path, class_name: str) -> Path:
    return safe_join(root, INTERMEDIATE_DIR_NAME, class_name)


def intermediate_path(root: Path, class_name: str, index: int, attempt: int, kind: str) -> Path:
    stem = f"{class_name}_{index}_{attempt}"
    suffix = SUBJECT_SUFFIX if kind == "subject" else CUT_SUFFIX
    return safe_join(root, INTERMEDIATE_DIR_NAME, class_name, stem + suffix)


def manifest_path(root: Path) -> Path:
    return safe_join(root, MANIFEST_NAME)


def description_path(root: Path) -> Path:
    return safe_join(root, DESCRIPTION_NAME)


def description_request_path(root: Path) -> Path:
    return safe_join(root, DESCRIPTION_REQUEST_NAME)


def relative_posix(root: Path, path: Path) -> str:
    """Путь относительно ``output_dir`` в POSIX-виде (никогда не абсолютный)."""

    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
