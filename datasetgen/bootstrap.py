"""Двухрежимный shim: единый источник истины о корне и legacy-модулях.

Пакет ``datasetgen`` честно устанавливается через ``pip`` и работает в двух режимах:

* **source checkout** — пакет лежит в дереве репозитория, рядом ``pyproject.toml``,
  legacy-модули (``configs``, ``cuda_mps_gens``, ``cutter``, ``diffusors_core``,
  ``filesystems_core``) и каталоги ``archive``/``gen``/``old sh``/``examples``.
  Сюда же попадает editable-установка: ``__file__`` указывает в исходники.
* **установленный wheel** — пакет лежит в ``site-packages``, legacy-модули
  установлены рядом с ним как package data (см. ``force-include`` в
  ``pyproject.toml``), а каталогов репозитория рядом нет.

``IS_SOURCE_CHECKOUT`` различает режимы по наличию ``pyproject.toml`` рядом с
пакетом: в checkout он есть (в т.ч. при editable-установке), в ``site-packages``
его нет. От этого флага зависят защищённые пути и формулировки ошибок в
:mod:`datasetgen.paths`.

Сам по себе модуль ничего не импортирует и не создаёт побочных эффектов.
"""

from __future__ import annotations

import sys
from pathlib import Path

#: Каталог самого пакета ``datasetgen`` (checkout или ``site-packages``).
PACKAGE_DIR = Path(__file__).resolve().parent

#: Родитель ``PACKAGE_DIR``: корень checkout либо каталог установленного пакета.
LEGACY_ROOT = PACKAGE_DIR.parent

#: ``True`` только в source checkout: рядом с пакетом лежит ``pyproject.toml``.
IS_SOURCE_CHECKOUT: bool = (LEGACY_ROOT / "pyproject.toml").is_file()


def package_dir() -> Path:
    """Каталог пакета ``datasetgen``."""

    return PACKAGE_DIR


def repository_root() -> Path:
    """Корень source checkout либо каталог установленного пакета (родитель ``datasetgen``)."""

    return LEGACY_ROOT


def examples_dir() -> Path:
    """Каталог с примерами YAML-а задач.

    В checkout это ``<корень>/examples``; в установленном режиме те же файлы
    приезжают как package data в ``datasetgen/examples``.
    """

    return LEGACY_ROOT / "examples" if IS_SOURCE_CHECKOUT else PACKAGE_DIR / "examples"


def ensure_legacy_importable() -> None:
    """Идемпотентно добавить каталог legacy-модулей в ``sys.path``.

    Логика одинакова в обоих режимах и менять её не нужно: в ``site-packages``
    этот каталог уже находится в ``sys.path``, поэтому append не происходит; в
    source checkout это единственное место, которое делает ``import configs``
    (и остальные legacy-модули) детерминированным.
    """

    root = str(repository_root())
    if root not in sys.path:
        sys.path.append(root)
