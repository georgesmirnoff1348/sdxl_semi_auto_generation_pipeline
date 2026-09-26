"""Детерминированный расчёт material-строк, зерна и выбора значений переменных.

Формулы (SALT = "datasetgen/v1")::

    material = "|".join([SALT, job.seed, task, class_name, index, attempt])
    seed     = int.from_bytes(sha256(material)[:4], "big") % 2_147_483_647
    value    = values[name][int.from_bytes(sha256(material + "|" + name)[:8], "big") % len(values)]
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from .schema import SALT

#: Модуль(seed) = 2**31 - 1, диапазон результата 1..2147483646.
SEED_MODULUS = 2_147_483_647

#: Суффикс material-строки слоя инпейнтинга (``two_layer``).
INPAINT_SUFFIX = "inpaint"
#: Суффикс material-строки слоя фонового промпта (``two_layer``).
BACKGROUND_SUFFIX = "bg"


def material(job_seed: int, task: str, class_name: str, index: int, attempt: int) -> str:
    """Стабильная material-строка попытки генерации."""

    return "|".join([SALT, str(job_seed), task, class_name, str(index), str(attempt)])


def derive_seed(material_text: str) -> int:
    """Зерно пайплайна из material-строки."""

    digest = hashlib.sha256(material_text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % SEED_MODULUS


def pick_variable(material_text: str, var_name: str, values: Sequence[str]) -> str:
    """Детерминированный выбор одного значения переменной (обход — по sorted(name))."""

    if not values:
        raise ValueError(f"variable '{var_name}' has no values")
    digest = hashlib.sha256((material_text + "|" + var_name).encode("utf-8")).digest()
    position = int.from_bytes(digest[:8], "big") % len(values)
    return values[position]
