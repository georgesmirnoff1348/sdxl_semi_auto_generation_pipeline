"""Ручной рендер шаблонов промптов и валидация плейсхолдеров.

``str.format`` запрещён: разбор выполняется вручную, чтобы
``{{``/``}}`` давали литеральные скобки, а ``{name}`` — подстановку
детерминированно выбранного значения переменной.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from .errors import ConfigError

#: Допустимое имя переменной/плейсхолдера.
PLACEHOLDER_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")


def extract_placeholders(template: str, label: str = "template") -> tuple[str, ...]:
    """Уникальные имена плейсхолдеров в порядке появления.

    ``{{`` и ``}}`` — экранирование, результат — литеральная скобка.
    """

    names: list[str] = []
    position = 0
    length = len(template)
    while position < length:
        char = template[position]
        if char == "{":
            if template.startswith("{{", position):
                position += 2
                continue
            end = template.find("}", position + 1)
            if end == -1:
                raise ConfigError(f"{label}.template: unterminated placeholder at position {position}")
            name = template[position + 1 : end]
            if not PLACEHOLDER_RE.match(name):
                raise ConfigError(
                    f"{label}.template: invalid placeholder '{{{name}}}' "
                    f"(expected {PLACEHOLDER_RE.pattern})"
                )
            if name not in names:
                names.append(name)
            position = end + 1
            continue
        if char == "}":
            position += 2 if template.startswith("}}", position) else 1
            continue
        position += 1
    return tuple(names)


def render(template: str, values: Mapping[str, str], label: str = "template") -> str:
    """Подставить значения в шаблон (``{{``/``}}`` → ``{``/``}``)."""

    chunks: list[str] = []
    position = 0
    length = len(template)
    while position < length:
        char = template[position]
        if char == "{":
            if template.startswith("{{", position):
                chunks.append("{")
                position += 2
                continue
            end = template.find("}", position + 1)
            if end == -1:
                raise ConfigError(f"{label}.template: unterminated placeholder at position {position}")
            name = template[position + 1 : end]
            if not PLACEHOLDER_RE.match(name):
                raise ConfigError(
                    f"{label}.template: invalid placeholder '{{{name}}}' "
                    f"(expected {PLACEHOLDER_RE.pattern})"
                )
            if name not in values:
                raise ConfigError(f"{label}.template: unknown variable '{name}'")
            chunks.append(str(values[name]))
            position = end + 1
            continue
        if char == "}":
            if template.startswith("}}", position):
                chunks.append("}")
                position += 2
                continue
            chunks.append("}")
            position += 1
            continue
        chunks.append(char)
        position += 1
    return "".join(chunks)


def validate_template(template: str, variables: Mapping[str, Sequence[str]], label: str) -> None:
    """Проверить взаимное соответствие плейсхолдеров шаблона и ключей ``variables``."""

    for name in extract_placeholders(template, label):
        if name not in variables:
            raise ConfigError(f"{label}.template: unknown variable '{name}'")
    for name in variables:
        if name not in extract_placeholders(template, label):
            raise ConfigError(f"{label}.variables: unused variable '{name}'")


def validate_variables(
    variables: Mapping[str, Sequence[str]], label: str
) -> dict[str, tuple[str, ...]]:
    """Проверить значения переменных: непустой список непустых строк без переводов строк."""

    result: dict[str, tuple[str, ...]] = {}
    for name, values in variables.items():
        if not isinstance(values, (list, tuple)):
            raise ConfigError(f"{label}.variables['{name}']: expected a non-empty list of strings")
        if len(values) == 0:
            raise ConfigError(f"{label}.variables['{name}']: must not be an empty list")
        for value in values:
            if not isinstance(value, str):
                raise ConfigError(
                    f"{label}.variables['{name}']: every value must be a string, got {type(value).__name__}"
                )
            if not value.strip():
                raise ConfigError(f"{label}.variables['{name}']: values must not be empty")
            if "\n" in value or "\r" in value:
                raise ConfigError(f"{label}.variables['{name}']: values must not contain newlines")
        result[name] = tuple(values)
    return result
