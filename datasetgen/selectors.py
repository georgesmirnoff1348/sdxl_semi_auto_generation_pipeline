"""Грамматика и разбор селектора ``--remove`` для команды ``regenerate``.

    SELECTOR := SEGMENT (";" SEGMENT)*        # хотя бы один SEGMENT
    SEGMENT  := [CLASS ":"] ITEM ("," ITEM)+  # хотя бы один ITEM
    ITEM     := INT | INT "-" INT              # включительно
    INT      := [0-9]+                          # >= 1
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from .errors import SelectorError
from .schema import TASK_ANOMALY

_INT_RE = re.compile(r"[0-9]+")
_RANGE_RE = re.compile(r"([0-9]+)\s*-\s*([0-9]+)")


def _known_classes_text(class_names: Sequence[str]) -> str:
    return ", ".join(class_names)


def _parse_item(item: str) -> tuple[int, int]:
    """Синтаксис одного ITEM (без знания класса): ``INT`` или ``INT "-" INT``."""

    if not item:
        raise SelectorError("unexpected token '' in --remove")
    if _INT_RE.fullmatch(item):
        number = int(item)
        if number < 1:
            raise SelectorError(f"index {number} is invalid (indices are 1-based)")
        return (number, number)

    match = _RANGE_RE.fullmatch(item)
    if match:
        start, end = int(match.group(1)), int(match.group(2))
        if start < 1 or end < 1:
            raise SelectorError(f"index {min(start, end)} is invalid (indices are 1-based)")
        if end < start:
            raise SelectorError(f"invalid range '{item}' in --remove")
        return (start, end)

    raise SelectorError(f"unexpected token '{item}' in --remove")


def _expand_item(start: int, end: int, class_name: str, counts: Mapping[str, int]) -> list[int]:
    count = counts[class_name]
    if end > count:
        raise SelectorError(
            f"index {end} is out of range for class '{class_name}' (count {count})"
        )
    return list(range(start, end + 1))


def _default_class(class_names: Sequence[str], task: str) -> str:
    if task == TASK_ANOMALY and len(class_names) == 1:
        return class_names[0]
    raise SelectorError(f"class is required in --remove for task '{task}'")


def parse_remove(
    selector: str,
    class_names: Sequence[str],
    counts: Mapping[str, int],
    task: str,
) -> tuple[tuple[str, int], ...]:
    """Разобрать ``--remove`` и вернуть нормализованный список ``(class, index)``.

    Порядок результата: классы в порядке YAML, затем индекс по возрастанию.
    Дубликаты и пересечения диапазонов — ошибка, а не тихий dedupe.
    """

    text = (selector or "").strip()
    if not text:
        raise SelectorError("empty selector")

    order = {name: position for position, name in enumerate(class_names)}
    selected: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()

    for raw_segment in text.split(";"):
        segment = raw_segment.strip()
        if not segment:
            raise SelectorError("unexpected token '' in --remove")

        class_name: str | None = None
        if ":" in segment:
            head, _, tail = segment.partition(":")
            head = head.strip()
            if head:
                if head not in order:
                    raise SelectorError(
                        f"unknown class '{head}' in --remove (known: {_known_classes_text(class_names)})"
                    )
                class_name = head
                segment = tail

        segment = segment.strip()
        if not segment:
            raise SelectorError("unexpected token '' in --remove")

        # синтаксис ITEM проверяется до разрешения класса: сообщения об ошибках
        # не зависят от того, указан ли класс в сегменте
        items = [_parse_item(raw_item.strip()) for raw_item in segment.split(",")]
        if class_name is None:
            class_name = _default_class(class_names, task)

        for start, end in items:
            for number in _expand_item(start, end, class_name, counts):
                key = (class_name, number)
                if key in seen:
                    raise SelectorError(
                        f"duplicate index {number} for class '{class_name}' in --remove"
                    )
                seen.add(key)
                selected.append(key)

    return tuple(sorted(selected, key=lambda key: (order[key[0]], key[1])))
