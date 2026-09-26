"""Append-only журнал ``manifest.jsonl``: запись, чтение, самовосстановление хвоста.

Правила:
* записи только дописываются (никаких перезаписей, никаких удалений);
* единственная мутация существующего файла — усечение неполного хвоста (R9);
* битые строки пропускаются с предупреждением и никогда не «чинятся»;
* если файл непуст, но не дал ни одной валидной записи — ошибка ``MANIFEST``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from .errors import ManifestError
from .schema import STATUS_PLANNED, STATUS_REJECTED, STATUS_SUCCEEDED

SCHEMA_VERSION = 1
RECORD_TYPES = ("run", "image")
REALIZATION_STATUSES = (STATUS_PLANNED, STATUS_SUCCEEDED, STATUS_REJECTED)

Key = tuple[str, int]


def _format_record(record: Mapping) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_valid_record(parsed: object) -> bool:
    """Годна ли запись: известный ``type`` + обязательные поля верного типа.

    Валидный JSON с ``"index": "1"`` — это битая строка (её пропускаем с
    предупреждением), а не повод ронять прогон через ``int()``.
    """

    if not isinstance(parsed, dict):
        return False
    kind = parsed.get("type")
    if kind not in RECORD_TYPES:
        return False
    if kind == "run":
        return True
    return (
        isinstance(parsed.get("class"), str)
        and _is_int(parsed.get("index"))
        and _is_int(parsed.get("attempt"))
    )


@dataclass(frozen=True)
class ManifestState:
    """Эффективное состояние датасета по последним ``image``-записям."""

    last: Mapping[Key, Mapping] = field(default_factory=dict)
    max_attempt: Mapping[Key, int] = field(default_factory=dict)
    records: Mapping[Key, tuple[Mapping, ...]] = field(default_factory=dict)

    def last_record(self, key: Key) -> Mapping | None:
        return self.last.get(key)

    def attempts(self, key: Key) -> int:
        """Максимальный известный attempt для (class, index); 0 — записей не было."""

        return self.max_attempt.get(key, 0)

    def realizations(self, key: Key) -> tuple[Mapping, ...]:
        """Все записи, фиксирующие реализацию (planned/succeeded/rejected), по порядку.

        Guard сверяет новую пару ``(seed, prompt)`` со **всеми** прошлыми
        реализациями, а не только с последней: номер попытки входит в material,
        поэтому совпадение почти невозможно, но инвариант «никогда не
        воспроизводим отклонённое» должен быть исполнимым, а не вероятностным.
        """

        return tuple(
            record
            for record in self.records.get(key, ())
            if record.get("status") in REALIZATION_STATUSES
        )

    def owns_output(self, key: Key, relative: str) -> bool:
        """R5: манифест управляет этим путём, если на него ссылается хоть одна запись.

        Так остатки упавшей попытки (0 байт / частичный файл) считаются своими,
        а не чужими, и не блокируют батч.
        """

        return any(record.get("output") == relative for record in self.records.get(key, ()))

    def has_succeeded(self, key: Key) -> bool:
        record = self.last.get(key)
        return bool(record) and record.get("status") == STATUS_SUCCEEDED

    def is_rejected(self, key: Key) -> bool:
        record = self.last.get(key)
        return bool(record) and record.get("status") == STATUS_REJECTED

    def planned_record(self, key: Key, attempt: int) -> Mapping | None:
        """Последняя planned-запись ровно этой попытки (для проверки R5)."""

        found: Mapping | None = None
        for record in self.records.get(key, ()):
            if record.get("status") == STATUS_PLANNED and int(record.get("attempt", 0)) == attempt:
                found = record
        return found

    def rejected_count(self) -> int:
        return sum(1 for record in self.last.values() if record.get("status") == STATUS_REJECTED)

    def produced_count(self) -> int:
        return sum(1 for record in self.last.values() if record.get("status") == STATUS_SUCCEEDED)


class Manifest:
    """Обёртка над ``manifest.jsonl`` (append-only)."""

    def __init__(
        self,
        path: Path,
        records: tuple[Mapping, ...] = (),
        corrupt_lines: tuple[int, ...] = (),
        truncated_tail: bool = False,
    ) -> None:
        self.path = Path(path)
        self.records: list[Mapping] = list(records)
        self.corrupt_lines: tuple[int, ...] = tuple(corrupt_lines)
        self.truncated_tail = truncated_tail

    # ---------------------------------------------------------------- загрузка
    @classmethod
    def load(cls, path: str | Path, warn: Callable[[str], None] | None = None) -> "Manifest":
        """Прочитать манифест. Отсутствующий/пустой файл — чистый старт."""

        target = Path(path)
        if not target.exists():
            return cls(target)
        payload = target.read_bytes()
        if not payload.strip():
            return cls(target)
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ManifestError(f"manifest is not valid JSONL: {target} (line 1)") from exc

        # R9: неполный (без завершающего \n) хвост записи не считается.
        truncated_tail = not text.endswith("\n")
        body = text[: text.rfind("\n") + 1] if truncated_tail else text

        records: list[Mapping] = []
        corrupt: list[int] = []
        for number, line in enumerate(body.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                corrupt.append(number)
                continue
            if not _is_valid_record(parsed):
                corrupt.append(number)
                continue
            records.append(parsed)

        if corrupt and warn is not None:
            for number in corrupt:
                warn(f"manifest: skipping corrupt line {number}")
        if not records:
            if truncated_tail and not corrupt:
                # R9: весь файл — одна неполная строка; repair_truncated_tail()
                # обрежет её целиком, то есть это чистый старт, а не ошибка.
                return cls(target, (), (), True)
            raise ManifestError(
                f"manifest is not valid JSONL: {target} (line {corrupt[0] if corrupt else 1})"
            )
        return cls(target, tuple(records), tuple(corrupt), truncated_tail)

    # ------------------------------------------------------------- восстановление
    def repair_truncated_tail(self) -> bool:
        """R9: обрезать файл до последнего полного ``\\n`` (единственная мутация)."""

        if not self.truncated_tail:
            return False
        payload = self.path.read_bytes()
        keep = payload.rfind(b"\n") + 1
        with open(self.path, "r+b") as handle:
            handle.truncate(keep)
        self.truncated_tail = False
        return True

    # ------------------------------------------------------------------ запись
    def append(self, record: Mapping) -> Mapping:
        """Дописать одну запись: одна строка + ``flush`` + ``fsync``."""

        payload = _format_record(record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(payload + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        stored = dict(record)
        self.records.append(stored)
        return stored

    # ------------------------------------------------------------------ доступ
    def run_records(self) -> tuple[Mapping, ...]:
        return tuple(record for record in self.records if record.get("type") == "run")

    def image_records(self) -> tuple[Mapping, ...]:
        return tuple(record for record in self.records if record.get("type") == "image")

    def last_run_record(self) -> Mapping | None:
        runs = self.run_records()
        return runs[-1] if runs else None

    def state(self) -> ManifestState:
        last: dict[Key, Mapping] = {}
        max_attempt: dict[Key, int] = {}
        by_key: dict[Key, list[Mapping]] = {}
        for record in self.image_records():
            key = (str(record.get("class", "")), int(record.get("index", 0)))
            last[key] = record
            attempt = int(record.get("attempt", 1))
            if attempt > max_attempt.get(key, 0):
                max_attempt[key] = attempt
            by_key.setdefault(key, []).append(record)
        return ManifestState(
            last=last,
            max_attempt=max_attempt,
            records={key: tuple(items) for key, items in by_key.items()},
        )
