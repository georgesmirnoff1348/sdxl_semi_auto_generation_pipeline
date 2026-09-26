"""Иерархия ошибок datasetgen и таблица кодов выхода CLI.

Формат вывода ошибки (stderr)::

    datasetgen: error: [CODE] <сообщение>

Предупреждения (stderr, код возврата не меняется)::

    datasetgen: warning: <текст>
"""

from __future__ import annotations

PROGRAM = "datasetgen"

EXIT_OK = 0
EXIT_INTERNAL = 1
EXIT_USAGE = 2
EXIT_JOB_NOT_FOUND = 3
EXIT_CONFIG = 4
EXIT_SELECTOR = 5
EXIT_UNSAFE_PATH = 6
EXIT_GENERATION_FAILED = 7
EXIT_MANIFEST = 8
EXIT_DEPENDENCY_MISSING = 9


class DatasetGenError(Exception):
    """Базовая ошибка пакета: код в сообщении + код выхода процесса."""

    code = "INTERNAL"
    exit_code = EXIT_INTERNAL

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def render(self) -> str:
        return f"{PROGRAM}: error: [{self.code}] {self.message}"


class UsageError(DatasetGenError):
    code = "USAGE"
    exit_code = EXIT_USAGE


class JobNotFoundError(DatasetGenError):
    code = "JOB_NOT_FOUND"
    exit_code = EXIT_JOB_NOT_FOUND


class ConfigError(DatasetGenError):
    code = "CONFIG"
    exit_code = EXIT_CONFIG


class SelectorError(DatasetGenError):
    code = "SELECTOR"
    exit_code = EXIT_SELECTOR


class UnsafePathError(DatasetGenError):
    code = "UNSAFE_PATH"
    exit_code = EXIT_UNSAFE_PATH


class GenerationFailedError(DatasetGenError):
    code = "GENERATION_FAILED"
    exit_code = EXIT_GENERATION_FAILED


class ManifestError(DatasetGenError):
    code = "MANIFEST"
    exit_code = EXIT_MANIFEST


class DependencyMissingError(DatasetGenError):
    """Preflight реальных зависимостей режима генерации не пройден."""

    code = "DEPENDENCY_MISSING"
    exit_code = EXIT_DEPENDENCY_MISSING
