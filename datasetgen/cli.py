"""CLI ``datasetgen``: argparse, коды выхода, рендер ошибок и предупреждений.

Модуль не импортирует тяжёлые зависимости: реальные обёртки создаются
только внутри :mod:`datasetgen.engines` в момент фактической генерации.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .config import load_yaml
from .errors import EXIT_INTERNAL, EXIT_OK, EXIT_USAGE, DatasetGenError
from .runner import regenerate, run_job
from .selectors import parse_remove

PROGRAM = "datasetgen"

DRY_RUN_NOTE = (
    "--dry-run does not construct pipelines, download models, generate images, or require a GPU"
)

USAGE_EXAMPLES = f"""\
examples:
  uv run --locked datasetgen --help
  uv run --locked datasetgen --version

  # classification: безопасно посмотреть план (ничего не пишется, GPU не нужен)
  uv run --locked datasetgen run examples/classification.yaml \\
      --model-id stabilityai/stable-diffusion-xl-base-1.0 --dry-run

  # реальная генерация на GPU (Kaggle/Colab)
  uv run --locked datasetgen run examples/classification.yaml \\
      --model-id stabilityai/stable-diffusion-xl-base-1.0

  # anomaly detection: resume — команда повторяется, готовые файлы не трогаются
  uv run --locked datasetgen run examples/anomaly.yaml \\
      --model-id stabilityai/stable-diffusion-xl-base-1.0

  # перегенерация: один класс + диапазон
  uv run --locked datasetgen regenerate examples/classification.yaml \\
      --model-id stabilityai/stable-diffusion-xl-base-1.0 \\
      --remove "class_a:3,7-9;class_b:4"

  # перегенерация: без класса (для task 'anomaly_detection' это класс 'normal')
  uv run --locked datasetgen regenerate examples/anomaly.yaml \\
      --model-id stabilityai/stable-diffusion-xl-base-1.0 \\
      --remove "3,7-9"

  # другой каталог вывода
  uv run --locked datasetgen run examples/classification.yaml \\
      --model-id stabilityai/stable-diffusion-xl-base-1.0 --output-dir ./out/try2

{DRY_RUN_NOTE}.
"""

COMMAND_HELP = {
    "run": "Сгенерировать все недостающие выходы (resume: готовое не перезаписывается).",
    "regenerate": "Отклонить выбранные индексы (файлы перемещаются в _rejected/) и перегенерировать их.",
}

EPILOG_EXIT_CODES = """\
exit codes:
  0 OK (включая --dry-run и «всё уже готово»)   2 USAGE (argparse)
  1 INTERNAL   3 JOB_NOT_FOUND   4 CONFIG   5 SELECTOR
  6 UNSAFE_PATH   7 GENERATION_FAILED   8 MANIFEST   9 DEPENDENCY_MISSING

extras:
  pip install 'ai-basic-alina[single-layer]'   # torch/diffusers/cv2: generation.mode: single_layer
  pip install 'ai-basic-alina[two-layer]'      # + rembg/onnxruntime: generation.mode: two_layer
  локальное колесо: pip install 'dist/<wheel-file>[two-layer]'
  #   (имя файла даёт `uv build --wheel --out-dir dist`; подставьте то, что он напечатал)

output layout (output_dir):
  <class>/<class>_<index>.png            финальные изображения
  _rejected/<class>_<index>_<attempt>.png  отклонённые (перемещены, не удалены)
  _intermediate/<class>/...             только для generation.mode: two_layer
  manifest.jsonl                         append-only журнал попыток
  dataset_description.md                 отчёт (dataset_description.enabled: true)
  dataset_description_request.md         заготовка для внешнего агента
"""


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("job", metavar="JOB.yaml", help="путь к YAML-задаче (относительно CWD)")
    parser.add_argument(
        "--model-id",
        required=True,
        metavar="DIFFUSERS_MODEL_ID",
        help="id модели для OrdinaryGen(model=...) и запись в манифест (в YAML не пишется)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "напечатать план и выйти: ничего не пишется, пайплайны не строятся, "
            "модели не скачиваются, GPU не нужен"
        ),
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        default=None,
        help="переопределить output_dir из YAML (абсолютный или относительный CWD)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROGRAM,
        description="datasetgen — генерация синтетических CV-датасетов из одного YAML.",
        epilog=USAGE_EXAMPLES + "\n" + EPILOG_EXIT_CODES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=f"{PROGRAM} {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command", metavar="{run,regenerate}")

    run_parser = subparsers.add_parser(
        "run",
        help=COMMAND_HELP["run"],
        description=COMMAND_HELP["run"],
        epilog=USAGE_EXAMPLES + "\n" + EPILOG_EXIT_CODES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common_arguments(run_parser)

    regenerate_parser = subparsers.add_parser(
        "regenerate",
        help=COMMAND_HELP["regenerate"],
        description=(
            COMMAND_HELP["regenerate"]
            + "\n\n--remove SELECTOR: SEGMENT (';' SEGMENT)*, "
            "SEGMENT := [CLASS ':'] INT (',' INT | ',' INT '-' INT)+"
        ),
        epilog=USAGE_EXAMPLES + "\n" + EPILOG_EXIT_CODES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common_arguments(regenerate_parser)
    regenerate_parser.add_argument(
        "--remove",
        required=True,
        metavar="SELECTOR",
        help='индексы к перегенерации, напр. "class_a:3,7-9;class_b:4" или "3,7-9"',
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    deps: Any = None,
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    """Точка входа. Возвращает код выхода (0 OK, 1 INTERNAL, 2 USAGE, ...)."""

    out = stdout if stdout is not None else sys.stdout
    err = stderr if stderr is not None else sys.stderr
    parser = build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:  # --help / --version / usage-ошибки argparse
        return EXIT_OK if exc.code is None else int(exc.code)

    if not getattr(args, "command", None):
        parser.print_help(out)
        return EXIT_USAGE

    def warn(message: str) -> None:
        err.write(f"{PROGRAM}: warning: {message}\n")

    def report_error(message: str) -> None:
        err.write(message + "\n")

    try:
        # Признак реальных бэкендов: в тестах фейки передаются явно (deps != None).
        real_backends = deps is None
        if deps is None:
            from .engines import default_dependencies

            deps = default_dependencies()

        warnings: list[str] = []
        job = load_yaml(args.job, warnings=warnings)
        for message in warnings:
            warn(message)

        # Preflight реальных зависимостей режима: до любых записей на диск и до
        # legacy-импортов. --dry-run честно остаётся «ничего не требует».
        if real_backends and not args.dry_run:
            from .engines import ensure_runtime_available

            ensure_runtime_available(job.generation.mode)

        output_dir = Path(args.output_dir) if args.output_dir else None
        if args.command == "run":
            result = run_job(
                job,
                args.model_id,
                output_dir=output_dir,
                dry_run=bool(args.dry_run),
                deps=deps,
                out=out,
                warn=warn,
                error=report_error,
            )
        else:
            selection = parse_remove(
                args.remove, job.class_names(), job.class_counts(), job.task
            )
            result = regenerate(
                job,
                args.model_id,
                selection,
                output_dir=output_dir,
                dry_run=bool(args.dry_run),
                deps=deps,
                out=out,
                warn=warn,
                error=report_error,
            )
        return result.exit_code
    except DatasetGenError as exc:
        err.write(exc.render() + "\n")
        return exc.exit_code
    except KeyboardInterrupt:  # pragma: no cover - интерактивное прерывание
        err.write(f"{PROGRAM}: error: [INTERNAL] interrupted by user\n")
        return EXIT_INTERNAL
    except Exception as exc:  # noqa: BLE001 - INTERNAL + traceback
        traceback.print_exc(file=err)
        err.write(f"{PROGRAM}: error: [INTERNAL] {type(exc).__name__}: {exc}\n")
        return EXIT_INTERNAL


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
