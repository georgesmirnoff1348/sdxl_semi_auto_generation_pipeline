#!/usr/bin/env python3
"""Воспроизводимая проверка: ``pip install <wheel>`` даёт рабочий пакет.

Скрипт доказывает, что пакет не «работает только из checkout»:

1. собирает wheel из исходников во временный каталог (``uv build``);
2. проверяет СОДЕРЖИМОЕ колеса через ``zipfile`` (без установки);
3. создаёт чистый venv и ставит туда только базовую зависимость ``pyyaml``
   (никаких extras);
4. запускает консольный скрипт с ``cwd`` вне репозитория и с вычищенным
   окружением (``PYTHONPATH``/``VIRTUAL_ENV`` удалены, ``PYTHONNOUSERSITE=1``);
5. проверяет, что legacy-модули (``configs``, ``cutter``, ``cuda_mps_gens``,
   ``diffusors_core``, ``filesystems_core``) резолвятся ИЗ УСТАНОВЛЕННОГО пакета;
6. проверяет preflight зависимостей режима ``two_layer`` (exit 9).

Скрипт использует ТОЛЬКО стандартную библиотеку плюс внешний ``uv``
вызывается через ``subprocess``. Репозиторий не изменяется: все файлы
создаются во временном каталоге, который удаляется на выходе.

Запуск::

    uv run --locked python tools/verify_wheel.py

Полезные флаги (по умолчанию всё автоматически, без аргументов):

* ``--keep-temporary`` — не удалять временный каталог (отладка);
* ``--python`` — версия интерпретатора для venv (по умолчанию ``3.13``);
* ``--source-root`` — корень исходников для сборки (по умолчанию корень
  репозитория; позволяет собрать «сломанный» wheel из копии репозитория,
  не трогая рабочее дерево);
* ``--timeout`` — таймаут одной внешней команды в секундах.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

#: Корень репозитория (каталог с ``pyproject.toml``).
REPO_ROOT = Path(__file__).resolve().parent.parent

#: Ожидаемое имя каталога метаданных в колесе.
DIST_INFO = "ai_basic_alina-0.1.0.dist-info"

#: Обязательные файлы колеса: пакет, примеры и legacy-модули.
REQUIRED_ENTRIES: tuple[str, ...] = (
    "datasetgen/__init__.py",
    "datasetgen/cli.py",
    "datasetgen/bootstrap.py",
    "datasetgen/paths.py",
    "datasetgen/engines.py",
    "datasetgen/examples/classification.yaml",
    "datasetgen/examples/anomaly.yaml",
    "configs.py",
    "cutter.py",
    "cuda_mps_gens.py",
    "diffusors_core.py",
    "filesystems_core.py",
    f"{DIST_INFO}/entry_points.txt",
    f"{DIST_INFO}/METADATA",
)

#: Точные имена, которых в колесе быть НЕ должно.
FORBIDDEN_EXACT: tuple[str, ...] = ("main.py",)

#: Префиксы путей, которых в колесе быть НЕ должно (мусор из checkout).
FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "tests/",
    "archive/",
    "gen/",
    "old sh/",
    "examples/",
    "dist/",
    "build/",
    ".venv/",
)

#: Legacy-модули, необходимые для старта реального запуска.
LEGACY_MODULES: tuple[str, ...] = (
    "configs",
    "cutter",
    "cuda_mps_gens",
    "diffusors_core",
    "filesystems_core",
)

#: Тяжёлые зависимости extras: их НЕ должно быть в базовой установке.
HEAVY_MODULES: tuple[str, ...] = ("torch", "diffusers", "rembg", "onnxruntime")

#: Дистрибутивы, обязательные экстрами: в METADATA только с маркером ``extra ==``.
EXTRA_ONLY_REQUIREMENTS: tuple[str, ...] = (
    "torch",
    "diffusers",
    "rembg",
    "onnxruntime",
    "opencv-python",
)

#: Ожидаемая строка консольного скрипта в ``entry_points.txt``.
ENTRY_POINT_LINE = "datasetgen = datasetgen.cli:main"

#: Ожидаемая версия в ``--version``.
EXPECTED_VERSION = "datasetgen 0.1.0"

#: Ожидаемый код выхода preflight-зависимостей.
EXIT_DEPENDENCY_MISSING = 9

#: Минимальная задача ``generation.mode: two_layer``, собираемая САМОСТОЯТЕЛЬНО
#: (не копируется из репозитория): так проверка дополнительно доказывает, что
#: пакет самодостаточен и не нуждается в ``examples/`` рядом с исходниками.
TWO_LAYER_JOB = """\
version: 1
task: anomaly_detection
output_dir: ./out/two_layer
seed: 20260926

dataset_description:
  enabled: false
  request_file: null
  max_prompt_rows: 50

generation:
  mode: two_layer
  cutter: u2net
  cutter_model_name: null
  background:
    model_id: test/background-model
    template: "background photography of {place}, 1980s, casual photo"
    negative_prompt: "people, illustration, human, outdoor, text"
    variables:
      place: [hospital room with painted wall, industrial hall]
    inference:
      num_inference_steps: 20
      guidance_scale: 7.0
      width: 1024
      height: 1024
      extra:
        inner_pad: 20
        strength: 0.9

inference:
  num_inference_steps: 20
  guidance_scale: 7.0
  width: 1024
  height: 1024
  extra: {}

classes:
  # task: anomaly_detection требует ровно один класс с именем `normal`.
  - name: normal
    count: 2
    template: "a {worker} wearing {uniform}, studio photography"
    negative_prompt: "3d render, anime, blurry, text, watermark"
    variables:
      worker: [factory worker, industrial engineer]
      uniform: [complete protective equipment, helmet and high-visibility workwear]
"""

#: Сниппет для ``find_spec`` — модуль НАХОДИТСЯ, но не импортируется.
FIND_SPEC_SNIPPET = """
import importlib.util, json, sys


def probe(name):
    try:
        spec = importlib.util.find_spec(name)
    except Exception as exc:  # noqa: BLE001 - диагностика
        return {"origin": None, "error": type(exc).__name__ + ": " + str(exc)}
    return {"origin": None if spec is None else spec.origin}


print(json.dumps({name: probe(name) for name in sys.argv[1:]}))
"""

#: Сниппет для отчёта о каталоге примеров установленного пакета.
EXAMPLES_SNIPPET = """
import json

from datasetgen import paths

directory = paths.examples_dir()
print(
    json.dumps(
        {
            "path": str(directory),
            "is_dir": directory.is_dir(),
            "files": sorted(entry.name for entry in directory.iterdir())
            if directory.is_dir()
            else [],
        }
    )
)
"""


class VerificationError(Exception):
    """Проверка не прошла: печатается как ``FAIL: ...`` и даёт код 1."""


def ok(message: str) -> None:
    """Отметить успешно выполненную проверку."""

    print(f"OK: {message}", flush=True)


def expect(condition: bool, expected: str, actual: object = None) -> None:
    """``OK: expected`` либо ``FAIL`` с фактическим значением."""

    if condition:
        ok(expected)
    else:
        raise VerificationError(f"{expected} (actual: {actual!r})")


def _describe(command: list[str], proc: subprocess.CompletedProcess) -> str:
    return (
        f"$ {' '.join(command)}\n"
        f"exit={proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout}\n"
        f"--- stderr ---\n{proc.stderr}"
    )


def run(
    command: list[str],
    *,
    cwd: Path,
    timeout: int,
    step: str,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """Запустить внешнюю команду и вернуть результат (без проверки кода)."""

    try:
        return subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:  # нет uv / нет интерпретатора
        raise VerificationError(f"{step}: executable not found: {command[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise VerificationError(f"{step}: timed out after {timeout}s") from exc


def require_success(
    command: list[str], proc: subprocess.CompletedProcess, step: str
) -> None:
    """Провалиться с полным выводом команды, если код возврата ненулевой."""

    if proc.returncode != 0:
        print(_describe(command, proc), file=sys.stderr, flush=True)
        raise VerificationError(f"{step}: exit code {proc.returncode}, expected 0")


def clean_env() -> dict[str, str]:
    """Окружение ребёнка: без ``PYTHONPATH``/``VIRTUAL_ENV``, без user-site.

    Это главная защита от «работает из checkout»: ни ``PYTHONPATH``, ни
    editable-установка не должны подмешивать исходники репозитория.
    """

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("VIRTUAL_ENV", None)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def venv_bin(venv: Path) -> Path:
    """Каталог исполняемых файлов venv (``bin`` на POSIX, ``Scripts`` на Windows)."""

    return venv / ("Scripts" if os.name == "nt" else "bin")


def build_wheel(source_root: Path, dist_dir: Path, timeout: int) -> Path:
    """Собрать wheel через ``uv build`` и вернуть путь к единственному ``.whl``."""

    build = run(
        ["uv", "build", "--wheel", "--out-dir", str(dist_dir)],
        cwd=source_root,
        timeout=timeout,
        step="build wheel",
    )
    if build.returncode != 0:
        print(_describe(["uv", "build", "--wheel"], build), file=sys.stderr, flush=True)
        raise VerificationError(
            "uv build failed, so the wheel was not produced; this is a BUILD "
            "problem, not a wheel defect. The build backend (hatchling) is pinned "
            "in pyproject.toml but is not part of uv.lock, so the isolated build "
            "environment is still downloaded from PyPI on first use: re-run with "
            "access to PyPI (no network / offline / air-gapped machine is the "
            "usual cause). Alternatively pre-populate the uv cache."
        )
    wheels = sorted(dist_dir.glob("*.whl"))
    if not wheels:
        raise VerificationError(f"uv build produced no *.whl in {dist_dir}")
    if len(wheels) > 1:
        raise VerificationError(
            f"uv build produced {len(wheels)} wheels, expected exactly 1: "
            f"{[wheel.name for wheel in wheels]}"
        )
    return wheels[0]


def check_zip_names(names: list[str]) -> None:
    """Структура архива: нет мусора из checkout, абсолютных путей и ``..``."""

    present = set(names)
    for entry in REQUIRED_ENTRIES:
        expect(entry in present, f"wheel contains {entry}", f"missing; got {sorted(present)}")

    for entry in FORBIDDEN_EXACT:
        expect(entry not in present, f"wheel does not contain {entry}", sorted(present))

    offending = [name for name in names if name.startswith(FORBIDDEN_PREFIXES)]
    expect(
        not offending,
        "wheel has no checkout leftovers "
        f"({', '.join(FORBIDDEN_PREFIXES)})",
        offending,
    )

    absolute = [
        name
        for name in names
        if name.startswith("/") or re.match(r"^[A-Za-z]:[\\/]", name) or ".." in name
    ]
    expect(
        not absolute,
        "wheel entry names are relative and contain no '..'",
        absolute,
    )


def requirement_name(line: str) -> str:
    """Имя дистрибутива из строки ``Requires-Dist`` (без маркера и extras)."""

    body = line.split(":", 1)[1].strip() if ":" in line else line.strip()
    head = body.split(";", 1)[0].strip()
    head = head.split("[", 1)[0].strip()
    match = re.match(r"^[A-Za-z0-9._-]+", head)
    return match.group(0).lower() if match else ""


def requirement_marker(line: str) -> str:
    """Часть строки ``Requires-Dist`` после ``;`` (пустая строки — нет маркера)."""

    body = line.split(":", 1)[1].strip() if ":" in line else line.strip()
    parts = body.split(";", 1)
    return parts[1].strip() if len(parts) == 2 else ""


def check_metadata(metadata: str) -> None:
    """``METADATA``: ``Requires-Python``, extras и разделение base/extra-зависимостей."""

    expect(
        "Requires-Python: >=3.12" in metadata,
        "METADATA declares 'Requires-Python: >=3.12'",
        [line for line in metadata.splitlines() if line.startswith("Requires-Python")],
    )
    for extra in ("single-layer", "two-layer"):
        expect(
            f"Provides-Extra: {extra}" in metadata,
            f"METADATA declares 'Provides-Extra: {extra}'",
            [line for line in metadata.splitlines() if line.startswith("Provides-Extra")],
        )

    requires = [line for line in metadata.splitlines() if line.startswith("Requires-Dist:")]
    expect(bool(requires), "METADATA contains Requires-Dist lines", requires)

    plain_pyyaml = [
        line
        for line in requires
        if requirement_name(line) == "pyyaml" and "extra ==" not in requirement_marker(line)
    ]
    expect(
        bool(plain_pyyaml),
        "METADATA requires 'pyyaml' without an 'extra ==' marker",
        requires,
    )

    for name in EXTRA_ONLY_REQUIREMENTS:
        matching = [line for line in requires if requirement_name(line) == name]
        expect(
            bool(matching),
            f"METADATA requires '{name}' (via extras)",
            requires,
        )
        without_marker = [
            line for line in matching if "extra ==" not in requirement_marker(line)
        ]
        expect(
            not without_marker,
            f"METADATA requires '{name}' only with an 'extra ==' marker "
            "(base install must stay lightweight)",
            without_marker,
        )


def check_wheel_content(wheel: Path) -> None:
    """Проверить содержимое колеса через ``zipfile``, ничего не устанавливая."""

    with zipfile.ZipFile(wheel) as archive:
        names = sorted(archive.namelist())
        check_zip_names(names)
        entry_points = archive.read(f"{DIST_INFO}/entry_points.txt").decode("utf-8")
        metadata = archive.read(f"{DIST_INFO}/METADATA").decode("utf-8", "replace")

    expect(
        ENTRY_POINT_LINE in entry_points,
        f"entry_points.txt contains '{ENTRY_POINT_LINE}'",
        entry_points,
    )
    check_metadata(metadata)
    ok(f"wheel content verified without installation ({len(names)} entries)")


def python_json(
    python: Path,
    snippet: str,
    work: Path,
    env: dict[str, str],
    timeout: int,
    arguments: tuple[str, ...] = (),
) -> tuple[dict, subprocess.CompletedProcess]:
    """Выполнить сниппет на venv-интерпретаторе и разобрать JSON из stdout."""

    command = [str(python), "-c", snippet, *arguments]
    proc = run(command, cwd=work, env=env, timeout=timeout, step="python snippet")
    require_success(command, proc, "python snippet")
    try:
        return json.loads(proc.stdout.strip()), proc
    except json.JSONDecodeError as exc:
        raise VerificationError(
            f"python snippet did not print valid JSON: {exc} ({proc.stdout!r})"
        ) from exc


def check_installed(site_packages: Path) -> None:
    """Проверить, что у чистого venv вообще есть ``site-packages``."""

    expect(
        site_packages.is_dir(),
        f"venv has site-packages at {site_packages}",
        "directory does not exist",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="verify_wheel.py",
        description="Prove that 'pip install <wheel>' yields a working datasetgen.",
    )
    parser.add_argument(
        "--keep-temporary",
        action="store_true",
        help="do not delete the temporary workspace (debugging)",
    )
    parser.add_argument(
        "--python",
        default="3.13",
        help="interpreter version for the clean venv (default: 3.13)",
    )
    parser.add_argument(
        "--source-root",
        default=str(REPO_ROOT),
        help="source tree to build from (default: repository root)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="per-command timeout in seconds (default: 1800)",
    )
    args = parser.parse_args(argv)

    source_root = Path(args.source_root).resolve()
    if not (source_root / "pyproject.toml").is_file():
        print(
            f"FAIL: --source-root {source_root} does not contain pyproject.toml",
            flush=True,
        )
        return 1
    if shutil.which("uv") is None:
        print("FAIL: 'uv' was not found on PATH; it is required to build and install", flush=True)
        return 1

    temporary: tempfile.TemporaryDirectory | None = None
    if args.keep_temporary:
        workspace = Path(tempfile.mkdtemp(prefix="datasetgen-wheel-"))
    else:
        temporary = tempfile.TemporaryDirectory(prefix="datasetgen-wheel-")
        workspace = Path(temporary.name)
    dist_dir = workspace / "dist"
    venv = workspace / "venv"
    work = workspace / "work"
    for directory in (dist_dir, work):
        directory.mkdir(parents=True, exist_ok=True)
    env = clean_env()

    try:
        wheel = build_wheel(source_root, dist_dir, args.timeout)
        ok(f"built wheel {wheel.name} into {dist_dir}")

        check_wheel_content(wheel)

        create = run(
            ["uv", "venv", "--python", args.python, str(venv)],
            cwd=workspace,
            timeout=args.timeout,
            step="create venv",
        )
        require_success(["uv", "venv"], create, "create venv")
        python = venv_bin(venv) / ("python.exe" if os.name == "nt" else "python")
        expect(python.is_file(), f"clean venv has an interpreter at {python}", "missing")
        ok(f"created clean venv {venv} (python {args.python})")

        install = run(
            ["uv", "pip", "install", "--python", str(python), str(wheel)],
            cwd=workspace,
            timeout=args.timeout,
            step="install wheel",
        )
        require_success(["uv", "pip", "install"], install, "install wheel")
        ok("installed the wheel into the clean venv (base install, no extras)")

        # ------------------------------------------------------- smoke: --help
        cli = venv_bin(venv) / ("datasetgen.exe" if os.name == "nt" else "datasetgen")
        expect(cli.is_file(), f"console script installed at {cli}", "missing")

        help_proc = run([str(cli), "--help"], cwd=work, env=env, timeout=args.timeout, step="--help")
        require_success([str(cli), "--help"], help_proc, "datasetgen --help")
        for marker in ("run", "regenerate", "DEPENDENCY_MISSING", "extras:"):
            expect(
                marker in help_proc.stdout,
                f"'datasetgen --help' mentions '{marker}'",
                help_proc.stdout,
            )

        version_proc = run(
            [str(cli), "--version"], cwd=work, env=env, timeout=args.timeout, step="--version"
        )
        require_success([str(cli), "--version"], version_proc, "datasetgen --version")
        expect(
            EXPECTED_VERSION in version_proc.stdout,
            f"'datasetgen --version' prints '{EXPECTED_VERSION}'",
            version_proc.stdout,
        )

        purelib_proc = run(
            [str(python), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
            cwd=work,
            env=env,
            timeout=args.timeout,
            step="locate site-packages",
        )
        require_success([str(python), "-c", "..."], purelib_proc, "locate site-packages")
        # resolve(): на macOS /var -> /private/var, а __file__ уже разыменован.
        site_packages = Path(purelib_proc.stdout.strip()).resolve()
        check_installed(site_packages)

        # --------------------------------------- two modes: installed vs checkout
        origin_proc = run(
            [str(python), "-c", "import datasetgen; print(datasetgen.__file__)"],
            cwd=work,
            env=env,
            timeout=args.timeout,
            step="locate datasetgen",
        )
        require_success([str(python), "-c", "..."], origin_proc, "locate datasetgen")
        origin = Path(origin_proc.stdout.strip()).resolve()
        expect(
            origin.is_relative_to(site_packages),
            f"datasetgen is imported from site-packages ({origin})",
            f"{origin} is not inside {site_packages}",
        )
        expect(
            not origin.is_relative_to(source_root),
            f"datasetgen is NOT imported from the repository ({source_root})",
            origin,
        )

        flags_proc = run(
            [str(python), "-c", "from datasetgen import paths; print(paths.IS_SOURCE_CHECKOUT)"],
            cwd=work,
            env=env,
            timeout=args.timeout,
            step="read IS_SOURCE_CHECKOUT",
        )
        require_success([str(python), "-c", "..."], flags_proc, "read IS_SOURCE_CHECKOUT")
        expect(
            flags_proc.stdout.strip() == "False",
            "IS_SOURCE_CHECKOUT is False in the installed venv",
            flags_proc.stdout.strip(),
        )

        examples, _ = python_json(python, EXAMPLES_SNIPPET, work, env, args.timeout)
        expect(
            examples["is_dir"],
            f"paths.examples_dir() exists at {examples['path']}",
            examples,
        )
        for name in ("classification.yaml", "anomaly.yaml"):
            expect(
                name in examples["files"],
                f"examples_dir() contains {name}",
                examples["files"],
            )
        examples_dir = Path(examples["path"])

        # ------------------------------------------------------------- dry run
        dry = run(
            [
                str(cli),
                "run",
                str(examples_dir / "classification.yaml"),
                "--model-id",
                "test/model",
                "--dry-run",
            ],
            cwd=work,
            env=env,
            timeout=args.timeout,
            step="dry-run",
        )
        require_success([str(cli), "run", "...", "--dry-run"], dry, "datasetgen run --dry-run")
        expect("plan:" in dry.stdout, "dry-run prints the plan", dry.stdout)
        expect("dry-run" in dry.stdout, "dry-run announces 'dry-run'", dry.stdout)
        expect(
            not (work / "out").exists(),
            "dry-run did not create out/",
            sorted(str(entry) for entry in work.iterdir()),
        )

        # ------------------------------------------- import resolution: legacy
        legacy, _ = python_json(
            python, FIND_SPEC_SNIPPET, work, env, args.timeout, LEGACY_MODULES
        )
        for name in LEGACY_MODULES:
            origin_value = legacy[name].get("origin")
            expect(
                bool(origin_value),
                f"legacy module '{name}' resolves in the installed venv",
                legacy[name],
            )
            resolved = Path(origin_value).resolve()
            expect(
                resolved.is_relative_to(site_packages),
                f"legacy module '{name}' is importable from site-packages ({resolved})",
                f"{resolved} is not inside {site_packages}",
            )
            expect(
                not resolved.is_relative_to(source_root),
                f"legacy module '{name}' does NOT come from the repository",
                resolved,
            )
        ok("legacy modules resolve from the installed package, not from the checkout")

        # ------------------------------------------ base install stays lightweight
        heavy, _ = python_json(
            python, FIND_SPEC_SNIPPET, work, env, args.timeout, HEAVY_MODULES
        )
        for name in HEAVY_MODULES:
            expect(
                heavy[name].get("origin") is None,
                f"'{name}' is absent in the base install (find_spec -> None)",
                heavy[name],
            )

        # ------------------------------------------------- preflight exit code 9
        job_file = work / "job.yaml"
        job_file.write_text(TWO_LAYER_JOB, encoding="utf-8")
        preflight = run(
            [str(cli), "run", "job.yaml", "--model-id", "test/model"],
            cwd=work,
            env=env,
            timeout=args.timeout,
            step="two_layer preflight",
        )
        expect(
            preflight.returncode == EXIT_DEPENDENCY_MISSING,
            f"two_layer run without extras exits with {EXIT_DEPENDENCY_MISSING}",
            f"exit={preflight.returncode}; stderr={preflight.stderr!r}",
        )
        expect(
            "datasetgen: error: [DEPENDENCY_MISSING]" in preflight.stderr,
            "preflight reports '[DEPENDENCY_MISSING]' on stderr",
            preflight.stderr,
        )
        expect(
            "onnxruntime" in preflight.stderr,
            "preflight names the missing module 'onnxruntime'",
            preflight.stderr,
        )
        expect(
            not (work / "out").exists(),
            "preflight failure did not create the output directory",
            sorted(str(entry) for entry in work.iterdir()),
        )

        print("OK: wheel verified", flush=True)
        return 0
    except VerificationError as exc:
        print(f"FAIL: {exc}", flush=True)
        return 1
    finally:
        if temporary is not None:
            temporary.cleanup()
        else:
            print(f"temporary workspace kept at {workspace}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
