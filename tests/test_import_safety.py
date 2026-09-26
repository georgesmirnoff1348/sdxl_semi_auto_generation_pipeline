"""TC-66..TC-70, TC-89, TC-90: инварианты безопасности (нет удалений, лёгкий импорт)."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = REPO_ROOT / "datasetgen"
FORBIDDEN_CALLS = {"rmtree", "remove", "unlink", "rmdir", "removedirs", "removedirs"}

FORBIDDEN_TOKENS = (
    "from_pretrained",
    "StableDiffusion",
    "DPMSolver",
    "new_session",
    "torch.",
    "AutoencoderKL",
    "ControlNetModel",
)

FORBIDDEN_IMPORTS = ("main",)


def module_files() -> list[Path]:
    return sorted(PACKAGE.glob("*.py"))


def read_module(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class TestSourceInvariants(unittest.TestCase):
    """TC-66, TC-69, TC-70: AST/grep-инварианты исходников пакета."""

    def test_tc66_no_deletion_calls(self) -> None:
        offenders: list[str] = []
        for path in module_files():
            tree = ast.parse(read_module(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func = node.func
                    name = None
                    if isinstance(func, ast.Attribute):
                        name = func.attr
                    elif isinstance(func, ast.Name):
                        name = func.id
                    if name in FORBIDDEN_CALLS:
                        offenders.append(f"{path.name}:{node.lineno} -> {name}")
        self.assertEqual(offenders, [], offenders)

    #: Атрибуты, которые действительно меняют состояние файловой системы.
    MUTATING_METHODS = {
        "replace",
        "rename",
        "truncate",
        "mkdir",
        "write_text",
        "write_bytes",
        "open",
        "touch",
        "symlink_to",
        "hardlink_to",
        "chmod",
        "chown",
    }

    def test_only_expected_filesystem_mutations(self) -> None:
        allowed = {"replace", "rename", "truncate", "mkdir", "write_text", "write_bytes", "open"}
        used = set()
        for path in module_files():
            tree = ast.parse(read_module(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    used.add(node.func.attr)
        self.assertTrue({"replace", "truncate"} <= used)
        self.assertFalse(used & FORBIDDEN_CALLS, used & FORBIDDEN_CALLS)
        # каждая мутация ФС в пакете обязана быть в белом списке
        unexpected = used & self.MUTATING_METHODS - allowed
        self.assertEqual(unexpected, set(), f"непредвиденные мутации ФС: {sorted(unexpected)}")

    def test_tc69_no_pipeline_construction(self) -> None:
        for path in module_files():
            text = read_module(path)
            for token in FORBIDDEN_TOKENS:
                self.assertNotIn(token, text, f"{path.name} содержит {token!r}")

    def test_tc69_legacy_imports_are_lazy(self) -> None:
        for path in module_files():
            tree = ast.parse(read_module(path))
            for node in tree.body:  # только верхний уровень модуля
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertNotIn(alias.name, FORBIDDEN_IMPORTS, f"{path.name}: {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    self.assertNotIn(node.module, FORBIDDEN_IMPORTS, f"{path.name}: {node.module}")

    def test_tc70_no_main_import_and_no_subprocess(self) -> None:
        for path in module_files():
            tree = ast.parse(read_module(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertNotIn(alias.name, ("subprocess", "main"), f"{path.name}: {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    self.assertNotIn(node.module, ("subprocess", "main"), f"{path.name}: {node.module}")
                elif isinstance(node, ast.Attribute) and node.attr in ("system", "popen", "run"):
                    self.assertNotEqual(
                        getattr(node.value, "attr", getattr(node.value, "id", "")), "os",
                        f"{path.name}: os.{node.attr}",
                    )

    def test_heavy_modules_are_not_imported_by_engines(self) -> None:
        """engines.py — единственная точка legacy-импортов, и только внутри функций."""

        tree = ast.parse(read_module(PACKAGE / "engines.py"))
        legacy = {"cuda_mps_gens", "cutter", "configs"}
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                name = node.module if isinstance(node, ast.ImportFrom) else None
                if name is None:
                    for alias in node.names:
                        self.assertNotIn(alias.name, legacy)
                else:
                    self.assertNotIn(name, legacy)


class TestImportSafety(unittest.TestCase):
    """TC-67, TC-68: ``--dry-run`` и импорт CLI не тянут тяжёлые модули."""

    HEAVY = "{'torch','diffusers','rembg','cutter','cuda_mps_gens','configs','diffusors_core'}"

    def run_python(self, code: str, **kwargs) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            **kwargs,
        )

    def test_tc67_importing_cli_runner_planner(self) -> None:
        code = (
            "import datasetgen.cli, datasetgen.runner, datasetgen.planner, sys;"
            f"assert not {self.HEAVY} & set(sys.modules), sorted({self.HEAVY} & set(sys.modules));"
            "print('ok')"
        )
        result = self.run_python(code)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ok", result.stdout)

    def test_tc68_dry_run_two_layer_does_not_import_heavy_modules(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            job_path = Path(folder) / "job.yaml"
            job_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "task": "classification",
                        "output_dir": str(Path(folder) / "out"),
                        "seed": 20260926,
                        "generation": {
                            "mode": "two_layer",
                            "cutter": "birefnet",
                            "cutter_model_name": None,
                            "background": {
                                "model_id": "test/background",
                                "template": "background of {place}",
                                "negative_prompt": "people",
                                "variables": {"place": ["a hall"]},
                                "inference": {"extra": {"inner_pad": 20}},
                            },
                        },
                        "classes": [
                            {
                                "name": "class_a",
                                "count": 1,
                                "template": "a {worker}",
                                "negative_prompt": "",
                                "variables": {"worker": ["factory worker"]},
                            },
                            {
                                "name": "class_b",
                                "count": 1,
                                "template": "a {worker} with {anomaly}",
                                "negative_prompt": "",
                                "variables": {"worker": ["engineer"], "anomaly": ["crust"]},
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            code = (
                "import sys;"
                "from datasetgen.cli import main;"
                f"code = main(['run', {str(job_path)!r}, '--model-id', 'test/model', '--dry-run']);"
                "assert code == 0, code;"
                f"assert not {self.HEAVY} & set(sys.modules), sorted({self.HEAVY} & set(sys.modules));"
                "print('ok')"
            )
            result = self.run_python(code)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("ok", result.stdout)
            self.assertFalse((Path(folder) / "out").exists())


class TestBootstrap(unittest.TestCase):
    """TC-89: shim идемпотентен."""

    def test_tc89_ensure_legacy_importable_is_idempotent(self) -> None:
        """Shim добавляет корень legacy-модулей не более одного раза и только в конец.

        Проверка не зависит от состояния окружения. В реальном прогоне корень
        репозитория лежит в ``sys.path`` дважды и без участия shim: его добавляет
        ``_editable_impl_*.pth`` editable-установки, а ``python -m unittest``
        кладёт абсолютный рабочий каталог в ``sys.path[0]``. Поэтому «сколько
        всего вхождений корня» — характеристика окружения, а не поведения shim;
        здесь изолированно проверяется именно поведение: сколько записей shim
        добавил, куда, и что повторные вызовы не меняют ничего.
        """
        from datasetgen.bootstrap import ensure_legacy_importable, repository_root

        root = str(repository_root())

        def shim_call_on(entries: list[str]) -> list[str]:
            """Вызвать shim на песочнице и вернуть получившийся ``sys.path``."""
            saved = list(sys.path)
            sys.path[:] = entries
            try:
                ensure_legacy_importable()
                return list(sys.path)
            finally:
                sys.path[:] = saved

        # 1. Реальное окружение: сколько бы раз ни звали shim, новых записей нет.
        before = list(sys.path)
        ensure_legacy_importable()
        after_first = list(sys.path)
        ensure_legacy_importable()
        ensure_legacy_importable()
        self.assertEqual(after_first, sys.path)
        self.assertEqual(before, after_first[: len(before)])
        self.assertLessEqual(len(after_first) - len(before), 1)
        self.assertIn(root, after_first)
        self.assertEqual(sys.path.count(root), after_first.count(root))

        # 2. Чистое окружение (корня в sys.path нет): ровно одна новая запись, строго в конце.
        clean = ["", "/python313.zip", "/site-packages"]
        self.assertEqual(shim_call_on(clean), [*clean, root])

        # 3. Ключевая регрессия: повторные вызовы на уже дополненном пути — no-op.
        primed = [*clean, root]
        self.assertEqual(shim_call_on(primed), primed)
        self.assertEqual(shim_call_on(primed), primed)

        # 4. Корень уже в sys.path (editable-режим/колесо) — не дублируется и не переставляется.
        preloaded = ["/site-packages", root, ""]
        self.assertEqual(shim_call_on(preloaded), preloaded)
        self.assertEqual(shim_call_on(preloaded).count(root), 1)

        # 5. Песочница не оставила следов в реальном sys.path.
        self.assertEqual(list(sys.path), after_first)


class TestStructure(unittest.TestCase):
    """TC-90: компиляция и структура пакета."""

    EXPECTED = {
        "__init__.py",
        "__main__.py",
        "bootstrap.py",
        "cli.py",
        "config.py",
        "description.py",
        "determinism.py",
        "engines.py",
        "errors.py",
        "layout.py",
        "manifest.py",
        "paths.py",
        "planner.py",
        "prompts.py",
        "runner.py",
        "schema.py",
        "selectors.py",
    }

    def test_module_tree(self) -> None:
        found = {path.name for path in module_files()}
        self.assertEqual(found, self.EXPECTED)

    def test_tc90_compileall(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "compileall", "-q", "datasetgen"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_runner_does_not_import_engines_at_module_level(self) -> None:
        tree = ast.parse(read_module(PACKAGE / "runner.py"))
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module == "engines":
                self.fail("runner.py must not import engines at module level")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertNotEqual(alias.name, "engines")

    def test_entry_point_is_declared(self) -> None:
        text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('datasetgen = "datasetgen.cli:main"', text)
        self.assertIn("pyyaml", text)
        self.assertIn('packages = ["datasetgen"]', text)

    #: legacy-модули, которые обязаны ехать в колесо рядом с пакетом.
    LEGACY_MODULES = (
        "configs.py",
        "cutter.py",
        "cuda_mps_gens.py",
        "diffusors_core.py",
        "filesystems_core.py",
    )

    def test_build_config_ships_legacy_modules_and_extras(self) -> None:
        """Колесо везёт legacy-модули и примеры, тяжёлое живёт в extras."""

        with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
            config = tomllib.load(handle)

        wheel = config["tool"]["hatch"]["build"]["targets"]["wheel"]
        force = wheel["force-include"]

        for legacy in self.LEGACY_MODULES:
            self.assertEqual(force.get(legacy), legacy, legacy)
        self.assertEqual(
            force.get("examples/classification.yaml"), "datasetgen/examples/classification.yaml"
        )
        self.assertEqual(force.get("examples/anomaly.yaml"), "datasetgen/examples/anomaly.yaml")
        for source in force:
            self.assertTrue((REPO_ROOT / source).is_file(), f"force-include: {source}")

        # main.py не должен попадать в колесо ни одним из способов
        self.assertNotIn("main.py", force)
        self.assertEqual(wheel["packages"], ["datasetgen"])

        project = config["project"]
        self.assertEqual(project["dependencies"], ["pyyaml"])
        extras = project["optional-dependencies"]
        self.assertIn("single-layer", extras)
        self.assertIn("two-layer", extras)
        single = " ".join(extras["single-layer"])
        two = " ".join(extras["two-layer"])
        for heavy in ("onnxruntime", "rembg"):
            self.assertIn(heavy, two)
            self.assertNotIn(heavy, single)
            self.assertNotIn(heavy, " ".join(project["dependencies"]))

    def test_gitignore_has_single_out_entry(self) -> None:
        lines = [
            line.strip()
            for line in (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(lines.count("/out/"), 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
