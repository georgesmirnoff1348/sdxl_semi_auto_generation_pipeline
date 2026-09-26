"""TC-32, TC-56, TC-79..TC-85: CLI (коды выхода, help, полный прогон на фейках)."""

from __future__ import annotations

import contextlib
import io
import json
import unittest
from pathlib import Path
from unittest import mock

from datasetgen import cli
from datasetgen.errors import (
    EXIT_CONFIG,
    EXIT_DEPENDENCY_MISSING,
    EXIT_GENERATION_FAILED,
    EXIT_JOB_NOT_FOUND,
    EXIT_OK,
    EXIT_SELECTOR,
    EXIT_UNSAFE_PATH,
    EXIT_USAGE,
)
from tests.fakes import MODEL_ID, FailingStage, TmpCase, fake_find_spec, two_layer_generation

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPO_ROOT / "examples"


class CliTestCase(TmpCase):
    def call(self, argv, deps=None):
        """Вызывает CLI, захватывая и argparse-вывод (help/usage идёт в sys.*)."""

        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(list(argv), deps=deps, stdout=out, stderr=err)
        return code, out.getvalue() + err.getvalue(), err.getvalue()


class TestHelpAndVersion(CliTestCase):
    def test_tc79_help(self) -> None:
        code, out, _ = self.call(["--help"])
        self.assertEqual(code, EXIT_OK)
        for needle in (
            "run",
            "regenerate",
            "--model-id",
            "--dry-run",
            "--remove",
            "--output-dir",
            "--version",
            "examples/classification.yaml",
            'class_a:3,7-9;class_b:4',
            '3,7-9',
            "--dry-run does not construct pipelines, download models, generate images, or require a GPU",
        ):
            self.assertIn(needle, out)

    def test_help_keeps_wheel_instructions_version_agnostic(self) -> None:
        """Epilog не зашивает имя колеса: версия и платформа меняются, инструкция — нет."""

        code, out, _ = self.call(["--help"])
        self.assertEqual(code, EXIT_OK)
        self.assertNotIn("0.1.0", out)
        self.assertNotIn(".whl", out)
        self.assertIn("dist/<wheel-file>[two-layer]", out)

    def test_tc80_version(self) -> None:
        code, out, _ = self.call(["--version"])
        self.assertEqual(code, EXIT_OK)
        self.assertEqual(out.strip(), "datasetgen 0.1.0")

    def test_tc81_unknown_command_and_flag(self) -> None:
        self.assertEqual(self.call(["bogus"])[0], EXIT_USAGE)
        self.assertEqual(self.call(["run", "--nope"])[0], EXIT_USAGE)

    def test_no_command_prints_help(self) -> None:
        code, out, _ = self.call([])
        self.assertEqual(code, EXIT_USAGE)
        self.assertIn("usage:", out)

    def test_tc32_dry_run_requires_model_id(self) -> None:
        path = self.job_path(count=1)
        code, _, err = self.call(["run", str(path), "--dry-run"])
        self.assertEqual(code, EXIT_USAGE)
        self.assertIn("--model-id", err)

    def test_tc56_regenerate_requires_remove(self) -> None:
        path = self.job_path(count=1)
        code, _, _ = self.call(["regenerate", str(path), "--model-id", MODEL_ID])
        self.assertEqual(code, EXIT_USAGE)


class TestExampleDryRuns(CliTestCase):
    def test_tc01_classification_dry_run(self) -> None:
        deps = self.make_deps()
        code, out, err = self.call(
            ["run", str(EXAMPLES / "classification.yaml"), "--model-id", MODEL_ID, "--dry-run"],
            deps=deps,
        )
        self.assertEqual(code, EXIT_OK, err)
        self.assertIn("plan: 24 output(s) to process", out)
        self.assertIn("class_a/class_a_1.png", out)
        self.assertIn("dry-run: nothing was written", out)
        deps.backends.raise_if_called()
        self.assertFalse((REPO_ROOT / "out").exists())

    def test_anomaly_dry_run(self) -> None:
        deps = self.make_deps()
        code, out, err = self.call(
            ["run", str(EXAMPLES / "anomaly.yaml"), "--model-id", MODEL_ID, "--dry-run"],
            deps=deps,
        )
        self.assertEqual(code, EXIT_OK, err)
        self.assertIn("plan: 20 output(s) to process", out)
        deps.backends.raise_if_called()

    def test_output_dir_override(self) -> None:
        path = self.job_path(count=1)
        target = self.root / "elsewhere"
        code, out, err = self.call(
            ["run", str(path), "--model-id", MODEL_ID, "--dry-run", "--output-dir", str(target)],
            deps=self.make_deps(),
        )
        self.assertEqual(code, EXIT_OK, err)
        self.assertIn(str(target), out)
        self.assertFalse(target.exists())


class TestExitCodes(CliTestCase):
    def test_tc82_job_not_found(self) -> None:
        code, _, err = self.call(
            ["run", str(self.root / "missing.yaml"), "--model-id", MODEL_ID, "--dry-run"]
        )
        self.assertEqual(code, EXIT_JOB_NOT_FOUND)
        self.assertIn("[JOB_NOT_FOUND] job file not found:", err)

    def test_job_file_is_directory(self) -> None:
        code, _, err = self.call(["run", str(self.root), "--model-id", MODEL_ID, "--dry-run"])
        self.assertEqual(code, EXIT_JOB_NOT_FOUND)
        self.assertIn("job file is not a regular file:", err)

    def test_config_error(self) -> None:
        path = self.job_path(count=1, version=2)
        code, _, err = self.call(["run", str(path), "--model-id", MODEL_ID, "--dry-run"])
        self.assertEqual(code, EXIT_CONFIG)
        self.assertIn("[CONFIG] version: expected 1, got 2", err)

    def test_selector_error(self) -> None:
        path = self.job_path(count=3)
        code, _, err = self.call(
            ["regenerate", str(path), "--model-id", MODEL_ID, "--remove", "class_a:9-7", "--dry-run"]
        )
        self.assertEqual(code, EXIT_SELECTOR)
        self.assertIn("[SELECTOR] invalid range '9-7' in --remove", err)

    def test_unsafe_path_error(self) -> None:
        path = self.job_path(count=1, output_dir=str(REPO_ROOT))
        code, _, err = self.call(["run", str(path), "--model-id", MODEL_ID, "--dry-run"])
        self.assertEqual(code, EXIT_UNSAFE_PATH)
        self.assertIn("must not be the repository root", err)

    def test_manifest_error(self) -> None:
        path = self.job_path(count=1)
        self.out.mkdir(parents=True)
        (self.out / "manifest.jsonl").write_bytes(b"\x00\x01\x02 not json\n")
        code, _, err = self.call(["run", str(path), "--model-id", MODEL_ID, "--dry-run"])
        self.assertEqual(code, 8)
        self.assertIn("[MANIFEST] manifest is not valid JSONL", err)

    def test_l10_single_incomplete_manifest_line_is_a_clean_start(self) -> None:
        """R9: файл из одной незавершённой строки обрезается целиком, а не падает."""

        path = self.job_path(count=1)
        self.out.mkdir(parents=True)
        manifest = self.out / "manifest.jsonl"
        manifest.write_bytes(b'{"type": "image", "class": "class_a"')
        code, _, err = self.call(
            ["run", str(path), "--model-id", MODEL_ID], deps=self.make_deps()
        )
        self.assertEqual(code, EXIT_OK, err)
        self.assertFalse(manifest.read_bytes().startswith(b'{"type"'))
        self.assertTrue((self.out / "class_a" / "class_a_1.png").is_file())

    def test_l09_manifest_with_wrong_field_types_is_skipped(self) -> None:
        """Валидный JSON с неверным типом поля — битая строка, а не traceback."""

        path = self.job_path(count=1)
        self.out.mkdir(parents=True)
        run_record = {
            "schema_version": 1,
            "type": "run",
            "run_id": "seeded",
            "command": "run",
            "started_at": "2026-09-26T13:59:42Z",
            "job_file": str(path),
            "job_sha256": "0" * 64,
            "output_dir": str(self.out),
            "task": "classification",
            "generation_mode": "single_layer",
            "model_id": MODEL_ID,
            "job_seed": 20260926,
            "classes": [],
            "inference": {},
            "two_layer": None,
        }
        (self.out / "manifest.jsonl").write_text(
            json.dumps(run_record)
            + "\n"
            + json.dumps(
                {"schema_version": 1, "type": "image", "class": "class_a", "index": "x", "attempt": 1}
            )
            + "\n",
            encoding="utf-8",
        )
        deps = self.make_deps()
        code, _, err = self.call(["run", str(path), "--model-id", MODEL_ID], deps=deps)
        self.assertEqual(code, EXIT_OK, err)
        self.assertIn("manifest: skipping corrupt line 2", err)
        self.assertNotIn("Traceback", err)


class TestDependencyPreflight(CliTestCase):
    """Preflight реальных зависимостей до любых записей на диск (выход 9)."""

    MISSING = frozenset({"onnxruntime", "rembg"})

    def two_layer_job(self) -> Path:
        return self.job_path(count=1, generation=two_layer_generation())

    def test_real_backends_without_two_layer_stack_fail_before_writing(self) -> None:
        path = self.two_layer_job()
        with mock.patch("importlib.util.find_spec", fake_find_spec(self.MISSING)):
            code, _, err = self.call(["run", str(path), "--model-id", MODEL_ID])
        self.assertEqual(code, EXIT_DEPENDENCY_MISSING)
        self.assertEqual(code, 9)
        self.assertIn("[DEPENDENCY_MISSING]", err)
        self.assertIn("onnxruntime", err)
        self.assertNotIn("Traceback", err)
        self.assertFalse(self.out.exists())

    def test_dry_run_still_requires_nothing(self) -> None:
        path = self.two_layer_job()
        with mock.patch("importlib.util.find_spec", fake_find_spec(self.MISSING)):
            code, out, err = self.call(
                ["run", str(path), "--model-id", MODEL_ID, "--dry-run"]
            )
        self.assertEqual(code, EXIT_OK, err)
        self.assertIn("plan:", out)
        self.assertFalse(self.out.exists())


class TestFullRuns(CliTestCase):
    def test_tc83_full_run_with_fakes(self) -> None:
        path = self.job_path(count=2)
        deps = self.make_deps()
        code, out, err = self.call(["run", str(path), "--model-id", MODEL_ID], deps=deps)
        self.assertEqual(code, EXIT_OK, err)
        self.assertIn("summary:", out)
        self.assertTrue((self.out / "class_a" / "class_a_1.png").is_file())
        self.assertTrue((self.out / "class_a" / "class_a_2.png").is_file())
        self.assertTrue((self.out / "class_b" / "class_b_1.png").is_file())
        self.assertTrue((self.out / "manifest.jsonl").is_file())
        self.assertEqual(deps.backends.counters["make_generator"], 1)

    def test_tc84_regenerate_with_fakes(self) -> None:
        path = self.job_path(count=4)
        self.call(["run", str(path), "--model-id", MODEL_ID], deps=self.make_deps())
        deps = self.make_deps()
        code, out, err = self.call(
            ["regenerate", str(path), "--model-id", MODEL_ID, "--remove", "class_a:3,4;class_b:2"],
            deps=deps,
        )
        self.assertEqual(code, EXIT_OK, err)
        self.assertTrue((self.out / "_rejected" / "class_a_3_1.png").is_file())
        self.assertTrue((self.out / "_rejected" / "class_a_4_1.png").is_file())
        self.assertTrue((self.out / "_rejected" / "class_b_2_1.png").is_file())
        self.assertIn("rejected_moved=3", out)

    def test_tc85_regenerate_with_failing_backend(self) -> None:
        path = self.job_path(count=2)
        self.call(["run", str(path), "--model-id", MODEL_ID], deps=self.make_deps())
        deps = self.make_deps(fail_at=FailingStage.SINGLE)
        code, _, err = self.call(
            ["regenerate", str(path), "--model-id", MODEL_ID, "--remove", "class_a:1"],
            deps=deps,
        )
        self.assertEqual(code, EXIT_GENERATION_FAILED)
        self.assertFalse((self.out / "class_a" / "class_a_1.png").exists())
        self.assertTrue((self.out / "_rejected" / "class_a_1_1.png").is_file())

    def test_warnings_go_to_stderr(self) -> None:
        path = self.job_path(
            count=1, description={"enabled": True, "request_file": str(self.root / "absent.md")}
        )
        code, _, err = self.call(["run", str(path), "--model-id", MODEL_ID], deps=self.make_deps())
        self.assertEqual(code, EXIT_OK)
        self.assertIn("datasetgen: warning:", err)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
