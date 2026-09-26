"""TC-57..TC-65: безопасность путей (R1..R6, R10)."""

from __future__ import annotations

import contextlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from datasetgen.bootstrap import IS_SOURCE_CHECKOUT, PACKAGE_DIR
from datasetgen.errors import ConfigError, UnsafePathError
from datasetgen.paths import (
    REPO_ROOT,
    examples_dir,
    protected_paths,
    safe_join,
    validate_class_name,
    validate_output_dir,
)


class SafeJoinTest(unittest.TestCase):
    def setUp(self) -> None:
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        self.root = Path(holder.name).resolve()

    def test_joins_inside_root(self) -> None:
        self.assertEqual(safe_join(self.root, "class_a"), self.root / "class_a")
        self.assertEqual(
            safe_join(self.root, "class_a", "class_a_1.png"),
            self.root / "class_a" / "class_a_1.png",
        )

    def test_tc65_rejects_escape(self) -> None:
        with self.assertRaises(UnsafePathError):
            safe_join(self.root, "../../etc")
        with self.assertRaises(UnsafePathError):
            safe_join(self.root, "..", "..", "etc", "passwd")


class ClassNameTest(unittest.TestCase):
    def test_valid_names(self) -> None:
        for name in ("a", "class_a", "Class-A", "class_b2", "A" * 64):
            validate_class_name(name)

    def test_invalid_names(self) -> None:
        for name in ("", "Class A", "_hidden", "-dash", "имя", "a" * 65, "a.b"):
            with self.assertRaises(ConfigError):
                validate_class_name(name, "classes[0].name")

    def test_reserved_names(self) -> None:
        for name in (
            "_rejected",
            "_intermediate",
            "manifest.jsonl",
            "dataset_description.md",
            "dataset_description_request.md",
        ):
            with self.assertRaises(ConfigError) as ctx:
                validate_class_name(name, "classes[0].name")
            self.assertIn("reserved name", ctx.exception.message)


class OutputDirTest(unittest.TestCase):
    def setUp(self) -> None:
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        self.root = Path(holder.name).resolve()
        self.job_file = self.root / "job.yaml"
        self.job_file.write_text("version: 1\n", encoding="utf-8")

    def assertUnsafe(self, path: Path, *needles: str) -> str:
        with self.assertRaises(UnsafePathError) as ctx:
            validate_output_dir(path, self.job_file)
        message = ctx.exception.message
        for needle in needles:
            self.assertIn(needle, message)
        return message

    def test_tc57_repository_root(self) -> None:
        self.assertUnsafe(REPO_ROOT, "must not be the repository root")

    def test_tc58_protected_paths(self) -> None:
        for protected in protected_paths():
            self.assertUnsafe(protected, "must not be a protected path")

    def test_tc59_existing_file(self) -> None:
        target = self.root / "out.png"
        target.write_bytes(b"x")
        self.assertUnsafe(target, "output_dir is an existing file")

    def test_tc60_non_empty_without_manifest(self) -> None:
        target = self.root / "out"
        target.mkdir()
        (target / "stray.txt").write_text("hello", encoding="utf-8")
        self.assertUnsafe(target, "is not empty and has no manifest.jsonl")

    def test_empty_directory_is_allowed(self) -> None:
        target = self.root / "out"
        target.mkdir()
        self.assertEqual(validate_output_dir(target, self.job_file), target)

    def test_directory_with_manifest_is_allowed(self) -> None:
        target = self.root / "out"
        target.mkdir()
        (target / "manifest.jsonl").write_text("", encoding="utf-8")
        self.assertEqual(validate_output_dir(target, self.job_file), target)

    def test_output_dir_may_not_contain_the_job_file(self) -> None:
        self.assertUnsafe(self.root, "must not contain the job file")

    def test_output_dir_may_not_be_ancestor_of_repository(self) -> None:
        self.assertUnsafe(
            REPO_ROOT.parent, "must not be an ancestor of the repository root"
        )

    def test_package_directory_is_protected(self) -> None:
        """R2: в каталог самого пакета datasetgen писать нельзя никогда."""

        self.assertIn(PACKAGE_DIR.resolve(), tuple(path.resolve() for path in protected_paths()))
        self.assertUnsafe(PACKAGE_DIR, "must not be a protected path")

    def test_source_checkout_detection(self) -> None:
        """Двухрежимность: из checkout пакет видит корень и legacy-модули рядом."""

        self.assertIs(IS_SOURCE_CHECKOUT, True)
        self.assertTrue((REPO_ROOT / "cutter.py").is_file())


class InstalledWheelOutputDirTest(unittest.TestCase):
    """Пути после ``pip install``: ``site-packages`` — НЕ корень исходного репозитория.

    Режим подменяется фейковым ``site-packages`` во временном каталоге (через
    ``mock.patch``), поэтому тест не зависит от реально установленного колеса и
    ничего не пишет в окружение.
    """

    def setUp(self) -> None:
        install = tempfile.TemporaryDirectory()
        self.addCleanup(install.cleanup)
        work = tempfile.TemporaryDirectory()
        self.addCleanup(work.cleanup)

        # Фейковая установка: <tmp>/site-packages/datasetgen/, рядом нет pyproject.toml.
        self.site_packages = Path(install.name).resolve() / "site-packages"
        self.package = self.site_packages / "datasetgen"
        self.package.mkdir(parents=True)
        (self.package / "__init__.py").write_text("", encoding="utf-8")

        # Рабочий каталог пользователя: job-файл рядом, но вне output_dir.
        self.work = Path(work.name).resolve()
        self.job_file = self.work / "job.yaml"
        self.job_file.write_text("version: 1\n", encoding="utf-8")

        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(mock.patch("datasetgen.paths.IS_SOURCE_CHECKOUT", False))
        stack.enter_context(mock.patch("datasetgen.paths.REPO_ROOT", self.site_packages))
        stack.enter_context(
            mock.patch("datasetgen.paths.package_dir", return_value=self.package)
        )

    def assertUnsafe(self, path: Path, *needles: str) -> str:
        with self.assertRaises(UnsafePathError) as ctx:
            validate_output_dir(path, self.job_file)
        message = ctx.exception.message
        for needle in needles:
            self.assertIn(needle, message)
        return message

    def test_installation_directory_is_never_a_valid_output_dir(self) -> None:
        self.assertUnsafe(
            self.site_packages, "must not be the installation directory"
        )

    def test_package_directory_and_its_ancestors_stay_forbidden(self) -> None:
        """R2: писать в установленный код и выше по дереву нельзя никогда."""

        self.assertEqual(protected_paths(), (self.package,))
        self.assertUnsafe(self.package, "must not be a protected path")
        message = self.assertUnsafe(
            self.site_packages.parent, "must not contain a protected path"
        )
        # Ключевое отличие от checkout: правило «предок корня репозитория» здесь
        # НЕ применяется — site-packages не считается корнем исходного репозитория.
        self.assertNotIn("repository root", message)

    def test_ordinary_working_directory_is_allowed(self) -> None:
        target = self.work / "out"
        target.mkdir()
        self.assertEqual(validate_output_dir(target, self.job_file), target)


class ExamplesDirTest(unittest.TestCase):
    """``examples_dir()``: каталог репозитория в checkout, package data в wheel."""

    def test_checkout_variant_returns_repository_examples(self) -> None:
        self.assertEqual(examples_dir(), REPO_ROOT / "examples")
        for name in ("classification.yaml", "anomaly.yaml"):
            self.assertTrue((examples_dir() / name).is_file(), name)

    def test_installed_variant_returns_package_data_directory(self) -> None:
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        package = Path(holder.name).resolve() / "site-packages" / "datasetgen"
        package_data = package / "examples"
        package_data.mkdir(parents=True)
        for name in ("classification.yaml", "anomaly.yaml"):
            (package_data / name).write_text("version: 1\n", encoding="utf-8")

        with (
            mock.patch("datasetgen.bootstrap.IS_SOURCE_CHECKOUT", False),
            mock.patch("datasetgen.bootstrap.PACKAGE_DIR", package),
        ):
            directory = examples_dir()

        self.assertEqual(directory, package_data)
        self.assertTrue(directory.is_dir())
        self.assertEqual(
            sorted(entry.name for entry in directory.iterdir()),
            ["anomaly.yaml", "classification.yaml"],
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
