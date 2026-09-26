"""Preflight реальных зависимостей режима генерации (exit 9 DEPENDENCY_MISSING)."""

from __future__ import annotations

import re
import tomllib
import unittest
from pathlib import Path
from unittest import mock

from datasetgen.engines import (
    EXTRA_FOR_MODE,
    RUNTIME_MODULES,
    ensure_runtime_available,
    missing_runtime_modules,
)
from datasetgen.errors import EXIT_DEPENDENCY_MISSING, DependencyMissingError
from datasetgen.schema import MODE_SINGLE, MODE_TWO
from tests.fakes import fake_find_spec

#: Корень репозитория (каталог с ``pyproject.toml``).
REPO_ROOT = Path(__file__).resolve().parent.parent

#: Имя нашего дистрибутива: self-reference ``ai-basic-alina[single-layer]`` внутри
#: extra ``two-layer`` — это не внешняя зависимость, а «уже проверенное single_layer».
SELF_DISTRIBUTION = "ai-basic-alina"

#: Единственный источник истины для несовпадений «имя дистрибутива != имя модуля».
#: Новое расхождение обязано появиться здесь, иначе падает
#: :meth:`ExtrasMatchPreflightTest.test_every_extra_requirement_is_resolvable`.
DISTRIBUTION_TO_IMPORT: dict[str, str] = {
    "pillow": "PIL",
    "opencv-python": "cv2",
}


def read_extras() -> dict[str, tuple[str, ...]]:
    """Extras из ``[project.optional-dependencies]`` (только стандартная библиотека)."""

    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)["project"]
    return {
        name: tuple(requirements)
        for name, requirements in project.get("optional-dependencies", {}).items()
    }


def distribution_name(requirement: str) -> str:
    """Имя дистрибутива из строки PEP 508: без версии, extras и маркеров."""

    return re.split(r"[\s<>=!~;\[(]", requirement.strip(), maxsplit=1)[0].strip().lower()


def expand_self_reference(
    extra: str,
    extras: dict[str, tuple[str, ...]],
    seen: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Требования extra с рекурсивным разворотом ``ai-basic-alina[<extra>]``.

    ``two-layer`` объявляет не весь список заново, а ``ai-basic-alina[single-layer]``,
    поэтому для сравнения с preflight self-reference нужно развернуть.
    """

    if extra in seen:  # pragma: no cover - защита от «two-layer тянет two-layer»
        raise AssertionError(f"cyclic extras reference: {seen + (extra,)}")
    requirements: list[str] = []
    for requirement in extras[extra]:
        if distribution_name(requirement) != SELF_DISTRIBUTION:
            requirements.append(requirement)
            continue
        inner = re.search(r"\[([^]]+)]", requirement)
        if inner is None:  # pragma: no cover - защита от битой pyproject.toml
            raise AssertionError(f"self-reference without extras: {requirement!r}")
        requirements.extend(expand_self_reference(inner.group(1), extras, seen + (extra,)))
    return tuple(requirements)


def unresolved_requirements(extra: str, extras: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    """Требования extra, для которых имя модуля не выводится автоматически."""

    unresolved = []
    for requirement in expand_self_reference(extra, extras):
        name = distribution_name(requirement)
        if name in DISTRIBUTION_TO_IMPORT:
            continue
        if name.isidentifier():  # имя дистрибутива совпадает с именем модуля
            continue
        unresolved.append(requirement)
    return tuple(unresolved)


def import_names(extra: str, extras: dict[str, tuple[str, ...]]) -> frozenset[str]:
    """Имена модулей preflight, которые даёт extra (включая self-reference)."""

    return frozenset(
        DISTRIBUTION_TO_IMPORT.get(distribution_name(requirement), distribution_name(requirement))
        for requirement in expand_self_reference(extra, extras)
    )


class MissingRuntimeModulesTest(unittest.TestCase):
    """Поиск отсутствующих модулей через find_spec: без единого тяжёлого импорта."""

    def test_single_layer_reports_exactly_the_missing_modules(self) -> None:
        missing = frozenset({"cv2", "numpy", "torch"})
        with mock.patch("importlib.util.find_spec", fake_find_spec(missing)):
            found = missing_runtime_modules(MODE_SINGLE)
        self.assertEqual(found, ("cv2", "numpy", "torch"))
        self.assertEqual(set(found), set(RUNTIME_MODULES[MODE_SINGLE]) & set(missing))
        with mock.patch("importlib.util.find_spec", fake_find_spec(frozenset())):
            self.assertEqual(missing_runtime_modules(MODE_SINGLE), ())

    def test_two_layer_adds_the_cutter_stack(self) -> None:
        missing = frozenset({"onnxruntime", "rembg"})
        with mock.patch("importlib.util.find_spec", fake_find_spec(missing)):
            found = missing_runtime_modules(MODE_TWO)
            self.assertEqual(missing_runtime_modules(MODE_SINGLE), ())
        self.assertEqual(found, ("onnxruntime", "rembg"))
        self.assertEqual(
            set(RUNTIME_MODULES[MODE_TWO]) - set(RUNTIME_MODULES[MODE_SINGLE]),
            {"onnxruntime", "rembg"},
        )
        self.assertEqual(EXTRA_FOR_MODE, {MODE_SINGLE: "single-layer", MODE_TWO: "two-layer"})


class EnsureRuntimeAvailableTest(unittest.TestCase):
    """Диагностика: exit 9, имя режима, имена модулей и подсказка про extra."""

    def test_raises_with_actionable_message_and_is_a_noop_otherwise(self) -> None:
        missing = frozenset({"onnxruntime", "rembg"})
        with mock.patch("importlib.util.find_spec", fake_find_spec(missing)):
            with self.assertRaises(DependencyMissingError) as ctx:
                ensure_runtime_available(MODE_TWO)
        exc = ctx.exception
        self.assertEqual(exc.exit_code, EXIT_DEPENDENCY_MISSING)
        self.assertEqual(exc.exit_code, 9)
        self.assertEqual(exc.code, "DEPENDENCY_MISSING")
        for needle in ("two_layer", "onnxruntime", "rembg", "ai-basic-alina[two-layer]"):
            self.assertIn(needle, exc.message)
        self.assertEqual(
            exc.render(),
            "datasetgen: error: [DEPENDENCY_MISSING] generation.mode: two_layer requires "
            "modules that are not installed: onnxruntime, rembg "
            "(install the extra: pip install 'ai-basic-alina[two-layer]')",
        )
        # окружение готово — preflight молчит и ничего не бросает
        with mock.patch("importlib.util.find_spec", fake_find_spec(frozenset())):
            self.assertIsNone(ensure_runtime_available(MODE_SINGLE))
            self.assertIsNone(ensure_runtime_available(MODE_TWO))


class ExtrasMatchPreflightTest(unittest.TestCase):
    """Extras в ``pyproject.toml`` и список модулей preflight — одно и то же множество.

    Без этого теста можно добавить runtime-зависимость в extra, забыв про
    :data:`datasetgen.engines.RUNTIME_MODULES`, и тогда подсказка «install the
    extra» перестанет быть точной: генерация падала бы уже внутри legacy-обёрток.
    """

    def setUp(self) -> None:
        self.extras = read_extras()

    def test_every_extra_requirement_is_resolvable(self) -> None:
        for extra in self.extras:
            with self.subTest(extra=extra):
                self.assertEqual(
                    unresolved_requirements(extra, self.extras),
                    (),
                    f"extra '{extra}' requires distributions whose import name is "
                    "unknown: add them to DISTRIBUTION_TO_IMPORT and to "
                    "RUNTIME_MODULES in datasetgen/engines.py",
                )

    def test_single_layer_extra_matches_preflight_modules(self) -> None:
        self.assertEqual(
            import_names("single-layer", self.extras),
            frozenset(RUNTIME_MODULES[MODE_SINGLE]),
        )

    def test_two_layer_extra_matches_preflight_modules(self) -> None:
        self.assertEqual(
            import_names("two-layer", self.extras),
            frozenset(RUNTIME_MODULES[MODE_TWO]),
        )

    def test_mode_hints_point_at_existing_extras(self) -> None:
        """Подсказка «install the extra» называет extra, который реально есть в TOML."""

        self.assertEqual(set(EXTRA_FOR_MODE.values()), set(self.extras))
        for mode, extra in EXTRA_FOR_MODE.items():
            with self.subTest(mode=mode):
                with mock.patch(
                    "datasetgen.engines.missing_runtime_modules", return_value=("zzz",)
                ):
                    with self.assertRaises(DependencyMissingError) as ctx:
                        ensure_runtime_available(mode)
                self.assertIn(f"{SELF_DISTRIBUTION}[{extra}]", ctx.exception.message)

    def test_known_mismatches_stay_explicit_and_used(self) -> None:
        self.assertEqual(DISTRIBUTION_TO_IMPORT, {"pillow": "PIL", "opencv-python": "cv2"})
        preflight = {name for names in RUNTIME_MODULES.values() for name in names}
        declared = {
            distribution_name(requirement)
            for requirements in self.extras.values()
            for requirement in requirements
        }
        for distribution, module in DISTRIBUTION_TO_IMPORT.items():
            with self.subTest(distribution=distribution):
                self.assertIn(distribution, declared)
                self.assertIn(module, preflight)
                self.assertNotEqual(distribution, module)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
