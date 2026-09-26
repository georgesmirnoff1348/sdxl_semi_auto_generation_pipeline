"""TC-71..TC-74: совместимость с legacy-обёртками и правка F-1.

Модули импортируются БЕЗ конструирования моделей: проверяются только
классы/сигнатуры/исходники.
"""

from __future__ import annotations

import ast
import inspect
import sys
import unittest
from collections.abc import Mapping
from dataclasses import is_dataclass
from pathlib import Path

from datasetgen.engines import default_dependencies
from datasetgen.schema import InferenceSpec

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def cutter_remove_background_parameters() -> list[str]:
    """Сигнатура ``Cutter.remove_background``.

    ``cutter.py`` тянет rembg/onnxruntime, которых может не быть в dev-окружении,
    поэтому при неудаче импорта сигнатура читается из AST исходника.
    """

    try:
        from cutter import Cutter
    except BaseException:  # noqa: BLE001 - rembg без onnxruntime делает sys.exit(1)
        tree = ast.parse((REPO_ROOT / "cutter.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "remove_background":
                return [argument.arg for argument in node.args.args]
        raise AssertionError("remove_background not found in cutter.py")
    return list(inspect.signature(Cutter.remove_background).parameters)


class TestLegacyTypes(unittest.TestCase):
    """TC-71: адаптер возвращает настоящие dataclass-конфиги репозитория."""

    def test_legacy_types_produce_dataclasses(self) -> None:
        from configs import FactorInferenceParameters, FactorPrompts

        legacy = default_dependencies().legacy_types
        prompts = legacy.prompts("a factory worker", "3d render")
        self.assertIsInstance(prompts, FactorPrompts)
        self.assertTrue(is_dataclass(prompts))
        self.assertIsInstance(prompts, Mapping)

        spec = InferenceSpec(num_inference_steps=15, guidance_scale=6.0, width=768, height=512,
                             extra={"inner_pad": 20, "strength": 0.9})
        config = legacy.params(4242, spec)
        self.assertIsInstance(config, FactorInferenceParameters)
        self.assertTrue(is_dataclass(config))
        self.assertIsInstance(config, Mapping)

        prompts_dict = dict(prompts._as_dict())
        self.assertIn("prompt", prompts_dict)
        self.assertIn("negative_prompt", prompts_dict)

        config_dict = dict(config._as_dict())
        for key in ("num_inference_steps", "guidance_scale", "height", "width", "seed"):
            self.assertIn(key, config_dict)
        self.assertEqual(config_dict["inner_pad"], 20)
        self.assertEqual(config_dict["strength"], 0.9)
        self.assertEqual(config_dict["seed"], 4242)
        self.assertNotIn("extra", config_dict)

    def test_config_is_a_mapping_of_pipelines_kwargs(self) -> None:
        from configs import FactorInferenceParameters

        config = FactorInferenceParameters(extra={"inner_pad": 20})
        self.assertEqual(dict(config), dict(config._as_dict()))

    def test_tc74_procedural_prompts_still_work(self) -> None:
        from configs import FactorPrompts

        healthy = FactorPrompts.soviet_citizen()
        self.assertTrue(healthy.prompt)
        self.assertTrue(healthy.negative_prompt)
        spore = FactorPrompts.spore_syndrome()
        self.assertTrue(spore.prompt)
        self.assertTrue(spore.negative_prompt)


class TestLegacySources(unittest.TestCase):
    """TC-72, TC-73: F-1 исправлен и ничего больше не сломано."""

    def setUp(self) -> None:
        import cuda_mps_gens  # импорт модуля не строит пайплайны

        self.module = cuda_mps_gens

    def test_tc72_ordinary_gen_uses_as_dict(self) -> None:
        source = inspect.getsource(self.module.OrdinaryGen.generate_image)
        self.assertIn("_as_dict()", source)
        self.assertNotIn("asdict(", source)

    def test_tc72_controlnet_inpainter_uses_as_dict(self) -> None:
        source = inspect.getsource(self.module.ControlNetInpainter.inpaint_ControlNet)
        self.assertIn("_as_dict()", source)
        self.assertNotIn("asdict(", source)

    def test_working_code_still_uses_as_dict(self) -> None:
        for method in (self.module.ControlNetGen.generate_image, self.module.BackInpainter.inpaint_image):
            source = inspect.getsource(method)
            self.assertIn("_as_dict()", source)
            self.assertNotIn("asdict(", source)

    def test_tc73_signatures_are_unchanged(self) -> None:
        expected = {
            self.module.OrdinaryGen.generate_image: ["self", "prompts", "config", "save_path"],
            self.module.ControlNetGen.generate_image: [
                "self", "prompts", "config", "control_image", "save_path",
            ],
            self.module.BackInpainter.inpaint_image: [
                "self", "prompts", "config", "image", "mask", "save_path",
            ],
        }
        for method, parameters in expected.items():
            self.assertEqual(list(inspect.signature(method).parameters), parameters, method)

        self.assertEqual(cutter_remove_background_parameters(), ["self", "image", "save_path"])

    def test_tc73_decorators_are_preserved(self) -> None:
        expected = {
            self.module.OrdinaryGen.generate_image: "generate_image",
            self.module.ControlNetGen.generate_image: "generate_image",
            self.module.BackInpainter.inpaint_image: "inpaint_image",
            self.module.ControlNetInpainter.inpaint_ControlNet: "inpaint_ControlNet",
        }
        for method, name in expected.items():
            self.assertTrue(hasattr(method, "__wrapped__"), method)
            self.assertEqual(method.__name__, name, method)

    def test_public_class_names_are_unchanged(self) -> None:
        for name in ("OrdinaryGen", "ControlNetGen", "BackInpainter", "ControlNetInpainter"):
            self.assertTrue(hasattr(self.module, name), name)
        source = (REPO_ROOT / "cutter.py").read_text(encoding="utf-8")
        for name in ("Cutter", "U2netCutter", "BirefNetCutter"):
            self.assertIn(f"class {name}", source, name)


class TestLegacyFilesystemHelpers(unittest.TestCase):
    """Legacy filesystems_core остаётся рабочим (используется для сверки диска)."""

    def setUp(self) -> None:
        super().setUp()
        import tempfile

        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        self.root = Path(holder.name)
        self.directory = self.root / "class_a"
        self.directory.mkdir()

    def test_comrade_manager_reports_missing_numbers(self) -> None:
        from filesystems_core import ComradeManager, DirectoryScanner

        for index in (1, 2, 5):
            (self.directory / f"class_a_{index}.png").write_bytes(b"x")
        manager = ComradeManager(self.directory, prefix="class_a")
        self.assertEqual(manager.find_missing_numbers(), [3, 4])
        self.assertEqual(manager.get_next_filename(extension=".png"), "class_a_3.png")
        stats = DirectoryScanner(self.directory).analyze_pattern_files()
        self.assertEqual(stats["class_a"]["count"], 3)
        self.assertEqual(stats["class_a"]["max_number"], 5)

    def test_scanner_requires_existing_directory(self) -> None:
        from filesystems_core import DirectoryScanner

        with self.assertRaises(ValueError):
            DirectoryScanner(self.root / "absent")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
