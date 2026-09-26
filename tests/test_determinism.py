"""TC-17..TC-21: детерминизм material/seed/выбора переменных."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from datasetgen.determinism import derive_seed, material, pick_variable
from datasetgen.planner import inpaint_seed, resolve_prompt, settings_for
from datasetgen.schema import PromptSpec
from tests.fakes import class_entry, job_dict, write_job

from tempfile import TemporaryDirectory

from datasetgen.config import load_yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def spec(**variables) -> PromptSpec:
    return PromptSpec("a {worker} wearing {uniform}", "3d render", variables)


class TestMaterialAndSeed(unittest.TestCase):
    def test_tc18_seed_range_and_stability(self) -> None:
        for index in range(1, 51):
            text = material(20260926, "classification", "class_a", index, 1)
            first = derive_seed(text)
            self.assertEqual(first, derive_seed(text))
            self.assertGreaterEqual(first, 0)
            self.assertLess(first, 2_147_483_647)

    def test_tc18_seed_stable_across_processes(self) -> None:
        text = material(20260926, "anomaly_detection", "normal", 7, 2)
        expected = derive_seed(text)
        code = (
            "from datasetgen.determinism import derive_seed;"
            f"print(derive_seed({text!r}))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(int(result.stdout.strip()), expected)

    def test_tc19_seed_depends_on_every_component(self) -> None:
        base = derive_seed(material(1, "classification", "class_a", 1, 1))
        variants = {
            "job.seed": derive_seed(material(2, "classification", "class_a", 1, 1)),
            "task": derive_seed(material(1, "anomaly_detection", "class_a", 1, 1)),
            "class": derive_seed(material(1, "classification", "class_b", 1, 1)),
            "index": derive_seed(material(1, "classification", "class_a", 2, 1)),
            "attempt": derive_seed(material(1, "classification", "class_a", 1, 2)),
        }
        for name, value in variants.items():
            self.assertNotEqual(base, value, name)

    def test_material_contains_salt(self) -> None:
        self.assertTrue(material(1, "classification", "class_a", 1, 1).startswith("datasetgen/v1|"))

    def test_tc20_inpaint_seed_differs(self) -> None:
        text = material(20260926, "classification", "class_a", 3, 1)
        self.assertNotEqual(inpaint_seed(text), derive_seed(text))
        self.assertEqual(inpaint_seed(text), inpaint_seed(text))


class TestVariableSelection(unittest.TestCase):
    def test_pick_variable_is_stable(self) -> None:
        values = ("a", "b", "c")
        chosen = pick_variable("material", "worker", values)
        self.assertEqual(chosen, pick_variable("material", "worker", values))
        self.assertIn(chosen, values)

    def test_tc21_coverage(self) -> None:
        values = ("low", "mid", "high")
        chosen = {
            derive_seed(material(20260926, "classification", "class_a", index, 1)) % 3
            for index in range(1, 51)
        }
        self.assertGreaterEqual(len(chosen), 2)

    def test_resolve_prompt_uses_sorted_variable_order(self) -> None:
        prompt_spec = spec(worker=("w1", "w2"), uniform=("u1", "u2"), pose=("p1", "p2"))
        first = resolve_prompt(prompt_spec, "material")
        self.assertEqual(first, resolve_prompt(prompt_spec, "material"))
        self.assertNotIn("{", first)


class TestPromptDeterminism(unittest.TestCase):
    """TC-17: промпт зависит от всех компонентов material."""

    def setUp(self) -> None:
        holder = TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        self.root = Path(holder.name)

    def load(self, **kwargs):
        path = write_job(self.root, job_dict(**kwargs))
        return load_yaml(path)

    def test_tc17_same_inputs_same_prompt(self) -> None:
        job = self.load()
        first = resolve_prompt(job.classes[0].prompt, material(job.seed, job.task, "class_a", 1, 1))
        second = resolve_prompt(job.classes[0].prompt, material(job.seed, job.task, "class_a", 1, 1))
        self.assertEqual(first, second)

    def test_tc17_different_attempt_class_index(self) -> None:
        job = self.load()
        spec_a = job.classes[0].prompt
        spec_b = job.classes[1].prompt
        attempts = [
            resolve_prompt(spec_a, material(job.seed, job.task, "class_a", index, attempt))
            for attempt in (1, 2, 3)
            for index in range(1, 13)
        ]
        self.assertGreaterEqual(len(set(attempts)), 2, attempts)
        other_class = resolve_prompt(spec_b, material(job.seed, job.task, "class_b", 1, 1))
        self.assertNotEqual(
            resolve_prompt(spec_a, material(job.seed, job.task, "class_a", 1, 1)),
            other_class,
        )
        # реализация привязана к material, а не к порядку обхода
        first = material(job.seed, job.task, "class_a", 1, 1)
        self.assertEqual(
            resolve_prompt(spec_a, first),
            resolve_prompt(spec_a, first),
        )

    def test_settings_exclude_seed(self) -> None:
        job = self.load()
        settings = settings_for(job.inference)
        self.assertEqual(
            sorted(settings), ["extra", "guidance_scale", "height", "num_inference_steps", "width"]
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
