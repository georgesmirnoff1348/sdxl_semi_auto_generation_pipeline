"""TC-01..TC-16: загрузка примеров и строгая валидация YAML-схемы."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from datasetgen.config import load_yaml
from datasetgen.errors import ConfigError, JobNotFoundError
from tests.fakes import class_entry, job_dict, two_layer_generation, write_job

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPO_ROOT / "examples"


class ConfigTestCase(unittest.TestCase):
    def setUp(self) -> None:
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        self.root = Path(holder.name)

    def write(self, data: dict, name: str = "job.yaml") -> Path:
        return write_job(self.root, data, name=name)

    def load(self, data: dict, **kwargs):
        return load_yaml(self.write(data), **kwargs)

    def assertConfigError(self, data: dict, *needles: str) -> str:
        with self.assertRaises(ConfigError) as ctx:
            self.load(data)
        message = ctx.exception.message
        for needle in needles:
            self.assertIn(needle, message)
        return message


class TestExamples(ConfigTestCase):
    """TC-01: примеры из репозитория должны грузиться без ошибок."""

    def test_classification_example(self) -> None:
        job = load_yaml(EXAMPLES / "classification.yaml")
        self.assertEqual(job.task, "classification")
        self.assertEqual(len(job.classes), 2)
        self.assertEqual([spec.name for spec in job.classes], ["class_a", "class_b"])
        self.assertEqual(job.generation.mode, "single_layer")
        self.assertTrue(job.description.enabled)
        self.assertEqual(job.version, 1)
        self.assertEqual(len(job.raw_sha256), 64)

    def test_anomaly_example(self) -> None:
        job = load_yaml(EXAMPLES / "anomaly.yaml")
        self.assertEqual(job.task, "anomaly_detection")
        self.assertEqual(len(job.classes), 1)
        self.assertEqual(job.classes[0].name, "normal")
        self.assertFalse(job.description.enabled)

    def test_missing_job_file(self) -> None:
        with self.assertRaises(JobNotFoundError):
            load_yaml(self.root / "nope.yaml")

    def test_job_file_is_directory(self) -> None:
        with self.assertRaises(JobNotFoundError):
            load_yaml(self.root)


class TestStructuralRules(ConfigTestCase):
    """TC-02, TC-03, TC-09, TC-16."""

    def test_tc02_wrong_version(self) -> None:
        self.assertConfigError(job_dict(version=2), "version: expected 1, got 2")

    def test_tc03_unknown_root_key(self) -> None:
        data = job_dict()
        data["outputdir"] = "./out"
        self.assertConfigError(
            data,
            "unknown key in root: 'outputdir'",
            "(allowed: classes, dataset_description, generation, inference, output_dir, seed, task, version)",
        )

    def test_tc09_count_values(self) -> None:
        for bad in (0, -1, "12", True, 1.5):
            data = job_dict(classes=[class_entry("class_a", bad), class_entry("class_b", 2)])
            self.assertConfigError(data, "classes[0].count")

    def test_tc16_empty_variable_list(self) -> None:
        entry = class_entry("class_a", 2, variables={"worker": [], "place": ["a hall"]})
        data = job_dict(classes=[entry, class_entry("class_b", 2)])
        self.assertConfigError(data, "classes[0].variables['worker']", "must not be an empty list")

    def test_bool_is_not_int(self) -> None:
        data = job_dict()
        data["seed"] = True
        self.assertConfigError(data, "seed: expected an integer, got a boolean")

    def test_seed_must_be_non_negative(self) -> None:
        self.assertConfigError(job_dict(seed=-1), "seed: expected an integer >= 0")

    def test_unknown_key_in_inference(self) -> None:
        self.assertConfigError(
            job_dict(inference={"steps": 10}), "unknown key in inference: 'steps'"
        )

    def test_width_must_be_multiple_of_eight(self) -> None:
        self.assertConfigError(job_dict(inference={"width": 1001}), "inference.width")


class TestTaskRules(ConfigTestCase):
    """TC-04, TC-05, TC-06."""

    def test_tc04_three_classes(self) -> None:
        data = job_dict(classes=[class_entry(name, 2) for name in ("class_a", "class_b", "class_c")])
        self.assertConfigError(data, "task 'classification' requires exactly 2 classes, got 3")

    def test_tc05_one_class(self) -> None:
        self.assertConfigError(
            job_dict(classes=[class_entry("class_a", 2)]),
            "task 'classification' requires exactly 2 classes, got 1",
        )

    def test_tc06_anomaly_class_must_be_normal(self) -> None:
        self.assertConfigError(
            job_dict(task="anomaly_detection", classes=[class_entry("abnormal", 2)]),
            "task 'anomaly_detection' requires exactly 1 class named 'normal', got 1",
        )

    def test_anomaly_two_classes(self) -> None:
        self.assertConfigError(
            job_dict(
                task="anomaly_detection",
                classes=[class_entry("normal", 2), class_entry("abnormal", 2)],
            ),
            "task 'anomaly_detection' requires exactly 1 class named 'normal', got 2",
        )


class TestClassNameRules(ConfigTestCase):
    """TC-07, TC-08."""

    def test_tc07_invalid_class_name(self) -> None:
        data = job_dict(classes=[class_entry("Class A", 2), class_entry("class_b", 2)])
        self.assertConfigError(
            data,
            "classes[0].name: invalid class name 'Class A'",
            "(expected ^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$)",
        )

    def test_tc08_reserved_name(self) -> None:
        data = job_dict(classes=[class_entry("class_a", 2), class_entry("manifest.jsonl", 2)])
        self.assertConfigError(data, "classes[1].name: reserved name 'manifest.jsonl'")

    def test_name_starting_with_underscore(self) -> None:
        data = job_dict(classes=[class_entry("_hidden", 2), class_entry("class_b", 2)])
        self.assertConfigError(data, "invalid class name '_hidden'")

    def test_duplicate_class_names(self) -> None:
        data = job_dict(classes=[class_entry("class_a", 2), class_entry("class_a", 2)])
        self.assertConfigError(data, "classes[1].name: duplicate class name 'class_a'")


class TestTemplateRules(ConfigTestCase):
    """TC-10, TC-11, TC-12."""

    def test_tc10_unknown_variable(self) -> None:
        entry = class_entry("class_a", 2, template="a {worker} wearing {uniform}")
        data = job_dict(classes=[entry, class_entry("class_b", 2)])
        self.assertConfigError(data, "classes[0].template: unknown variable 'uniform'")

    def test_tc11_unused_variable(self) -> None:
        entry = class_entry(
            "class_a",
            2,
            template="a {worker} wearing {uniform}",
            variables={
                "worker": ["factory worker", "engineer"],
                "uniform": ["helmet"],
                "background": ["a hall"],
            },
        )
        data = job_dict(classes=[entry, class_entry("class_b", 2)])
        self.assertConfigError(data, "classes[0].variables: unused variable 'background'")

    def test_tc12_escaped_braces(self) -> None:
        entry = class_entry(
            "class_a",
            2,
            template="a {worker} wearing {{uniform}} suit",
            variables={"worker": ["factory worker", "engineer"]},
        )
        data = job_dict(classes=[entry, class_entry("class_b", 2)])
        job = self.load(data)
        self.assertEqual(job.classes[0].prompt.template, "a {worker} wearing {{uniform}} suit")
        from datasetgen.prompts import render

        self.assertEqual(
            render(job.classes[0].prompt.template, {"worker": "comrade"}),
            "a comrade wearing {uniform} suit",
        )

    def test_invalid_placeholder_name(self) -> None:
        entry = class_entry("class_a", 2, template="a {Worker}")
        data = job_dict(classes=[entry, class_entry("class_b", 2)])
        self.assertConfigError(data, "classes[0].template: invalid placeholder '{Worker}'")

    def test_background_template_is_validated(self) -> None:
        generation = two_layer_generation()
        generation["background"]["template"] = "background of {place}"
        generation["background"]["variables"] = {"spot": ["a hall"]}
        self.assertConfigError(
            job_dict(generation=generation), "generation.background.template: unknown variable 'place'"
        )


class TestGenerationRules(ConfigTestCase):
    """TC-13, TC-14, TC-15."""

    def test_tc13_two_layer_without_background(self) -> None:
        generation = {"mode": "two_layer", "cutter": "u2net", "cutter_model_name": None}
        self.assertConfigError(
            job_dict(generation=generation),
            "generation.background: required when generation.mode is 'two_layer'",
        )

    def test_tc14_custom_cutter_without_model_name(self) -> None:
        generation = two_layer_generation(cutter="custom", cutter_model_name=None)
        self.assertConfigError(
            job_dict(generation=generation),
            "'custom' requires generation.cutter_model_name",
        )

    def test_custom_cutter_with_model_name(self) -> None:
        job = self.load(job_dict(generation=two_layer_generation(cutter="custom", cutter_model_name="isnet")))
        self.assertEqual(job.generation.cutter_model_name, "isnet")

    def test_tc15_single_layer_background_is_a_warning(self) -> None:
        generation = two_layer_generation()
        generation["mode"] = "single_layer"
        warnings: list[str] = []
        job = self.load(job_dict(generation=generation), warnings=warnings)
        self.assertEqual(job.generation.mode, "single_layer")
        self.assertTrue(any("unused two_layer block" in text for text in warnings), warnings)

    def test_unknown_cutter(self) -> None:
        generation = {"mode": "single_layer", "cutter": "magic"}
        self.assertConfigError(job_dict(generation=generation), "generation.cutter: expected one of")


class TestDescriptionRules(ConfigTestCase):
    def test_max_prompt_rows_must_be_positive(self) -> None:
        self.assertConfigError(
            job_dict(description={"enabled": True, "max_prompt_rows": 0}),
            "dataset_description.max_prompt_rows",
        )

    def test_missing_request_file_is_a_warning(self) -> None:
        warnings: list[str] = []
        job = self.load(
            job_dict(description={"enabled": True, "request_file": str(self.root / "absent.md")}),
            warnings=warnings,
        )
        self.assertEqual(job.description.request_file, str(self.root / "absent.md"))
        self.assertTrue(any("request_file" in text for text in warnings), warnings)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
