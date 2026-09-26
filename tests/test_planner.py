"""Планирование выходов: порядок, attempt, complete, режим regenerate."""

from __future__ import annotations

import unittest

from datasetgen.determinism import derive_seed, material
from datasetgen.planner import (
    all_keys,
    background_prompt,
    build_plan,
    cutter_model_name,
    inpaint_seed,
    make_output,
)
from tests.fakes import MODEL_ID, TmpCase, two_layer_generation


class TestBuildPlan(TmpCase):
    def test_run_plan_covers_every_output(self) -> None:
        job = self.job(count=3)
        plan = build_plan(job, MODEL_ID, "run")
        self.assertEqual(len(plan.outputs), 6)
        self.assertEqual(plan.complete, ())
        self.assertEqual(
            [(item.class_name, item.index) for item in plan.outputs],
            [("class_a", 1), ("class_a", 2), ("class_a", 3), ("class_b", 1), ("class_b", 2), ("class_b", 3)],
        )
        self.assertEqual([item.filename for item in plan.outputs[:2]], ["class_a_1.png", "class_a_2.png"])

    def test_all_keys_follows_yaml_order(self) -> None:
        job = self.job(count=2)
        self.assertEqual(all_keys(job), (("class_a", 1), ("class_a", 2), ("class_b", 1), ("class_b", 2)))

    def test_complete_flags_move_outputs(self) -> None:
        job = self.job(count=2)
        plan = build_plan(
            job,
            MODEL_ID,
            "run",
            max_attempts={("class_a", 1): 1},
            complete_flags={("class_a", 1): True},
        )
        self.assertEqual([item.key for item in plan.complete], [("class_a", 1)])
        self.assertEqual(plan.complete[0].status, "complete")
        self.assertNotIn(("class_a", 1), [item.key for item in plan.outputs])

    def test_attempt_increments_after_known_attempts(self) -> None:
        job = self.job(count=1)
        plan = build_plan(job, MODEL_ID, "run", max_attempts={("class_a", 1): 3, ("class_b", 1): 0})
        attempts = {item.key: item.attempt for item in plan.outputs}
        self.assertEqual(attempts[("class_a", 1)], 4)
        self.assertEqual(attempts[("class_b", 1)], 1)

    def test_regenerate_selection_is_respected(self) -> None:
        job = self.job(count=12)
        selection = (("class_a", 3), ("class_b", 4))
        plan = build_plan(job, MODEL_ID, "regenerate", selection=selection)
        self.assertEqual([item.key for item in plan.outputs], list(selection))
        self.assertEqual(plan.complete, ())

    def test_settings_do_not_contain_seed(self) -> None:
        job = self.job(inference={"num_inference_steps": 12, "extra": {"strength": 0.4}})
        item = make_output(job, "class_a", 1, 1)
        self.assertEqual(item.settings["num_inference_steps"], 12)
        self.assertEqual(item.settings["extra"], {"strength": 0.4})
        self.assertNotIn("seed", item.settings)

    def test_seed_and_prompt_come_from_material(self) -> None:
        job = self.job(count=1)
        item = make_output(job, "class_a", 2, 3)
        text = material(job.seed, job.task, "class_a", 2, 3)
        self.assertEqual(item.material, text)
        self.assertEqual(item.seed, derive_seed(text))


class TestTwoLayerHelpers(TmpCase):
    def test_background_prompt_uses_own_material(self) -> None:
        job = self.job(count=1, generation=two_layer_generation())
        item = make_output(job, "class_a", 1, 1)
        background = background_prompt(job, item.material)
        self.assertIsNotNone(background)
        self.assertIn("background photography of", background)
        self.assertNotEqual(background, item.prompt)

    def test_background_prompt_is_none_for_single_layer(self) -> None:
        job = self.job(count=1)
        item = make_output(job, "class_a", 1, 1)
        self.assertIsNone(background_prompt(job, item.material))

    def test_inpaint_seed_differs_from_subject_seed(self) -> None:
        job = self.job(count=1, generation=two_layer_generation())
        item = make_output(job, "class_a", 1, 1)
        self.assertNotEqual(inpaint_seed(item.material), item.seed)

    def test_cutter_model_names(self) -> None:
        job = self.job(count=1, generation=two_layer_generation())
        self.assertEqual(cutter_model_name(job.generation), "u2net")
        job_biref = self.job(count=1, generation=two_layer_generation(cutter="birefnet"), name="b.yaml")
        self.assertEqual(cutter_model_name(job_biref.generation), "birefnet-general")
        job_custom = self.job(
            count=1, generation=two_layer_generation(cutter="custom", cutter_model_name="isnet"), name="c.yaml"
        )
        self.assertEqual(cutter_model_name(job_custom.generation), "isnet")

    def test_background_prompt_uses_sorted_variables(self) -> None:
        """Порядок обхода переменных — по именам, поэтому промпт не зависит от YAML."""

        forward = self.job(
            count=1,
            name="forward.yaml",
            classes=[
                {
                    "name": "class_a",
                    "count": 1,
                    "template": "{alpha} and {beta}",
                    "negative_prompt": "",
                    "variables": {"alpha": ["A1", "A2"], "beta": ["B1", "B2"]},
                },
                {"name": "class_b", "count": 1, "template": "{alpha} and {beta}",
                 "negative_prompt": "", "variables": {"alpha": ["A1", "A2"], "beta": ["B1", "B2"]}},
            ],
        )
        reverse = self.job(
            count=1,
            name="reverse.yaml",
            classes=[
                {
                    "name": "class_a",
                    "count": 1,
                    "template": "{beta} and {alpha}",
                    "negative_prompt": "",
                    "variables": {"alpha": ["A1", "A2"], "beta": ["B1", "B2"]},
                },
                {"name": "class_b", "count": 1, "template": "{beta} and {alpha}",
                 "negative_prompt": "", "variables": {"alpha": ["A1", "A2"], "beta": ["B1", "B2"]}},
            ],
        )
        # одинаковый набор значений -> одинаковая пара «значение alpha, значение beta»
        pairs_forward = {
            tuple(make_output(forward, "class_a", index, 1).prompt.split(" and "))
            for index in range(1, 13)
        }
        pairs_reverse = {
            tuple(make_output(reverse, "class_a", index, 1).prompt.split(" and "))
            for index in range(1, 13)
        }
        swapped = {tuple(reversed(pair)) for pair in pairs_forward}
        self.assertTrue(
            pairs_reverse <= swapped,
            "выбор значений обязан зависеть только от material и имён переменных",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
