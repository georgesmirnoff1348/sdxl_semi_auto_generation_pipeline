"""TC-23..TC-34, TC-38..TC-40, TC-61..TC-64, TC-86..TC-88: прогон ``run``."""

from __future__ import annotations

import unittest
from dataclasses import replace

from datasetgen import runner
from datasetgen.errors import EXIT_GENERATION_FAILED, EXIT_OK, UnsafePathError
from datasetgen.layout import final_image_path
from datasetgen.manifest import Manifest
from tests.fakes import (
    BACKGROUND_MODEL_ID,
    MODEL_ID,
    FailingStage,
    TmpCase,
    image_bytes,
    make_dependencies,
    two_layer_generation,
    write_png,
)


class TestHappyPath(TmpCase):
    def test_tc23_classification_layout(self) -> None:
        job = self.job(count=12)
        result = self.execute(job, self.make_deps())
        self.assertEqual(result.exit_code, EXIT_OK)
        self.assertEqual(result.succeeded, 24)
        self.assertTrue((self.out / "class_a").is_dir())
        self.assertTrue((self.out / "class_b").is_dir())
        self.assertTrue(self.manifest_path().is_file())
        for index in range(1, 13):
            self.assertTrue((self.out / "class_a" / f"class_a_{index}.png").is_file())
            self.assertTrue((self.out / "class_b" / f"class_b_{index}.png").is_file())
        self.assertFalse((self.out / "_intermediate").exists())

    def test_tc24_anomaly_layout(self) -> None:
        job = self.job(task="anomaly_detection", count=3)
        self.execute(job, self.make_deps())
        self.assertTrue((self.out / "normal" / "normal_1.png").is_file())
        self.assertEqual(
            sorted(item.name for item in self.out.iterdir()),
            ["manifest.jsonl", "normal"],
        )

    def test_tc25_single_layer_has_no_intermediate_dir(self) -> None:
        job = self.job(count=2)
        self.execute(job, self.make_deps())
        self.assertFalse((self.out / "_intermediate").exists())

    def test_tc33_manifest_happy_path(self) -> None:
        job = self.job(count=2)
        self.execute(job, self.make_deps())
        self.assertEqual(len(self.run_records()), 1)
        self.assertEqual(self.statuses("class_a", 1), ["planned", "succeeded"])
        self.assertEqual(len(self.image_records()), 8)
        succeeded = [
            record
            for record in self.image_records()
            if record["status"] == "succeeded" and record["class"] == "class_a" and record["index"] == 1
        ][0]
        self.assertEqual(succeeded["output"], "class_a/class_a_1.png")
        self.assertEqual(succeeded["model_id"], MODEL_ID)
        self.assertEqual(succeeded["bytes"], (self.out / "class_a" / "class_a_1.png").stat().st_size)
        self.assertIsNone(succeeded["error"])
        run_record = self.run_records()[0]
        self.assertEqual(run_record["command"], "run")
        self.assertEqual(run_record["task"], "classification")
        self.assertEqual(run_record["output_dir"], str(self.out))
        self.assertEqual(run_record["job_seed"], job.seed)
        self.assertEqual(run_record["generation_mode"], "single_layer")

    def test_run_record_two_layer_block(self) -> None:
        job = self.job(count=1, generation=two_layer_generation())
        self.execute(job, self.make_deps())
        block = self.run_records()[0]["two_layer"]
        self.assertEqual(block["background_model_id"], BACKGROUND_MODEL_ID)
        self.assertEqual(block["cutter"], "u2net")
        self.assertEqual(block["background_inference"]["extra"], {"inner_pad": 20, "strength": 0.9})


class TestDispatch(TmpCase):
    def test_tc27_single_layer_dispatch(self) -> None:
        job = self.job(count=2)
        deps = self.make_deps()
        self.execute(job, deps)
        counters = deps.backends.counters
        self.assertEqual(counters["make_generator"], 1)
        self.assertEqual(counters["make_cutter"], 0)
        self.assertEqual(counters["make_inpainter"], 0)

    def test_tc28_two_layer_dispatch(self) -> None:
        job = self.job(count=1, generation=two_layer_generation())
        deps = self.make_deps()
        self.execute(job, deps)
        counters = deps.backends.counters
        self.assertEqual(counters["make_generator"], 2)
        self.assertEqual(counters["make_cutter"], 2)
        self.assertEqual(counters["make_inpainter"], 2)
        kinds = [entry["kind"] for entry in deps.backends.log if entry["kind"] in
                 ("generate_image", "remove_background", "inpaint_image")]
        self.assertEqual(
            kinds,
            ["generate_image", "remove_background", "inpaint_image"] * 2,
        )

    def test_tc26_two_layer_intermediates(self) -> None:
        job = self.job(count=2, generation=two_layer_generation())
        self.execute(job, self.make_deps())
        intermediate = self.out / "_intermediate" / "class_a"
        self.assertTrue((intermediate / "class_a_1_1_subject.png").is_file())
        self.assertTrue((intermediate / "class_a_1_1_cut.png").is_file())
        names = sorted(item.name for item in (self.out / "class_a").iterdir())
        self.assertEqual(names, ["class_a_1.png", "class_a_2.png"])

    def test_tc29_arguments_reach_legacy_wrappers(self) -> None:
        job = self.job(
            count=1,
            inference={"num_inference_steps": 12, "guidance_scale": 6.5, "extra": {"inner_pad": 20}},
            generation=two_layer_generation(extra={"inner_pad": 20, "strength": 0.9}),
        )
        deps = self.make_deps()
        self.execute(job, deps)
        subject = deps.backends.calls("generate_image")[0]
        self.assertEqual(subject["config"].seed, subject["config"]._as_dict()["seed"])
        self.assertEqual(subject["config"].num_inference_steps, 12)
        self.assertEqual(subject["config"].extra, {"inner_pad": 20})
        self.assertTrue(str(subject["save_path"]).endswith("class_a_1_1_subject.png"))
        self.assertEqual(subject["prompts"].prompt, subject["prompts"]._as_dict()["prompt"])

        inpaint = deps.backends.calls("inpaint_image")[0]
        self.assertEqual(inpaint["config"].extra, {"inner_pad": 20, "strength": 0.9})
        self.assertIn("background photography of", inpaint["prompts"].prompt)
        self.assertTrue(str(inpaint["save_path"]).endswith("class_a/class_a_1.png"))

        plan_seed = runner.build_plan(job, MODEL_ID, "run").outputs[0].seed
        self.assertEqual(deps.legacy_types.params_created[0].seed, plan_seed)
        self.assertNotEqual(inpaint["config"].seed, plan_seed)


class TestDryRun(TmpCase):
    def test_tc30_dry_run_writes_nothing(self) -> None:
        job = self.job(count=2)
        deps = self.make_deps()
        result = self.execute(job, deps, dry_run=True)
        self.assertEqual(result.exit_code, EXIT_OK)
        self.assertFalse(result.written)
        self.assertFalse(self.out.exists())
        self.assertEqual(list(self.root.iterdir()), [self.job_path()])
        deps.backends.raise_if_called()

    def test_tc31_dry_run_plan_matches_real_run(self) -> None:
        job = self.job(count=2)
        dry_plan = runner.build_plan(job, MODEL_ID, "run")
        deps = self.make_deps()
        self.execute(job, deps, dry_run=True)
        real_plan = runner.build_plan(job, MODEL_ID, "run", complete_flags={
            (name, index): True for name, index in
            [(spec.name, index) for spec in job.classes for index in range(1, spec.count + 1)]
        })
        self.assertEqual(
            [(item.class_name, item.index, item.attempt, item.seed, item.prompt, item.filename)
             for item in dry_plan.outputs],
            [(item.class_name, item.index, item.attempt, item.seed, item.prompt, item.filename)
             for item in real_plan.complete],
        )

    def test_dry_run_two_layer_prints_background_prompt(self) -> None:
        job = self.job(count=1, generation=two_layer_generation())
        deps = self.make_deps()
        self.execute(job, deps, dry_run=True)
        self.assertIn("background prompt:", self.stdout.getvalue())
        deps.backends.raise_if_called()
        self.assertFalse(self.out.exists())


class TestFailures(TmpCase):
    def test_tc34_failure_is_recorded_and_batch_continues(self) -> None:
        job = self.job(count=2)
        deps = self.make_deps(fail_at=FailingStage.SINGLE)
        result = self.execute(job, deps)
        self.assertEqual(result.exit_code, EXIT_GENERATION_FAILED)
        self.assertEqual(result.failed, 4)
        self.assertEqual(self.statuses("class_a", 1), ["planned", "failed"])
        failure = [
            record
            for record in self.image_records()
            if record["status"] == "failed" and record["class"] == "class_a" and record["index"] == 1
        ][0]
        self.assertEqual(failure["error"]["type"], "RuntimeError")
        self.assertIn("injected failure", failure["error"]["message"])
        self.assertEqual(failure["error"]["stage"], "single")
        self.assertEqual(len(self.image_records()), 8)

    def test_failure_stage_for_two_layer_cut(self) -> None:
        job = self.job(count=1, generation=two_layer_generation())
        deps = self.make_deps(fail_at=FailingStage.CUT)
        result = self.execute(job, deps)
        self.assertEqual(result.exit_code, EXIT_GENERATION_FAILED)
        failed = [record for record in self.image_records() if record["status"] == "failed"]
        self.assertEqual(failed[0]["error"]["stage"], "cut")
        self.assertFalse(final_image_path(self.out, "class_a", 1).exists())

    def test_save_stage_when_backend_writes_nothing(self) -> None:
        job = self.job(count=1)
        deps = self.make_deps(save=False)
        result = self.execute(job, deps)
        self.assertEqual(result.exit_code, EXIT_GENERATION_FAILED)
        failed = [record for record in self.image_records() if record["status"] == "failed"]
        self.assertEqual(failed[0]["error"]["stage"], "save")


class TestResume(TmpCase):
    def test_tc38_resume_processes_only_missing_outputs(self) -> None:
        job = self.job(count=2)
        self.seed_manifest(
            job,
            [
                self.make_image_record(job, "class_a", 1, status="succeeded"),
                self.make_image_record(job, "class_a", 2, status="planned"),
                self.make_image_record(job, "class_b", 1, status="succeeded"),
                self.make_image_record(job, "class_b", 2, status="planned"),
            ],
        )
        for name in ("class_a", "class_b"):
            write_png(self.out / name / f"{name}_1.png", 1)
        before = {
            name: (
                image_bytes(self.out / name / f"{name}_1.png"),
                (self.out / name / f"{name}_1.png").stat().st_mtime_ns,
            )
            for name in ("class_a", "class_b")
        }

        deps = self.make_deps()
        result = self.execute(job, deps)
        self.assertEqual(result.exit_code, EXIT_OK, self.errors)
        self.assertEqual(result.planned, 2)
        self.assertEqual(result.skipped_complete, 2)
        self.assertEqual(
            sorted(entry["save_path"].name for entry in deps.backends.calls("generate_image")),
            ["class_a_2.png", "class_b_2.png"],
        )
        for name in ("class_a", "class_b"):
            path = self.out / name / f"{name}_1.png"
            self.assertEqual((image_bytes(path), path.stat().st_mtime_ns), before[name])

    def test_tc39_succeeded_but_missing_file_is_regenerated(self) -> None:
        job = self.job(count=1)
        self.seed_manifest(
            job,
            [
                self.make_image_record(job, "class_a", 1, status="succeeded"),
                self.make_image_record(job, "class_b", 1, status="succeeded"),
            ],
        )
        deps = self.make_deps()
        result = self.execute(job, deps)
        self.assertEqual(result.planned, 2)
        self.assertEqual(self.statuses("class_a", 1), ["succeeded", "planned", "succeeded"])
        self.assertEqual(
            [
                record["attempt"]
                for record in self.image_records()
                if record["class"] == "class_a" and record["index"] == 1
            ],
            [1, 2, 2],
        )
        self.assertTrue((self.out / "class_a" / "class_a_1.png").is_file())

    def test_tc40_rejected_index_is_untouched_by_run(self) -> None:
        job = self.job(count=1)
        self.seed_manifest(
            job,
            [
                self.make_image_record(job, "class_a", 1, status="succeeded"),
                self.make_image_record(
                    job, "class_b", 1, status="rejected", rejected_file="_rejected/class_b_1_1.png"
                ),
            ],
        )
        write_png(self.out / "class_a" / "class_a_1.png", 1)
        deps = self.make_deps()
        result = self.execute(job, deps)
        self.assertEqual(result.planned, 0)
        self.assertEqual(result.skipped_complete, 1)
        self.assertEqual(
            [entry["save_path"].name for entry in deps.backends.calls("generate_image")],
            [],
        )
        self.assertEqual(list((self.out / "class_b").iterdir()), [])
        self.assertTrue(any("rejected" in text for text in self.warnings), self.warnings)

    def test_tc87_growing_count_generates_only_new_indexes(self) -> None:
        job = self.job(count=4)
        self.execute(job, self.make_deps())
        grown = self.job(count=6)
        deps = self.make_deps()
        result = self.execute(grown, deps)
        self.assertEqual(result.planned, 4)
        self.assertEqual(result.skipped_complete, 8)
        self.assertEqual(
            sorted(entry["save_path"].name for entry in deps.backends.calls("generate_image")),
            ["class_a_5.png", "class_a_6.png", "class_b_5.png", "class_b_6.png"],
        )

    def test_tc88_shrinking_count_keeps_extra_files(self) -> None:
        job = self.job(count=6)
        self.execute(job, self.make_deps())
        marker = (self.out / "class_a" / "class_a_5.png").read_bytes()
        shrunk = self.job(count=4)
        result = self.execute(shrunk, self.make_deps())
        self.assertEqual(result.exit_code, EXIT_OK)
        self.assertEqual(result.planned, 0)
        self.assertTrue((self.out / "class_a" / "class_a_5.png").is_file())
        self.assertEqual((self.out / "class_a" / "class_a_5.png").read_bytes(), marker)
        self.assertTrue(any("changed since the last run" in text for text in self.warnings), self.warnings)


class TestSafetyRules(TmpCase):
    def test_tc61_unmanaged_file_is_not_overwritten(self) -> None:
        job = self.job(count=3)
        self.seed_manifest(job, [self.make_image_record(job, "class_a", 1, status="succeeded")])
        write_png(self.out / "class_a" / "class_a_1.png", 1)
        stray = self.out / "class_a" / "class_a_3.png"
        write_png(stray, 7)
        before = image_bytes(stray)
        # индекс 3 есть на диске, но манифест о нём ничего не знает
        with self.assertRaises(UnsafePathError) as ctx:
            self.execute(job, self.make_deps())
        self.assertIn("refusing to overwrite unmanaged file", ctx.exception.message)
        self.assertIn("no matching manifest record", ctx.exception.message)
        self.assertEqual(image_bytes(stray), before)

    def test_tc62_unfinished_planned_attempt_may_be_overwritten(self) -> None:
        job = self.job(count=1)
        from datasetgen.planner import make_output

        planned = make_output(job, "class_a", 1, 1)
        self.seed_manifest(
            job,
            [
                self.make_image_record(
                    job,
                    "class_a",
                    1,
                    attempt=1,
                    status="planned",
                    seed=planned.seed,
                    prompt=planned.prompt,
                )
            ],
        )
        class_dir = self.out / "class_a"
        class_dir.mkdir(parents=True)
        write_png(class_dir / "class_a_1.png", 3)
        result = self.execute(job, self.make_deps())
        self.assertEqual(result.exit_code, EXIT_OK, self.errors)
        # R5 соблюдён: незавершённая planned-попытка (тот же attempt, seed, prompt)
        # перезаписывается, а не отклоняется как чужой файл
        self.assertEqual(result.planned, 2)
        self.assertEqual(
            [record["attempt"] for record in self.image_records() if record["class"] == "class_a"],
            [1, 1, 1],
        )

    def test_tc64_output_dir_must_match_manifest(self) -> None:
        job = self.job(count=1)
        other = self.root / "other"
        manifest = Manifest.load(other / "manifest.jsonl")
        manifest.append(
            {
                "schema_version": 1,
                "type": "run",
                "run_id": "seeded",
                "command": "run",
                "started_at": "2026-09-26T13:59:42Z",
                "job_file": str(job.source_path),
                "job_sha256": job.raw_sha256,
                "output_dir": str(self.out),
                "task": job.task,
                "generation_mode": job.generation.mode,
                "model_id": MODEL_ID,
                "job_seed": job.seed,
                "classes": [],
                "inference": {},
                "two_layer": None,
            }
        )
        with self.assertRaises(UnsafePathError) as ctx:
            self.execute(job, self.make_deps(), output_dir=other)
        self.assertIn("output_dir differs from the manifest record", ctx.exception.message)

    def test_tc86_job_file_change_is_a_warning(self) -> None:
        job = self.job(count=1)
        self.execute(job, self.make_deps())
        self.warnings.clear()
        path = self.job_path(count=2)
        from datasetgen.config import load_yaml

        changed = load_yaml(path)
        result = self.execute(changed, self.make_deps())
        self.assertEqual(result.exit_code, EXIT_OK)
        self.assertTrue(any("changed since the last run" in text for text in self.warnings), self.warnings)

    def test_second_run_with_everything_done_is_a_noop(self) -> None:
        job = self.job(count=2)
        self.execute(job, self.make_deps())
        deps = self.make_deps()
        result = self.execute(self.job(count=2), deps)
        self.assertEqual(result.exit_code, EXIT_OK)
        self.assertEqual(result.planned, 0)
        self.assertEqual(result.skipped_complete, 4)
        self.assertIn("summary:", self.stdout.getvalue())
        # ничего делать нечего — пайплайн грузить незачем
        self.assertEqual(deps.backends.counters["make_generator"], 0)
        deps.backends.raise_if_called()

    def test_m01_noop_run_survives_a_broken_backend(self) -> None:
        """Даже бэкенд, падающий на конструировании пайплайна, не ломает no-op прогон.

        Это ровно тот случай на машине без CUDA/MPS, где ``OrdinaryGen.__init__``
        бросает ``AttributeError`` (известное ограничение F-2).
        """

        class ExplodingBackends:
            def make_generator(self, model_id):
                raise AttributeError("'OrdinaryGen' object has no attribute 'device'")

            def make_cutter(self, model_name):  # pragma: no cover - не должен вызываться
                raise AssertionError("cutter must not be created")

            def make_inpainter(self, model_id):  # pragma: no cover - не должен вызываться
                raise AssertionError("inpainter must not be created")

        self.execute(self.job(count=1), self.make_deps())
        deps = replace(make_dependencies(), backends=ExplodingBackends())
        result = self.execute(self.job(count=1), deps)
        self.assertEqual(result.exit_code, EXIT_OK)
        self.assertEqual(result.planned, 0)

    def test_m03_leftover_of_a_failed_attempt_does_not_block_the_batch(self) -> None:
        """Остаток упавшей попытки описан манифестом, поэтому он не «чужой» файл.

        Раньше такой индекс ронял весь прогон ``UNSAFE_PATH`` ещё до генерации.
        """

        job = self.job(count=2)
        self.seed_manifest(
            job,
            [
                self.make_image_record(job, "class_a", 1, status="failed"),
                self.make_image_record(job, "class_a", 2, status="succeeded"),
                self.make_image_record(job, "class_b", 1, status="succeeded"),
                self.make_image_record(job, "class_b", 2, status="succeeded"),
            ],
        )
        for name in ("class_a", "class_b"):
            for index in (1, 2):
                write_png(self.out / name / f"{name}_{index}.png", index)
        leftover = self.out / "class_a" / "class_a_1.png"
        before = image_bytes(leftover)
        deps = self.make_deps()
        result = self.execute(job, deps)
        self.assertEqual(result.exit_code, EXIT_OK, self.errors)
        self.assertEqual(result.planned, 1)
        self.assertEqual(self.statuses("class_a", 1), ["failed", "planned", "succeeded"])
        self.assertNotEqual(image_bytes(leftover), before)
        # остальные файлы не тронуты
        self.assertEqual(
            sorted(entry["save_path"].name for entry in deps.backends.calls("generate_image")),
            ["class_a_1.png"],
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
