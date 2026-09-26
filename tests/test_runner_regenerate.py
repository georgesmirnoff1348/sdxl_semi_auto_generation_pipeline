"""TC-22, TC-41..TC-45, TC-63, TC-85: отклонение и перегенерация."""

from __future__ import annotations

import unittest

from datasetgen.errors import EXIT_GENERATION_FAILED, EXIT_OK, UnsafePathError
from datasetgen.manifest import Manifest
from datasetgen.planner import build_plan, make_output
from tests.fakes import MODEL_ID, FailingStage, TmpCase, image_bytes, write_png


class RegenerateBase(TmpCase):
    """Состояние «уже сгенерировано» готовится детерминированно (без реальных картинок)."""

    def prepared(self, count: int = 3, name: str = "job.yaml"):
        job = self.job(count=count, name=name)
        records = []
        for spec in job.classes:
            for index in range(1, spec.count + 1):
                records.append(self.make_image_record(job, spec.name, index, status="succeeded"))
                write_png(self.out / spec.name / f"{spec.name}_{index}.png", index)
        self.seed_manifest(job, records)
        return job

    def rejected_copy(self, index: int, attempt: int = 1, class_name: str = "class_a"):
        return self.out / "_rejected" / f"{class_name}_{index}_{attempt}.png"


class TestRejectAndRegenerate(RegenerateBase):
    def test_tc41_file_is_moved_and_kept(self) -> None:
        job = self.prepared()
        original = self.out / "class_a" / "class_a_3.png"
        payload = image_bytes(original)
        result = self.execute_regenerate(job, self.make_deps(), "class_a:3")
        self.assertEqual(result.exit_code, EXIT_OK, self.errors)
        self.assertEqual(result.rejected_moved, 1)
        # старый файл перемещён (его байты — в _rejected), новый создан заново
        self.assertTrue(self.rejected_copy(3).is_file())
        self.assertEqual(image_bytes(self.rejected_copy(3)), payload)
        self.assertNotEqual(image_bytes(original), payload)

    def test_tc42_manifest_transitions(self) -> None:
        job = self.prepared()
        self.execute_regenerate(job, self.make_deps(), "class_a:3")
        statuses = self.statuses("class_a", 3)
        self.assertEqual(statuses, ["succeeded", "rejected", "planned", "succeeded"])
        records = [
            record for record in self.image_records() if record["class"] == "class_a" and record["index"] == 3
        ]
        self.assertEqual({record["index"] for record in records}, {3})
        self.assertEqual([record["attempt"] for record in records], [1, 1, 2, 2])
        self.assertTrue(all(record["output"] == "class_a/class_a_3.png" for record in records))
        rejected = [record for record in records if record["status"] == "rejected"][0]
        self.assertEqual(rejected["rejected_file"], "_rejected/class_a_3_1.png")
        self.assertEqual(rejected["reason"], "cli --remove")
        self.assertIn("rejected_at", rejected)

    def test_tc43_new_realization_differs(self) -> None:
        job = self.prepared()
        rejected_before = [
            record for record in self.image_records() if record["class"] == "class_a" and record["index"] == 3
        ][-1]
        self.execute_regenerate(job, self.make_deps(), "class_a:3")
        records = [
            record for record in self.image_records() if record["class"] == "class_a" and record["index"] == 3
        ]
        # rejected-запись ДОЛЖНА зафиксировать реализацию отклонённой попытки
        rejected = [record for record in records if record["status"] == "rejected"][0]
        self.assertEqual(rejected["seed"], rejected_before["seed"])
        self.assertEqual(rejected["prompt"], rejected_before["prompt"])
        # новая попытка детерминирована формулой и обязана отличаться по seed
        second = [record for record in records if record["attempt"] == 2][-1]
        expected = make_output(job, "class_a", 3, 2)
        self.assertEqual(second["seed"], expected.seed)
        self.assertEqual(second["prompt"], expected.prompt)
        self.assertNotEqual(second["seed"], rejected["seed"])

    def test_tc44_second_reject_keeps_both_copies(self) -> None:
        job = self.prepared()
        self.execute_regenerate(job, self.make_deps(), "class_a:3")
        payload = image_bytes(self.out / "class_a" / "class_a_3.png")
        result = self.execute_regenerate(job, self.make_deps(), "class_a:3")
        self.assertEqual(result.exit_code, EXIT_OK, self.errors)
        self.assertTrue(self.rejected_copy(3, 1).is_file())
        self.assertTrue(self.rejected_copy(3, 2).is_file())
        self.assertEqual(image_bytes(self.rejected_copy(3, 2)), payload)
        self.assertTrue((self.out / "class_a" / "class_a_3.png").is_file())

    def test_tc45_index_without_file_is_not_moved(self) -> None:
        job = self.prepared()
        (self.out / "class_a" / "class_a_3.png").unlink()
        result = self.execute_regenerate(job, self.make_deps(), "class_a:3")
        self.assertEqual(result.exit_code, EXIT_OK, self.errors)
        self.assertEqual(result.rejected_moved, 0)
        self.assertTrue(any("nothing to reject" in text for text in self.warnings), self.warnings)
        attempts = [
            record["attempt"]
            for record in self.image_records()
            if record["class"] == "class_a" and record["index"] == 3
        ]
        self.assertEqual(attempts[-2:], [2, 2])
        self.assertTrue((self.out / "class_a" / "class_a_3.png").is_file())
        # L-01: даже когда переносить нечего, rejected-запись документирует отказ
        rejected = [record for record in self.image_records() if record["status"] == "rejected"][0]
        self.assertIn("rejected_file", rejected)
        self.assertIsNone(rejected["rejected_file"])
        self.assertEqual(rejected["reason"], "cli --remove")
        self.assertIn("rejected_at", rejected)

    def test_m02_failed_attempt_is_not_marked_rejected(self) -> None:
        """В графе статусов нет перехода failed -> rejected: сразу planned (attempt+1)."""

        job = self.job(count=2)
        self.seed_manifest(
            job,
            [
                self.make_image_record(job, "class_a", 1, status="failed"),
                self.make_image_record(job, "class_a", 2, status="succeeded"),
            ],
        )
        write_png(self.out / "class_a" / "class_a_2.png", 2)
        result = self.execute_regenerate(job, self.make_deps(), "class_a:1")
        self.assertEqual(result.exit_code, EXIT_OK, self.errors)
        self.assertEqual(result.rejected_moved, 0)
        self.assertEqual(self.statuses("class_a", 1), ["failed", "planned", "succeeded"])
        self.assertTrue((self.out / "class_a" / "class_a_1.png").is_file())
        self.assertFalse((self.out / "_rejected").exists())

    def test_m02_leftover_of_failed_attempt_is_replaced_not_moved(self) -> None:
        job = self.job(count=1)
        self.seed_manifest(job, [self.make_image_record(job, "class_a", 1, status="failed")])
        leftover = self.out / "class_a" / "class_a_1.png"
        write_png(leftover, 99)
        before = image_bytes(leftover)
        result = self.execute_regenerate(job, self.make_deps(), "class_a:1")
        self.assertEqual(result.exit_code, EXIT_OK, self.errors)
        self.assertEqual(result.rejected_moved, 0)
        self.assertEqual(self.statuses("class_a", 1), ["failed", "planned", "succeeded"])
        self.assertFalse((self.out / "_rejected").exists())
        self.assertNotEqual(image_bytes(leftover), before)

    def test_tc63_existing_rejected_target_is_fatal(self) -> None:
        job = self.prepared()
        write_png(self.rejected_copy(3), 5)
        with self.assertRaises(UnsafePathError) as ctx:
            self.execute_regenerate(job, self.make_deps(), "class_a:3")
        self.assertIn("rejected target already exists", ctx.exception.message)

    def test_tc85_failing_backend_leaves_no_partial_files(self) -> None:
        job = self.prepared()
        result = self.execute_regenerate(
            job, self.make_deps(fail_at=FailingStage.SINGLE), "class_a:3"
        )
        self.assertEqual(result.exit_code, EXIT_GENERATION_FAILED)
        self.assertEqual(result.failed, 1)
        failed = [record for record in self.image_records() if record["status"] == "failed"]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["error"]["stage"], "single")
        self.assertFalse((self.out / "class_a" / "class_a_3.png").exists())
        self.assertTrue(self.rejected_copy(3).is_file())

    def test_dry_run_regenerate_writes_nothing(self) -> None:
        job = self.prepared(count=9)
        payload = image_bytes(self.out / "class_a" / "class_a_3.png")
        deps = self.make_deps()
        result = self.execute_regenerate(job, deps, "class_a:3,7-9;class_b:4", dry_run=True)
        self.assertEqual(result.exit_code, EXIT_OK, self.errors)
        self.assertEqual(result.planned, 5)
        self.assertEqual(image_bytes(self.out / "class_a" / "class_a_3.png"), payload)
        self.assertFalse((self.out / "_rejected").exists())
        deps.backends.raise_if_called()

    def test_run_record_contains_removed_list(self) -> None:
        job = self.prepared()
        self.execute_regenerate(job, self.make_deps(), "class_a:2,3")
        record = self.run_records()[-1]
        self.assertEqual(record["command"], "regenerate")
        self.assertEqual(
            record["removed"], [{"class": "class_a", "index": 2}, {"class": "class_a", "index": 3}]
        )

    def test_other_indexes_are_untouched(self) -> None:
        job = self.prepared()
        payload = image_bytes(self.out / "class_a" / "class_a_1.png")
        self.execute_regenerate(job, self.make_deps(), "class_a:3")
        self.assertEqual(image_bytes(self.out / "class_a" / "class_a_1.png"), payload)
        self.assertEqual(self.statuses("class_a", 1), ["succeeded"])

    def test_regenerate_accepts_string_selector(self) -> None:
        job = self.prepared()
        result = self.execute_regenerate(job, self.make_deps(), "class_a:1")
        self.assertEqual(result.exit_code, EXIT_OK, self.errors)
        self.assertEqual(result.rejected_moved, 1)


class TestGuard(RegenerateBase):
    """TC-22: никогда не воспроизводим уже существующую реализацию (seed, prompt)."""

    def seed_identical_future(self, job, class_name: str, index: int, attempt: int) -> None:
        """Дописывает запись, полностью совпадающую с будущей попыткой ``attempt``."""

        upcoming = build_plan(
            job,
            MODEL_ID,
            "regenerate",
            max_attempts={(class_name, index): attempt - 1},
            selection=((class_name, index),),
        ).outputs[0]
        manifest = Manifest.load(self.manifest_path())
        manifest.append(
            self.make_image_record(
                job,
                class_name,
                index,
                attempt=attempt - 1,
                status="succeeded",
                seed=upcoming.seed,
                prompt=upcoming.prompt,
            )
        )

    def test_tc22_identical_realization_is_refused(self) -> None:
        job = self.prepared()
        self.seed_identical_future(job, "class_a", 3, 3)
        payload = image_bytes(self.out / "class_a" / "class_a_3.png")

        result = self.execute_regenerate(job, self.make_deps(), "class_a:3")
        self.assertEqual(result.exit_code, EXIT_GENERATION_FAILED)
        self.assertEqual(result.failed, 1)
        self.assertEqual(result.rejected_moved, 0)
        # файл не перезаписан и не перенесён
        self.assertEqual(image_bytes(self.out / "class_a" / "class_a_3.png"), payload)
        failed = [
            record
            for record in self.image_records()
            if record["status"] == "failed" and record["attempt"] == 3
        ]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["error"]["stage"], "guard")
        self.assertIn(
            "refusing to reproduce an identical (seed, prompt) realization",
            failed[0]["error"]["message"],
        )
        self.assertIn("class_a index 3 attempt 3", failed[0]["error"]["message"])
        self.assertIn("add more alternatives to variables", failed[0]["error"]["message"])
        self.assertTrue(
            any("[CONFIG] refusing to reproduce" in text for text in self.errors), self.errors
        )

    def test_guard_does_not_stop_other_outputs(self) -> None:
        job = self.prepared()
        self.seed_identical_future(job, "class_a", 1, 2)
        result = self.execute_regenerate(job, self.make_deps(), "class_a:1;class_b:1")
        self.assertEqual(result.failed, 1, self.errors)
        self.assertEqual(result.succeeded, 1)
        self.assertTrue((self.out / "class_b" / "class_b_1.png").is_file())

    def test_l02_guard_covers_every_past_realization(self) -> None:
        """Совпадение с ЛЮБОЙ прошлой реализацией блокирует попытку, не только с последней."""

        job = self.prepared()
        upcoming = build_plan(
            job,
            MODEL_ID,
            "regenerate",
            max_attempts={("class_a", 3): 2},
            selection=(("class_a", 3),),
        ).outputs[0]
        manifest = Manifest.load(self.manifest_path())
        # совпадающая реализация приписана НЕ последней записи ключа
        manifest.append(
            self.make_image_record(
                job,
                "class_a",
                3,
                attempt=1,
                status="succeeded",
                seed=upcoming.seed,
                prompt=upcoming.prompt,
            )
        )
        manifest.append(
            self.make_image_record(
                job, "class_a", 3, attempt=2, status="succeeded", seed=7, prompt="unrelated"
            )
        )
        payload = image_bytes(self.out / "class_a" / "class_a_3.png")

        result = self.execute_regenerate(job, self.make_deps(), "class_a:3")
        self.assertEqual(result.exit_code, EXIT_GENERATION_FAILED)
        self.assertEqual(result.failed, 1)
        self.assertEqual(result.rejected_moved, 0)
        self.assertEqual(image_bytes(self.out / "class_a" / "class_a_3.png"), payload)
        failed = [record for record in self.image_records() if record["status"] == "failed"]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["error"]["stage"], "guard")
        self.assertIn("class_a index 3 attempt 3", failed[0]["error"]["message"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
