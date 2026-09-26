"""TC-33..TC-37: формат манифеста, битые строки, неполный хвост (R9)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from datasetgen.errors import ManifestError
from datasetgen.manifest import SCHEMA_VERSION, Manifest


def run_record(run_id: str = "run-1", output_dir: str = "/tmp/out") -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "type": "run",
        "run_id": run_id,
        "command": "run",
        "started_at": "2026-09-26T13:59:42Z",
        "job_file": "/abs/repo/examples/classification.yaml",
        "job_sha256": "3b1f",
        "output_dir": output_dir,
        "task": "classification",
        "generation_mode": "single_layer",
        "model_id": "test/model-id",
        "job_seed": 20260926,
        "classes": [{"name": "class_a", "count": 12}],
        "inference": {
            "num_inference_steps": 20,
            "guidance_scale": 7.0,
            "width": 1024,
            "height": 1024,
            "extra": {},
        },
        "two_layer": None,
    }


def image_record(index: int = 1, status: str = "succeeded", attempt: int = 1) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "type": "image",
        "run_id": "run-1",
        "class": "class_a",
        "index": index,
        "attempt": attempt,
        "status": status,
        "output": f"class_a/class_a_{index}.png",
        "prompt": "a factory worker",
        "negative_prompt": "3d render",
        "seed": 1837465920,
        "model_id": "test/model-id",
        "settings": {
            "num_inference_steps": 20,
            "guidance_scale": 7.0,
            "width": 1024,
            "height": 1024,
            "extra": {},
        },
        "generation_mode": "single_layer",
        "intermediate": [],
        "timestamp": "2026-09-26T13:59:43Z",
        "error": None,
        "bytes": 1042381 if status == "succeeded" else None,
        "duration_sec": 9.41 if status == "succeeded" else None,
    }


class ManifestTestCase(unittest.TestCase):
    def setUp(self) -> None:
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        self.root = Path(holder.name)
        self.path = self.root / "manifest.jsonl"

    def write_lines(self, *lines: str) -> None:
        self.path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")

    def dump(self, record: dict) -> str:
        return json.dumps(record, ensure_ascii=False, sort_keys=True)


class TestAppendAndRead(ManifestTestCase):
    def test_missing_file_is_a_clean_start(self) -> None:
        manifest = Manifest.load(self.path)
        self.assertEqual(list(manifest.records), [])
        self.assertEqual(dict(manifest.state().last), {})

    def test_tc33_records_round_trip(self) -> None:
        manifest = Manifest.load(self.path)
        manifest.append(run_record())
        manifest.append(image_record(status="planned"))
        manifest.append(image_record(status="succeeded"))
        lines = self.path.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 3)
        for line in lines:
            payload = json.loads(line)
            self.assertEqual(payload["schema_version"], SCHEMA_VERSION)
        # sort_keys=True и ensure_ascii=False
        self.assertTrue(lines[0].startswith('{"classes"'))
        self.assertIn('"output": "class_a/class_a_1.png"', lines[2])

    def test_unicode_is_not_escaped(self) -> None:
        manifest = Manifest.load(self.path)
        record = image_record()
        record["prompt"] = "рабочий на заводе"
        manifest.append(record)
        raw = self.path.read_text(encoding="utf-8")
        self.assertIn("рабочий на заводе", raw)
        self.assertNotIn("\\u", raw)

    def test_output_paths_are_relative_posix(self) -> None:
        manifest = Manifest.load(self.path)
        record = image_record()
        manifest.append(record)
        payload = json.loads(self.path.read_text(encoding="utf-8").strip())
        self.assertEqual(payload["output"], "class_a/class_a_1.png")
        self.assertNotIn("\\", payload["output"])
        self.assertFalse(payload["output"].startswith("/"))

    def test_state_reports_last_status(self) -> None:
        self.write_lines(
            self.dump(image_record(status="planned")),
            self.dump(image_record(status="failed")),
            self.dump(image_record(status="planned", attempt=2)),
            self.dump(image_record(status="succeeded", attempt=2)),
        )
        state = Manifest.load(self.path).state()
        key = ("class_a", 1)
        self.assertTrue(state.has_succeeded(key))
        self.assertEqual(state.attempts(key), 2)
        self.assertEqual(state.last_record(key)["status"], "succeeded")

    def test_state_reports_rejected(self) -> None:
        self.write_lines(
            self.dump(image_record(status="succeeded")),
            self.dump({**image_record(status="rejected"), "rejected_file": "_rejected/class_a_1_1.png"}),
        )
        state = Manifest.load(self.path).state()
        self.assertTrue(state.is_rejected(("class_a", 1)))


class TestCorruption(ManifestTestCase):
    def test_tc35_corrupt_line_is_skipped(self) -> None:
        self.write_lines(
            self.dump(run_record()),
            "{not json at all",
            self.dump(image_record(status="succeeded")),
        )
        warnings: list[str] = []
        manifest = Manifest.load(self.path, warn=warnings.append)
        self.assertEqual(len(manifest.records), 2)
        self.assertEqual(manifest.corrupt_lines, (2,))
        self.assertIn("manifest: skipping corrupt line 2", warnings)

    def test_unknown_record_type_is_corrupt(self) -> None:
        self.write_lines(self.dump(run_record()), json.dumps({"type": "mystery"}))
        warnings: list[str] = []
        manifest = Manifest.load(self.path, warn=warnings.append)
        self.assertEqual(len(manifest.records), 1)
        self.assertEqual(manifest.corrupt_lines, (2,))

    def test_tc37_binary_garbage_is_manifest_error(self) -> None:
        self.path.write_bytes(b"\x00\x01\x02\xff\xfe not json \x80\x81")
        with self.assertRaises(ManifestError) as ctx:
            Manifest.load(self.path)
        self.assertIn("manifest is not valid JSONL", ctx.exception.message)
        self.assertIn(str(self.path), ctx.exception.message)

    def test_only_corrupt_lines_is_manifest_error(self) -> None:
        self.write_lines("garbage", "more garbage")
        with self.assertRaises(ManifestError) as ctx:
            Manifest.load(self.path)
        self.assertIn("manifest is not valid JSONL", ctx.exception.message)

    def test_l09_valid_json_with_wrong_field_types_is_corrupt(self) -> None:
        """Правильный JSON с неверным типом поля — битая строка, а не повод для traceback."""

        self.write_lines(
            self.dump(run_record()),
            json.dumps({"type": "image", "class": "class_a", "index": "1", "attempt": 1}),
            json.dumps({"type": "image", "class": "class_a", "index": 1, "attempt": None}),
            json.dumps({"type": "image", "class": 5, "index": 1, "attempt": 1}),
            json.dumps({"type": "image", "class": "class_a", "index": True, "attempt": 1}),
            self.dump(image_record()),
        )
        warnings: list[str] = []
        manifest = Manifest.load(self.path, warn=warnings.append)
        self.assertEqual(manifest.corrupt_lines, (2, 3, 4, 5))
        self.assertEqual(len(manifest.records), 2)
        self.assertEqual(
            warnings,
            [f"manifest: skipping corrupt line {number}" for number in (2, 3, 4, 5)],
        )
        # state() обязан пережить такие строки и восстановить рабочие записи
        state = manifest.state()
        self.assertTrue(state.has_succeeded(("class_a", 1)))
        self.assertEqual(state.attempts(("class_a", 1)), 1)


class TestTailRepair(ManifestTestCase):
    def test_tc36_incomplete_tail_is_truncated(self) -> None:
        self.path.write_text(
            self.dump(run_record()) + "\n" + '{"type": "image", "class": "class_a"',
            encoding="utf-8",
        )
        manifest = Manifest.load(self.path)
        self.assertTrue(manifest.truncated_tail)
        self.assertEqual(len(manifest.records), 1)
        self.assertTrue(manifest.repair_truncated_tail())
        text = self.path.read_text(encoding="utf-8")
        self.assertTrue(text.endswith("\n"))
        self.assertEqual(len(text.strip().splitlines()), 1)
        manifest.append(image_record())
        reread = Manifest.load(self.path)
        self.assertEqual(len(reread.records), 2)

    def test_l10_file_that_is_one_incomplete_line_is_a_clean_start(self) -> None:
        """R9: последней полной строки нет — файл обрезается целиком, а не падает."""

        self.path.write_text('{"type": "image", "class": "class_a"', encoding="utf-8")
        manifest = Manifest.load(self.path)
        self.assertTrue(manifest.truncated_tail)
        self.assertEqual(list(manifest.records), [])
        self.assertTrue(manifest.repair_truncated_tail())
        self.assertEqual(self.path.read_bytes(), b"")
        manifest.append(run_record())
        self.assertEqual(len(Manifest.load(self.path).records), 1)

    def test_l10_incomplete_line_after_corrupt_line_is_still_an_error(self) -> None:
        self.path.write_text("garbage\n" + '{"type": "image", "class": "class_a"', encoding="utf-8")
        with self.assertRaises(ManifestError):
            Manifest.load(self.path)

    def test_complete_file_is_not_touched(self) -> None:
        self.write_lines(self.dump(run_record()))
        before = self.path.read_bytes()
        manifest = Manifest.load(self.path)
        self.assertFalse(manifest.truncated_tail)
        self.assertFalse(manifest.repair_truncated_tail())
        self.assertEqual(self.path.read_bytes(), before)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
