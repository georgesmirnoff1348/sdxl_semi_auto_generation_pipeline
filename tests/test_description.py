"""TC-75..TC-78: локальный отчёт и opt-in заготовка для внешнего агента."""

from __future__ import annotations

import unittest

from datasetgen.description import write_reports
from tests.fakes import TmpCase, two_layer_generation


class TestDescription(TmpCase):
    def description_file(self) -> str:
        return (self.out / "dataset_description.md").read_text(encoding="utf-8")

    def request_file(self) -> str:
        return (self.out / "dataset_description_request.md").read_text(encoding="utf-8")

    def test_tc75_disabled_creates_nothing(self) -> None:
        job = self.job(count=1, description={"enabled": False})
        self.execute(job, self.make_deps())
        self.assertFalse((self.out / "dataset_description.md").exists())
        self.assertFalse((self.out / "dataset_description_request.md").exists())

    def test_tc76_report_contains_facts(self) -> None:
        job = self.job(count=2, description={"enabled": True, "max_prompt_rows": 50})
        self.execute(job, self.make_deps())
        text = self.description_file()
        self.assertIn("classification", text)
        self.assertIn("single_layer", text)
        self.assertIn("class_a", text)
        self.assertIn("class_b", text)
        self.assertIn("test/model-id", text)
        self.assertIn("num_inference_steps", text)
        self.assertIn("datasetgen/v1", text)
        self.assertIn("Воспроизводимость", text)
        self.assertIn("Сгенерировано локально командой `datasetgen`", text)
        self.assertIn("внешние агенты не вызывались", text)
        self.assertIn("successes: 4", text)
        self.assertIn("failures: 0", text)
        self.assertIn("| класс | запрошено | произведено | отклонено |", text)
        # реализации промптов содержат seed и сам промпт
        self.assertIn("| класс | index | attempt | seed | prompt |", text)
        self.assertIn("factory worker", text)

    def test_tc76_two_layer_report_has_background_column(self) -> None:
        job = self.job(
            count=1,
            description={"enabled": True},
            generation=two_layer_generation(),
        )
        self.execute(job, self.make_deps())
        text = self.description_file()
        self.assertIn("| класс | index | attempt | seed | prompt | background prompt |", text)
        self.assertIn("test/background-model-id", text)
        self.assertIn("generation.cutter: `u2net`", text)

    def test_m04_produced_counts_effective_state_not_every_attempt(self) -> None:
        """После regenerate «произведено» = файлы в каталоге, а не число попыток."""

        job = self.job(count=2, description={"enabled": True})
        self.execute(job, self.make_deps())
        self.execute_regenerate(job, self.make_deps(), "class_a:1")
        text = self.description_file()
        # 2 файла на класс: произведено = 2, хотя succeeded-попыток было 3
        self.assertIn("| `class_a` | 2 | 2 | 0 |", text)
        self.assertIn("| `class_b` | 2 | 2 | 0 |", text)
        # таблица реализаций и счётчик не противоречат друг другу
        section = text.split("## 4.")[1].split("## 5.")[0]
        rows = [line for line in section.splitlines() if line.startswith("| `class_a` |")]
        self.assertEqual(len(rows), 2, rows)

    def test_prompt_table_is_truncated(self) -> None:
        job = self.job(count=5, description={"enabled": True, "max_prompt_rows": 3})
        self.execute(job, self.make_deps())
        text = self.description_file()
        self.assertIn("показаны первые 3 из 10", text)

    def test_tc77_missing_request_file_warns_but_report_is_written(self) -> None:
        job = self.job(
            count=1,
            description={"enabled": True, "request_file": str(self.root / "absent.md")},
        )
        self.execute(job, self.make_deps())
        self.assertTrue((self.out / "dataset_description.md").is_file())
        self.assertFalse((self.out / "dataset_description_request.md").exists())
        self.assertTrue(
            any("request_file" in text for text in self.warnings), self.warnings
        )

    def test_tc78_request_file_is_embedded_and_untouched(self) -> None:
        source = self.root / "brief.md"
        source.write_text("Нужен датасет рабочих завода.\n", encoding="utf-8")
        before = source.read_bytes()
        job = self.job(
            count=1,
            description={"enabled": True, "request_file": str(source)},
        )
        self.execute(job, self.make_deps())
        text = self.request_file()
        self.assertIn("ЗАДАЧА ДЛЯ ВНЕШНЕГО АГЕНТА", text)
        self.assertIn("Нужен датасет рабочих завода.", text)
        self.assertIn("datasetgen` не выполнял", text)
        self.assertIn("Исходный материал пользователя", text)
        self.assertIn("class_a", text)
        self.assertEqual(source.read_bytes(), before)

    def test_write_reports_is_a_noop_when_disabled(self) -> None:
        job = self.job(count=1, description={"enabled": False})
        self.assertFalse(
            write_reports(
                job=job,
                model_id="test/model-id",
                root=self.out,
                state=self._empty_state(),
                succeeded=0,
                failed=0,
                rejected_moved=0,
                errors=[],
            )
        )
        self.assertFalse((self.out / "dataset_description.md").exists())

    def _empty_state(self):
        from datasetgen.manifest import ManifestState

        return ManifestState()

    def test_no_subprocess_or_network_in_package(self) -> None:
        from datasetgen import description as module

        source = module.__file__
        with open(source, encoding="utf-8") as handle:
            text = handle.read()
        for forbidden in ("subprocess", "os.system", "urllib", "requests", "http"):
            self.assertNotIn(forbidden, text)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
