"""TC-46..TC-55: грамматика и разбор ``--remove``."""

from __future__ import annotations

import unittest

from datasetgen.errors import SelectorError
from datasetgen.selectors import parse_remove

CLASSIFICATION = ("class_a", "class_b")
COUNTS = {"class_a": 12, "class_b": 12}


def parse(selector: str, task: str = "classification", counts=None, names=None) -> tuple:
    return parse_remove(
        selector,
        names or CLASSIFICATION,
        counts or COUNTS,
        task,
    )


class TestParsing(unittest.TestCase):
    def test_tc46_two_classes(self) -> None:
        self.assertEqual(
            parse("class_a:3,7-9;class_b:4"),
            (
                ("class_a", 3),
                ("class_a", 7),
                ("class_a", 8),
                ("class_a", 9),
                ("class_b", 4),
            ),
        )

    def test_tc47_anomaly_without_class(self) -> None:
        self.assertEqual(
            parse("3,7-9", task="anomaly_detection", names=("normal",), counts={"normal": 12}),
            (("normal", 3), ("normal", 7), ("normal", 8), ("normal", 9)),
        )

    def test_tc48_spaces_are_trimmed(self) -> None:
        self.assertEqual(
            parse("class_a: 1-3, 5"),
            (("class_a", 1), ("class_a", 2), ("class_a", 3), ("class_a", 5)),
        )

    def test_equal_range_is_allowed(self) -> None:
        self.assertEqual(parse("class_a:4-4"), (("class_a", 4),))

    def test_order_is_normalized(self) -> None:
        self.assertEqual(
            parse("class_b:2;class_a:9,1"),
            (("class_a", 1), ("class_a", 9), ("class_b", 2)),
        )


class TestErrors(unittest.TestCase):
    def assertSelectorError(self, selector: str, *needles: str, **kwargs) -> str:
        with self.assertRaises(SelectorError) as ctx:
            parse(selector, **kwargs)
        message = ctx.exception.message
        for needle in needles:
            self.assertIn(needle, message)
        return message

    def test_tc49_malformed_selectors(self) -> None:
        self.assertSelectorError("  ", "empty selector")
        self.assertSelectorError(";", "unexpected token '' in --remove")
        self.assertSelectorError("1,,2", "unexpected token '' in --remove")
        self.assertSelectorError("class_a:", "unexpected token '' in --remove")
        self.assertSelectorError("class_a:1,,2", "unexpected token '' in --remove")

    def test_tc49_unknown_class_token(self) -> None:
        self.assertSelectorError(
            "nope:", "unknown class 'nope' in --remove (known: class_a, class_b)"
        )

    def test_tc50_zero_index(self) -> None:
        self.assertSelectorError("0", "index 0 is invalid (indices are 1-based)")

    def test_tc51_reversed_range(self) -> None:
        self.assertSelectorError("9-7", "invalid range '9-7' in --remove")

    def test_tc52_out_of_range(self) -> None:
        self.assertSelectorError(
            "class_a:25", "index 25 is out of range for class 'class_a' (count 12)"
        )

    def test_tc53_unknown_class(self) -> None:
        self.assertSelectorError(
            "class_z:3", "unknown class 'class_z' in --remove (known: class_a, class_b)"
        )

    def test_tc54_class_required_for_classification(self) -> None:
        self.assertSelectorError(
            "3,7-9", "class is required in --remove for task 'classification'"
        )

    def test_tc55_duplicates_and_overlaps(self) -> None:
        for selector in ("class_a:3,7-9,8", "class_a:3;class_a:3", "class_a:1-5,3-7"):
            self.assertSelectorError(
                selector, "duplicate index", "for class 'class_a' in --remove"
            )

    def test_unexpected_token(self) -> None:
        self.assertSelectorError("class_a:abc", "unexpected token 'abc' in --remove")

    def test_range_out_of_range(self) -> None:
        self.assertSelectorError(
            "class_a:10-13", "index 13 is out of range for class 'class_a' (count 12)"
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
