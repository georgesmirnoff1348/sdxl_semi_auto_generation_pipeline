"""TC-12: ручной рендер шаблонов, экранирование скобок, валидация плейсхолдеров."""

from __future__ import annotations

import unittest

from datasetgen.errors import ConfigError
from datasetgen.prompts import extract_placeholders, render, validate_template, validate_variables


class TestRender(unittest.TestCase):
    def test_tc12_escaped_braces_render_as_literals(self) -> None:
        rendered = render("a {{literal}} prompt for {worker}", {"worker": "comrade"})
        self.assertEqual(rendered, "a {literal} prompt for comrade")

    def test_escaped_and_variable_together(self) -> None:
        rendered = render("{{ {worker} }} in {place}", {"worker": "worker", "place": "hall"})
        self.assertEqual(rendered, "{ worker } in hall")

    def test_lone_closing_brace_is_literal(self) -> None:
        self.assertEqual(render("100% } ok", {}), "100% } ok")

    def test_unknown_variable(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            render("a {worker}", {}, label="classes[0]")
        self.assertEqual(ctx.exception.message, "classes[0].template: unknown variable 'worker'")

    def test_invalid_placeholder(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            render("a {Worker}", {}, label="classes[0]")
        self.assertIn("invalid placeholder '{Worker}'", ctx.exception.message)

    def test_unterminated_placeholder(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            render("a {worker", {})
        self.assertIn("unterminated placeholder", ctx.exception.message)


class TestExtractAndValidate(unittest.TestCase):
    def test_extract_keeps_order_and_uniqueness(self) -> None:
        names = extract_placeholders("{b} then {a} then {b} done")
        self.assertEqual(names, ("b", "a"))

    def test_extract_ignores_escaped(self) -> None:
        self.assertEqual(extract_placeholders("{{a}} {b}"), ("b",))

    def test_validate_template_reports_unknown_variable(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            validate_template("a {worker} in {place}", {"worker": ("w",)}, "classes[0]")
        self.assertEqual(ctx.exception.message, "classes[0].template: unknown variable 'place'")

    def test_validate_template_reports_unused_variable(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            validate_template("a {worker}", {"worker": ("w",), "background": ("b",)}, "classes[0]")
        self.assertEqual(
            ctx.exception.message, "classes[0].variables: unused variable 'background'"
        )

    def test_validate_variables_rejects_empty_value(self) -> None:
        with self.assertRaises(ConfigError):
            validate_variables({"worker": ("",)}, "classes[0]")

    def test_validate_variables_rejects_newlines(self) -> None:
        with self.assertRaises(ConfigError):
            validate_variables({"worker": ("a\nb",)}, "classes[0]")

    def test_validate_variables_rejects_non_strings(self) -> None:
        with self.assertRaises(ConfigError):
            validate_variables({"worker": (1,)}, "classes[0]")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
