"""Opt-in end-to-end check: a built wheel must be installable and runnable.

По умолчанию тест ВЫКЛЮЧЕН: сборка wheel и создание venv требуют сети
(``hatchling`` зафиксирован в ``pyproject.toml``, но не в ``uv.lock``, поэтому
изолированное build-окружение всё равно скачивается с PyPI) и занимают минуты.
Лёгкий прогон ``python -m unittest discover -s tests -t .`` остаётся быстрым.

Включение::

    DATASETGEN_WHEEL_E2E=1 uv run --locked python -m unittest tests.test_wheel_install -v
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VERIFY_SCRIPT = REPO_ROOT / "tools" / "verify_wheel.py"

#: Сборка + venv + установка занимают минуты; таймаут с большим запасом.
TIMEOUT_SECONDS = 3600


@unittest.skipUnless(
    os.environ.get("DATASETGEN_WHEEL_E2E") == "1",
    "wheel install check is opt-in (set DATASETGEN_WHEEL_E2E=1): it builds a wheel "
    "with `uv build`, which downloads the isolated build backend (hatchling is "
    "pinned in pyproject.toml, but not part of uv.lock), then creates a fresh "
    "venv, installs the base package and runs the console script — slow and "
    "network-dependent, so it is excluded from the default test run",
)
class WheelInstallTest(unittest.TestCase):
    """Делегирует всю проверку ``tools/verify_wheel.py``."""

    def test_wheel_installs_and_runs_outside_checkout(self) -> None:
        self.assertTrue(
            VERIFY_SCRIPT.is_file(),
            f"verification script is missing: {VERIFY_SCRIPT}",
        )
        completed = subprocess.run(
            [sys.executable, str(VERIFY_SCRIPT)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            check=False,
        )
        if completed.returncode != 0:
            self.fail(
                f"tools/verify_wheel.py failed with exit code {completed.returncode} "
                f"(expected 0 and a final 'OK: wheel verified' line)\n"
                f"--- stdout ---\n{completed.stdout}\n"
                f"--- stderr ---\n{completed.stderr}"
            )
        self.assertIn(
            "OK: wheel verified",
            completed.stdout,
            msg=f"tools/verify_wheel.py exited 0 without the success line\n{completed.stdout}",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
