"""Tests for tests/conftest.py, the switch that skips ``@pytest.mark.slow`` tests by default
(kept next to the rules tests because the rules worker owns the conftest)."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

CONFTEST = Path(__file__).resolve().parent / "conftest.py"
_spec = importlib.util.spec_from_file_location("hivemind_tests_conftest", CONFTEST)
assert _spec is not None and _spec.loader is not None
conftest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(conftest)  # by path: independent of how pytest imported tests/conftest.py
SLOW_ENV, slow_selected = conftest.SLOW_ENV, conftest.slow_selected


@pytest.mark.parametrize("expr,selected", [
    ("", False),
    ("slow", True),
    ("not slow", False),
    ("slow and network", True),
    ("network and not slow", False),
    ("slow or not slow", True),
    ("(slow)", True),
    ("network", False),
    ("slowish", False),
])
def test_slow_selected(expr: str, selected: bool) -> None:
    assert slow_selected(expr) is selected


def _run(tmp_path: Path, *args: str, env_slow: str | None = None) -> str:
    env = {k: v for k, v in os.environ.items() if k != SLOW_ENV}
    if env_slow is not None:
        env[SLOW_ENV] = env_slow
    out = subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-rs", *args],
                         cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    return out.stdout


def test_slow_tests_run_only_when_selected(tmp_path: Path) -> None:
    shutil.copy(CONFTEST, tmp_path / "conftest.py")
    (tmp_path / "pytest.ini").write_text("[pytest]\nmarkers =\n    slow: long\n")
    (tmp_path / "test_x.py").write_text(
        "import pytest\n\n"
        "def test_fast():\n    pass\n\n"
        "@pytest.mark.slow\n"
        "def test_slow():\n    pass\n")
    default = _run(tmp_path)
    assert "1 passed, 1 skipped" in default and "run with -m slow or HIVEMIND_SLOW=1" in default
    assert "1 passed, 1 deselected" in _run(tmp_path, "-m", "slow")
    assert "1 passed, 1 deselected" in _run(tmp_path, "-m", "not slow")
    assert "2 passed" in _run(tmp_path, env_slow="1")
    assert "1 passed, 1 skipped" in _run(tmp_path, env_slow="0")
