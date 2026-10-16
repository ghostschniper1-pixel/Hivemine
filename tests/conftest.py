"""Shared pytest configuration.

Tests marked ``@pytest.mark.slow`` are skipped unless they are selected: run them with
``-m slow`` (any marker expression that asks for ``slow``, such as ``-m "slow or not slow"``)
or with the environment variable ``HIVEMIND_SLOW=1``. The default run stays fast (contracts 0:
about 60 s), and a slow test needs no skip logic of its own.

The hook sees every collected item, including the bridge tests under ``bridge/server``.
"""

from __future__ import annotations

import os
import re

import pytest

SLOW_ENV = "HIVEMIND_SLOW"
_TRUE = ("1", "true", "yes", "on")


def slow_selected(markexpr: str) -> bool:
    """Whether a ``-m`` expression asks for slow tests: it names ``slow`` somewhere not directly
    after ``not``. ``"slow"``, ``"slow and network"`` and ``"slow or not slow"`` do;
    ``""``, ``"not slow"`` and ``"network"`` do not."""
    tokens = re.findall(r"\w+", markexpr)
    return any(t == "slow" and (i == 0 or tokens[i - 1] != "not") for i, t in enumerate(tokens))


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get(SLOW_ENV, "").strip().lower() in _TRUE:
        return
    if slow_selected(config.getoption("markexpr") or ""):
        return
    skip = pytest.mark.skip(reason=f"slow: run with -m slow or {SLOW_ENV}=1")
    for item in items:
        if item.get_closest_marker("slow") is not None:
            item.add_marker(skip)
