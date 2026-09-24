"""A session-level guard that asks pytest what it is about to run.

`test_predeclaration.py` checks the kill test by parsing it, which catches every spelling that
appears in the file's own source. It cannot catch a mark applied from a plugin, from a conftest, or
from a command line — those never appear in the file at all.

So the real guard is here, and it inspects collected items rather than text. `tryfirst` so it sees
them **before** `-m` deselection removes them: deselection is not disablement, and a lane that
legitimately deselects the kill test must still not be able to silence it. A `UsageError` rather
than a failing test, because a failing test can be deselected too.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

KILL_TEST_FILE = "test_kill_criteria.py"


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if importlib.util.find_spec("warranty_claim_recovery") is None:
        return

    kill = [item for item in items if Path(str(item.fspath)).name == KILL_TEST_FILE]
    if not kill:
        return

    disabled = sorted(
        item.nodeid
        for item in kill
        if item.get_closest_marker("skip") is not None
        or item.get_closest_marker("skipif") is not None
        or item.get_closest_marker("xfail") is not None
    )
    if disabled:
        raise pytest.UsageError(
            "the predeclared kill test has been disabled by a mark, so the criteria that decide "
            "whether this project ships are not being evaluated: " + ", ".join(disabled)
        )
