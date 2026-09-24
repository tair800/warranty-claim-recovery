"""The kill test is predeclared, and this is what stops it quietly ceasing to be.

`test_kill_criteria.py` is only worth anything if three things stay true of it, and none of them is
guaranteed by the fact that it passes:

1. **It grades the artifacts, not the system.** A grader that imports the package under test can be
   made to pass by changing the package's own definition of the thing being graded.
2. **Its thresholds are module-level constants.** A threshold computed at runtime, read from a file,
   or defaulted from the data can be moved without the diff showing a moved threshold.
3. **The zeros are zero and the floors have not fallen.** The values are asserted here by number, so
   lowering one requires editing this file too — which a reviewer sees.

It also bans skipping. `tests/conftest.py` catches marks applied from a plugin or a conftest, which
never appear in the file's own source; this catches the spellings that do.

This file parses. It does not import. Parsing a file that imports the package would import the
package, and then a broken import in the system under test would take the predeclaration guard down
with it — at precisely the moment it is most needed.
"""

from __future__ import annotations

import ast
from pathlib import Path

TESTS = Path(__file__).resolve().parent
KILL_TEST = TESTS / "test_kill_criteria.py"

#: Spellings that disable a test from inside the file. `conftest.py` covers the ones applied from
#: outside it.
BANNED_NODES = (
    "pytest.mark.skip",
    "pytest.mark.skipif",
    "pytest.mark.xfail",
    "pytest.importorskip",
    "pytest.skip",
    "unittest.skip",
)

#: Every threshold, with the value ADR-001 fixed. Written out rather than imported, so that changing
#: one in the kill test and not here fails the build.
DECLARED = {
    "MAX_RESUME_NODE_DIVERGENCES": 0,
    "MAX_TOOL_REEXECUTIONS": 0,
    "MAX_LOST_HUMAN_DECISIONS": 0,
    "MAX_UNAPPROVED_SUBMISSIONS": 0,
    "EXACT_SUBMISSION_EFFECTS_PER_IDENTITY": 1,
    "MAX_DUPLICATE_SUBMISSIONS": 0,
    "MIN_CONCURRENT_ATTEMPTS": 16,
    "MAX_AMOUNT_MISMATCHES": 0,
    "MAX_FALSE_RECOVERIES": 0,
    "MAX_UNGROUNDED_REQUIREMENTS": 0,
    "MAX_UNFAITHFUL_CITATIONS": 0,
    "MIN_HOLDOUT_RECALL_AT_5": 0.85,
    "RETRIEVAL_K": 5,
    "MAX_DOUBLE_LEASES_WITHOUT_REDIS": 0,
    "MAX_SUBMISSIONS_WITHOUT_REDIS": 0,
    "MAX_RESUBMISSIONS_AFTER_WINDOW_CLOSED": 0,
    "MIN_DENOMINATOR": 1,
}

#: The ones that are absolute. A rate would invite a denominator, and a denominator invites an
#: argument about which cases counted.
MUST_BE_ZERO = tuple(name for name, value in DECLARED.items() if value == 0)


def parse() -> ast.Module:
    return ast.parse(KILL_TEST.read_text(encoding="utf-8"), filename=str(KILL_TEST))


def module_constants(tree: ast.Module) -> dict[str, object]:
    found: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant):
                found[target.id] = node.value.value
    return found


def test_the_kill_test_imports_nothing_from_the_package() -> None:
    tree = parse()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("warranty_claim_recovery"), (
                    f"the kill test imports {alias.name}; it must grade the committed artifacts, "
                    f"not the system that produced them"
                )
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert not module.startswith("warranty_claim_recovery"), (
                f"the kill test imports from {module}; it must grade the committed artifacts, "
                f"not the system that produced them"
            )


def test_no_threshold_is_computed_at_runtime() -> None:
    """Each threshold is a literal assigned at module level, so a change to one is visible."""
    found = module_constants(parse())
    for name in DECLARED:
        assert name in found, (
            f"{name} is no longer a module-level constant assigned a literal. A threshold that is "
            f"computed, imported or defaulted can be moved without the diff showing a moved "
            f"threshold."
        )


def test_no_threshold_has_been_lowered() -> None:
    found = module_constants(parse())
    for name, declared in DECLARED.items():
        assert found[name] == declared, (
            f"{name} is {found[name]}, and ADR-001 fixed it at {declared}. Thresholds may be "
            f"raised. They may never be lowered."
        )


def test_the_absolute_zeros_are_still_zero() -> None:
    found = module_constants(parse())
    for name in MUST_BE_ZERO:
        assert found[name] == 0, f"{name} is {found[name]}; ADR-001 declares it absolute"


def test_the_kill_test_cannot_disable_itself() -> None:
    source = KILL_TEST.read_text(encoding="utf-8")
    for banned in BANNED_NODES:
        assert banned not in source, (
            f"{banned} appears in the kill test. A predeclared criterion that skips is a criterion "
            f"that was quietly deleted."
        )


def test_every_declared_criterion_has_a_test_that_reads_it() -> None:
    """A constant nothing asserts on is decoration, and would let a criterion be silently retired."""
    tree = parse()
    names_used = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    functions = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    tests = [name for name in functions if name.startswith("test_")]

    for name in DECLARED:
        if name == "MIN_DENOMINATOR":
            continue  # used inside the shared `denominator` helper rather than in a test body
        assert name in names_used, f"{name} is declared but nothing asserts on it"

    # ADR-001 declares thirteen conditions plus the hold-out precondition.
    assert len(tests) >= 14, (
        f"the kill test defines {len(tests)} tests; ADR-001 declares thirteen kill conditions and "
        f"the hold-out precondition"
    )


def test_the_vacuity_guard_exists_and_is_used() -> None:
    """Project 7's kill condition G passed with an empty numerator. That cannot happen silently here."""
    tree = parse()
    helpers = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    assert "denominator" in helpers, "the vacuity guard has been removed from the kill test"

    calls = sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "denominator"
    )
    assert calls >= 13, (
        f"the vacuity guard is called {calls} times; every graded criterion must state the "
        f"population it was graded over"
    )
