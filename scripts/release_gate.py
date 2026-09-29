"""Grade ADR-001 §5's thirteen kill conditions against the committed artifacts, and report.

This script reports. `tests/test_kill_criteria.py` grades. They read the same files and they are
two different jobs, and the difference decides this script's exit code, so it is worth stating
plainly.

**A missed threshold exits zero.** ADR-001 §5 places exactly one build-failing obligation on this
file — *fails the build if any graded criterion's denominator is zero* — and the `Makefile` records
why the release gate is not part of `make test`: "`test` asks whether the software works and must be
green; the release gate asks whether the three claims in ADR-001 §4 hold, and its honest answer may
be no." A gate that exits non-zero the moment a claim does not hold is a gate whose red build is a
disclosed, expected red, and a disclosed red is a red nobody reads — taking the undisclosed one with
it. The predeclared kill test is what fails the build on a missed threshold, which is the right
place for it: that file is committed, its thresholds cannot be lowered, and it imports nothing from
this repository.

**A criterion graded over nothing exits non-zero.** That is the vacuity guard and it is the one
thing this script refuses to report as a result. Project 7's kill condition G passed because its
numerator was empty by construction, and nothing in the artifact said so; a criterion that could not
have failed is not a criterion, and reporting one as a pass is worse than reporting a failure. So a
zero denominator, a missing artifact and an unreadable one all produce a non-zero exit and a verdict
that is not `PASSED`. The distinction being drawn is between *the claim does not hold*, which is a
result, and *nothing measured the claim*, which is a broken build.

**The thresholds are restated here rather than imported from the kill test.** Importing them would
mean this script and the grader could only ever agree, so the table below would prove nothing about
the grader. Restating them creates the opposite risk — the two tables drifting apart — and
`tests/test_evaluation.py` closes it by parsing the kill test's own source and asserting that every
threshold here matches the literal there. `src/warranty_claim_recovery/api/app.py` takes the same
decision for the same reason and records it in the same words.

**Every criterion states the population it was graded over, in the table.** A verdict without a
denominator is the shape of the failure this whole guard exists to prevent, and a reader should not
have to open a JSON file to find out whether `PASSED` meant anything.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final, NamedTuple

REPO_ROOT: Final = Path(__file__).resolve().parents[1]

# ------------------------------------------------------------------------------------------------
# The thresholds, quoted from ADR-001 §5. Restated, never imported; the module docstring says why,
# and `tests/test_evaluation.py` asserts that each one still matches the literal in the kill test.
# ------------------------------------------------------------------------------------------------

MAX_RESUME_NODE_DIVERGENCES: Final = 0
MAX_TOOL_REEXECUTIONS: Final = 0
MAX_LOST_HUMAN_DECISIONS: Final = 0
MAX_UNAPPROVED_SUBMISSIONS: Final = 0
EXACT_SUBMISSION_EFFECTS_PER_IDENTITY: Final = 1
MAX_DUPLICATE_SUBMISSIONS: Final = 0
MIN_CONCURRENT_ATTEMPTS: Final = 16
MAX_AMOUNT_MISMATCHES: Final = 0
MAX_FALSE_RECOVERIES: Final = 0
MAX_UNGROUNDED_REQUIREMENTS: Final = 0
MAX_UNFAITHFUL_CITATIONS: Final = 0
MIN_HOLDOUT_RECALL_AT_5: Final = 0.85
RETRIEVAL_K: Final = 5
REQUIRED_DISTANCE_OPERATOR: Final = "<=>"
MAX_DOUBLE_LEASES_WITHOUT_REDIS: Final = 0
MAX_SUBMISSIONS_WITHOUT_REDIS: Final = 0
MAX_RESUBMISSIONS_AFTER_WINDOW_CLOSED: Final = 0
MIN_DENOMINATOR: Final = 1

#: The same mapping `tests/test_predeclaration.py` keeps against the kill test's own source, so that
#: a drift between this script and the grader is a failing test rather than a discrepancy somebody
#: notices while reading two tables side by side.
THRESHOLDS: Final[dict[str, object]] = {
    "MAX_RESUME_NODE_DIVERGENCES": MAX_RESUME_NODE_DIVERGENCES,
    "MAX_TOOL_REEXECUTIONS": MAX_TOOL_REEXECUTIONS,
    "MAX_LOST_HUMAN_DECISIONS": MAX_LOST_HUMAN_DECISIONS,
    "MAX_UNAPPROVED_SUBMISSIONS": MAX_UNAPPROVED_SUBMISSIONS,
    "EXACT_SUBMISSION_EFFECTS_PER_IDENTITY": EXACT_SUBMISSION_EFFECTS_PER_IDENTITY,
    "MAX_DUPLICATE_SUBMISSIONS": MAX_DUPLICATE_SUBMISSIONS,
    "MIN_CONCURRENT_ATTEMPTS": MIN_CONCURRENT_ATTEMPTS,
    "MAX_AMOUNT_MISMATCHES": MAX_AMOUNT_MISMATCHES,
    "MAX_FALSE_RECOVERIES": MAX_FALSE_RECOVERIES,
    "MAX_UNGROUNDED_REQUIREMENTS": MAX_UNGROUNDED_REQUIREMENTS,
    "MAX_UNFAITHFUL_CITATIONS": MAX_UNFAITHFUL_CITATIONS,
    "MIN_HOLDOUT_RECALL_AT_5": MIN_HOLDOUT_RECALL_AT_5,
    "RETRIEVAL_K": RETRIEVAL_K,
    "REQUIRED_DISTANCE_OPERATOR": REQUIRED_DISTANCE_OPERATOR,
    "MAX_DOUBLE_LEASES_WITHOUT_REDIS": MAX_DOUBLE_LEASES_WITHOUT_REDIS,
    "MAX_SUBMISSIONS_WITHOUT_REDIS": MAX_SUBMISSIONS_WITHOUT_REDIS,
    "MAX_RESUBMISSIONS_AFTER_WINDOW_CLOSED": MAX_RESUBMISSIONS_AFTER_WINDOW_CLOSED,
    "MIN_DENOMINATOR": MIN_DENOMINATOR,
}

PASSED: Final = "PASSED"
FAILED: Final = "FAILED"
#: A criterion whose population was empty. Reported separately from `FAILED` because the remedies
#: are different: a failure is a system to fix, a vacuous criterion is a harness that measured
#: nothing and a number nobody may quote.
VACUOUS: Final = "VACUOUS"
#: The artifact is absent or unreadable, so nothing graded the criterion at all.
NOT_GRADED: Final = "NOT_GRADED"

#: Exit codes. Zero is reserved for "this script did its job", including when the answer is no.
EXIT_REPORTED: Final = 0
EXIT_VACUOUS: Final = 2


class _Absent:
    """A key that was not in the artifact, kept distinct from a key whose value is `None`.

    A sentinel class rather than `None`, because `None` is a value an artifact can legitimately
    hold — an example reason that did not exist, a plan that was not captured — and treating the two
    the same is how a criterion comes to be reported as measured-and-null rather than as absent.
    """

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return "<absent>"


ABSENT: Final = _Absent()


class Check(NamedTuple):
    """One comparison between a value in an artifact and a threshold ADR-001 fixed.

    Declarative rather than a predicate function, so that the threshold is a literal in a table a
    reviewer reads rather than a number inside a lambda. `path` is a key path because two of the
    criteria read a nested value, and a flat key plus a convention for nesting is a convention that
    will one day be spelled two ways.
    """

    path: tuple[str, ...]
    relation: str
    limit: Any

    @property
    def rendered_path(self) -> str:
        return ".".join(self.path)


class Criterion(NamedTuple):
    """One of ADR-001 §5's thirteen, plus the artifact and the population that grade it."""

    letter: str
    fails_if: str
    artifact: str
    denominator: str
    checks: tuple[Check, ...]
    graded_over: str


#: Relations. Small and closed on purpose: a criterion needing a relation this list does not hold is
#: a criterion whose shape changed, and that should be a visible edit here rather than a lambda
#: somebody slid into the table.
AT_MOST: Final = "<="
AT_LEAST: Final = ">="
EQUALS: Final = "=="
IS_TRUE: Final = "is true"
CONTAINS: Final = "contains"
#: Kill condition J's second half. It is not a comparison against a literal — the thing the system
#: has to beat is measured in the same run — so it reads the artifact's own baseline table. The
#: limit is the key that table lives under, so the relation still carries everything it needs.
ABOVE_EVERY: Final = "strictly above every"


CRITERIA: Final[tuple[Criterion, ...]] = (
    Criterion(
        letter="A",
        fails_if="a resumed case continues at a different node from the checkpoint's",
        artifact="durability.json",
        denominator="cases_killed",
        graded_over="every killed case, every node",
        checks=(Check(("node_divergences",), AT_MOST, MAX_RESUME_NODE_DIVERGENCES),),
    ),
    Criterion(
        letter="B",
        fails_if="a tool that completed before the kill runs again after the resume",
        artifact="durability.json",
        denominator="tool_invocations_observed",
        graded_over="every killed case",
        checks=(Check(("tool_reexecutions",), AT_MOST, MAX_TOOL_REEXECUTIONS),),
    ),
    Criterion(
        letter="C",
        fails_if="a human decision recorded before the kill is absent after the resume",
        artifact="durability.json",
        denominator="human_decisions_before_kill",
        graded_over="every killed case",
        checks=(Check(("human_decisions_lost",), AT_MOST, MAX_LOST_HUMAN_DECISIONS),),
    ),
    Criterion(
        letter="D",
        fails_if="a submission appears with no approval for the same case and the same version",
        artifact="submission.json",
        denominator="submissions_observed",
        graded_over="whole corpus",
        checks=(
            Check(("submissions_without_approval",), AT_MOST, MAX_UNAPPROVED_SUBMISSIONS),
            Check(
                ("submissions_with_stale_version_approval",), AT_MOST, MAX_UNAPPROVED_SUBMISSIONS
            ),
        ),
    ),
    Criterion(
        letter="E",
        fails_if="concurrent identical submissions produce more than one submission effect",
        artifact="submission.json",
        denominator="identities_raced",
        graded_over="at least 16 concurrent attempts per identity",
        checks=(
            Check(("concurrent_attempts_per_identity",), AT_LEAST, MIN_CONCURRENT_ATTEMPTS),
            Check(("max_effects_for_one_identity",), EQUALS, EXACT_SUBMISSION_EFFECTS_PER_IDENTITY),
            Check(("duplicate_submissions",), AT_MOST, MAX_DUPLICATE_SUBMISSIONS),
        ),
    ),
    Criterion(
        letter="F",
        fails_if="a computed recoverable amount differs from the generator's ground truth",
        artifact="recovery.json",
        denominator="cases_scored",
        graded_over="whole corpus",
        checks=(
            Check(("amount_mismatches",), AT_MOST, MAX_AMOUNT_MISMATCHES),
            Check(("amount_exact_match_rate",), EQUALS, 1.0),
        ),
    ),
    Criterion(
        letter="G",
        fails_if="a case is authorised whose ground truth is NOT_RECOVERABLE — a false recovery",
        artifact="recovery.json",
        denominator="holdout_not_recoverable_cases",
        graded_over="hold-out",
        checks=(Check(("holdout_false_recoveries",), AT_MOST, MAX_FALSE_RECOVERIES),),
    ),
    Criterion(
        letter="H",
        fails_if="a requirement is reported satisfied with no citation",
        artifact="groundedness.json",
        denominator="requirements_marked_satisfied",
        graded_over="whole corpus",
        checks=(
            Check(
                ("requirements_satisfied_without_citation",), AT_MOST, MAX_UNGROUNDED_REQUIREMENTS
            ),
        ),
    ),
    Criterion(
        letter="I",
        fails_if="a cited span is not verbatim at its recorded offsets in the version it names",
        artifact="groundedness.json",
        denominator="citations_checked",
        graded_over="whole corpus",
        checks=(Check(("unfaithful_citations",), AT_MOST, MAX_UNFAITHFUL_CITATIONS),),
    ),
    Criterion(
        letter="J",
        fails_if="hold-out recall@5 is below the floor, or is not strictly above every baseline",
        artifact="retrieval.json",
        denominator="holdout_queries",
        graded_over="hold-out",
        checks=(
            Check(("k",), EQUALS, RETRIEVAL_K),
            Check(("system", "recall_at_k"), AT_LEAST, MIN_HOLDOUT_RECALL_AT_5),
            Check(("system", "recall_at_k"), ABOVE_EVERY, "baselines"),
        ),
    ),
    Criterion(
        letter="K",
        fails_if="the dense stage does not use the pgvector operator, by the server's own plan",
        artifact="pgvector.json",
        denominator="statements_explained",
        graded_over="live PostgreSQL",
        checks=(
            Check(("extension_installed",), IS_TRUE, True),
            Check(("column_is_vector_type",), IS_TRUE, True),
            Check(("executed_statement",), CONTAINS, REQUIRED_DISTANCE_OPERATOR),
            Check(("explain_uses_distance_operator",), IS_TRUE, True),
        ),
    ),
    Criterion(
        letter="L",
        fails_if="with Redis unavailable a case is leased twice or a submission proceeds",
        artifact="redis.json",
        denominator="injections",
        graded_over="failure injection",
        checks=(
            Check(("double_leases_without_redis",), AT_MOST, MAX_DOUBLE_LEASES_WITHOUT_REDIS),
            Check(("submissions_without_redis",), AT_MOST, MAX_SUBMISSIONS_WITHOUT_REDIS),
            Check(("queue_is_load_bearing",), IS_TRUE, True),
        ),
    ),
    Criterion(
        letter="M",
        fails_if="a case is resubmitted after the manufacturer's claim window closed",
        artifact="recovery.json",
        denominator="cases_with_a_closed_window",
        graded_over="whole corpus",
        checks=(
            Check(
                ("resubmissions_after_window_closed",),
                AT_MOST,
                MAX_RESUBMISSIONS_AFTER_WINDOW_CLOSED,
            ),
        ),
    ),
    # Not one of the thirteen. It is the precondition that makes G and J mean anything, and the kill
    # test grades it in the same file for the same reason: a leak should fail the build that the
    # scores are published in, not the one after it.
    Criterion(
        letter="H/O",
        fails_if="the hold-out is not frozen, is split by the wrong unit, or has leaked",
        artifact="holdout.json",
        denominator="programs_total",
        graded_over="precondition for G and J",
        checks=(
            Check(("split_unit",), EQUALS, "warranty_program"),
            Check(("leaked_programs",), EQUALS, 0),
            Check(("leaked_cases",), EQUALS, 0),
            Check(("digest_matches_corpus",), IS_TRUE, True),
        ),
    ),
)


class CheckResult(NamedTuple):
    passed: bool
    rendered: str


class Verdict(NamedTuple):
    criterion: Criterion
    verdict: str
    denominator: int | None
    results: tuple[CheckResult, ...]
    note: str | None

    @property
    def population(self) -> str:
        if self.denominator is None:
            return f"{self.criterion.denominator}=?"
        return f"{self.criterion.denominator}={self.denominator}"


def read_artifact(artifacts_dir: Path, name: str) -> dict[str, Any] | None:
    """The artifact, or `None` when it is absent or is not an object.

    A malformed file reads as absent rather than raising. The criterion then reports `NOT_GRADED`
    and the run exits non-zero, which is the same outcome a missing file gets and is the honest one:
    in both cases nothing graded the claim. Raising would stop the report at the first bad file and
    hide the state of the other twelve, which is exactly the information a person fixing a build
    needs.
    """
    path = artifacts_dir / name
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _value(payload: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return ABSENT
        current = current[key]
    return current


def _rendered(check: Check, found: Any) -> str:
    return f"{check.rendered_path}={found!r} {check.relation} {check.limit!r}"


def _is_true(check: Check, found: Any) -> CheckResult:
    """Identity with `True`, not truthiness. A count of one is not a proof that something holds."""
    return CheckResult(found is True, f"{check.rendered_path}={found!r} is true")


def _contains(check: Check, found: Any) -> CheckResult:
    passed = isinstance(found, str) and str(check.limit) in found
    return CheckResult(passed, f"{check.rendered_path} contains {check.limit!r}")


def _equals(check: Check, found: Any) -> CheckResult:
    return CheckResult(found == check.limit, _rendered(check, found))


def _numeric(found: Any) -> float | None:
    """The value as a float, or `None` when it is not a number.

    `bool` is excluded although it is an `int` in Python. A criterion whose artifact holds `true`
    where a count belongs is an artifact somebody wrote by hand, and `True <= 0` being `False` would
    report that as an ordinary failure rather than as a malformed file.
    """
    if isinstance(found, bool) or not isinstance(found, int | float):
        return None
    return float(found)


def _at_most(check: Check, found: Any) -> CheckResult:
    value = _numeric(found)
    if value is None:
        return CheckResult(False, f"{check.rendered_path}={found!r} is not a number")
    return CheckResult(value <= float(check.limit), _rendered(check, found))


def _at_least(check: Check, found: Any) -> CheckResult:
    value = _numeric(found)
    if value is None:
        return CheckResult(False, f"{check.rendered_path}={found!r} is not a number")
    return CheckResult(value >= float(check.limit), _rendered(check, found))


#: One comparator per relation, so that adding a relation is adding a row here rather than a branch
#: in the middle of `evaluate`. `ABOVE_EVERY` is absent because it is the only relation that needs
#: the whole artifact rather than the one value it read.
_RELATIONS: Final[dict[str, Callable[[Check, Any], CheckResult]]] = {
    AT_MOST: _at_most,
    AT_LEAST: _at_least,
    EQUALS: _equals,
    IS_TRUE: _is_true,
    CONTAINS: _contains,
}


def evaluate(payload: Mapping[str, Any], check: Check) -> CheckResult:
    """One comparison, with the measured value rendered beside the threshold it was compared to.

    The rendered string carries both numbers. A report that printed only a verdict would make every
    investigation start by reopening the artifact, and the first thing anybody wants to know about a
    failed criterion is by how much.

    An unknown relation fails rather than raising. This gate's job is to produce a report, and a
    table with one row saying "this gate does not know that relation" is more use to the person who
    just added it than a traceback that hides the other thirteen rows.
    """
    found = _value(payload, check.path)
    if isinstance(found, _Absent):
        return CheckResult(False, f"{check.rendered_path} is absent from the artifact")
    if check.relation == ABOVE_EVERY:
        return _above_every(payload, check, found)
    comparator = _RELATIONS.get(check.relation)
    if comparator is None:
        return CheckResult(False, f"{check.relation!r} is not a relation this gate knows")
    return comparator(check, found)


def _above_every(payload: Mapping[str, Any], check: Check, found: Any) -> CheckResult:
    """Kill condition J's comparison against the baselines measured in the same run.

    An empty baseline table fails. "Above every baseline" over no baselines is vacuously true, and a
    criterion that is satisfied by measuring nothing is the exact failure ADR-001's vacuity guard
    was written against — here it would be invisible, because the recall figure beside it would be
    real.
    """
    table = _value(payload, (str(check.limit),))
    if not isinstance(table, Mapping) or not table:
        return CheckResult(
            False, f"{check.limit} holds no baseline, so 'above every' graded nothing"
        )
    floors: dict[str, float] = {}
    for name, row in table.items():
        recall = row.get("recall_at_k") if isinstance(row, Mapping) else None
        if not isinstance(recall, int | float) or isinstance(recall, bool):
            return CheckResult(False, f"baseline {name} has no numeric recall_at_k")
        floors[str(name)] = float(recall)
    system = float(found)
    best = max(floors, key=lambda name: floors[name])
    passed = all(system > floor for floor in floors.values())
    return CheckResult(
        passed,
        f"{check.rendered_path}={system} against the best baseline {best}={floors[best]}",
    )


def grade(artifacts_dir: Path, criteria: Sequence[Criterion] = CRITERIA) -> list[Verdict]:
    """Every criterion, in ADR-001 §5's order, with its population and its comparisons.

    The denominator is read first and decides whether the comparisons are worth making. A criterion
    whose population is empty is reported `VACUOUS` **and its checks are still run and printed**,
    because the numbers are diagnostic even when the verdict is not a result: a reader looking at a
    harness that measured nothing wants to see what it did produce.
    """
    verdicts: list[Verdict] = []
    for criterion in criteria:
        payload = read_artifact(artifacts_dir, criterion.artifact)
        if payload is None:
            verdicts.append(
                Verdict(
                    criterion=criterion,
                    verdict=NOT_GRADED,
                    denominator=None,
                    results=(),
                    note=(
                        f"artifacts/{criterion.artifact} is absent or unreadable, so nothing "
                        f"graded this criterion"
                    ),
                )
            )
            continue

        raw = _value(payload, (criterion.denominator,))
        denominator = int(raw) if isinstance(raw, int) and not isinstance(raw, bool) else None
        results = tuple(evaluate(payload, check) for check in criterion.checks)

        if denominator is None:
            verdict = NOT_GRADED
            note = (
                f"{criterion.denominator} is not an integer in artifacts/{criterion.artifact}, so "
                f"the population this criterion was graded over cannot be established"
            )
        elif denominator < MIN_DENOMINATOR:
            verdict = VACUOUS
            note = (
                f"{criterion.denominator} is {denominator}: this criterion was graded over an "
                f"empty population and therefore could not have failed"
            )
        else:
            verdict = PASSED if all(result.passed for result in results) else FAILED
            note = None
        verdicts.append(
            Verdict(
                criterion=criterion,
                verdict=verdict,
                denominator=denominator,
                results=results,
                note=note,
            )
        )
    return verdicts


def _column(values: Sequence[str], minimum: int) -> int:
    return max([minimum, *(len(value) for value in values)])


def render(verdicts: Sequence[Verdict]) -> list[str]:
    """The table, aligned, with the population beside every verdict.

    Fourteen rows: ADR-001 §5's thirteen kill conditions and the hold-out precondition, which is not
    one of them and is printed with them because G and J mean nothing without it.
    """
    populations = [verdict.population for verdict in verdicts]
    width = _column(populations, len("graded over"))
    lines = [
        "ADR-001 §5 — the thirteen predeclared kill conditions, and the hold-out precondition",
        "",
        f"  {'id':<4} {'verdict':<10} {'graded over':<{width}}  the project fails if",
        f"  {'-' * 4} {'-' * 10} {'-' * width}  {'-' * 60}",
    ]
    for verdict in verdicts:
        lines.append(
            f"  {verdict.criterion.letter:<4} {verdict.verdict:<10} "
            f"{verdict.population:<{width}}  {verdict.criterion.fails_if}"
        )
    lines.append("")
    for verdict in verdicts:
        if verdict.verdict == PASSED:
            continue
        lines.append(f"  {verdict.criterion.letter} — {verdict.verdict}")
        if verdict.note:
            lines.append(f"      {verdict.note}")
        for result in verdict.results:
            marker = "ok  " if result.passed else "MISS"
            lines.append(f"      {marker} {result.rendered}")
        lines.append(f"      artifact: artifacts/{verdict.criterion.artifact}")
        lines.append(f"      population: {verdict.criterion.graded_over}")
    return lines


def report(verdicts: Sequence[Verdict]) -> dict[str, Any]:
    """The payload `artifacts/release_gate.json` carries, and the console reads.

    `criteria` is a list of records each naming its own letter rather than a mapping keyed by one.
    A list keeps ADR-001 §5's order in the file, and the order is how a reader checks that thirteen
    criteria are still thirteen. `api/app.py` reads either shape and defaults an unmentioned
    criterion to `NOT_RUN`, so a criterion quietly dropped from this file shows on the console as
    ungraded rather than disappearing from the table.
    """
    vacuous = [v.criterion.letter for v in verdicts if v.verdict in {VACUOUS, NOT_GRADED}]
    failed = [v.criterion.letter for v in verdicts if v.verdict == FAILED]
    overall = PASSED if not vacuous and not failed else FAILED
    return {
        "is_synthetic_corpus": True,
        "what_this_is": (
            "the verdict on ADR-001 §5's thirteen predeclared kill conditions and the hold-out "
            "precondition, read from the committed artifacts by scripts/release_gate.py"
        ),
        "verdict": overall,
        "detail": (
            "every criterion passed over a non-empty population"
            if overall == PASSED
            else (
                f"failed: {', '.join(failed) or 'none'}; not graded or graded over nothing: "
                f"{', '.join(vacuous) or 'none'}"
            )
        ),
        "failed": failed,
        "not_graded_or_vacuous": vacuous,
        "thresholds": dict(THRESHOLDS),
        "threshold_source": (
            "quoted from ADR-001 §5 and restated here rather than imported from "
            "tests/test_kill_criteria.py, which is predeclared. tests/test_evaluation.py asserts "
            "the two tables agree by parsing the kill test's source."
        ),
        "vacuity_guard": (
            "a criterion whose denominator is below "
            f"{MIN_DENOMINATOR} is reported {VACUOUS} and this script exits non-zero. A criterion "
            "that could not have failed is not a criterion, and reporting one as a pass is worse "
            "than reporting a failure."
        ),
        "exit_code_policy": (
            "zero when this script graded every criterion, including when the answer is no; "
            f"{EXIT_VACUOUS} when any criterion was graded over nothing or its artifact was "
            "absent. A missed threshold fails the build through tests/test_kill_criteria.py, "
            "which is "
            "predeclared and imports nothing from this repository."
        ),
        "criteria": [
            {
                "letter": v.criterion.letter,
                "verdict": v.verdict,
                "artifact": v.criterion.artifact,
                "fails_if": v.criterion.fails_if,
                "graded_over": v.criterion.graded_over,
                "denominator_key": v.criterion.denominator,
                "denominator": v.denominator,
                "note": v.note,
                "checks": [
                    {"measured": result.rendered, "passed": result.passed} for result in v.results
                ],
            }
            for v in verdicts
        ],
    }


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", default=None, help="defaults to WCR_ARTIFACTS_DIR")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.artifacts_dir is not None:
        artifacts_dir = Path(args.artifacts_dir)
    else:
        # Read from the environment directly rather than through `config.get_settings`, so that this
        # script can grade a directory of artifacts without importing the package it is grading.
        # The kill test takes the same position for the same reason.
        artifacts_dir = Path(os.environ.get("WCR_ARTIFACTS_DIR", "artifacts"))
    if not artifacts_dir.is_absolute():
        artifacts_dir = REPO_ROOT / artifacts_dir

    verdicts = grade(artifacts_dir)
    payload = report(verdicts)

    for line in render(verdicts):
        print(line)
    print()
    print(f"ORIGINAL RELEASE GATE: {payload['verdict']}")
    print(payload["detail"])

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    destination = artifacts_dir / "release_gate.json"
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {destination}")

    if payload["not_graded_or_vacuous"]:
        print(
            "exiting non-zero: a criterion was graded over an empty population or its artifact was "
            "absent. ADR-001 §5's vacuity guard fails the build for this and not for a missed "
            "threshold, because a criterion that could not have failed is not a criterion.",
            file=sys.stderr,
        )
        return EXIT_VACUOUS
    return EXIT_REPORTED


if __name__ == "__main__":
    raise SystemExit(main())
