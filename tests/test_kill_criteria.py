"""The thirteen predeclared kill conditions, graded against the committed evidence.

**PREDECLARED. This file was committed before any source file existed.** `git log --diff-filter=A`
on it precedes the first commit under `src/`, which is the only thing that makes the thresholds below
mean anything: a criterion written after seeing a score is a description, not a test.

Thresholds may be **raised**. They may never be lowered, deleted, renamed, relaxed, skipped or
`xfail`ed. If a measurement misses, the honest outcomes are to fix the system or to record the
failure in `DECISIONS.md` — not to move the line.

**This file imports nothing from `warranty_claim_recovery`.** It reads `artifacts/*.json` and the
standard library. A grader that imports the system under test can be made to pass by changing the
system's own definition of the thing it is grading, which is how a project comes to grade itself.
`tests/test_predeclaration.py` asserts that property over this file's AST.

The absolute zeros are deliberate and are absolute. Every one of them describes an event that must
not be possible rather than an event that should be rare: money computed wrongly, a submission with
no approval, a submission twice, a claim resubmitted after its window shut, a requirement satisfied
by nothing, a citation pointing at text that is not there. A rate would invite a denominator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"

# --------------------------------------------------------------------------------------------
# The thresholds. Module-level constants so `test_predeclaration.py` can read them without running
# anything, and so a diff that changes one is a diff a reviewer sees.
# --------------------------------------------------------------------------------------------

#: A — the resumed case must continue at the node the checkpoint recorded.
MAX_RESUME_NODE_DIVERGENCES = 0

#: B — a tool that completed before the kill must not run again after the resume.
MAX_TOOL_REEXECUTIONS = 0

#: C — a human decision recorded before the kill must survive it.
MAX_LOST_HUMAN_DECISIONS = 0

#: D — every submission must carry an approval for the same case AND the same case version.
MAX_UNAPPROVED_SUBMISSIONS = 0

#: E — one recovery identity, one effect, however many callers race for it.
EXACT_SUBMISSION_EFFECTS_PER_IDENTITY = 1
MAX_DUPLICATE_SUBMISSIONS = 0
MIN_CONCURRENT_ATTEMPTS = 16

#: F — the money is exact or it is wrong. Decimal equality, no tolerance.
MAX_AMOUNT_MISMATCHES = 0

#: G — a false recovery is an attempt to take money from a manufacturer on a basis that does not
#: exist. It is the expensive failure and its budget is zero.
MAX_FALSE_RECOVERIES = 0

#: H — a requirement reported satisfied with nothing behind it.
MAX_UNGROUNDED_REQUIREMENTS = 0

#: I — a cited span that is not in the document version it names, at the offsets it names.
MAX_UNFAITHFUL_CITATIONS = 0

#: J — retrieval must find the governing clause, and must beat every predeclared baseline. The
#: baselines are fixed in ADR-001 §6 and each differs in the retrieval stage itself.
MIN_HOLDOUT_RECALL_AT_5 = 0.85
RETRIEVAL_K = 5

#: K — the dense stage must run in pgvector, proven by the server's own plan.
REQUIRED_DISTANCE_OPERATOR = "<=>"

#: L — with Redis gone, the queue fails closed.
MAX_DOUBLE_LEASES_WITHOUT_REDIS = 0
MAX_SUBMISSIONS_WITHOUT_REDIS = 0

#: M — the claim window is arithmetic, not judgement.
MAX_RESUBMISSIONS_AFTER_WINDOW_CLOSED = 0

#: The vacuity guard. Project 7's kill condition G passed with an empty numerator and the pass was
#: worthless. Every criterion here states the population it was graded over, and a criterion graded
#: over nothing fails rather than passes.
MIN_DENOMINATOR = 1


def load(name: str) -> dict[str, Any]:
    path = ARTIFACTS / name
    if not path.is_file():
        pytest.fail(
            f"{name} is missing. It is produced by `make artifacts`, and a kill test that cannot "
            f"find its evidence must fail rather than silently grade nothing."
        )
    parsed: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return parsed


def denominator(payload: dict[str, Any], key: str) -> int:
    """Read a criterion's population, and refuse a criterion that was graded over nothing."""
    value = int(payload[key])
    assert value >= MIN_DENOMINATOR, (
        f"{key} is {value}: this criterion was graded over an empty population and therefore "
        f"could not have failed. A criterion that cannot fail is not a criterion."
    )
    return value


# --------------------------------------------------------------------------------------------
# A, B, C — durability. The blueprint's headline, and the portfolio's only stateful-agent evidence.
# --------------------------------------------------------------------------------------------


def test_a_resume_continues_at_the_recorded_node() -> None:
    evidence = load("durability.json")
    denominator(evidence, "cases_killed")
    assert evidence["node_divergences"] <= MAX_RESUME_NODE_DIVERGENCES


def test_b_no_completed_tool_runs_twice() -> None:
    evidence = load("durability.json")
    denominator(evidence, "tool_invocations_observed")
    assert evidence["tool_reexecutions"] <= MAX_TOOL_REEXECUTIONS


def test_c_no_human_decision_is_lost_across_a_kill() -> None:
    evidence = load("durability.json")
    denominator(evidence, "human_decisions_before_kill")
    assert evidence["human_decisions_lost"] <= MAX_LOST_HUMAN_DECISIONS


# --------------------------------------------------------------------------------------------
# D, E — nothing leaves without approval, and nothing leaves twice.
# --------------------------------------------------------------------------------------------


def test_d_every_submission_has_an_approval_for_the_same_case_version() -> None:
    evidence = load("submission.json")
    denominator(evidence, "submissions_observed")
    assert evidence["submissions_without_approval"] <= MAX_UNAPPROVED_SUBMISSIONS
    assert evidence["submissions_with_stale_version_approval"] <= MAX_UNAPPROVED_SUBMISSIONS


def test_e_concurrent_identical_submissions_produce_exactly_one_effect() -> None:
    evidence = load("submission.json")
    denominator(evidence, "identities_raced")
    assert evidence["concurrent_attempts_per_identity"] >= MIN_CONCURRENT_ATTEMPTS
    assert evidence["max_effects_for_one_identity"] == EXACT_SUBMISSION_EFFECTS_PER_IDENTITY
    assert evidence["duplicate_submissions"] <= MAX_DUPLICATE_SUBMISSIONS


# --------------------------------------------------------------------------------------------
# F, G, M — the money, and the two ways of getting it wrong that cost real money.
# --------------------------------------------------------------------------------------------


def test_f_every_recoverable_amount_equals_the_ground_truth_exactly() -> None:
    evidence = load("recovery.json")
    denominator(evidence, "cases_scored")
    assert evidence["amount_mismatches"] <= MAX_AMOUNT_MISMATCHES
    assert evidence["amount_exact_match_rate"] == 1.0


def test_g_no_false_recovery_on_the_holdout() -> None:
    evidence = load("recovery.json")
    denominator(evidence, "holdout_not_recoverable_cases")
    assert evidence["holdout_false_recoveries"] <= MAX_FALSE_RECOVERIES


def test_m_nothing_is_resubmitted_after_the_claim_window_closed() -> None:
    evidence = load("recovery.json")
    denominator(evidence, "cases_with_a_closed_window")
    assert evidence["resubmissions_after_window_closed"] <= MAX_RESUBMISSIONS_AFTER_WINDOW_CLOSED


# --------------------------------------------------------------------------------------------
# H, I — evidence. A requirement with nothing behind it, and a citation that points nowhere.
# --------------------------------------------------------------------------------------------


def test_h_no_requirement_is_satisfied_without_a_citation() -> None:
    evidence = load("groundedness.json")
    denominator(evidence, "requirements_marked_satisfied")
    assert evidence["requirements_satisfied_without_citation"] <= MAX_UNGROUNDED_REQUIREMENTS


def test_i_every_cited_span_is_verbatim_at_the_offsets_it_names() -> None:
    evidence = load("groundedness.json")
    denominator(evidence, "citations_checked")
    assert evidence["unfaithful_citations"] <= MAX_UNFAITHFUL_CITATIONS


# --------------------------------------------------------------------------------------------
# J — retrieval, measured, and against baselines that differ in the retrieval stage itself.
# --------------------------------------------------------------------------------------------


def test_j_holdout_retrieval_clears_the_floor_and_beats_every_baseline() -> None:
    evidence = load("retrieval.json")
    denominator(evidence, "holdout_queries")
    assert evidence["k"] == RETRIEVAL_K

    system = float(evidence["system"]["recall_at_k"])
    assert system >= MIN_HOLDOUT_RECALL_AT_5

    baselines = evidence["baselines"]
    assert baselines, "no baselines were measured, so 'above every baseline' graded nothing"
    for name, result in baselines.items():
        assert system > float(result["recall_at_k"]), (
            f"the system does not beat baseline {name}: "
            f"{system} against {result['recall_at_k']}"
        )


# --------------------------------------------------------------------------------------------
# K, L — the two pieces of infrastructure the skill matrix says this project alone must prove.
# --------------------------------------------------------------------------------------------


def test_k_the_dense_stage_executes_in_pgvector() -> None:
    evidence = load("pgvector.json")
    denominator(evidence, "statements_explained")
    assert evidence["extension_installed"] is True
    assert evidence["column_is_vector_type"] is True
    assert REQUIRED_DISTANCE_OPERATOR in evidence["executed_statement"]
    assert evidence["explain_uses_distance_operator"] is True


def test_l_the_queue_fails_closed_when_redis_is_gone() -> None:
    evidence = load("redis.json")
    denominator(evidence, "injections")
    assert evidence["double_leases_without_redis"] <= MAX_DOUBLE_LEASES_WITHOUT_REDIS
    assert evidence["submissions_without_redis"] <= MAX_SUBMISSIONS_WITHOUT_REDIS
    assert evidence["queue_is_load_bearing"] is True


# --------------------------------------------------------------------------------------------
# The hold-out itself. Not one of the thirteen: it is the precondition that makes G and J mean
# anything, and it is checked here so that a leak fails the same build the scores do.
# --------------------------------------------------------------------------------------------


def test_the_holdout_is_frozen_split_by_program_and_unleaked() -> None:
    evidence = load("holdout.json")
    denominator(evidence, "programs_total")
    assert evidence["split_unit"] == "warranty_program"
    assert evidence["leaked_programs"] == 0
    assert evidence["leaked_cases"] == 0
    assert evidence["digest_matches_corpus"] is True
