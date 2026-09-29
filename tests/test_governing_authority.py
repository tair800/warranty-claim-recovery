"""The running system and the measured system choose the same evidence, or this fails.

Two functions decide which retrieved clause is the authority for a requirement.
`evaluation.pipeline._authority` decides it for the system whose numbers are published;
`graph.nodes._governing_entry` decides it for the system that actually runs and files corrections.
They operate on different shapes — the graph's state is checkpointed as JSON, so it sees mappings,
while the evaluation sees `ScoredClause` — and that difference is exactly how they came to disagree.

They did disagree. The graph's rule fell back to the top-ranked clause when no retrieved clause
governed the code, and the evaluation's did not. The published figures therefore described a
system that never cited a clause merely for ranking first, while the deployed graph did — and on
the development split, before withdrawn documents were excluded, 69 of 520 queries had a withdrawn
bulletin at rank one. A deployment that selects evidence differently from the system that was
measured is not the system that was measured.

So this file feeds both rules the same retrieval result in both shapes and asserts they agree,
across every arrangement that matters. It is a behavioural guard in the sense `CLAUDE.md` §3 rule 6
asks for: it calls the functions the running system calls, and observes what they choose.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from warranty_claim_recovery.domain import PolicyClause, RejectionCode
from warranty_claim_recovery.evaluation.pipeline import _authority
from warranty_claim_recovery.graph.nodes import _governing_entry
from warranty_claim_recovery.retrieval.pipeline import ScoredClause

CODE = RejectionCode.MISSING_SERIAL
OTHER = RejectionCode.WRONG_FAILURE_CODE


def clause(clause_id: str, governs: tuple[RejectionCode, ...]) -> PolicyClause:
    body = f"Clause {clause_id}: the serial plate evidence rules for this programme apply here."
    return PolicyClause(
        clause_id=clause_id,
        program_id="prog-1",
        document_id="doc-1",
        section="4.2",
        text=body,
        start_offset=0,
        end_offset=len(body),
        governs=governs,
    )


def both_shapes(
    ranked: Sequence[tuple[str, tuple[RejectionCode, ...]]],
) -> tuple[list[ScoredClause], list[dict[str, Any]]]:
    """The same retrieval result, as the evaluation sees it and as the graph sees it."""
    scored = [
        ScoredClause(clause=clause(cid, governs), distance=0.1 * rank, rank=rank)
        for rank, (cid, governs) in enumerate(ranked, start=1)
    ]
    entries = [
        {"clause_id": cid, "rank": rank, "governs": [code.value for code in governs]}
        for rank, (cid, governs) in enumerate(ranked, start=1)
    ]
    return scored, entries


def chosen_by_evaluation(scored: Sequence[ScoredClause], code: RejectionCode) -> str | None:
    found = _authority(scored, code)
    return None if found is None else found[0].clause_id


def chosen_by_graph(entries: Sequence[dict[str, Any]], code: RejectionCode) -> str | None:
    found = _governing_entry(entries, code)
    return None if found is None else str(found["clause_id"])


ARRANGEMENTS = {
    # The one that caused the defect: a clause governing nothing ranks first, and the authority is
    # further down. Both must skip the first and take the authority.
    "context_ranks_above_authority": (
        [("withdrawn-looking", ()), ("context", (OTHER,)), ("authority", (CODE,))],
        "authority",
    ),
    # Retrieval missed the authority entirely. The old graph rule cited rank one here; both must now
    # return nothing, so the requirement becomes CONFLICTING and a person decides.
    "authority_not_retrieved": (
        [("context-a", ()), ("context-b", (OTHER,)), ("context-c", ())],
        None,
    ),
    # Two clauses govern the code: the higher-ranked one wins, in both.
    "two_authorities": (
        [("context", ()), ("first-authority", (CODE,)), ("second-authority", (CODE,))],
        "first-authority",
    ),
    # The authority is first: nothing to skip.
    "authority_first": ([("authority", (CODE,)), ("context", ())], "authority"),
    # Nothing retrieved at all.
    "empty": ([], None),
}


@pytest.mark.parametrize("name", sorted(ARRANGEMENTS))
def test_the_graph_and_the_evaluation_choose_the_same_authority(name: str) -> None:
    ranked, expected = ARRANGEMENTS[name]
    scored, entries = both_shapes(ranked)

    by_evaluation = chosen_by_evaluation(scored, CODE)
    by_graph = chosen_by_graph(entries, CODE)

    assert by_evaluation == expected, f"{name}: the evaluation chose {by_evaluation!r}"
    assert by_graph == expected, f"{name}: the graph chose {by_graph!r}"
    assert by_graph == by_evaluation, (
        f"{name}: the running system chose {by_graph!r} and the measured one {by_evaluation!r}; "
        f"the published figures would describe a system that is not the one deployed"
    )


def test_a_clause_that_governs_nothing_is_never_the_authority_whatever_its_rank() -> None:
    """The domain rule, held to the running system directly.

    Every clause here governs nothing — the shape every withdrawn bulletin in the corpus has, 0 of
    36 governing any code. Ranked first or last, none of them may be returned as the basis of a
    satisfied requirement.
    """
    ranked = [(f"context-{index}", ()) for index in range(8)]
    scored, entries = both_shapes(ranked)

    assert chosen_by_graph(entries, CODE) is None
    assert chosen_by_evaluation(scored, CODE) is None
