"""Which retrieved clause is the authority for a requirement — one rule, called from both sides.

Two functions used to answer this. `graph.nodes._governing_entry` answered it for the system that
runs and files corrections; `evaluation.pipeline._authority` answered it for the system whose
numbers are published. They were written separately, over different shapes, and they disagreed:
the graph fell back to the top-ranked clause when no retrieved clause governed the code, and the
evaluation did not. So the published figures described a system that was not the one deployed.

That class of defect is not repaired by making two implementations agree today. It is repaired by
having one. Both call sites now map their candidates onto `ClauseView` and ask this module, and
`tests/test_governing_authority.py` pins that they do.

### The rule, in the order it is applied

Among the retrieved clauses — already filtered to the claim's programme, its policy version and
documents currently in force, and already ranked by pgvector — keep only those that **govern the
rejection code**, then take, in rank order within each tier:

1. **a clause that names the claim's own part family**;
2. otherwise **a clause that names no family at all** — a general policy provision;
3. otherwise **nothing**. A clause that names only a *different* family is not authority for this
   claim, and returning it would cite another machine's rule against this one.

`None` makes the requirement `CONFLICTING`, which the gate sends to a person. That is deliberate:
a missed authority is a question for an adjudicator, not something to paper over with the nearest
paragraph.

### Why tier 1 exists: a bulletin that amends the policy for one family

Found on the development split after the first scoring, and recorded in ADR-002. Service bulletins
in this corpus amend the policy for **one part family**: *"For the hydraulic pump family
KH-PMP-6017 only, this bulletin replaces clause 2 of policy version 2019.1."* Both the bulletin's
clause and the policy clause it replaces govern the same rejection code, and `PolicyClause.governs`
carries no family scope, so "the highest-ranked clause that governs the code" picked whichever
ranked first. On the development split that was the **replaced policy clause** in 66 of the 78
queries whose governing clause is such a bulletin. The correction would then have quoted "the
component identification plate" to a manufacturer whose own current bulletin says that, for this
family, the serial is on the secondary plate — a correction refused on sight.

The scope is read from the clause's **section heading**, where the corpus states it as an exact
identifier — *"B1. Amended serial identification for the KH-PMP-6017 family"* — and matched against
the claim's part number, which is that family identifier plus a variant suffix
(`corpus.programs.part_number`: `f"{family_id}-{variant}"`). An exact identifier compared with an
exact identifier; no free-text reading of what a sentence means.

### What this rule cannot move

It chooses **which** clause is cited. It does not rank, so recall@k is untouched; it does not
decide whether a requirement is satisfied — that comes from the evidence — so no requirement status,
no gate outcome and no amount moves. That is what made it safe to fix after the hold-out had been
computed once (ADR-002): it changes the text of a correction and no number any kill condition reads.
The change is measured on the development split and published in `artifacts/retrieval.json`.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Sequence
from typing import Final, NamedTuple

__all__ = [
    "FAMILY_IDENTIFIER",
    "ClauseView",
    "authority_index",
    "families_named_in",
    "family_of",
]

#: A part family identifier as the corpus writes it: two letters, three letters, four digits —
#: `KH-PMP-6017`. Anchored on word boundaries so a part number (`KH-PMP-6017-A`) inside a heading is
#: read as its family and not as a longer token.
FAMILY_IDENTIFIER: Final = re.compile(r"\b([A-Z]{2}-[A-Z]{3}-\d{4})\b")


class ClauseView(NamedTuple):
    """The two facts about a candidate this rule reads, whatever shape the caller holds it in."""

    governs: Collection[str]
    section: str


def family_of(part_number: str) -> str:
    """The family a part belongs to: its part number without the variant suffix.

    `corpus.programs.part_number` builds a part number as `f"{family_id}-{variant}"`, so the family
    is everything before the last hyphen. Raises on a part number with no suffix rather than
    treating the whole string as a family, because a part number that is not in the corpus's shape
    is one this rule has no business scoping.
    """
    family, separator, variant = part_number.rpartition("-")
    if not separator or not family or not variant:
        raise ValueError(f"{part_number!r} is not a family identifier followed by a variant")
    return family


def families_named_in(section: str) -> frozenset[str]:
    """Every family identifier a section heading names. Empty for a general provision."""
    return frozenset(FAMILY_IDENTIFIER.findall(section))


def authority_index(
    candidates: Sequence[ClauseView], *, rejection_code: str, part_number: str
) -> int | None:
    """The position of the authority among `candidates`, or `None`. See the module docstring.

    `candidates` must be in rank order; the rule never re-ranks, it only chooses within the order
    it was given, which is what keeps recall untouched.
    """
    family = family_of(part_number)
    governing = [
        (index, families_named_in(view.section))
        for index, view in enumerate(candidates)
        if rejection_code in view.governs
    ]
    for index, named in governing:
        if family in named:
            return index
    for index, named in governing:
        if not named:
            return index
    return None
