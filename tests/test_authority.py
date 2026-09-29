"""The authority rule on hand-written inputs, including the bulletin case that motivated it.

`authority.authority_index` decides which retrieved clause a requirement cites, for the running
graph and for the evaluation alike. Each test below is one arrangement a correction can meet, and
each is named for the failure it prevents. The rule reads only an exact family identifier and the
`governs` codes; no test here depends on the database, the encoder or the corpus.
"""

from __future__ import annotations

import pytest

from warranty_claim_recovery.authority import (
    ClauseView,
    authority_index,
    families_named_in,
    family_of,
)

SERIAL = "MISSING_SERIAL"
LABOUR = "LABOUR_RATE_EXCEEDED"
PUMP = "KH-PMP-6017"
VALVE = "KH-VLV-6137"

POLICY = ClauseView(governs=(SERIAL,), section="2. Serial identification on the component plate")
PUMP_BULLETIN = ClauseView(
    governs=(SERIAL,), section=f"B1. Amended serial identification for the {PUMP} family"
)
VALVE_BULLETIN = ClauseView(
    governs=(SERIAL,), section=f"B1. Amended serial identification for the {VALVE} family"
)
CONTEXT = ClauseView(governs=(), section="7. General conditions")


def pick(candidates: list[ClauseView], part_number: str, code: str = SERIAL) -> int | None:
    return authority_index(candidates, rejection_code=code, part_number=part_number)


def test_a_family_bulletin_outranks_the_policy_clause_it_amends() -> None:
    """The defect: the replaced policy clause ranked first and was cited in 66 of 78 cases.

    The bulletin says it replaces the policy clause for this family, so for a part of this family
    it is the authority, whichever of the two the vector search happened to put first.
    """
    assert pick([POLICY, PUMP_BULLETIN], f"{PUMP}-A") == 1


def test_a_bulletin_for_another_family_is_not_authority_for_this_one() -> None:
    """The other direction: a valve's amendment must never be cited against a pump claim."""
    assert pick([VALVE_BULLETIN, POLICY], f"{PUMP}-A") == 1


def test_only_another_familys_bulletin_governs_so_there_is_no_authority() -> None:
    """Returning the valve bulletin here would cite another machine's rule. Nothing is right."""
    assert pick([VALVE_BULLETIN, CONTEXT], f"{PUMP}-A") is None


def test_the_general_policy_governs_a_family_no_bulletin_amends() -> None:
    assert pick([CONTEXT, POLICY, VALVE_BULLETIN], f"{PUMP}-B") == 1


def test_a_clause_that_governs_nothing_is_never_the_authority() -> None:
    """Every withdrawn bulletin in the corpus has this shape: 0 of 36 govern any code."""
    assert pick([CONTEXT, CONTEXT, CONTEXT], f"{PUMP}-A") is None


def test_governing_a_different_code_does_not_count() -> None:
    labour_bulletin = ClauseView(
        governs=(LABOUR,), section=f"B2. Labour rate for the {PUMP} family"
    )
    assert pick([labour_bulletin, POLICY], f"{PUMP}-A") == 1


def test_rank_order_is_kept_within_a_tier() -> None:
    """The rule chooses; it never re-ranks. Two general provisions: the higher-ranked wins."""
    second_policy = ClauseView(governs=(SERIAL,), section="9. Serial evidence, supplementary")
    assert pick([CONTEXT, second_policy, POLICY], f"{PUMP}-A") == 1


def test_nothing_retrieved_means_no_authority() -> None:
    assert pick([], f"{PUMP}-A") is None


def test_the_family_is_read_from_the_part_number_by_its_last_hyphen() -> None:
    assert family_of(f"{PUMP}-A") == PUMP
    assert family_of(f"{PUMP}-B") == PUMP


@pytest.mark.parametrize("malformed", ["KHPMP6017", "-A", "KH-PMP-6017-"])
def test_a_part_number_not_in_the_corpus_shape_is_refused_rather_than_guessed(
    malformed: str,
) -> None:
    with pytest.raises(ValueError, match="variant"):
        family_of(malformed)


def test_a_part_number_inside_a_heading_is_read_as_its_family() -> None:
    """A heading that quotes a full part number still names the family, not a longer token."""
    assert families_named_in(f"Note on part {PUMP}-A fitted after 2021") == frozenset({PUMP})
    assert families_named_in("7. General conditions") == frozenset()
