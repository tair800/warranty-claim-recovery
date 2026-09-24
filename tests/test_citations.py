"""Citation verification, and the one test that says why membership is not enough.

`test_a_drifted_offset_is_caught_although_the_quote_is_present` is the point of this file. It builds
the situation warranty policies actually produce — the same sentence appearing twice in one document
— and asserts both halves: that `quote in document_text` is **true**, and that `verify_citation` is
**false**. A shortcut implementation would pass the first assertion and fail the second, so the test
fails when the check is weakened rather than only when it is deleted.

Everything else here exists because kill condition I allows zero unfaithful citations, and a
criterion with a budget of zero is only as good as the predicate it counts.
"""

from __future__ import annotations

from warranty_claim_recovery.domain import Citation, PolicyClause, RejectionCode
from warranty_claim_recovery.retrieval.citations import (
    citation_failure,
    citation_for,
    verify_citation,
)

SYNTHETIC_NOTICE = (
    "SYNTHETIC DOCUMENT. Generated for the warranty-claim-recovery corpus. "
    "This is not a manufacturer publication and no clause here is legal text.\n\n"
)

#: The sentence that appears twice. Warranty policies are written from templates and the same
#: obligation is restated in the general conditions and again in the parts schedule; this is the
#: shape, shortened.
REPEATED = "A corrected claim must be filed within the period stated in the rejection notice."


def _document_with_a_repeat() -> tuple[str, int, int]:
    """A document holding `REPEATED` twice, and the offsets of the **first** occurrence."""
    head = SYNTHETIC_NOTICE + "1. General conditions\n\n"
    middle = "\n\n2. Parts schedule\n\n"
    text = head + REPEATED + middle + REPEATED + "\n"
    return text, len(head), len(head) + len(REPEATED)


def _clause(text: str, start: int, end: int) -> PolicyClause:
    return PolicyClause(
        clause_id="CLS-0001",
        program_id="PGM-0001",
        document_id="DOC-0001",
        section="1. General conditions",
        text=text,
        start_offset=start,
        end_offset=end,
        governs=(RejectionCode.MISSING_SERIAL,),
    )


def test_a_faithful_citation_verifies() -> None:
    text, start, end = _document_with_a_repeat()
    citation = citation_for(_clause(REPEATED, start, end), "v1")
    assert verify_citation(citation, text) is True
    assert citation_failure(citation, text) is None


def test_the_second_occurrence_verifies_at_its_own_offsets() -> None:
    """Two identical spans are both faithful — at their own coordinates, and only there."""
    text, first_start, _ = _document_with_a_repeat()
    second_start = text.index(REPEATED, first_start + 1)
    citation = citation_for(_clause(REPEATED, second_start, second_start + len(REPEATED)), "v1")
    assert verify_citation(citation, text) is True


def test_a_drifted_offset_is_caught_although_the_quote_is_present() -> None:
    """The whole argument of `citations.py`, as one assertion pair.

    The offsets are shifted by three characters — a heading marker inserted upstream by a chunker,
    an off-by-one in a splitter, a document re-exported with a different preamble. The quote is
    still somewhere in the document, twice over, so `in` says the citation is faithful. It is not:
    the coordinates address text that is not the quote, and an adjudicator sent to those coordinates
    reads something else.
    """
    text, start, end = _document_with_a_repeat()
    drifted = citation_for(_clause(REPEATED, start + 3, end + 3), "v1")

    assert drifted.quote in text, "the fixture must reproduce the situation membership gets wrong"
    assert verify_citation(drifted, text) is False

    failure = citation_failure(drifted, text)
    assert failure is not None
    assert "holds different text" in failure


def test_offsets_past_the_end_of_the_document_fail_and_say_so() -> None:
    text, _, _ = _document_with_a_repeat()
    beyond = len(text) + 500
    citation = citation_for(_clause(REPEATED, beyond, beyond + len(REPEATED)), "v1")
    assert verify_citation(citation, text) is False
    failure = citation_failure(citation, text)
    assert failure is not None
    assert str(len(text)) in failure


def test_an_inverted_span_fails_even_though_the_model_would_refuse_to_build_one() -> None:
    """The bounds check is exercised, not merely present.

    `Citation`'s validator makes an inverted or empty span unconstructible through the normal path,
    so this uses `model_construct` to bypass it. The branch is kept in `verify_citation` because
    that function is what kill condition I counts, and a grading predicate that is correct only
    while a validator in another module stays as it is today has a dependency nobody records.
    """
    text, start, end = _document_with_a_repeat()
    inverted = Citation.model_construct(
        clause_id="CLS-0001",
        document_id="DOC-0001",
        policy_version="v1",
        section="1. General conditions",
        quote=REPEATED,
        start_offset=end,
        end_offset=start,
    )
    assert verify_citation(inverted, text) is False
    failure = citation_failure(inverted, text)
    assert failure is not None
    assert "empty or inverted" in failure


def test_the_same_offsets_against_a_different_document_fail() -> None:
    """A correct quote checked against the wrong document version is the failure ADR-001 names."""
    text, start, end = _document_with_a_repeat()
    citation = citation_for(_clause(REPEATED, start, end), "v1")
    superseded = text.replace("1. General conditions", "1. General conditions (revised)")
    assert verify_citation(citation, superseded) is False


def test_offsets_are_character_offsets_and_survive_non_ascii_prose() -> None:
    """British policy prose carries en-dashes and pound signs; offsets must not be byte counts.

    A byte-based offset would work on the ASCII half of this corpus and drift on the rest, which is
    the worst possible failure distribution: correct often enough to look right.
    """
    preamble = SYNTHETIC_NOTICE + "Cover is limited to £2,500 per claim — parts and labour.\n\n"
    quote = "The distributor's labour rate may not exceed the published schedule — see 4.2."
    text = preamble + quote + "\n"
    citation = citation_for(_clause(quote, len(preamble), len(preamble) + len(quote)), "v2")
    assert verify_citation(citation, text) is True


def test_citation_for_takes_the_version_from_the_caller_and_quotes_the_whole_clause() -> None:
    text, start, end = _document_with_a_repeat()
    clause = _clause(REPEATED, start, end)
    citation = citation_for(clause, "2026-03")

    assert citation.policy_version == "2026-03"
    assert citation.quote == clause.text
    assert (citation.start_offset, citation.end_offset) == (clause.start_offset, clause.end_offset)
    assert citation.document_id == clause.document_id
    assert citation.section == clause.section
    assert verify_citation(citation, text) is True
