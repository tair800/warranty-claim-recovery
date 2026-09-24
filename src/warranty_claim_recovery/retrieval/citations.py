"""Checking a citation, and the one shortcut that makes the check worthless.

A citation in this system is a quote together with the coordinates it was taken from: a document
identifier, a policy version, and a pair of character offsets. Kill condition I allows zero
citations whose span is not present verbatim at the offsets it names, and that criterion is only as
good as the function below.

**The shortcut, and why it is refused.** The obvious implementation is
`citation.quote in document_text`. It is one line, it is faster, and it passes for every citation
this system will ever produce — including the broken ones. Warranty policies are written from
templates, and the same sentence about filing a corrected claim within the stated period appears in
the general conditions, in the parts schedule and in the labour schedule of a single document. A
membership test on that sentence succeeds no matter which of the three the offsets point at, so a
clause whose coordinates drifted by a chunking change — a paragraph reflowed, a heading inserted
upstream, an off-by-one in a splitter — reports as faithful. The citation then sends a
manufacturer's adjudicator to a section that does not say what the resubmission claims it says,
which is the failure the whole evidence chain exists to prevent, arriving through the check that was
supposed to catch it.

So the check **slices and compares**. `document_text[start:end] == quote`, and nothing else. The
offsets are the claim being made; verifying anything weaker verifies a different claim.

Rejected: normalising whitespace on both sides before comparing, to tolerate a document that was
re-wrapped after the clause was extracted. It would make the check pass more often, which is exactly
what is wrong with it. A re-wrapped document is a new document version, its clauses need
re-extracting, and a comparison that forgives the difference hides the fact that the stored offsets
now address the wrong characters. The half-open interval convention matches Python slicing and
`PolicyClause`'s own validator: `end_offset - start_offset == len(quote)`.
"""

from __future__ import annotations

from warranty_claim_recovery.domain import Citation, PolicyClause

__all__ = ["citation_failure", "citation_for", "verify_citation"]


def verify_citation(citation: Citation, document_text: str) -> bool:
    """True when the document says, at those exact offsets, exactly what the citation quotes.

    The bounds are tested before the slice even though Python would return a short slice rather than
    raise. A short slice compares unequal and would give the right answer anyway, but only by
    accident: `Citation` guarantees that the quote's length equals the span, so the lengths differ
    and the comparison fails. Relying on that is relying on a validator in another module staying
    the way it is, and the cost of not relying on it is one comparison.
    """
    if citation.end_offset > len(document_text):
        return False
    if citation.start_offset >= citation.end_offset:
        return False
    return document_text[citation.start_offset : citation.end_offset] == citation.quote


def citation_failure(citation: Citation, document_text: str) -> str | None:
    """Why a citation failed, in a sentence, or `None` when it did not fail.

    `verify_citation` answers the graded question and returns a boolean, because a criterion counted
    in an artifact should be counted from one unambiguous answer. This exists alongside it so that
    the artifact can also carry a reason: a run reporting "three unfaithful citations" with no
    indication of whether the offsets ran off the end of the document or landed on the wrong
    paragraph is a run that has to be repeated by hand before anyone can act on it.

    The reason never includes the document's surrounding text. A diagnostic that quotes the
    neighbourhood of a bad offset is a diagnostic that ends up in a log, and the corpus is synthetic
    today but this code path is not.
    """
    if citation.end_offset > len(document_text):
        return (
            f"{citation.clause_id}: the citation ends at offset {citation.end_offset} and "
            f"{citation.document_id} at version {citation.policy_version} is "
            f"{len(document_text)} characters long"
        )
    if citation.start_offset >= citation.end_offset:
        return (
            f"{citation.clause_id}: the span [{citation.start_offset}, {citation.end_offset}) is "
            f"empty or inverted and cannot address any text"
        )
    found = document_text[citation.start_offset : citation.end_offset]
    if found != citation.quote:
        return (
            f"{citation.clause_id}: {citation.document_id} at version {citation.policy_version} "
            f"holds different text at [{citation.start_offset}, {citation.end_offset}); the quote "
            f"is {len(citation.quote)} characters and does not match the document there"
        )
    return None


def citation_for(clause: PolicyClause, policy_version: str) -> Citation:
    """The citation a retrieved clause supports, at the policy version that was actually searched.

    `policy_version` is a parameter rather than a field of `PolicyClause` because the clause does
    not know it. A clause identifier and its offsets are properties of a document; which version of
    the policy was in force is a property of the *query*, and the retriever is the only party that
    holds both. Storing the version on the clause and reading it from there would mean a citation
    could name a version nobody searched — a correct quote from a superseded policy, which is the
    wrong authority, and the manufacturer will say so.

    The whole clause text is the quote. A shorter excerpt would need its own offsets, computed by
    finding the excerpt inside the clause, and `str.find` returning the first of several occurrences
    is precisely the drift this module refuses to tolerate elsewhere.
    """
    return Citation(
        clause_id=clause.clause_id,
        document_id=clause.document_id,
        policy_version=policy_version,
        section=clause.section,
        quote=clause.text,
        start_offset=clause.start_offset,
        end_offset=clause.end_offset,
    )
