"""The policy documents, the clauses cut from them, and the offsets that make a citation checkable.

Kill condition I is that every cited span is present verbatim at the offsets it names in the
document version it names. That guarantee is not something the retrieval layer can establish and it
is not something a test can bolt on afterwards: it has to be true of the corpus at the moment the
corpus is written, or every citation the system produces is unverifiable by construction. So the
document text is assembled here, character by character, with a cursor, and each clause records the
span it occupies in the assembled string. `generate.py` then slices every document at every
clause's offsets and refuses to write a corpus where one does not match.

Rejected: storing clauses without a document and treating the clause text as its own source. It is
simpler and it makes kill condition I vacuous — a citation would be checked against the very string
it was copied from, which is a comparison that cannot fail. A citation is only worth anything if the
document exists independently of the quote.

**Three documents per programme, and the third one is withdrawn.**

- the warranty policy, ten clauses, which is where all seven rejection codes find their governing
  authority;
- the current service bulletin, three clauses, two of which *amend* the policy for one part family —
  so the governing clause for a rejection code is not a property of the code alone;
- the superseded service bulletin, two clauses, which say almost the same thing in almost the same
  words and govern nothing.

The withdrawn bulletin is the reason `PolicyClause.governs` being empty is meaningful rather than
decorative. `domain.PolicyClause` documents that an empty `governs` means the clause is context and
may never be cited as the basis of a satisfied requirement; the withdrawn bulletin is what that rule
exists for. A retriever that ranks on text similarity alone will rank it near the clause that
replaced it, which is the point: a correct quotation from a withdrawn bulletin is still the wrong
authority, and the manufacturer will say so.

The amendment is also what stops the `exact_code_lookup` baseline in ADR-001 §6 from being
unbeatable by construction. If each rejection code had exactly one governing clause per programme,
a table lookup would score a perfect recall and kill condition J — which requires the system to
beat every baseline — could never be met by any retriever, however good. Two codes per programme
are amended for one named part family, so the governing clause depends on the code *and* the part,
and a code-keyed table is wrong for every claim about that family.

Every document opens with a sentence saying it is synthetic evaluation data and not a manufacturer
publication. It is in the document body rather than only in the JSON envelope, because the body is
what a citation shows a reader and an envelope is what a reader never sees.
"""

from __future__ import annotations

from typing import Final, NamedTuple

from warranty_claim_recovery.corpus.programs import ProgramSpec
from warranty_claim_recovery.domain import PolicyClause, RejectionCode

__all__ = [
    "AMENDED_CODES",
    "AMENDED_FAMILY_POSITION",
    "BULLETIN_CURRENT",
    "BULLETIN_SUPERSEDED",
    "POLICY",
    "SYNTHETIC_NOTICE",
    "BuiltDocument",
    "build_documents",
    "governing_clause_id",
]

#: Stated in the body of every document and in the body of every generated file. ADR-001 §1 and
#: `CLAUDE.md` rule 8 both require it by name, and it is a single constant so that the wording
#: cannot drift between the file that claims it and the file that carries it.
SYNTHETIC_NOTICE: Final = (
    "This is synthetic evaluation data generated from a committed seed. It is not a manufacturer "
    "publication, it describes no real warranty programme, and no clause here has legal effect."
)

POLICY: Final = "policy"
BULLETIN_CURRENT: Final = "bulletin-b"
BULLETIN_SUPERSEDED: Final = "bulletin-a"

#: Which part family the current bulletin amends. Position 0 in every programme — a fixed position
#: rather than a drawn one, because the claim recipe has to route claims at the amended family and
#: at the unamended ones in known proportions, and a drawn position would make that routing depend
#: on the seed in a way no reader could follow.
AMENDED_FAMILY_POSITION: Final = 0

#: The two codes the current bulletin takes authority over for the amended family.
AMENDED_CODES: Final[tuple[RejectionCode, ...]] = (
    RejectionCode.MISSING_SERIAL,
    RejectionCode.LABOUR_RATE_EXCEEDED,
)


class BuiltDocument(NamedTuple):
    """A document, its assembled text, and the clauses whose offsets index into that text."""

    document_id: str
    program_id: str
    policy_version: str
    kind: str
    title: str
    revision: str
    is_current: bool
    superseded_by: str | None
    text: str
    clauses: tuple[PolicyClause, ...]


class _Section(NamedTuple):
    section: str
    body: str
    governs: tuple[RejectionCode, ...]


def _policy_sections(spec: ProgramSpec) -> tuple[_Section, ...]:
    """The ten policy clauses.

    Each body is written to be longer than the two-hundred-character contract floor, to name the
    rejection code it governs in the manufacturer's own vocabulary, and to interpolate the terms
    that actually differ between programmes — the warranty length, the correction window, the
    labour cap, the deductible and the per-claim cap. Interpolating the real terms is not
    decoration: a corpus whose clause text is identical across eighteen programmes cannot
    distinguish a retriever that respects the programme filter from one that ignores it, because
    every candidate would be a correct answer to the wrong question.
    """
    program = spec.program
    maker = spec.manufacturer.name
    version = program.policy_version
    currency = program.currency.value
    excluded = spec.families[2]
    supplier_changed = spec.families[3]
    return (
        _Section(
            section="1. Scope and application",
            body=(
                f"This warranty policy, version {version}, is issued by {maker} and governs every "
                f"claim submitted by an authorised distributor against equipment supplied under a "
                f"{maker} distribution agreement. It replaces all earlier versions for claims "
                f"rejected on or after the date this version took effect. All amounts in this "
                f"policy are stated in {currency}; a claim presented in any other currency is "
                f"returned to the distributor without adjudication and no conversion is performed "
                f"by {maker} on the distributor's behalf."
            ),
            governs=(),
        ),
        _Section(
            section="2. Serial identification",
            body=(
                f"Every claim must carry the serial number stamped on the component identification "
                f"plate, transcribed in full and without separators. {maker} rejects a claim under "
                f"code MISSING_SERIAL where the serial is absent, illegible, or falls outside the "
                f"serial range recorded against the part number claimed. A photograph of the "
                f"component identification plate is the serial evidence of record. A serial read "
                f"from the machine chassis plate rather than from the component's own plate does "
                f"not satisfy this clause and will be rejected a second time."
            ),
            governs=(RejectionCode.MISSING_SERIAL,),
        ),
        _Section(
            section="3. Proof of installation and commissioning",
            body=(
                f"A claim must be supported by the installation certificate and the recorded "
                f"commissioning date for the equipment. {maker} rejects a claim under code "
                f"MISSING_INSTALL_PROOF where neither document is supplied, where the "
                f"commissioning "
                f"date is later than the recorded in-service date, or where the repair invoice "
                f"cannot be matched to an installation on file. The warranty period under clause 6 "
                f"runs from the in-service date evidenced here and from no other date."
            ),
            governs=(RejectionCode.MISSING_INSTALL_PROOF,),
        ),
        _Section(
            section="4. Failure coding and diagnosis",
            body=(
                f"The failure code recorded on the claim must be drawn from the {maker} failure "
                f"taxonomy current for policy version {version} and must be supported by the "
                f"diagnostic report produced at the time of the repair. A claim whose failure code "
                f"is absent, withdrawn, or contradicted by the diagnostic report is rejected under "
                f"code WRONG_FAILURE_CODE. Where a service bulletin has restated the taxonomy, the "
                f"code current on the failure date applies and not the code current on the date "
                f"the "
                f"claim was filed."
            ),
            governs=(RejectionCode.WRONG_FAILURE_CODE,),
        ),
        _Section(
            section="5. Covered parts and exclusions",
            body=(
                f"Cover extends only to the part numbers listed in the covered-parts schedule for "
                f"policy version {version}. The {excluded.label} family {excluded.family_id} is "
                f"excluded in full as a serviceable consumable. A component whose supplier changed "
                f"after the equipment manufacture date, including affected variants of "
                f"{supplier_changed.family_id}, is outside cover irrespective of the failure. A "
                f"claim naming a part number that is not listed, or a variant of a listed family "
                f"that is not itself listed, is rejected under code PART_NOT_COVERED."
            ),
            governs=(RejectionCode.PART_NOT_COVERED,),
        ),
        _Section(
            section="6. Warranty period",
            body=(
                f"Cover runs for {program.warranty_months} months from the in-service date "
                f"evidenced under clause 3. A failure occurring on or after the day the period "
                f"ends "
                f"is outside cover and is rejected under code OUTSIDE_WARRANTY_PERIOD, however "
                f"small the margin. {maker} does not extend the period for equipment that was "
                f"stored, idle or out of commission, and does not restart it after a repair "
                f"carried "
                f"out under this policy. The failure date, not the repair invoice date, decides "
                f"whether a claim falls inside the period."
            ),
            governs=(RejectionCode.OUTSIDE_WARRANTY_PERIOD,),
        ),
        _Section(
            section="7. Duplicate claims and prior recovery",
            body=(
                f"A recovery is identified by the claim reference together with the part number "
                f"and "
                f"the serial number. {maker} rejects a second claim carrying the same three values "
                f"under code DUPLICATE_CLAIM. Where an amount has already been recovered against "
                f"that identity, the amount already paid is deducted from any further settlement "
                f"and the balance alone is recoverable; where the amount already paid equals or "
                f"exceeds the settlement due, nothing further is recoverable and the claim is "
                f"closed without payment."
            ),
            governs=(RejectionCode.DUPLICATE_CLAIM,),
        ),
        _Section(
            section="8. Labour rates and time allowances",
            body=(
                f"Labour is reimbursed at the rate recorded in the distributor's labour rate "
                f"agreement, capped at {program.labour_rate_cap_per_hour} per hour under policy "
                f"version {version}. A claim presenting a higher rate is rejected under code "
                f"LABOUR_RATE_EXCEEDED; on resubmission the hours are reimbursed at the capped "
                f"rate "
                f"and the excess is borne by the distributor. Time in excess of the published "
                f"allowance for the operation is treated the same way and is not recoverable from "
                f"{maker} under any heading."
            ),
            governs=(RejectionCode.LABOUR_RATE_EXCEEDED,),
        ),
        _Section(
            section="9. Deductible, cap and settlement",
            body=(
                f"A deductible of {program.deductible} is applied to the eligible amount of every "
                f"claim under policy version {version}. The per-claim ceiling is "
                f"{program.claim_cap} and is applied after the deductible and never before it, so "
                f"that the ceiling limits what {maker} settles rather than what the distributor "
                f"claims. Amounts excluded under clauses 5 and 8 are removed before the deductible "
                f"is taken. No settlement is ever negative: where the deductible exceeds the "
                f"eligible amount the settlement is nil."
            ),
            governs=(),
        ),
        _Section(
            section="10. Correction window and resubmission",
            body=(
                f"A rejected claim may be corrected and resubmitted within "
                f"{program.correction_window_days} days of the rejection notice. A correction "
                f"received on the closing day is in time; one received on the following day is out "
                f"of time and the claim is written off. {maker} does not reopen a claim whose "
                f"correction window has closed, does not accept a partial correction as stopping "
                f"the clock, and counts the window from the date of the rejection notice and not "
                f"from the date the distributor became aware of it."
            ),
            governs=(),
        ),
    )


def _current_bulletin_sections(spec: ProgramSpec) -> tuple[_Section, ...]:
    program = spec.program
    maker = spec.manufacturer.name
    amended = spec.families[AMENDED_FAMILY_POSITION]
    supplier_changed = spec.families[3]
    return (
        _Section(
            section=f"B1. Amended serial identification for the {amended.family_id} family",
            body=(
                f"For the {amended.label} family {amended.family_id} only, this bulletin replaces "
                f"clause 2 of policy version {program.policy_version}. The serial is carried on "
                f"the "
                f"secondary plate behind the inspection cover and not on the primary plate, and a "
                f"claim rejected under code MISSING_SERIAL for this family is corrected by "
                f"supplying the secondary plate photograph. {maker} treats a serial transcribed "
                f"from the primary plate of this family as unreadable rather than as absent, and "
                f"the distinction decides which evidence the correction must carry."
            ),
            governs=(RejectionCode.MISSING_SERIAL,),
        ),
        _Section(
            section=f"B2. Amended labour allowance for the {amended.family_id} family",
            body=(
                f"For the {amended.family_id} family only, this bulletin replaces clause 8 of "
                f"policy version {program.policy_version}. The published allowance for a seal-pack "
                f"replacement on this family is revised, and a claim rejected under code "
                f"LABOUR_RATE_EXCEEDED for this family is adjudicated against the revised "
                f"allowance "
                f"and against the capped rate of {program.labour_rate_cap_per_hour} per hour. "
                f"Hours above the revised allowance remain the distributor's cost and {maker} will "
                f"not settle them on resubmission."
            ),
            governs=(RejectionCode.LABOUR_RATE_EXCEEDED,),
        ),
        _Section(
            section=f"B3. Supplier change notice for the {supplier_changed.family_id} family",
            body=(
                f"{maker} records that the component supplier for part variants within the "
                f"{supplier_changed.label} family {supplier_changed.family_id} changed after the "
                f"equipment manufacture date for the affected build range. This notice is "
                f"published "
                f"for information and takes no authority over any rejection code; the "
                f"covered-parts schedule and clause 5 of the policy decide cover for the affected "
                f"variants, and nothing in this bulletin extends cover to a variant the schedule "
                f"does not list."
            ),
            governs=(),
        ),
    )


def _superseded_bulletin_sections(spec: ProgramSpec) -> tuple[_Section, ...]:
    """The withdrawn bulletin.

    Both clauses govern nothing, and both read like the ones that replaced them — which is
    the whole reason they are in the corpus.
    """
    program = spec.program
    maker = spec.manufacturer.name
    amended = spec.families[AMENDED_FAMILY_POSITION]
    return (
        _Section(
            section=f"A1. Serial identification for the {amended.family_id} family (withdrawn)",
            body=(
                f"This clause is withdrawn and is retained for the record only. It formerly stated "
                f"that for the {amended.label} family {amended.family_id} the serial is carried on "
                f"the primary plate above the inspection cover, and that a claim rejected under "
                f"code MISSING_SERIAL for this family is corrected by supplying the primary plate "
                f"photograph. {maker} has replaced this clause in the current bulletin and a "
                f"correction filed on this authority is rejected a second time."
            ),
            governs=(),
        ),
        _Section(
            section=f"A2. Labour allowance for the {amended.family_id} family (withdrawn)",
            body=(
                f"This clause is withdrawn and is retained for the record only. It formerly set "
                f"the "
                f"published allowance for a seal-pack replacement on the {amended.family_id} "
                f"family under policy version {program.policy_version}, and stated that a claim "
                f"rejected under code LABOUR_RATE_EXCEEDED for this family is adjudicated against "
                f"that allowance. {maker} has replaced this clause in the current bulletin; "
                f"quoting "
                f"it in a correction cites an authority that no longer exists."
            ),
            governs=(),
        ),
    )


def _assemble(
    *,
    document_id: str,
    program_id: str,
    heading: str,
    subtitle: str,
    sections: tuple[_Section, ...],
    clause_prefix: str,
) -> tuple[str, tuple[PolicyClause, ...]]:
    """Build the document text and the clauses together, tracking one cursor.

    Deliberately not two passes. A first pass that renders the text and a second that searches it
    for each body would find the *first* occurrence of a body that appears twice, and the two
    withdrawn bulletin clauses are written to resemble the clauses that replaced them precisely so
    that a corpus built by searching would put a citation's offsets on the wrong clause. One cursor
    cannot make that mistake.
    """
    pieces: list[str] = []
    cursor = 0
    clauses: list[PolicyClause] = []

    def emit(chunk: str) -> int:
        nonlocal cursor
        start = cursor
        pieces.append(chunk)
        cursor += len(chunk)
        return start

    emit(SYNTHETIC_NOTICE)
    emit("\n\n")
    emit(heading)
    emit("\n")
    emit(subtitle)
    emit("\n\n")

    for index, section in enumerate(sections, start=1):
        emit("## ")
        emit(section.section)
        emit("\n\n")
        start = emit(section.body)
        clauses.append(
            PolicyClause(
                clause_id=f"{clause_prefix}-{index:02d}",
                program_id=program_id,
                document_id=document_id,
                section=section.section,
                text=section.body,
                start_offset=start,
                end_offset=start + len(section.body),
                governs=section.governs,
            )
        )
        emit("\n\n")

    return "".join(pieces), tuple(clauses)


def build_documents(spec: ProgramSpec) -> tuple[BuiltDocument, ...]:
    """The three documents of one programme, in a fixed order, with their clauses."""
    program = spec.program
    program_id = program.program_id
    maker = spec.manufacturer.name
    version = program.policy_version

    plans = (
        (
            POLICY,
            f"{maker} — Warranty Policy {version}",
            "Issued to authorised distributors. Synthetic evaluation document.",
            "r1",
            True,
            None,
            _policy_sections(spec),
        ),
        (
            BULLETIN_CURRENT,
            f"{maker} — Service Bulletin B, under Warranty Policy {version}",
            "Current bulletin. Amends the clauses it names and no others.",
            "B",
            True,
            None,
            _current_bulletin_sections(spec),
        ),
        (
            BULLETIN_SUPERSEDED,
            f"{maker} — Service Bulletin A, under Warranty Policy {version}",
            f"Withdrawn. Replaced in full by Service Bulletin B of policy version {version}.",
            "A",
            False,
            f"{program_id}-{BULLETIN_CURRENT}",
            _superseded_bulletin_sections(spec),
        ),
    )

    documents: list[BuiltDocument] = []
    for kind, heading, subtitle, revision, is_current, superseded_by, sections in plans:
        document_id = f"{program_id}-{kind}"
        text, clauses = _assemble(
            document_id=document_id,
            program_id=program_id,
            heading=heading,
            subtitle=subtitle,
            sections=sections,
            clause_prefix=document_id,
        )
        documents.append(
            BuiltDocument(
                document_id=document_id,
                program_id=program_id,
                policy_version=version,
                kind=kind,
                title=heading,
                revision=revision,
                is_current=is_current,
                superseded_by=superseded_by,
                text=text,
                clauses=clauses,
            )
        )
    return tuple(documents)


def governing_clause_id(
    documents: tuple[BuiltDocument, ...],
    *,
    code: RejectionCode,
    family_position: int,
) -> str:
    """Which clause actually decides this rejection, for this part family.

    The bulletin takes authority over `AMENDED_CODES` for the amended family; the policy decides
    everything else. Resolved by searching the built documents for the clause whose `governs`
    contains the code rather than by an index written down twice, so a change to the section list
    cannot leave a truth file pointing at a clause that has moved.
    """
    amended = family_position == AMENDED_FAMILY_POSITION and code in AMENDED_CODES
    wanted = BULLETIN_CURRENT if amended else POLICY
    for document in documents:
        if document.kind != wanted:
            continue
        for clause in document.clauses:
            if code in clause.governs:
                return clause.clause_id
    raise KeyError(
        f"no clause in the {wanted} document of {documents[0].program_id} governs {code}. Every "
        f"rejection code must have exactly one governing authority per programme and part family, "
        f"or the retrieval evaluation has no gold answer to score against."
    )
