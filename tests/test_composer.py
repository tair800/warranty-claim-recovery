"""The composer: a grounding property that is counted, and a model arm that refuses to pretend.

Nothing here touches a database, a retriever or an encoder. That is deliberate and it is the same
argument `tests/test_gate.py` makes about the gate: the properties being checked are properties of
assembly and of string arithmetic, so a test that needed infrastructure to run would fail for two
unrelated reasons and a reader could not tell which.

**The grounding tests are written around a distractor.** Every request below is validated against a
vocabulary that contains clause identifiers the case retrieved and the gate did **not** approve as
evidence. Checking the cited identifiers against themselves passes for every composer ever written,
including one that invents an authority; the only version of this check that can fail is one where
the vocabulary is wider than the citations. ADR-001's corpus supplies exactly that in the form of
withdrawn service bulletins, and `DISTRACTOR_CLAUSE` stands for one here.

**The failing case is planted by substituting a composer the running code calls**, not by asserting
over source text. `CLAUDE.md` §3 rule 6 requires a guard to be shown failing from behaviour, and
`LeakyComposer` is the smallest thing that produces the failure `validated` exists to catch: a
narrative that names a clause nobody cited.

**The abstractive arm is tested for raising, and its payload for what it would have sent.** ADR-001
§3 forbids a stub, so the only two things that can be asserted about it are that it refuses and that
the request it would have made carries the gate's approved evidence and nothing else. The second is
the stronger claim and it is the reason `prompt_payload` is public: a reviewer can read precisely
what would leave this system if a credential appeared, rather than a README promising that it would
be safe.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from decimal import Decimal
from typing import Any

import pytest

from warranty_claim_recovery.composer import (
    ARTICLE_50_DISCLOSURE,
    COMPOSER_ABSTRACTIVE,
    COMPOSER_EXTRACTIVE,
    PROMPT_PAYLOAD_KEYS,
    AbstractiveComposer,
    CorrectionRequest,
    DisclosureMissingError,
    ExtractiveComposer,
    LiveModelUnavailableError,
    UngroundedCorrectionError,
    correction_request,
    grounding_report,
    mentioned_clause_ids,
    validated,
)
from warranty_claim_recovery.deadline import claim_window
from warranty_claim_recovery.domain import (
    Citation,
    Claim,
    CorrectionProposal,
    GateDecision,
    RecoveryOutcome,
    RejectionCode,
    Requirement,
    RequirementStatus,
    WarrantyProgram,
)
from warranty_claim_recovery.eligibility import PartCoverage, assess
from warranty_claim_recovery.gate import decide
from warranty_claim_recovery.money import Currency, Money
from warranty_claim_recovery.recovery import compute

GBP = Currency.GBP
PART = "AB-1234-C"

FAILURE_DATE = date(2025, 6, 10)
REJECTED_ON = FAILURE_DATE + timedelta(days=16)
AS_OF = date(2025, 7, 1)

#: Two quotes, so that a proposal citing both can be told apart from one citing the first twice.
SERIAL_QUOTE = "Claims must cite the serial number stamped on the failed component."
RANGE_QUOTE = "The covered build range for each part is listed in schedule two of this policy."

#: A clause the case retrieved and the gate never approved. It is the whole reason the grounding
#: check has something to fail on; see the module docstring.
DISTRACTOR_CLAUSE = "CL-0099-WITHDRAWN"


def make_program() -> WarrantyProgram:
    return WarrantyProgram(
        program_id="PRG-ACME-V2",
        manufacturer="Acme Drivetrain (synthetic)",
        policy_version="v2",
        currency=GBP,
        correction_window_days=30,
        warranty_months=24,
        labour_rate_cap_per_hour=Money("55.00", GBP),
        deductible=Money("0.00", GBP),
        claim_cap=Money("5000.00", GBP),
    )


def make_claim() -> Claim:
    return Claim(
        claim_id="CLM-0001",
        program_id="PRG-ACME-V2",
        part_number=PART,
        serial_number="SN-004821",
        in_service_date=date(2024, 1, 15),
        failure_date=FAILURE_DATE,
        repair_invoice_date=FAILURE_DATE + timedelta(days=2),
        rejection_code=RejectionCode.MISSING_SERIAL,
        rejected_on=REJECTED_ON,
        claimed_parts=Money(Decimal("400.00"), GBP),
        claimed_labour_hours=4,
        claimed_labour_rate=Money(Decimal("50.00"), GBP),
    )


def citation(clause_id: str, section: str, quote: str) -> Citation:
    """A citation whose offsets span its own quote, because `Citation` refuses any other kind."""
    return Citation(
        clause_id=clause_id,
        document_id="DOC-ACME-V2",
        policy_version="v2",
        section=section,
        quote=quote,
        start_offset=120,
        end_offset=120 + len(quote),
    )


def satisfied(requirement_id: str, cited: Citation) -> Requirement:
    return Requirement(
        requirement_id=requirement_id,
        description="The evidence this rejection code demands, in the matrix's own words.",
        status=RequirementStatus.SATISFIED,
        detail="the evidence bundle carries this artefact",
        citation=cited,
    )


def make_decision(requirements: tuple[Requirement, ...] | None = None) -> GateDecision:
    """A real `GateDecision` from the real gate, not a hand-built one.

    Built by running the deterministic core, because `correction_request` refuses a decision whose
    requirements are not all satisfied and cited, and a hand-built decision would let this file
    assert things about a shape the gate cannot produce.
    """
    claim = make_claim()
    program = make_program()
    coverage = PartCoverage(part_number=PART, covered=True, serial_first=None, serial_last=None)
    eligibility = assess(claim, program, coverage)
    computation = compute(claim, program, eligibility)
    assessed = requirements or (
        satisfied("REQ-SERIAL-STAMPED", citation("CL-0007", "4.1", SERIAL_QUOTE)),
        satisfied("REQ-SERIAL-IN-COVERED-RANGE", citation("CL-0008", "4.2", RANGE_QUOTE)),
    )
    return decide(
        claim,
        program,
        claim_window(claim, program, AS_OF),
        eligibility,
        computation,
        assessed,
    )


def make_request(requirements: tuple[Requirement, ...] | None = None) -> CorrectionRequest:
    claim = make_claim()
    program = make_program()
    return correction_request(
        claim=claim,
        program=program,
        window=claim_window(claim, program, AS_OF),
        decision=make_decision(requirements),
    )


#: What the retrieval stage returned for this case: the two clauses the gate approved and one it
#: did not. `validated` is run against this everywhere below.
VOCABULARY = ("CL-0007", "CL-0008", DISTRACTOR_CLAUSE)


class LeakyComposer:
    """A composer that names a clause it does not cite, so the guard has something to catch.

    It exists because the shipped composer cannot produce this failure: `ExtractiveComposer`
    assembles from the citations it was handed, so there is no path by which an unapproved
    identifier could appear. A test that could only exercise the guard on grounded input would
    assert that a check which never fires does not fire.
    """

    __slots__ = ()

    @property
    def name(self) -> str:
        return "leaky-test-double"

    def propose(self, request: CorrectionRequest) -> CorrectionProposal:
        return CorrectionProposal(
            claim_id=request.claim_id,
            narrative=(
                f"The correction for {request.claim_id} additionally relies on clause "
                f"{DISTRACTOR_CLAUSE}, which nothing in this package cites."
            ),
            citations=request.citations,
            composer=self.name,
        )


# ------------------------------------------------------------------------------------------------
# The grounding property, as arithmetic over a vocabulary wider than the citations.
# ------------------------------------------------------------------------------------------------


def test_the_extractive_narrative_mentions_no_clause_it_does_not_cite() -> None:
    proposal = ExtractiveComposer().propose(make_request())
    report = grounding_report(proposal, VOCABULARY)

    assert set(report.mentioned) <= set(report.cited)
    assert report.ungrounded == ()
    assert report.grounded is True
    print(f"extractive grounding: mentioned {report.mentioned}, cited {report.cited}")


def test_a_retrieved_but_uncited_clause_never_reaches_the_narrative() -> None:
    """The distractor is in the vocabulary and must not be in the text.

    This is the assertion that gives the previous test its teeth: without it, a narrative that
    mentioned nothing at all would satisfy the subset property and prove nothing.
    """
    proposal = ExtractiveComposer().propose(make_request())

    assert DISTRACTOR_CLAUSE not in proposal.narrative
    assert "CL-0007" in proposal.narrative
    assert "CL-0008" in proposal.narrative


def test_every_cited_clause_is_quoted_verbatim_in_the_narrative() -> None:
    """The narrative is assembled from the spans, so each quote appears as the clause wrote it."""
    request = make_request()
    proposal = ExtractiveComposer().propose(request)

    for cited in request.citations:
        assert cited.quote in proposal.narrative
        assert str(cited.start_offset) in proposal.narrative
        assert str(cited.end_offset) in proposal.narrative


def test_the_narrative_carries_the_amount_and_the_deadline() -> None:
    request = make_request()
    narrative = ExtractiveComposer().narrative(request)

    assert str(request.recoverable_amount) in narrative
    assert request.closes_on.isoformat() in narrative
    assert request.rejection_code.value in narrative


def test_mentioned_clause_ids_is_matched_against_a_closed_vocabulary() -> None:
    """A pattern would invent matches out of prose and miss an identifier whose shape changed.

    Both directions are asserted: a clause named in the text but absent from the vocabulary is not
    reported, because the vocabulary is what the case actually retrieved and anything else is a
    guess about the corpus's naming.
    """
    text = "clauses CL-0007 and CL-2222 are discussed; ordinary prose mentions neither schedule"

    assert mentioned_clause_ids(text, VOCABULARY) == ("CL-0007",)
    assert mentioned_clause_ids(text, ()) == ()
    assert mentioned_clause_ids(text, ("CL-2222",)) == ("CL-2222",)


def test_a_composer_that_names_an_unapproved_clause_is_refused_on_the_way_out() -> None:
    """The guard, failing from behaviour rather than from a reading of the source."""
    request = make_request()
    leaked = LeakyComposer().propose(request)

    with pytest.raises(UngroundedCorrectionError, match=DISTRACTOR_CLAUSE):
        validated(leaked, vocabulary=VOCABULARY)

    report = grounding_report(leaked, VOCABULARY)
    assert report.ungrounded == (DISTRACTOR_CLAUSE,)
    assert report.grounded is False


def test_the_check_still_fires_when_the_distractor_is_absent_from_the_vocabulary() -> None:
    """A citation to a clause outside the retrieved set is folded in rather than excused.

    `grounding_report` adds the cited identifiers to the vocabulary, so the check does not depend on
    the retrieved set being a superset of the cited set. That is true today through
    `correction_request` and the check must not be the thing that assumes it.
    """
    leaked = LeakyComposer().propose(make_request())
    report = grounding_report(leaked, (DISTRACTOR_CLAUSE,))

    assert report.cited == ("CL-0007", "CL-0008")
    assert report.ungrounded == (DISTRACTOR_CLAUSE,)


def test_a_grounded_proposal_passes_through_validated_unchanged() -> None:
    proposal = ExtractiveComposer().propose(make_request())
    assert validated(proposal, vocabulary=VOCABULARY) is proposal


# ------------------------------------------------------------------------------------------------
# The Article 50 disclosure.
# ------------------------------------------------------------------------------------------------


def test_the_disclosure_constant_is_the_models_own_default() -> None:
    """One string, read from the field default, so the claim and the implementation cannot drift."""
    assert CorrectionProposal.model_fields["ai_disclosure"].default == ARTICLE_50_DISCLOSURE
    assert "AI system" in ARTICLE_50_DISCLOSURE


def test_every_proposal_carries_the_disclosure_without_being_asked() -> None:
    proposal = ExtractiveComposer().propose(make_request())
    assert proposal.ai_disclosure == ARTICLE_50_DISCLOSURE
    assert proposal.composer == COMPOSER_EXTRACTIVE


def test_a_stripped_disclosure_is_refused_at_the_boundary() -> None:
    request = make_request()
    stripped = CorrectionProposal(
        claim_id=request.claim_id,
        narrative=ExtractiveComposer().narrative(request),
        citations=request.citations,
        composer=COMPOSER_EXTRACTIVE,
        ai_disclosure="Produced by the claims team. No further disclosure is offered.",
    )

    with pytest.raises(DisclosureMissingError, match="Article 50"):
        validated(stripped, vocabulary=VOCABULARY)


# ------------------------------------------------------------------------------------------------
# What a composer is allowed to see.
# ------------------------------------------------------------------------------------------------


def test_a_refused_case_has_no_correction_to_compose() -> None:
    """`correction_request` refuses rather than composing prose somebody could file.

    The window is shut, so the gate's first rule fires and the outcome is `NOT_RECOVERABLE`. What a
    handler needs then is the reason and the requirement list, both already on the decision.
    """
    claim = make_claim()
    program = make_program()
    after_the_window = date(2026, 1, 1)
    window = claim_window(claim, program, after_the_window)
    coverage = PartCoverage(part_number=PART, covered=True, serial_first=None, serial_last=None)
    eligibility = assess(claim, program, coverage)
    decision = decide(
        claim,
        program,
        window,
        eligibility,
        compute(claim, program, eligibility),
        (satisfied("REQ-SERIAL-STAMPED", citation("CL-0007", "4.1", SERIAL_QUOTE)),),
    )

    assert decision.outcome is RecoveryOutcome.NOT_RECOVERABLE
    with pytest.raises(ValueError, match="no correction to file"):
        correction_request(claim=claim, program=program, window=window, decision=decision)


def test_the_request_carries_the_gates_citations_and_takes_none_from_its_caller() -> None:
    """There is no parameter through which a fourth clause could be added on the way to a model."""
    request = make_request()

    assert tuple(item.clause_id for item in request.citations) == ("CL-0007", "CL-0008")
    assert DISTRACTOR_CLAUSE not in {item.clause_id for item in request.citations}
    assert request.requirements == make_decision().requirements


def test_one_clause_satisfying_two_requirements_is_cited_once() -> None:
    """Deduplicated in requirement order: a clause cited twice reads as two separate authorities."""
    shared = citation("CL-0007", "4.1", SERIAL_QUOTE)
    request = make_request(
        (
            satisfied("REQ-SERIAL-STAMPED", shared),
            satisfied("REQ-SERIAL-IN-COVERED-RANGE", shared),
        )
    )

    assert tuple(item.clause_id for item in request.citations) == ("CL-0007",)
    assert len(request.requirements) == 2


# ------------------------------------------------------------------------------------------------
# The abstractive port: it raises, and its payload is inspectable.
# ------------------------------------------------------------------------------------------------


def test_the_abstractive_arm_raises_when_no_credential_exists() -> None:
    """ADR-001 §3: a stub here would be scored as a model output for a model nobody called."""
    arm = AbstractiveComposer()
    assert arm.available is False
    assert arm.name == COMPOSER_ABSTRACTIVE

    with pytest.raises(LiveModelUnavailableError, match="no model credential is configured"):
        arm.propose(make_request())


def test_the_abstractive_arm_raises_even_with_a_credential_and_says_what_is_missing() -> None:
    """A key changes the message, not the outcome: no provider client is wired to it."""
    arm = AbstractiveComposer(api_key="placeholder-not-a-real-key", model="some-model")
    assert arm.available is True

    with pytest.raises(LiveModelUnavailableError, match="no provider client is wired"):
        arm.propose(make_request())


def test_the_prompt_payload_has_exactly_the_declared_keys() -> None:
    payload = AbstractiveComposer().prompt_payload(make_request())
    assert set(payload) == set(PROMPT_PAYLOAD_KEYS)


def test_the_prompt_payload_carries_the_gates_approved_evidence_and_nothing_else() -> None:
    """The evidence in the payload is exactly the evidence the gate approved.

    Both directions. Every approved requirement appears with its clause and its verbatim quote, and
    nothing that was not approved appears anywhere in the serialised request — which is the check
    that would catch a distractor arriving through a future field.
    """
    request = make_request()
    payload = AbstractiveComposer().prompt_payload(request)
    evidence: list[dict[str, Any]] = payload["approved_evidence"]

    assert [item["requirement_id"] for item in evidence] == [
        requirement.requirement_id for requirement in request.requirements
    ]
    assert [item["clause_id"] for item in evidence] == ["CL-0007", "CL-0008"]
    assert [item["quote"] for item in evidence] == [SERIAL_QUOTE, RANGE_QUOTE]

    rendered = json.dumps(payload, sort_keys=True)
    assert DISTRACTOR_CLAUSE not in rendered
    assert "withdrawn" not in rendered.lower()


def test_the_prompt_payload_does_not_forward_the_requirements_internal_detail() -> None:
    """`Requirement.detail` is written for a technician chasing a document, not for a manufacturer.

    Sending it would invite a model to repeat an internal instruction back to the manufacturer as
    though it were part of the claim.
    """
    request = make_request()
    payload = AbstractiveComposer().prompt_payload(request)

    assert all("detail" not in item for item in payload["approved_evidence"])
    assert "chase the technician" not in json.dumps(payload)


def test_no_monetary_value_crosses_the_model_boundary_as_a_float() -> None:
    """The one boundary where being wrong is least recoverable, checked over the whole payload."""
    payload = AbstractiveComposer().prompt_payload(make_request())

    floats = list(_floats_in(payload))
    assert floats == [], f"the payload carries floating-point values: {floats}"
    assert payload["claimed_total"] == "600.00"
    assert payload["recoverable_amount"] == "600.00"
    assert payload["currency"] == GBP.value
    assert Decimal(payload["recoverable_amount"]) == make_request().recoverable_amount.amount


def test_the_payload_states_the_outcome_the_deterministic_gate_reached() -> None:
    payload = AbstractiveComposer().prompt_payload(make_request())

    assert payload["outcome"] == RecoveryOutcome.RECOVERABLE.value
    assert payload["rejection_code"] == RejectionCode.MISSING_SERIAL.value
    assert payload["policy_version"] == "v2"
    assert payload["disclosure"] == ARTICLE_50_DISCLOSURE


def _floats_in(value: object, path: str = "$") -> list[str]:
    """Every floating-point value in a nested payload, with the path that reaches it."""
    if isinstance(value, float):
        return [f"{path}={value!r}"]
    if isinstance(value, dict):
        found: list[str] = []
        for key, item in value.items():
            found.extend(_floats_in(item, f"{path}.{key}"))
        return found
    if isinstance(value, list):
        found = []
        for index, item in enumerate(value):
            found.extend(_floats_in(item, f"{path}[{index}]"))
        return found
    return []
