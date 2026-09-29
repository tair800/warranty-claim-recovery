"""What the checkpoint holds, and why it holds nothing richer than JSON.

The checkpoint is the durable record of a case. It is written by one process and read by another,
on another day, after a deployment, and the whole of ADR-001 §4.1 rests on that read succeeding.
Everything in this module follows from taking that sentence literally.

**Every value in the state is a JSON primitive.** LangGraph's own serialiser would happily store a
Pydantic model, a `Decimal` or a `date`, and the tempting version of this module is therefore no
module at all: annotate the state with the domain types and let the checkpointer work it out. It was
rejected, for two reasons that are not stylistic.

The first is that a checkpoint written that way is keyed to the class layout that wrote it. Rename a
field on `Claim`, add a validator, move a model between modules, and every case that was mid-flight
at deploy time fails to deserialise. The symptom is not an error at the point of the change — it is
a case that cannot be resumed, discovered days later by whoever is holding the manufacturer's
deadline. A checkpoint of plain JSON survives all four of those edits and fails loudly, at the one
function that rebuilds the object, when it genuinely cannot.

The second is `Money`. It is not a Pydantic model and not a dataclass, so a generic serialiser has
to fall back on something, and everything it could fall back on is worse than a decimal string: a
float reintroduces binary floating point at the one boundary `money.py` exists to keep it out of,
and a pickled object makes the checkpoint unreadable by anything but this exact interpreter. So
money crosses this boundary as `{"amount": "1234.00", "currency": "GBP"}` — the shape the corpus
already uses — and `parse_money` refuses a float on the way back in.

**The rebuild functions are one-way pairs, deliberately.** `claim_from_json` re-runs every validator
`domain.Claim` declares. That looks redundant, since the value was valid when it was written, and it
is the cheap half of a real guarantee: a checkpoint edited by hand, restored from a backup taken
against an older corpus, or written by a version of this code that had one fewer validator would
otherwise flow straight into the gate. The cost is a validation pass per node; the alternative is a
resumed case that is wrong in a way nothing reports.

**Three keys accumulate rather than replace.** `node_history`, `audit` and `tool_calls` are
annotated with `operator.add`, so a node returns only what it added and LangGraph concatenates. That
is not a convenience. `node_history` is what kill condition A is graded from — the sequence of nodes
that actually ran — and a node that re-executed after a resume appears in it twice. A reducer that
replaced the list would let a re-execution overwrite the evidence of itself.

**`CaseAudit` exists because `audit.AuditLog` assigns sequence numbers and this state carries
events across processes.** `verify_submissions` orders events by the numeric tail of their
identifier, so a node that started a fresh log would emit a second event `:000001` and an approval
would sort level with a submission. `CaseAudit` seeds a log from the events already in the
checkpoint, continues the numbering, and hands back only what this node added — which is exactly
what the `operator.add` reducer needs.
"""

from __future__ import annotations

import operator
from collections.abc import Iterable, Mapping
from datetime import date
from typing import Annotated, Any, Final, TypedDict

from warranty_claim_recovery.audit import AuditLog
from warranty_claim_recovery.domain import (
    AuditEvent,
    CaseState,
    Citation,
    Claim,
    ClaimWindow,
    CorrectionProposal,
    GateDecision,
    GateSignals,
    PolicyClause,
    RecoveryComputation,
    RecoveryOutcome,
    RejectionCode,
    Requirement,
    RequirementStatus,
    WarrantyProgram,
)
from warranty_claim_recovery.eligibility import Eligibility, PartCoverage
from warranty_claim_recovery.money import Currency, Money, parse_money

__all__ = [
    "CASE_OPENED",
    "CASE_REFUSED",
    "CASE_WRITTEN_OFF",
    "CaseAudit",
    "GraphState",
    "citation_from_json",
    "citation_json",
    "citations_of",
    "claim_from_json",
    "claim_json",
    "clause_from_json",
    "clause_json",
    "computation_from_json",
    "computation_json",
    "coverage_from_json",
    "coverage_json",
    "decision_from_json",
    "decision_json",
    "eligibility_from_json",
    "eligibility_json",
    "event_from_json",
    "event_json",
    "events_of",
    "initial_state",
    "money_from_json",
    "money_json",
    "program_from_json",
    "program_json",
    "proposal_from_json",
    "proposal_json",
    "requirement_from_json",
    "requirement_json",
    "requirements_of",
    "retrieved_clause_ids",
    "window_from_json",
    "window_json",
]

#: Audit kinds this package writes that `audit.py` does not name. They are free strings there by
#: design — an audit log that rejected an event it did not recognise would lose exactly the evidence
#: an incident needs — and they are constants here so a typo is an import error rather than a gap in
#: whatever reads them later.
CASE_OPENED: Final = "case.opened"
CASE_REFUSED: Final = "case.refused"
CASE_WRITTEN_OFF: Final = "case.written_off"


class GraphState(TypedDict, total=False):
    """The whole of a case, as it sits in PostgreSQL between two processes.

    `total=False` because a node returns a partial update and LangGraph merges it; declaring every
    key required would make each node's return type a lie about what it actually produces.
    `initial_state` fills every key, so a node reading one it did not write reads a value rather
    than a `KeyError`.
    """

    #: The thread identifier the checkpointer is keyed by, carried in the state as well so that a
    #: state read out of the store is self-describing. A case whose identity lives only in the
    #: caller's configuration is a case nobody can attribute after the fact.
    case_id: str
    #: Bumped whenever the decided content changes. An approval is bound to the version it approved
    #: and a submission is valid only against an approval for its own version; see `audit.py`.
    case_version: int
    #: The `CaseState` this case has reached, as its string value.
    node: str
    #: The date the case is reasoned as of. Explicit, stored, and never `date.today()`: a resumed
    #: case must reach the same answer as the killed one, and a wall clock guarantees it will not.
    as_of: str

    claim: dict[str, Any]
    program: dict[str, Any]
    coverage: dict[str, Any]
    evidence: dict[str, str]

    node_history: Annotated[list[str], operator.add]
    audit: Annotated[list[dict[str, Any]], operator.add]
    tool_calls: Annotated[list[dict[str, Any]], operator.add]

    #: Requirement identity and status, before any citation exists. See `nodes.py` for why the two
    #: halves of a `Requirement` are assembled at different nodes.
    requirement_status: list[dict[str, Any]]
    window: dict[str, Any] | None
    eligibility: dict[str, Any] | None
    computation: dict[str, Any] | None
    #: Set when the deterministic core cannot answer at all — today, only a claim denominated in a
    #: currency the programme is not. It is a string rather than a boolean so the reason travels
    #: with the refusal.
    blocked_reason: str | None

    retrieved: list[dict[str, Any]]
    requirements: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    proposal: dict[str, Any] | None
    #: Set when the composer produced something the post-validator refused on the way out — an
    #: ungrounded mention, or a stripped Article 50 disclosure. It is separate from `blocked_reason`
    #: because the two have different readers and different remedies: one is a claim this system
    #: cannot price, the other is a correction this system will not send.
    composer_refusal: str | None

    gate: dict[str, Any] | None
    outcome: str | None
    gate_reason: str | None

    approval: dict[str, Any] | None
    submission: dict[str, Any] | None


# ------------------------------------------------------------------------------------------------
# Money and dates. Two functions each, and neither has a default.
# ------------------------------------------------------------------------------------------------


def money_json(amount: Money) -> dict[str, str]:
    """The wire form: a decimal string and a currency, never a float. See `Money.as_json`."""
    return amount.as_json()


def money_from_json(payload: Mapping[str, Any]) -> Money:
    """Rebuild an amount, refusing a float at the boundary rather than converting one.

    `parse_money` raises on a `float`, and that is the point of routing through it: a float in this
    payload did not come from this system, so accepting it would launder arithmetic nobody can audit
    into a type that looks exact.
    """
    return parse_money(payload["amount"], payload["currency"])


def _date(value: str) -> date:
    return date.fromisoformat(value)


# ------------------------------------------------------------------------------------------------
# The inputs: claim, programme, coverage. Written in both directions because a test builds the
# objects and the corpus supplies the payloads, and neither may be the only supported shape.
# ------------------------------------------------------------------------------------------------


def claim_json(claim: Claim) -> dict[str, Any]:
    prior = claim.previously_recovered
    return {
        "claim_id": claim.claim_id,
        "program_id": claim.program_id,
        "part_number": claim.part_number,
        "serial_number": claim.serial_number,
        "in_service_date": claim.in_service_date.isoformat(),
        "failure_date": claim.failure_date.isoformat(),
        "repair_invoice_date": claim.repair_invoice_date.isoformat(),
        "rejection_code": claim.rejection_code.value,
        "rejected_on": claim.rejected_on.isoformat(),
        "claimed_parts": money_json(claim.claimed_parts),
        "claimed_labour_hours": claim.claimed_labour_hours,
        "claimed_labour_rate": money_json(claim.claimed_labour_rate),
        "previously_recovered": None if prior is None else money_json(prior),
    }


def claim_from_json(payload: Mapping[str, Any]) -> Claim:
    prior = payload.get("previously_recovered")
    return Claim(
        claim_id=str(payload["claim_id"]),
        program_id=str(payload["program_id"]),
        part_number=str(payload["part_number"]),
        serial_number=(
            None if payload.get("serial_number") is None else str(payload["serial_number"])
        ),
        in_service_date=_date(str(payload["in_service_date"])),
        failure_date=_date(str(payload["failure_date"])),
        repair_invoice_date=_date(str(payload["repair_invoice_date"])),
        rejection_code=RejectionCode(payload["rejection_code"]),
        rejected_on=_date(str(payload["rejected_on"])),
        claimed_parts=money_from_json(payload["claimed_parts"]),
        claimed_labour_hours=int(payload["claimed_labour_hours"]),
        claimed_labour_rate=money_from_json(payload["claimed_labour_rate"]),
        previously_recovered=None if prior is None else money_from_json(prior),
    )


def program_json(program: WarrantyProgram) -> dict[str, Any]:
    return {
        "program_id": program.program_id,
        "manufacturer": program.manufacturer,
        "policy_version": program.policy_version,
        "currency": program.currency.value,
        "correction_window_days": program.correction_window_days,
        "warranty_months": program.warranty_months,
        "labour_rate_cap_per_hour": money_json(program.labour_rate_cap_per_hour),
        "deductible": money_json(program.deductible),
        "claim_cap": money_json(program.claim_cap),
    }


def program_from_json(payload: Mapping[str, Any]) -> WarrantyProgram:
    return WarrantyProgram(
        program_id=str(payload["program_id"]),
        manufacturer=str(payload["manufacturer"]),
        policy_version=str(payload["policy_version"]),
        currency=Currency(payload["currency"]),
        correction_window_days=int(payload["correction_window_days"]),
        warranty_months=int(payload["warranty_months"]),
        labour_rate_cap_per_hour=money_from_json(payload["labour_rate_cap_per_hour"]),
        deductible=money_from_json(payload["deductible"]),
        claim_cap=money_from_json(payload["claim_cap"]),
    )


def coverage_json(coverage: PartCoverage) -> dict[str, Any]:
    return {
        "part_number": coverage.part_number,
        "covered": coverage.covered,
        "serial_first": coverage.serial_first,
        "serial_last": coverage.serial_last,
    }


def coverage_from_json(payload: Mapping[str, Any]) -> PartCoverage:
    """Rebuild a coverage record, keeping `None` distinct from a wide range.

    `serial_first` and `serial_last` are `None` when the schedule places no build restriction on the
    part, and `eligibility.PartCoverage` says at length why that is not spelled as a range from zero
    to a large number. Coercing them with `int(payload.get(...) or 0)` would turn "no restriction"
    into "from build zero", which is the same answer today and a different one the moment a schedule
    genuinely starts at zero.
    """
    first = payload.get("serial_first")
    last = payload.get("serial_last")
    return PartCoverage(
        part_number=str(payload["part_number"]),
        covered=bool(payload["covered"]),
        serial_first=None if first is None else int(first),
        serial_last=None if last is None else int(last),
    )


# ------------------------------------------------------------------------------------------------
# The derived facts.
# ------------------------------------------------------------------------------------------------


def window_json(window: ClaimWindow) -> dict[str, Any]:
    return {
        "rejected_on": window.rejected_on.isoformat(),
        "closes_on": window.closes_on.isoformat(),
        "as_of": window.as_of.isoformat(),
        # Derived, and stored anyway. A window that records only its three dates makes every reader
        # re-derive the answer, and kill condition M is graded from records written at the time.
        "is_open": window.is_open,
        "days_remaining": window.days_remaining,
    }


def window_from_json(payload: Mapping[str, Any]) -> ClaimWindow:
    return ClaimWindow(
        rejected_on=_date(str(payload["rejected_on"])),
        closes_on=_date(str(payload["closes_on"])),
        as_of=_date(str(payload["as_of"])),
    )


def eligibility_json(eligibility: Eligibility) -> dict[str, Any]:
    return {
        "within_warranty_period": eligibility.within_warranty_period,
        "part_covered": eligibility.part_covered,
        "serial_in_range": eligibility.serial_in_range,
        "labour_rate_within_cap": eligibility.labour_rate_within_cap,
        "already_recovered": eligibility.already_recovered,
        "uncovered_parts": money_json(eligibility.uncovered_parts),
        "labour_excess": money_json(eligibility.labour_excess),
    }


def eligibility_from_json(payload: Mapping[str, Any]) -> Eligibility:
    return Eligibility(
        within_warranty_period=bool(payload["within_warranty_period"]),
        part_covered=bool(payload["part_covered"]),
        serial_in_range=bool(payload["serial_in_range"]),
        labour_rate_within_cap=bool(payload["labour_rate_within_cap"]),
        already_recovered=bool(payload["already_recovered"]),
        uncovered_parts=money_from_json(payload["uncovered_parts"]),
        labour_excess=money_from_json(payload["labour_excess"]),
    )


#: The eight amounts of a `RecoveryComputation`, in the order `recovery.finalise` writes them.
#: Listed once so that a field added to the model and not to this tuple fails to construct on the
#: way back rather than arriving as a silently absent amount.
_COMPUTATION_AMOUNTS: Final[tuple[str, ...]] = (
    "claimed_total",
    "labour_excess",
    "uncovered_parts",
    "eligible_amount",
    "deductible",
    "capped_amount",
    "already_recovered",
    "recoverable_amount",
)


def computation_json(computation: RecoveryComputation) -> dict[str, Any]:
    payload: dict[str, Any] = {"currency": computation.currency.value}
    for name in _COMPUTATION_AMOUNTS:
        amount: Money = getattr(computation, name)
        payload[name] = money_json(amount)
    return payload


def computation_from_json(payload: Mapping[str, Any]) -> RecoveryComputation:
    amounts = {name: money_from_json(payload[name]) for name in _COMPUTATION_AMOUNTS}
    return RecoveryComputation(currency=Currency(payload["currency"]), **amounts)


def citation_json(citation: Citation) -> dict[str, Any]:
    return {
        "clause_id": citation.clause_id,
        "document_id": citation.document_id,
        "policy_version": citation.policy_version,
        "section": citation.section,
        "quote": citation.quote,
        "start_offset": citation.start_offset,
        "end_offset": citation.end_offset,
    }


def citation_from_json(payload: Mapping[str, Any]) -> Citation:
    return Citation(
        clause_id=str(payload["clause_id"]),
        document_id=str(payload["document_id"]),
        policy_version=str(payload["policy_version"]),
        section=str(payload["section"]),
        quote=str(payload["quote"]),
        start_offset=int(payload["start_offset"]),
        end_offset=int(payload["end_offset"]),
    )


def clause_json(clause: PolicyClause) -> dict[str, Any]:
    return {
        "clause_id": clause.clause_id,
        "program_id": clause.program_id,
        "document_id": clause.document_id,
        "section": clause.section,
        "text": clause.text,
        "start_offset": clause.start_offset,
        "end_offset": clause.end_offset,
        "governs": [code.value for code in clause.governs],
    }


def clause_from_json(payload: Mapping[str, Any]) -> PolicyClause:
    return PolicyClause(
        clause_id=str(payload["clause_id"]),
        program_id=str(payload["program_id"]),
        document_id=str(payload["document_id"]),
        section=str(payload["section"]),
        text=str(payload["text"]),
        start_offset=int(payload["start_offset"]),
        end_offset=int(payload["end_offset"]),
        governs=tuple(RejectionCode(code) for code in payload.get("governs") or ()),
    )


def requirement_json(requirement: Requirement) -> dict[str, Any]:
    citation = requirement.citation
    return {
        "requirement_id": requirement.requirement_id,
        "description": requirement.description,
        "status": requirement.status.value,
        "detail": requirement.detail,
        "citation": None if citation is None else citation_json(citation),
    }


def requirement_from_json(payload: Mapping[str, Any]) -> Requirement:
    citation = payload.get("citation")
    return Requirement(
        requirement_id=str(payload["requirement_id"]),
        description=str(payload["description"]),
        status=RequirementStatus(payload["status"]),
        detail=str(payload["detail"]),
        citation=None if citation is None else citation_from_json(citation),
    )


def decision_json(decision: GateDecision) -> dict[str, Any]:
    return {
        "outcome": decision.outcome.value,
        "reason": decision.reason,
        "signals": decision.signals.model_dump(),
        "computation": computation_json(decision.computation),
        "requirements": [requirement_json(item) for item in decision.requirements],
    }


def decision_from_json(payload: Mapping[str, Any]) -> GateDecision:
    return GateDecision(
        outcome=RecoveryOutcome(payload["outcome"]),
        reason=str(payload["reason"]),
        signals=GateSignals(**payload["signals"]),
        computation=computation_from_json(payload["computation"]),
        requirements=tuple(requirement_from_json(item) for item in payload["requirements"]),
    )


def proposal_json(proposal: CorrectionProposal) -> dict[str, Any]:
    return {
        "claim_id": proposal.claim_id,
        "narrative": proposal.narrative,
        "citations": [citation_json(item) for item in proposal.citations],
        "composer": proposal.composer,
        "ai_disclosure": proposal.ai_disclosure,
    }


def proposal_from_json(payload: Mapping[str, Any]) -> CorrectionProposal:
    return CorrectionProposal(
        claim_id=str(payload["claim_id"]),
        narrative=str(payload["narrative"]),
        citations=tuple(citation_from_json(item) for item in payload["citations"]),
        composer=str(payload["composer"]),
        ai_disclosure=str(payload["ai_disclosure"]),
    )


def event_json(event: AuditEvent) -> dict[str, Any]:
    return {
        "event_id": event.event_id,
        "case_id": event.case_id,
        "case_version": event.case_version,
        "kind": event.kind,
        "node": event.node.value,
        "at": event.at.isoformat(),
        "actor": event.actor,
        "detail": dict(event.detail),
    }


def event_from_json(payload: Mapping[str, Any]) -> AuditEvent:
    return AuditEvent(
        event_id=str(payload["event_id"]),
        case_id=str(payload["case_id"]),
        case_version=int(payload["case_version"]),
        kind=str(payload["kind"]),
        node=CaseState(payload["node"]),
        at=_date(str(payload["at"])),
        actor=str(payload["actor"]),
        detail=dict(payload.get("detail") or {}),
    )


# ------------------------------------------------------------------------------------------------
# The audit seam.
# ------------------------------------------------------------------------------------------------


class CaseAudit:
    """An append-only log seeded from the checkpoint that remembers only what this node added.

    Two properties, and both exist because the events cross a process boundary.

    **The sequence continues rather than restarting.** `verify_submissions` orders events by the
    numeric tail of `event_id` and treats "an approval that precedes a submission" as the thing that
    makes the approval count. A node that started a fresh `AuditLog` would emit its first event as
    `:000001`, which sorts level with the intake event of the same case, and an approval could then
    sort after the submission it authorised. `AuditLog.extend` sets the counter from the events it
    adopts, which is exactly what is needed here.

    **Only the new events are returned.** The `audit` key in the state is reduced with
    `operator.add`, so a node that returned the whole log would double every event it had read. The
    separation also makes a discarded node honest: a node that is killed contributes nothing,
    because nothing it appended was ever returned.
    """

    __slots__ = ("_added", "_log")

    def __init__(self, existing: Iterable[Mapping[str, Any]] = ()) -> None:
        self._log = AuditLog()
        self._log.extend(event_from_json(payload) for payload in existing)
        self._added: list[AuditEvent] = []

    def record(
        self,
        *,
        case_id: str,
        case_version: int,
        kind: str,
        node: CaseState,
        at: date,
        actor: str,
        detail: dict[str, Any] | None = None,
    ) -> AuditEvent:
        event = self._log.append(
            case_id=case_id,
            case_version=case_version,
            kind=kind,
            node=node,
            at=at,
            actor=actor,
            detail=detail,
        )
        self._added.append(event)
        return event

    @property
    def added(self) -> list[dict[str, Any]]:
        """What this node appended, in order, ready for the `operator.add` reducer."""
        return [event_json(event) for event in self._added]

    @property
    def events(self) -> tuple[AuditEvent, ...]:
        """Everything the log holds — the checkpoint's events and this node's, in one sequence.

        `submission.submit` reads this to find the approval that authorises a filing. It is the
        whole log rather than `added` on purpose: the approval was recorded by an earlier node, in
        an earlier process, and a submitter that could only see the current node's events would find
        no approval and refuse every case.
        """
        return tuple(self._log)


def initial_state(
    *,
    case_id: str,
    claim: Mapping[str, Any],
    program: Mapping[str, Any],
    coverage: Mapping[str, Any],
    evidence: Mapping[str, str],
    as_of: date,
) -> GraphState:
    """A case as it arrives, with every key present.

    Every key is filled, including the ones no node has written yet, because a node that reads a key
    another node has not reached should read `None` rather than raise `KeyError`. A missing key and
    an absent value are the same fact here and they should not have two spellings.

    `case_version` starts at one rather than zero. Zero is what an unset integer field looks like,
    and an approval recorded against version zero would be indistinguishable from an approval whose
    version nobody set — which is precisely the distinction kill condition D turns on.
    """
    return GraphState(
        case_id=case_id,
        case_version=1,
        node=CaseState.INTAKE.value,
        as_of=as_of.isoformat(),
        claim=dict(claim),
        program=dict(program),
        coverage=dict(coverage),
        evidence=dict(evidence),
        node_history=[],
        audit=[],
        tool_calls=[],
        requirement_status=[],
        window=None,
        eligibility=None,
        computation=None,
        blocked_reason=None,
        retrieved=[],
        requirements=[],
        citations=[],
        proposal=None,
        composer_refusal=None,
        gate=None,
        outcome=None,
        gate_reason=None,
        approval=None,
        submission=None,
    )


def requirements_of(state: GraphState) -> tuple[Requirement, ...]:
    """The assembled requirements, rebuilt from the checkpoint in the matrix's order."""
    return tuple(requirement_from_json(item) for item in state.get("requirements") or ())


def citations_of(state: GraphState) -> tuple[Citation, ...]:
    return tuple(citation_from_json(item) for item in state.get("citations") or ())


def events_of(state: GraphState) -> tuple[AuditEvent, ...]:
    return tuple(event_from_json(item) for item in state.get("audit") or ())


def retrieved_clause_ids(state: GraphState) -> tuple[str, ...]:
    """Every clause the retrieval stage returned, cited or not.

    This is the vocabulary the composer's grounding check is run against. It has to be the whole
    retrieved set rather than the cited set, because the failure being looked for is a narrative
    that mentions a clause the case saw and did not cite — a clause the gate never approved as
    evidence. Checking against the cited set alone would be checking a set against itself.
    """
    return tuple(str(item["clause_id"]) for item in state.get("retrieved") or ())
