"""The work of a case, one function per `domain.CaseState`, none of which knows it can be killed.

`CLAUDE.md` §2.1 fixes the order and the order is the argument. Everything deterministic happens
before anything is written, and the human is asked last, after the system has finished having
opinions.

### Two orderings that look wrong and are not

**The gate decision is taken in the `compose` node, and the `gate` node re-takes it.** The node
order puts `compose` before `gate`; `CLAUDE.md` §3 rule 3 says the gate is computed before any text
is composed. Both are honoured by deciding first and publishing second: `compose` calls
`gate.decide` before a character exists, composes only from the evidence that decision approved, and
`gate` recomputes the decision from the same checkpointed inputs and **refuses the case if the two
disagree**. So the `gate` node is a check as well as a publication. If any text the composer
produced had been able to move the outcome, the two computations would differ and the case would
stop at `REVIEW` rather than quietly filing on the newer answer. The rejected alternative — have
`gate` simply read what `compose` wrote — passes the same tests and proves nothing, because the
property being claimed is precisely that the second computation is unnecessary.

**A requirement's status is decided at `requirements` and its citation is attached at `retrieval`.**
`domain.Requirement` refuses to be constructed `SATISFIED` without a citation, and a citation cannot
exist before the search. So the `requirements` node publishes statuses — a fact about the evidence
bundle, which is known at intake — and the `retrieval` node assembles the `Requirement` objects once
both halves exist. Deciding the status later, from the retrieved text, would be the model-shaped
mistake `requirements.py` was written to prevent: the evidence a claim carries is not a question the
policy corpus answers.

### What happens when a requirement is evidenced and nothing can be cited for it

It becomes `CONFLICTING`, which sends the case to `REVIEW` and a person. `MISSING` was rejected: the
evidence *is* there, and `MISSING` sends a technician to fetch a document that is already in the
bundle — a day of the correction window spent on an errand that cannot succeed. `SATISFIED` is not
available, because kill condition H allows zero requirements reported satisfied with nothing behind
them and `domain.Requirement` refuses to construct one. With the shipped index this branch does not
fire; it exists because the alternative to naming it is discovering it on an empty database.

### The kill switch

`CaseDependencies.kill_switch` is a seam, and it is the only concession in this module to the fact
that these nodes are killed on purpose. It is called at each node's entry and after each tool
completes, with the node and the number of tool calls that have finished, and in ordinary operation
it is `None` and costs one attribute read. `scripts/durability_evidence.py` installs one that calls
`os._exit`, which is a real process death.

A timer was rejected. A kill placed by a timer lands somewhere nobody can name, so the evidence
could not say which node was interrupted or whether a tool had completed — and kill conditions A and
B are both statements about exactly that. A kill that cannot be placed is a kill that cannot be
graded.

### Secrets

The approver's token is compared and then discarded. Neither `state["approval"]` nor the ledger row
the approval node writes carries it — both hold the decision, the actor, the version approved and a
note, and nothing else. Both are written to PostgreSQL, read by the console and printed into
evidence, and a credential that travels with the record it authenticates ends up in all three.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Final, Protocol, cast

from langgraph.types import interrupt

from warranty_claim_recovery import gate as gate_rules
from warranty_claim_recovery import recovery
from warranty_claim_recovery.audit import APPROVAL_GRANTED, APPROVAL_REQUESTED
from warranty_claim_recovery.composer import (
    Composer,
    DisclosureMissingError,
    ExtractiveComposer,
    UngroundedCorrectionError,
    correction_request,
    validated,
)
from warranty_claim_recovery.deadline import claim_window
from warranty_claim_recovery.domain import (
    CaseState,
    Citation,
    Claim,
    GateDecision,
    RecoveryOutcome,
    RejectionCode,
    Requirement,
    RequirementStatus,
    WarrantyProgram,
)
from warranty_claim_recovery.eligibility import assess
from warranty_claim_recovery.graph.state import (
    CASE_OPENED,
    CASE_REFUSED,
    CASE_WRITTEN_OFF,
    CaseAudit,
    GraphState,
    citation_from_json,
    claim_from_json,
    computation_from_json,
    computation_json,
    coverage_from_json,
    decision_from_json,
    decision_json,
    eligibility_from_json,
    eligibility_json,
    program_from_json,
    proposal_json,
    requirement_json,
    requirements_of,
    retrieved_clause_ids,
    window_from_json,
    window_json,
)
from warranty_claim_recovery.graph.tools import ToolLedger
from warranty_claim_recovery.money import CurrencyMismatchError
from warranty_claim_recovery.queue.idempotency import SubmissionGuard
from warranty_claim_recovery.requirements import required_for
from warranty_claim_recovery.retrieval.citations import citation_failure
from warranty_claim_recovery.submission import (
    ManufacturerPortal,
    SubmissionOutcome,
    SubmissionRefusal,
    outcome_from_json,
    outcome_json,
    record_outcome,
)
from warranty_claim_recovery.submission import submit as file_correction

__all__ = [
    "APPROVAL_NOT_REQUIRED",
    "APPROVAL_REFUSED",
    "AUTHORISING_OUTCOMES",
    "NODE_SEQUENCE",
    "SYSTEM_ACTOR",
    "TOOL_RECORD_DECISION",
    "TOOL_RETRIEVE_CLAUSES",
    "TOOL_SUBMIT_CORRECTION",
    "TOOL_VERIFY_CITATIONS",
    "CaseDependencies",
    "CaseNodes",
    "EvidenceSource",
    "KillSwitch",
    "route_after_approval",
]

#: The actor recorded against everything this system does on its own. A single value rather than one
#: per node, because the audit's `node` field already says where it happened, and a second
#: spelling of the same fact is a second thing to keep in step.
SYSTEM_ACTOR: Final = "case-machine"

#: Node name to the `CaseState` it reaches. The names are LangGraph's and the states are the
#: domain's, and the pairing is declared once, here, so "resumes at the same node" has one
#: vocabulary. `domain.CaseState` says why a state enum that drifts from the node names makes the
#: durability claim uncheckable.
NODE_SEQUENCE: Final[tuple[tuple[str, CaseState], ...]] = (
    ("intake", CaseState.INTAKE),
    ("requirements", CaseState.REQUIREMENTS),
    ("deadline", CaseState.DEADLINE),
    ("eligibility", CaseState.ELIGIBILITY),
    ("retrieval", CaseState.RETRIEVAL),
    ("compose", CaseState.COMPOSE),
    ("gate", CaseState.GATE),
    ("approval", CaseState.AWAITING_APPROVAL),
    ("submit", CaseState.SUBMITTED),
    ("write_off", CaseState.WRITTEN_OFF),
)

#: The two outcomes that let money leave. Imported from `domain` rather than spelled as strings, and
#: named here once so the approval node, the router and the submit node cannot come to hold three
#: slightly different opinions about what "authorised" means.
AUTHORISING_OUTCOMES: Final[frozenset[str]] = frozenset(
    {RecoveryOutcome.RECOVERABLE.value, RecoveryOutcome.PARTIALLY_RECOVERABLE.value}
)

#: Audit kinds this module writes that `audit.py` does not name. See `graph/state.py` for why free
#: strings are correct there and constants are correct here.
APPROVAL_NOT_REQUIRED: Final = "approval.not_required"
APPROVAL_REFUSED: Final = "approval.refused"

#: Tool names. They are half of an invocation identity, so renaming one makes every in-flight case
#: re-execute the tool it had already completed. Constants, so that is a deliberate edit rather than
#: a typo.
TOOL_RETRIEVE_CLAUSES: Final = "retrieve_clauses"
TOOL_VERIFY_CITATIONS: Final = "verify_citations"
TOOL_SUBMIT_CORRECTION: Final = "submit_correction"
#: Asking a person is the most expensive call this system makes and the one that must never be made
#: twice, so it goes through the ledger like every other. See `approval` for the measurement that
#: forced it.
TOOL_RECORD_DECISION: Final = "record_decision"

_APPROVED: Final = "APPROVED"


class EvidenceSource(Protocol):
    """Where clauses and document text come from.

    A protocol so the graph can be exercised without a 285MB ONNX session and a seeded index, and
    so the evidence run can hand it the real pgvector retriever. The two methods are together
    because they answer one question — *what does the policy say, and is it really there* — and a
    citation verified against a document fetched from somewhere other than the corpus the clause
    came from is a citation verified against the wrong thing.
    """

    def clauses(
        self,
        *,
        program_id: str,
        policy_version: str,
        rejection_code: RejectionCode,
        query: str,
        k: int,
    ) -> list[dict[str, Any]]: ...

    def document_text(self, document_id: str) -> str | None: ...


class KillSwitch(Protocol):
    """A seam the durability evidence uses to abandon the process at a point it can name."""

    def __call__(self, node: CaseState, completed_tool_calls: int) -> None: ...


@dataclass(frozen=True)
class CaseDependencies:
    """Everything the nodes reach outside themselves, in one frozen object.

    Frozen because a graph is compiled once and run by many workers; a mutable dependency bundle
    would let one case's configuration become another's. The composer defaults to the shipped
    extractive arm so that the ordinary construction is the one ADR-001 §3 describes, and the
    abstractive port has to be passed deliberately.
    """

    ledger: ToolLedger
    evidence: EvidenceSource
    guard: SubmissionGuard
    portal: ManufacturerPortal
    #: No default, and `None` means no approval is possible. `config.Settings.approver_token` has no
    #: default for the same reason: a deployment that forgot it must approve nothing rather than
    #: approve everything.
    approver_token: str | None
    composer: Composer = field(default_factory=ExtractiveComposer)
    retrieval_k: int = 5
    kill_switch: KillSwitch | None = None


def _as_date(value: str) -> date:
    return date.fromisoformat(value)


def _evidence_status(raw: str | None) -> tuple[RequirementStatus, str]:
    """Turn one entry of the evidence bundle into a requirement status and a sentence.

    An entry this system does not recognise becomes `CONFLICTING` rather than `MISSING`. The bundle
    says *something* about the artefact, and this system cannot say what: that is a decision for a
    person, and reporting it as absent would send a technician to fetch a document that may already
    be there in a state nobody has read.
    """
    if raw is None:
        return (
            RequirementStatus.MISSING,
            "the evidence bundle carries no entry at all for this artefact",
        )
    if raw == "PRESENT":
        return RequirementStatus.SATISFIED, "the evidence bundle carries this artefact"
    if raw == "ABSENT":
        return RequirementStatus.MISSING, "the evidence bundle records this artefact as absent"
    if raw == "CONFLICTING":
        return (
            RequirementStatus.CONFLICTING,
            "the evidence bundle records this artefact as contradicting another source",
        )
    return (
        RequirementStatus.CONFLICTING,
        f"the evidence bundle records this artefact as {raw!r}, which this system does not "
        f"recognise; a person has to read it",
    )


def _query_for(claim: Claim, program: WarrantyProgram) -> str:
    """The question put to the retriever, assembled the same way for every case.

    Fixed phrasing rather than a per-case sentence. `retrieval.pipeline.compose_query` makes the
    same argument about the rejection code: a query assembled differently at three call sites is
    three queries, and the recall measured for one of them is not the recall the system has. It is
    also a tuning knob, and ADR-001 §7 forbids turning one after the hold-out has been scored.
    """
    return (
        f"{program.manufacturer} warranty policy {program.policy_version}: what evidence does a "
        f"corrected claim need for part {claim.part_number} rejected as "
        f"{claim.rejection_code.value}?"
    )


def route_after_approval(state: GraphState) -> str:
    """Submit only with a granted approval and an authorising outcome; otherwise write the case off.

    Both conditions, not either. The outcome alone would file a case nobody approved; the approval
    alone would file a case the gate refused, which a person cannot authorise because the authority
    to authorise is not the authority to overrule the arithmetic.
    """
    approval = state.get("approval")
    granted = bool(approval and approval.get("granted"))
    return "submit" if granted and state.get("outcome") in AUTHORISING_OUTCOMES else "write_off"


class CaseNodes:
    """The ten node functions, bound to one set of dependencies."""

    __slots__ = ("_deps",)

    def __init__(self, deps: CaseDependencies) -> None:
        self._deps = deps

    def as_mapping(self) -> dict[str, Callable[[GraphState], dict[str, Any]]]:
        """Node name to function, in the declared order, checked against `NODE_SEQUENCE`.

        Built from `NODE_SEQUENCE` rather than written out again, so a node added to the pipeline
        without a function raises here — at import of the machine — rather than at the first case
        that reaches it.
        """
        functions: dict[str, Callable[[GraphState], dict[str, Any]]] = {
            "intake": self.intake,
            "requirements": self.requirements,
            "deadline": self.deadline,
            "eligibility": self.eligibility,
            "retrieval": self.retrieval,
            "compose": self.compose,
            "gate": self.gate,
            "approval": self.approval,
            "submit": self.submit,
            "write_off": self.write_off,
        }
        absent = [name for name, _ in NODE_SEQUENCE if name not in functions]
        if absent:
            raise RuntimeError(
                f"the pipeline declares nodes {absent} with no function behind them. A node that "
                f"cannot run is a case that stops at it with nothing to say why."
            )
        return functions

    # -- the seam ---------------------------------------------------------------------------

    def _reached(self, node: CaseState, completed_tool_calls: int) -> None:
        switch = self._deps.kill_switch
        if switch is not None:
            switch(node, completed_tool_calls)

    @staticmethod
    def _open(state: GraphState) -> tuple[CaseAudit, str, int, date]:
        """The four things every node needs before it does anything: the log, the case and the date.

        The audit log is seeded from the checkpoint on every node, which looks wasteful and is the
        mechanism: a node that started a fresh log would restart the sequence numbering and an
        approval could then sort level with the submission it authorised. See `state.CaseAudit`.
        """
        return (
            CaseAudit(state.get("audit") or ()),
            state["case_id"],
            state["case_version"],
            _as_date(state["as_of"]),
        )

    # -- the nodes --------------------------------------------------------------------------

    def intake(self, state: GraphState) -> dict[str, Any]:
        """Rebuild the case's inputs and refuse a case that was assembled wrongly.

        The three cross-checks raise rather than record a finding, which is the opposite of what
        every later node does. They are wiring faults — a claim assessed against another
        programme's policy, a coverage record for another part — and the plausible finding they
        would otherwise produce ("not covered") is indistinguishable downstream from a real
        exclusion. `eligibility.assess` makes the same argument about the same mismatch.
        """
        audit, case_id, case_version, at = self._open(state)
        self._reached(CaseState.INTAKE, 0)

        claim = claim_from_json(state["claim"])
        program = program_from_json(state["program"])
        coverage = coverage_from_json(state["coverage"])
        if claim.program_id != program.program_id:
            raise ValueError(
                f"{claim.claim_id} belongs to programme {claim.program_id} and was assembled "
                f"against {program.program_id}. Adjudicating it would apply another "
                f"manufacturer's deadline, cap and deductible to this claim."
            )
        if coverage.part_number != claim.part_number:
            raise ValueError(
                f"{claim.claim_id} claims part {claim.part_number} and carries a coverage record "
                f"for {coverage.part_number}; the finding would be about the wrong part."
            )

        audit.record(
            case_id=case_id,
            case_version=case_version,
            kind=CASE_OPENED,
            node=CaseState.INTAKE,
            at=at,
            actor=SYSTEM_ACTOR,
            detail={
                "claim_id": claim.claim_id,
                "program_id": program.program_id,
                "policy_version": program.policy_version,
                "rejection_code": claim.rejection_code.value,
                "recovery_identity": claim.recovery_identity,
            },
        )
        return {
            "node": CaseState.INTAKE.value,
            "node_history": ["intake"],
            "audit": audit.added,
        }

    def requirements(self, state: GraphState) -> dict[str, Any]:
        """What the rejection code demands, set against what the evidence bundle carries.

        The matrix is consulted by code and never by case. `requirements.py` argues why at length;
        the short version is that a per-case checklist makes the recovery rate a property of who
        was on shift.
        """
        audit, case_id, case_version, at = self._open(state)
        self._reached(CaseState.REQUIREMENTS, 0)

        claim = claim_from_json(state["claim"])
        evidence = state["evidence"]
        statuses: list[dict[str, Any]] = []
        for spec in required_for(claim.rejection_code):
            status, detail = _evidence_status(evidence.get(spec.evidence_key))
            statuses.append(
                {
                    "requirement_id": spec.requirement_id,
                    "description": spec.description,
                    "evidence_key": spec.evidence_key,
                    "status": status.value,
                    "detail": f"{detail} ({spec.evidence_key})",
                }
            )

        audit.record(
            case_id=case_id,
            case_version=case_version,
            kind="requirements.assessed",
            node=CaseState.REQUIREMENTS,
            at=at,
            actor=SYSTEM_ACTOR,
            detail={
                "requirements_total": len(statuses),
                "satisfied": sum(
                    1 for item in statuses if item["status"] == RequirementStatus.SATISFIED.value
                ),
            },
        )
        return {
            "node": CaseState.REQUIREMENTS.value,
            "node_history": ["requirements"],
            "requirement_status": statuses,
            "audit": audit.added,
        }

    def deadline(self, state: GraphState) -> dict[str, Any]:
        """The correction window, computed as of the date the case carries and never as of today."""
        audit, case_id, case_version, at = self._open(state)
        self._reached(CaseState.DEADLINE, 0)

        claim = claim_from_json(state["claim"])
        program = program_from_json(state["program"])
        window = claim_window(claim, program, at)

        audit.record(
            case_id=case_id,
            case_version=case_version,
            kind="deadline.computed",
            node=CaseState.DEADLINE,
            at=at,
            actor=SYSTEM_ACTOR,
            detail={
                "closes_on": window.closes_on.isoformat(),
                "is_open": window.is_open,
                "days_remaining": window.days_remaining,
            },
        )
        return {
            "node": CaseState.DEADLINE.value,
            "node_history": ["deadline"],
            "window": window_json(window),
            "audit": audit.added,
        }

    def eligibility(self, state: GraphState) -> dict[str, Any]:
        """What the policy covered, and what the money comes to.

        Both here, because the money is the second half of the same question and the pipeline has
        no separate node for it. A cross-currency claim is caught rather than propagated: there is
        no answer this system is entitled to give, so the case carries a `blocked_reason` forward
        and the gate turns it into `REVIEW` — a person decides — instead of the case crashing on a
        subtraction three nodes later.
        """
        audit, case_id, case_version, at = self._open(state)
        self._reached(CaseState.ELIGIBILITY, 0)

        claim = claim_from_json(state["claim"])
        program = program_from_json(state["program"])
        coverage = coverage_from_json(state["coverage"])

        update: dict[str, Any] = {
            "node": CaseState.ELIGIBILITY.value,
            "node_history": ["eligibility"],
        }
        try:
            assessed = assess(claim, program, coverage)
        except CurrencyMismatchError as error:
            audit.record(
                case_id=case_id,
                case_version=case_version,
                kind=CASE_REFUSED,
                node=CaseState.ELIGIBILITY,
                at=at,
                actor=SYSTEM_ACTOR,
                detail={"reason": "currency_mismatch"},
            )
            update["blocked_reason"] = str(error)
            update["audit"] = audit.added
            return update

        computation = recovery.compute(claim, program, assessed)
        audit.record(
            case_id=case_id,
            case_version=case_version,
            kind="eligibility.assessed",
            node=CaseState.ELIGIBILITY,
            at=at,
            actor=SYSTEM_ACTOR,
            detail={
                "part_covered": assessed.part_covered,
                "within_warranty_period": assessed.within_warranty_period,
                "recoverable_amount": str(computation.recoverable_amount.amount),
                "currency": computation.currency.value,
            },
        )
        update["eligibility"] = eligibility_json(assessed)
        update["computation"] = computation_json(computation)
        update["audit"] = audit.added
        return update

    def retrieval(self, state: GraphState) -> dict[str, Any]:
        """The governing clauses, their citations, and the requirements assembled from both.

        Two tool invocations, both through the ledger, and the kill switch is offered the node
        between them. That is the shape kill condition B is graded on: a resume that re-enters this
        node must replay the first call from its record and must not search again.
        """
        audit, case_id, case_version, at = self._open(state)
        self._reached(CaseState.RETRIEVAL, 0)

        claim = claim_from_json(state["claim"])
        program = program_from_json(state["program"])

        found = self._deps.ledger.invoke(
            case_id=case_id,
            case_version=case_version,
            node=CaseState.RETRIEVAL,
            tool=TOOL_RETRIEVE_CLAUSES,
            call_index=0,
            audit=audit,
            at=at,
            actor=SYSTEM_ACTOR,
            run=lambda: {
                "clauses": self._deps.evidence.clauses(
                    program_id=program.program_id,
                    policy_version=program.policy_version,
                    rejection_code=claim.rejection_code,
                    query=_query_for(claim, program),
                    k=self._deps.retrieval_k,
                )
            },
        )
        self._reached(CaseState.RETRIEVAL, 1)

        clauses: list[dict[str, Any]] = list(found.payload["clauses"])
        verified = self._deps.ledger.invoke(
            case_id=case_id,
            case_version=case_version,
            node=CaseState.RETRIEVAL,
            tool=TOOL_VERIFY_CITATIONS,
            call_index=1,
            audit=audit,
            at=at,
            actor=SYSTEM_ACTOR,
            run=lambda: self._verify(clauses, program.policy_version),
        )
        self._reached(CaseState.RETRIEVAL, 2)

        entries: list[dict[str, Any]] = list(verified.payload["citations"])
        governing = _governing_entry(entries, claim.rejection_code)
        assembled = [
            requirement_json(_requirement_from(item, governing))
            for item in state["requirement_status"]
        ]

        audit.record(
            case_id=case_id,
            case_version=case_version,
            kind="retrieval.completed",
            node=CaseState.RETRIEVAL,
            at=at,
            actor=SYSTEM_ACTOR,
            detail={
                "clauses": len(clauses),
                "citations": len(entries),
                "rejected_citations": len(verified.payload.get("rejected") or []),
                "governing_clause_id": None if governing is None else governing["clause_id"],
            },
        )
        return {
            "node": CaseState.RETRIEVAL.value,
            "node_history": ["retrieval"],
            "retrieved": clauses,
            "citations": entries,
            "requirements": assembled,
            "tool_calls": [
                {"invocation_id": found.invocation_id, "replayed": found.replayed},
                {"invocation_id": verified.invocation_id, "replayed": verified.replayed},
            ],
            "audit": audit.added,
        }

    def _verify(self, clauses: Sequence[Mapping[str, Any]], policy_version: str) -> dict[str, Any]:
        """Check every retrieved clause against the document it names, and drop the ones that fail.

        Dropped rather than repaired, and dropped here rather than counted later. A clause whose
        offsets do not address its own text cannot produce a faithful citation however good the
        retrieval was, so letting it through would put kill condition I's zero at the mercy of the
        corpus. `store.loader` refuses the same clause at load time for the same reason; this is the
        cheap half of the guarantee that the row has not moved since.
        """
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for rank, record in enumerate(clauses, start=1):
            citation = Citation(
                clause_id=str(record["clause_id"]),
                document_id=str(record["document_id"]),
                policy_version=policy_version,
                section=str(record["section"]),
                quote=str(record["text"]),
                start_offset=int(record["start_offset"]),
                end_offset=int(record["end_offset"]),
            )
            body = self._deps.evidence.document_text(citation.document_id)
            if body is None:
                failure: str | None = (
                    f"{citation.clause_id}: {citation.document_id} is not in the document corpus, "
                    f"so the quote cannot be checked against anything"
                )
            else:
                failure = citation_failure(citation, body)
            if failure is not None:
                rejected.append({"clause_id": citation.clause_id, "reason": failure})
                continue
            accepted.append(
                {
                    "clause_id": citation.clause_id,
                    "document_id": citation.document_id,
                    "policy_version": citation.policy_version,
                    "section": citation.section,
                    "quote": citation.quote,
                    "start_offset": citation.start_offset,
                    "end_offset": citation.end_offset,
                    "rank": rank,
                    "governs": [str(code) for code in record.get("governs") or ()],
                }
            )
        return {"citations": accepted, "rejected": rejected}

    def compose(self, state: GraphState) -> dict[str, Any]:
        """Decide first, then write — and write nothing for a case that is not going anywhere.

        The gate decision is taken here, before a character of the correction exists, and the module
        docstring argues why that is what honours both the node order and `CLAUDE.md` §3 rule 3. A
        composer that produces an ungrounded correction does not stop the case with an exception: it
        records a refusal, and the `gate` node turns that into `REVIEW`. An exception would strand
        the case at this node with the correction window still running.
        """
        audit, case_id, case_version, at = self._open(state)
        self._reached(CaseState.COMPOSE, 0)

        update: dict[str, Any] = {"node": CaseState.COMPOSE.value, "node_history": ["compose"]}
        if state.get("blocked_reason"):
            audit.record(
                case_id=case_id,
                case_version=case_version,
                kind="compose.skipped",
                node=CaseState.COMPOSE,
                at=at,
                actor=SYSTEM_ACTOR,
                detail={"reason": state["blocked_reason"]},
            )
            update["audit"] = audit.added
            return update

        decision = self._decide(state)
        update["gate"] = decision_json(decision)

        if not decision.authorises_recovery:
            audit.record(
                case_id=case_id,
                case_version=case_version,
                kind="compose.not_required",
                node=CaseState.COMPOSE,
                at=at,
                actor=SYSTEM_ACTOR,
                detail={"outcome": decision.outcome.value},
            )
            update["audit"] = audit.added
            return update

        request = correction_request(
            claim=claim_from_json(state["claim"]),
            program=program_from_json(state["program"]),
            window=window_from_json(_require(state, "window")),
            decision=decision,
        )
        try:
            proposal = validated(
                self._deps.composer.propose(request),
                vocabulary=retrieved_clause_ids(state),
            )
        except (UngroundedCorrectionError, DisclosureMissingError) as error:
            audit.record(
                case_id=case_id,
                case_version=case_version,
                kind="compose.refused",
                node=CaseState.COMPOSE,
                at=at,
                actor=SYSTEM_ACTOR,
                detail={"reason": str(error), "composer": self._deps.composer.name},
            )
            update["composer_refusal"] = str(error)
            update["audit"] = audit.added
            return update

        audit.record(
            case_id=case_id,
            case_version=case_version,
            kind="compose.completed",
            node=CaseState.COMPOSE,
            at=at,
            actor=SYSTEM_ACTOR,
            detail={"composer": proposal.composer, "citations": len(proposal.citations)},
        )
        update["proposal"] = proposal_json(proposal)
        update["audit"] = audit.added
        return update

    def gate(self, state: GraphState) -> dict[str, Any]:
        """Publish the decision, and refuse the case if recomputing it gives a different answer.

        The recomputation is the point. It runs after the correction has been written, over the
        same checkpointed deterministic inputs, and a disagreement means something between the two
        moved the outcome — which is exactly the failure `CLAUDE.md` §3 rule 3 forbids. It stops the
        case at `REVIEW` rather than raising, because the honest report to a handler is "this needs
        a person", not a traceback.
        """
        audit, case_id, case_version, at = self._open(state)
        self._reached(CaseState.GATE, 0)

        outcome, reason = self._publish(state)
        audit.record(
            case_id=case_id,
            case_version=case_version,
            kind="gate.decided",
            node=CaseState.GATE,
            at=at,
            actor=SYSTEM_ACTOR,
            detail={"outcome": outcome, "reason": reason},
        )
        return {
            "node": CaseState.GATE.value,
            "node_history": ["gate"],
            "outcome": outcome,
            "gate_reason": reason,
            "audit": audit.added,
        }

    def _publish(self, state: GraphState) -> tuple[str, str]:
        blocked = state.get("blocked_reason")
        if blocked:
            return RecoveryOutcome.REVIEW.value, blocked

        recorded = decision_from_json(_require(state, "gate"))
        recomputed = self._decide(state)
        if recomputed != recorded:
            return (
                RecoveryOutcome.REVIEW.value,
                f"the gate decided {recorded.outcome.value} before the correction was composed and "
                f"{recomputed.outcome.value} after it. Something between the two moved a "
                f"deterministic input, and no resubmission goes out on an outcome that changed "
                f"while text was being written.",
            )
        refusal = state.get("composer_refusal")
        if refusal:
            return (
                RecoveryOutcome.REVIEW.value,
                f"the gate authorised this case and the correction was refused on the way out: "
                f"{refusal}",
            )
        return recomputed.outcome.value, recomputed.reason

    def _decide(self, state: GraphState) -> GateDecision:
        """`gate.decide` over the checkpointed facts. One call site, used by two nodes."""
        return gate_rules.decide(
            claim_from_json(state["claim"]),
            program_from_json(state["program"]),
            window_from_json(_require(state, "window")),
            eligibility_from_json(_require(state, "eligibility")),
            computation_from_json(_require(state, "computation")),
            requirements_of(state),
        )

    def approval(self, state: GraphState) -> dict[str, Any]:
        """The interrupt. The graph stops here, and a different process answers it.

        **The interrupt is inside a tool invocation, and that is the whole of kill condition C.**
        The obvious shape — call `interrupt`, read the answer, return it in the node's update — was
        built first and then measured, and it loses the decision. LangGraph writes a resumed task's
        value nowhere durable before the task runs: it is carried in memory for that one run, and
        `scripts/durability_evidence.py` demonstrates the consequence directly. Kill the worker in
        the instant between a person answering and the node returning, and the resumed case raises
        the interrupt again and asks them a second time. Nothing in the checkpoint records that they
        ever answered. The earlier version of this docstring asserted the opposite; it was wrong,
        and it is recorded here as wrong because the next reader will be tempted by the same shape.
        A checkpoint alone cannot hold a decision the node that received it never returned.
        So the person's answer is written down by the same ledger that records every other
        side-effecting call — committed in its own transaction, outside the graph's — and the ledger
        replays it on the resume. Asking a warranty handler to adjudicate a case is the most
        expensive call this system makes and the one it must never make twice, which is exactly what
        `tools.ToolLedger` exists for; giving human approval a second, parallel durability mechanism
        would have been a second thing to keep correct across a kill.

        The kill switch is offered the node **after** the ledger has recorded the decision and
        before anything has been done with it. That is the moment the evidence run needs: a decision
        taken, a worker dead, and a resume that must neither lose it nor ask again.

        The token is compared inside `_ask` and never reaches the recorded payload. The ledger's
        rows are read by the console and copied into evidence, and a credential that travels with
        the record it authenticates ends up in both.
        """
        audit, case_id, case_version, at = self._open(state)
        if state.get("outcome") not in AUTHORISING_OUTCOMES:
            self._reached(CaseState.AWAITING_APPROVAL, 0)
            audit.record(
                case_id=case_id,
                case_version=case_version,
                kind=APPROVAL_NOT_REQUIRED,
                node=CaseState.AWAITING_APPROVAL,
                at=at,
                actor=SYSTEM_ACTOR,
                detail={"outcome": state.get("outcome")},
            )
            return {
                "node": CaseState.AWAITING_APPROVAL.value,
                "node_history": ["approval"],
                "approval": None,
                "audit": audit.added,
            }

        decided = self._deps.ledger.invoke(
            case_id=case_id,
            case_version=case_version,
            node=CaseState.AWAITING_APPROVAL,
            tool=TOOL_RECORD_DECISION,
            call_index=0,
            audit=audit,
            at=at,
            actor=SYSTEM_ACTOR,
            run=lambda: self._ask(state, case_version),
        )
        self._reached(CaseState.AWAITING_APPROVAL, 1)

        granted = bool(decided.payload["granted"])
        actor = str(decided.payload["actor"])
        note = str(decided.payload["note"])
        raw_refusal = decided.payload["refusal"]
        refusal = None if raw_refusal is None else str(raw_refusal)

        audit.record(
            case_id=case_id,
            case_version=case_version,
            kind=APPROVAL_REQUESTED,
            node=CaseState.AWAITING_APPROVAL,
            at=at,
            actor=SYSTEM_ACTOR,
            detail={"outcome": state.get("outcome")},
        )
        audit.record(
            case_id=case_id,
            case_version=case_version,
            kind=APPROVAL_GRANTED if granted else APPROVAL_REFUSED,
            node=CaseState.AWAITING_APPROVAL,
            at=at,
            actor=actor,
            detail={"note": note} if granted else {"note": note, "refusal": refusal},
        )
        return {
            "node": CaseState.AWAITING_APPROVAL.value,
            "node_history": ["approval"],
            # The token is compared and dropped. See the module docstring: this record is written to
            # PostgreSQL, rendered by the console and copied into evidence.
            "approval": {
                "granted": granted,
                "actor": actor,
                "case_version": case_version,
                "note": note,
                "refusal": refusal,
            },
            "tool_calls": [
                {"invocation_id": decided.invocation_id, "replayed": decided.replayed},
            ],
            "audit": audit.added,
        }

    def _ask(self, state: GraphState, case_version: int) -> dict[str, Any]:
        """Stop the graph, wait for a person, and hand back their decision without their token.

        This is the body of the `record_decision` invocation, so everything it returns is written to
        the ledger and replayed verbatim on a resume. That is why the token is compared here and
        discarded here rather than carried out to the caller: a payload is durable, and the one
        thing that must not become durable is the credential that authenticated it.

        `interrupt` raising out of this function leaves the invocation recorded as started and not
        completed, which `tools.py` already defines as a call that was interrupted in flight and
        must run again. That is the correct reading: the graph stopped before a person had answered,
        so there is no decision to replay and the next worker has to ask.
        """
        answer = interrupt(_approval_request(state))
        granted, actor, note, refusal = self._read_answer(answer, case_version)
        return {"granted": granted, "actor": actor, "note": note, "refusal": refusal}

    def _read_answer(self, answer: object, case_version: int) -> tuple[bool, str, str, str | None]:
        """Read the human's decision, failing closed on every doubt.

        Four ways to be refused and one way to be granted. The token check comes first because
        `config.Settings.approver_token` has no default: a deployment that did not configure one can
        approve nothing, and an approval that arrived without a token against an unconfigured system
        would otherwise be granted by two absences cancelling out.
        """
        expected = self._deps.approver_token
        if not isinstance(answer, Mapping):
            return False, "unknown", "the approval answer was not a mapping", "MALFORMED"
        actor = str(answer.get("actor") or "unknown")
        note = str(answer.get("note") or "")
        if not expected:
            return False, actor, note, "NO_APPROVER_TOKEN_CONFIGURED"
        if str(answer.get("token") or "") != expected:
            return False, actor, note, "TOKEN_REJECTED"
        approved_version = answer.get("case_version")
        if approved_version is None or int(approved_version) != case_version:
            return False, actor, note, "VERSION_MISMATCH"
        if str(answer.get("decision") or "") != _APPROVED:
            return False, actor, note, "DECLINED"
        return True, actor, note, None

    def submit(self, state: GraphState) -> dict[str, Any]:
        """File the correction, once, through the ledger and the Redis guard.

        The audit event is written by this node from the tool's result, on a fresh run and on a
        replay alike. That asymmetry with the ledger is the fix for a real defect: the ledger's rows
        survive a kill and the node's audit entries do not, so a resumed case that replayed the
        filing would otherwise end with no `SUBMISSION_MADE` in its log and
        `audit.verify_submissions` would grade a filing that happened as a filing that did not.
        """
        audit, case_id, case_version, at = self._open(state)
        self._reached(CaseState.SUBMITTED, 0)

        claim = claim_from_json(state["claim"])
        decision = decision_from_json(_require(state, "gate"))
        window = window_from_json(_require(state, "window"))
        proposal = state.get("proposal")

        if proposal is None:
            outcome = SubmissionOutcome(
                recovery_identity=claim.recovery_identity,
                accepted=False,
                duplicate=False,
                effect_id=None,
                refusal=SubmissionRefusal.NOT_AUTHORISED,
                detail=(
                    f"{case_id} reached the submit node with no correction to file; an authorised "
                    f"case without a composed correction is a wiring fault and nothing is filed"
                ),
            )
            record_outcome(
                audit,
                outcome,
                case_id=case_id,
                case_version=case_version,
                at=at,
                actor=SYSTEM_ACTOR,
            )
            return {
                "node": CaseState.SUBMITTED.value,
                "node_history": ["submit"],
                "submission": outcome_json(outcome),
                "audit": audit.added,
            }

        approvals = audit.events
        filed = self._deps.ledger.invoke(
            case_id=case_id,
            case_version=case_version,
            node=CaseState.SUBMITTED,
            tool=TOOL_SUBMIT_CORRECTION,
            call_index=0,
            audit=audit,
            at=at,
            actor=SYSTEM_ACTOR,
            run=lambda: outcome_json(
                file_correction(
                    claim=claim,
                    decision=decision,
                    window=window,
                    case_id=case_id,
                    case_version=case_version,
                    approvals=approvals,
                    guard=self._deps.guard,
                    portal=self._deps.portal,
                    filed_on=at,
                    narrative=str(proposal["narrative"]),
                    clause_ids=[str(item["clause_id"]) for item in proposal["citations"]],
                )
            ),
        )
        self._reached(CaseState.SUBMITTED, 1)

        outcome = outcome_from_json(filed.payload)
        record_outcome(
            audit, outcome, case_id=case_id, case_version=case_version, at=at, actor=SYSTEM_ACTOR
        )
        return {
            "node": CaseState.SUBMITTED.value,
            "node_history": ["submit"],
            "submission": outcome_json(outcome),
            "tool_calls": [{"invocation_id": filed.invocation_id, "replayed": filed.replayed}],
            "audit": audit.added,
        }

    def write_off(self, state: GraphState) -> dict[str, Any]:
        """Close the case without filing anything, and record why in the words the gate used."""
        audit, case_id, case_version, at = self._open(state)
        self._reached(CaseState.WRITTEN_OFF, 0)

        approval = state.get("approval")
        audit.record(
            case_id=case_id,
            case_version=case_version,
            kind=CASE_WRITTEN_OFF,
            node=CaseState.WRITTEN_OFF,
            at=at,
            actor=SYSTEM_ACTOR,
            detail={
                "outcome": state.get("outcome"),
                "reason": state.get("gate_reason"),
                "approval_granted": bool(approval and approval.get("granted")),
            },
        )
        return {
            "node": CaseState.WRITTEN_OFF.value,
            "node_history": ["write_off"],
            "audit": audit.added,
        }


def _require(state: GraphState, key: str) -> Mapping[str, Any]:
    """Read a key a later node depends on, naming the node that should have written it.

    A `KeyError` three frames inside a rebuild function says `'claimed_parts'`, which sends the
    reader to the money code. This says which stage of the pipeline did not run, which is the actual
    fault whenever it happens.
    """
    value = cast("Mapping[str, Any]", state).get(key)
    if not isinstance(value, Mapping):
        raise ValueError(
            f"the case reached a node that needs {key!r} and the checkpoint does not carry it. "
            f"That stage of the pipeline has not run, or its update was discarded."
        )
    return value


def _governing_entry(
    entries: Sequence[Mapping[str, Any]], code: RejectionCode
) -> Mapping[str, Any] | None:
    """The best-ranked verified citation that governs this rejection code, or `None`.

    **Never the best-ranked clause merely for ranking first.** `domain.PolicyClause` states the
    rule: a clause that governs no code is context rather than authority, and may never be cited as
    the basis of a requirement being satisfied. This function is where that rule is enforced for the
    running system, and it now agrees with `evaluation.pipeline._authority`, which enforced it for
    the evaluated one. `tests/test_governing_authority.py` feeds both the same retrieval result and
    asserts they choose the same clause — because a deployment that selects evidence differently
    from the system that was measured is not the system that was measured.

    **The fallback that used to be here, and why it went.** This function once returned the
    top-ranked clause when no retrieved clause governed the code, arguing that "a programme whose
    policy states its evidence requirements in prose rather than in a coded schedule still has a
    governing clause". Two measurements retired it:

    - **The case it defended does not exist here.** 0 of 126 (programme, rejection code) pairs in
      the committed corpus lack a current clause that governs them. Every programme states its
      requirements in a coded schedule, so the fallback never fired for the reason it was written.
    - **The case it did fire for was the dangerous one.** It fired only when retrieval had
      *missed* the governing clause, and then it cited whatever ranked first. On the development
      split, before withdrawn documents were excluded, 69 of 520 queries had a withdrawn bulletin at
      rank one — so a requirement could be marked satisfied on the authority of text that says of
      itself that it has been withdrawn.

    Returning `None` makes the requirement `CONFLICTING` rather than `SATISFIED`, which the gate
    sends to a person. That is the fail-closed direction and the right one: a missed authority is a
    question for an adjudicator, not something to paper over with the nearest paragraph. It can
    only make the system more cautious — it converts a would-be authorisation into a review, never
    the reverse — which is why it is safe to have found and fixed after the hold-out was first
    computed (ADR-002).
    """
    for entry in entries:
        if code.value in (entry.get("governs") or ()):
            return entry
    return None


def _requirement_from(
    status_record: Mapping[str, Any], governing: Mapping[str, Any] | None
) -> Requirement:
    """Assemble a `Requirement` from its status and the citation that supports it.

    A requirement the evidence satisfies with nothing to cite becomes `CONFLICTING`, not
    `SATISFIED`. The module docstring argues the choice; the enforcement is that
    `domain.Requirement` would refuse to construct the alternative, so kill condition H cannot be
    breached from here whatever this function decides.
    """
    status = RequirementStatus(status_record["status"])
    detail = str(status_record["detail"])
    if status is not RequirementStatus.SATISFIED:
        return Requirement(
            requirement_id=str(status_record["requirement_id"]),
            description=str(status_record["description"]),
            status=status,
            detail=detail,
            citation=None,
        )
    if governing is None:
        return Requirement(
            requirement_id=str(status_record["requirement_id"]),
            description=str(status_record["description"]),
            status=RequirementStatus.CONFLICTING,
            detail=(
                f"{detail}, and no clause of this policy version could be retrieved to establish "
                f"that it suffices; a person has to decide rather than a technician fetch anything"
            ),
            citation=None,
        )
    return Requirement(
        requirement_id=str(status_record["requirement_id"]),
        description=str(status_record["description"]),
        status=RequirementStatus.SATISFIED,
        detail=detail,
        citation=citation_from_json(governing),
    )


def _approval_request(state: GraphState) -> dict[str, Any]:
    """What a person is shown while the case waits, and what they are asked to answer.

    It carries the outcome, the amount, the deadline and the reason — everything needed to decide —
    and the shape of the answer, because an interrupt whose payload does not say what a valid reply
    looks like is an interrupt that gets replied to wrongly once and then never trusted.
    """
    computation = state.get("computation") or {}
    window = state.get("window") or {}
    recoverable = computation.get("recoverable_amount") or {}
    return {
        "case_id": state["case_id"],
        "case_version": state["case_version"],
        "outcome": state.get("outcome"),
        "reason": state.get("gate_reason"),
        "recoverable_amount": recoverable.get("amount"),
        "currency": recoverable.get("currency"),
        "closes_on": window.get("closes_on"),
        "days_remaining": window.get("days_remaining"),
        "answer_shape": {
            "decision": f"{_APPROVED} or anything else to decline",
            "actor": "who decided",
            "token": "the approver token; compared and then discarded",
            "case_version": "the version being approved; must equal case_version above",
            "note": "optional",
        },
    }
