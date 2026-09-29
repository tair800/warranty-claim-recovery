"""Filing the correction: four refusals, one effect, and a mock portal with a written contract.

ADR-001 §4.2 makes two promises about what leaves this system — *nothing leaves without a recorded
human approval for the same case version, and nothing leaves twice* — and this module is where both
are enforced against a real attempt rather than described.

**The portal is a mock, and ADR-001 §8 says so before anyone asks.** There is no manufacturer to
file with, and the idempotency claim is deliberately **about this system's own effects**, not about
a third party's behaviour. A real portal that deduplicated on its side would make the guarantee
here untestable and would move the property being claimed to somebody else's server. What is
claimed is: however many callers race one recovery identity, this system performs the filing exactly
once. `ManufacturerPortal` records every call it receives, so "exactly once" is counted at the point
of the effect rather than inferred from the absence of a complaint.

### The four refusals, in the order they are checked, and why that is the order

1. **The gate did not authorise.** `RECOVERABLE` and `PARTIALLY_RECOVERABLE` are the only outcomes
   that let money leave, and they are constructed in exactly one place in `gate.py`. Checked first
   because it is the cheapest question and because a refused case has nothing an approval could
   rescue.
2. **The correction window has closed.** Checked against the date the filing is *attempted*, not
   against the date the gate ran. Those are different days by design: the approval is an interrupt
   and a case sits at it while a person decides, so a window that was open at the gate can be shut
   by the time anybody says yes. This is kill condition M, budgeted at zero over the whole corpus,
   and checking it only at the gate would leave the entire human-decision interval unguarded — which
   is precisely the interval in which the deadline passes.
3. **There is no approval for this case at this version.** `audit.verify_submissions` grades this
   afterwards from the event stream; this check is what makes that grading come out clean. Both
   halves matter: an approval for an earlier version approved a different amount, and "approved" and
   "approved *this*" are different claims.
4. **Another caller already holds the identity.** `SubmissionGuard.claim_once` is `SET NX` inside
   Redis, so exactly one of sixteen simultaneous callers is told it won.

### Why the loser is given the winner's effect and not an error

Losing is not a failure, and `queue/idempotency.py` argues the general case. The consequence here is
concrete: a caller that received an exception would have to decide whether the submission had
happened, and the first caller to decide wrongly retries a filing that succeeded. So the loser waits
for the winner's effect identifier to appear against the identity and returns it, marked
`duplicate=True`, and records `SUBMISSION_SUPPRESSED` rather than `SUBMISSION_MADE`. Exactly one
`SUBMISSION_MADE` per identity is what keeps `verify_submissions` reporting zero duplicates, and the
suppression events are what let a reader see that the other fifteen callers arrived and were
stopped.

### Why this module writes no audit event itself

`submit` reads the audit and returns an outcome; `record_outcome` writes. The split exists because
`AuditLog` assigns sequence numbers from a mutable counter and the concurrency evidence calls
`submit` from sixteen threads at once. A submitter that appended would need the lock inside itself,
held across the Redis round trip and the filing, and the race being measured would be serialised by
the instrument measuring it. With the write separated, the lock covers an append and nothing else.

### Why the effect identifier is derived and the count is of calls

`effect_id` is a digest of the recovery identity and the case version, so a retry that legitimately
reaches the portal after an interrupted in-flight submission produces the same identifier rather
than a second one — which is what makes `SubmissionGuard.record_effect` able to accept a repeat in
silence and refuse a genuine second effect. It also means that counting *distinct* identifiers would
hide a duplicate filing, so the evidence counts **calls** to the portal instead. A guarantee
measured by a number that cannot go up is not a measurement.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Mapping, Sequence
from datetime import date
from enum import StrEnum
from hashlib import blake2b
from typing import Any, Final, NamedTuple, Protocol

from warranty_claim_recovery.audit import (
    APPROVAL_GRANTED,
    SUBMISSION_MADE,
    SUBMISSION_SUPPRESSED,
)
from warranty_claim_recovery.domain import (
    AuditEvent,
    CaseState,
    Claim,
    ClaimWindow,
    GateDecision,
)
from warranty_claim_recovery.money import Money
from warranty_claim_recovery.queue.idempotency import SubmissionGuard

__all__ = [
    "DEFAULT_RESOLVE_TIMEOUT_SECONDS",
    "PORTAL_CONTRACT",
    "AuditSink",
    "FilingRequest",
    "ManufacturerPortal",
    "Receipt",
    "SubmissionOutcome",
    "SubmissionRefusal",
    "approval_for",
    "contract_json",
    "effect_id_for",
    "outcome_from_json",
    "outcome_json",
    "record_outcome",
    "submit",
]

#: How long a losing caller waits for the winner's effect identifier to appear. Ten seconds because
#: the winner's remaining work between claiming the identity and recording the effect is one local
#: call and one Redis round trip; a loser still waiting after ten seconds is waiting on a winner
#: that has died, and the honest answer then is `EFFECT_UNRESOLVED` rather than a longer wait. It
#: is not a retry budget: nothing here retries the filing.
DEFAULT_RESOLVE_TIMEOUT_SECONDS: Final = 10.0

#: How often the loser asks. Short enough that the common case — the winner finishing within a few
#: milliseconds — costs one or two polls, long enough that sixteen losers do not saturate Redis with
#: the question while the winner is trying to answer it.
_POLL_INTERVAL_SECONDS: Final = 0.02

#: The mock portal's contract, written down because a mock with no stated contract is a mock that
#: quietly becomes whatever the caller needed that day. `docs/` and the console both publish this,
#: and `tests/test_submission.py` asserts the request the portal actually receives against it, so
#: the description and the behaviour cannot drift.
PORTAL_CONTRACT: Final[dict[str, Any]] = {
    "is_mock": True,
    "notice": (
        "A local mock of a manufacturer's correction portal. It files nothing with anybody. "
        "ADR-001 §8 records that the idempotency guarantee this project makes is about its own "
        "effects and not about a third party's behaviour."
    ),
    "request_fields": [
        "recovery_identity",
        "case_id",
        "case_version",
        "program_id",
        "claim_id",
        "rejection_code",
        "amount",
        "currency",
        "filed_on",
        "narrative_digest",
        "clause_ids",
    ],
    "response_fields": ["effect_id", "recovery_identity", "case_id", "case_version", "filed_on"],
    "guarantees": [
        "every call is recorded, including a call this system should not have made",
        "the effect identifier is a digest of the recovery identity and the case version, so a "
        "retry of one filing is not a second effect",
        "nothing is deduplicated inside the portal: one call in, one receipt out",
    ],
}


class SubmissionRefusal(StrEnum):
    """Why a filing did not happen. One member per check, and none of them is a generic failure.

    A closed enum rather than a free string because the console groups by it and the evidence counts
    by it. `EFFECT_UNRESOLVED` is the only one that is not a policy decision: it means the winner
    held the identity and did not record an effect inside the timeout, which is an operational fact
    about a dead worker rather than a judgement about the claim.
    """

    NOT_AUTHORISED = "NOT_AUTHORISED"
    WINDOW_CLOSED = "WINDOW_CLOSED"
    NO_APPROVAL = "NO_APPROVAL"
    STALE_APPROVAL = "STALE_APPROVAL"
    EFFECT_UNRESOLVED = "EFFECT_UNRESOLVED"


class FilingRequest(NamedTuple):
    """What is put in front of the portal. Every field is in `PORTAL_CONTRACT['request_fields']`."""

    recovery_identity: str
    case_id: str
    case_version: int
    program_id: str
    claim_id: str
    rejection_code: str
    amount: Money
    filed_on: date
    #: A digest of the correction rather than the correction. The portal is a mock and its receipt
    #: log is read in tests and in the console; carrying several kilobytes of prose through it would
    #: make the evidence unreadable, and the digest still proves that the text which was approved is
    #: the text that was filed.
    narrative_digest: str
    clause_ids: tuple[str, ...]

    def as_json(self) -> dict[str, Any]:
        return {
            "recovery_identity": self.recovery_identity,
            "case_id": self.case_id,
            "case_version": self.case_version,
            "program_id": self.program_id,
            "claim_id": self.claim_id,
            "rejection_code": self.rejection_code,
            "amount": str(self.amount.quantize().amount),
            "currency": self.amount.currency.value,
            "filed_on": self.filed_on.isoformat(),
            "narrative_digest": self.narrative_digest,
            "clause_ids": list(self.clause_ids),
        }


class Receipt(NamedTuple):
    """What the portal hands back. Every field is in `PORTAL_CONTRACT['response_fields']`."""

    effect_id: str
    recovery_identity: str
    case_id: str
    case_version: int
    filed_on: date


def effect_id_for(recovery_identity: str, case_version: int) -> str:
    """The identifier one filing produces, derived so that a retry of it is not a second effect.

    `blake2b` over the identity and the version, rather than a random token, because
    `SubmissionGuard.record_effect` accepts a repeat of the same effect in silence and refuses a
    different one. A random token would make the honest retry — a worker that filed, died before
    recording, and came back — indistinguishable from a genuine duplicate, and the guard would raise
    on the case it exists to forgive.
    """
    digest = blake2b(f"{recovery_identity}|{case_version}".encode(), digest_size=8)
    return f"EFF-{digest.hexdigest()}"


class ManufacturerPortal:
    """A local mock that files nothing and records everything, including what it should not receive.

    **It does not deduplicate.** That is the one thing it must not do: a portal that rejected a
    second filing for an identity would make this system's guarantee untestable, because every
    duplicate would be absorbed by the instrument rather than counted by it. One call in, one
    receipt out, and the evidence counts receipts.

    Thread-safe, because sixteen callers race it by design. The lock covers the append to the
    receipt list and nothing else, so the race being measured is still a race.
    """

    __slots__ = ("_lock", "_receipts")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._receipts: list[Receipt] = []

    def file(self, request: FilingRequest) -> Receipt:
        receipt = Receipt(
            effect_id=effect_id_for(request.recovery_identity, request.case_version),
            recovery_identity=request.recovery_identity,
            case_id=request.case_id,
            case_version=request.case_version,
            filed_on=request.filed_on,
        )
        with self._lock:
            self._receipts.append(receipt)
        return receipt

    @property
    def receipts(self) -> tuple[Receipt, ...]:
        with self._lock:
            return tuple(self._receipts)

    def filings_for(self, recovery_identity: str) -> tuple[Receipt, ...]:
        """Every call this identity produced — one entry per call, not per distinct effect.

        Per call on purpose. The effect identifier is derived from the identity, so two filings
        carry the same one and a count of distinct identifiers would report a duplicate as a single
        effect. Kill condition E allows exactly one, and the number it is graded on has to be able
        to reach two.
        """
        return tuple(item for item in self.receipts if item.recovery_identity == recovery_identity)


class SubmissionOutcome(NamedTuple):
    """What one attempt did, in the terms the audit log and the graph both need.

    `accepted` and `duplicate` are separate booleans rather than one tri-state because they answer
    different questions. `accepted` says whether *this* caller performed the filing; `duplicate`
    says whether the identity already had one. A loser is `accepted=False, duplicate=True` and holds
    the winner's effect identifier, which is the answer it needs and is not an error.
    """

    recovery_identity: str
    accepted: bool
    duplicate: bool
    effect_id: str | None
    refusal: SubmissionRefusal | None
    detail: str


class AuditSink(Protocol):
    """Whatever the caller appends audit events to.

    A protocol rather than `AuditLog` itself because the graph writes through `state.CaseAudit`,
    which seeds its sequence from the checkpoint, and the concurrency evidence writes through a
    lock-wrapped log. Both are the same shape and neither is the other.
    """

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
    ) -> AuditEvent: ...


def approval_for(
    events: Sequence[AuditEvent], *, case_id: str, case_version: int
) -> AuditEvent | None:
    """The approval that authorises this case at this version, or `None`.

    An approval for a *later* version counts, an approval for an earlier one does not. That
    asymmetry is deliberate and matches `audit.verify_submissions`, which compares the highest
    approved version against the submission's: approving version three and filing version two means
    a person looked at more than was filed, which is not the failure the control exists to catch,
    while approving version two and filing version three means they looked at less.
    """
    best: AuditEvent | None = None
    for event in events:
        if event.kind != APPROVAL_GRANTED or event.case_id != case_id:
            continue
        if event.case_version < case_version:
            continue
        if best is None or event.case_version < best.case_version:
            best = event
    return best


def _refused(recovery_identity: str, refusal: SubmissionRefusal, detail: str) -> SubmissionOutcome:
    return SubmissionOutcome(
        recovery_identity=recovery_identity,
        accepted=False,
        duplicate=False,
        effect_id=None,
        refusal=refusal,
        detail=detail,
    )


def submit(
    *,
    claim: Claim,
    decision: GateDecision,
    window: ClaimWindow,
    case_id: str,
    case_version: int,
    approvals: Sequence[AuditEvent],
    guard: SubmissionGuard,
    portal: ManufacturerPortal,
    filed_on: date,
    narrative: str,
    clause_ids: Sequence[str],
    resolve_timeout_seconds: float = DEFAULT_RESOLVE_TIMEOUT_SECONDS,
) -> SubmissionOutcome:
    """File the correction, or say precisely why it was not filed.

    `filed_on` is a parameter with no default. A default of `date.today()` would make every
    evaluation of kill condition M depend on the day the build ran, and the criterion — nothing
    resubmitted after the window closed — would pass or fail on the calendar rather than on the
    system. It is also the honest value: the date the filing is attempted is a fact the caller
    holds, and this module is not entitled to guess it.

    `approvals` is the whole event stream rather than a boolean, because the question is not "has
    somebody approved" but "is there an approval for this case at this version", and a caller that
    reduced it to a boolean would be the one deciding what counted.
    """
    identity = claim.recovery_identity

    if not decision.authorises_recovery:
        return _refused(
            identity,
            SubmissionRefusal.NOT_AUTHORISED,
            f"{case_id} was gated {decision.outcome.value}; only RECOVERABLE and "
            f"PARTIALLY_RECOVERABLE let money leave this system",
        )

    if filed_on > window.closes_on:
        return _refused(
            identity,
            SubmissionRefusal.WINDOW_CLOSED,
            f"the correction window for {case_id} closed on {window.closes_on.isoformat()} and "
            f"the filing was attempted on {filed_on.isoformat()}, "
            f"{(filed_on - window.closes_on).days} day(s) later; it would be refused on receipt",
        )

    approval = approval_for(approvals, case_id=case_id, case_version=case_version)
    if approval is None:
        held = [event.case_version for event in approvals if event.kind == APPROVAL_GRANTED]
        refusal = SubmissionRefusal.STALE_APPROVAL if held else SubmissionRefusal.NO_APPROVAL
        return _refused(
            identity,
            refusal,
            f"{case_id} has no recorded approval for version {case_version}; the log holds "
            f"approvals for versions {sorted(held)}. An approval of an earlier version approved a "
            f"different amount.",
        )

    if not guard.claim_once(identity):
        return _resolve_losing_caller(identity, guard, resolve_timeout_seconds)

    request = FilingRequest(
        recovery_identity=identity,
        case_id=case_id,
        case_version=case_version,
        program_id=claim.program_id,
        claim_id=claim.claim_id,
        rejection_code=claim.rejection_code.value,
        amount=decision.computation.recoverable_amount,
        filed_on=filed_on,
        narrative_digest=blake2b(narrative.encode("utf-8"), digest_size=16).hexdigest(),
        clause_ids=tuple(clause_ids),
    )
    receipt = portal.file(request)
    # Recorded after the effect, never before. The claim is what stops a second caller; the effect
    # identifier is what the second caller is given, and writing it before the portal had produced
    # it would hand out an identifier for a filing that had not happened.
    guard.record_effect(identity, receipt.effect_id)
    return SubmissionOutcome(
        recovery_identity=identity,
        accepted=True,
        duplicate=False,
        effect_id=receipt.effect_id,
        refusal=None,
        detail=(
            f"filed {request.amount} for {identity} on {filed_on.isoformat()} against approval "
            f"{approval.event_id} of case version {approval.case_version}"
        ),
    )


def _resolve_losing_caller(
    identity: str, guard: SubmissionGuard, timeout_seconds: float
) -> SubmissionOutcome:
    """Wait for the winner's effect identifier, and hand it back rather than raising.

    `time.monotonic` rather than a fixed number of polls, so the bound is a duration a reader can
    reason about instead of a count that means different things on different machines. Monotonic
    rather than wall clock because a clock adjustment during the wait would either end it
    immediately or never — and this loop runs on the path a warranty correction is filed on.
    """
    deadline = time.monotonic() + timeout_seconds
    while True:
        effect_id = guard.effect_for(identity)
        if effect_id is not None:
            return SubmissionOutcome(
                recovery_identity=identity,
                accepted=False,
                duplicate=True,
                effect_id=effect_id,
                refusal=None,
                detail=(
                    f"{identity} was already claimed; this caller returns the winner's effect "
                    f"{effect_id} and performs none of its own"
                ),
            )
        if time.monotonic() >= deadline:
            return _refused(
                identity,
                SubmissionRefusal.EFFECT_UNRESOLVED,
                f"{identity} is claimed by another caller that has not recorded an effect within "
                f"{timeout_seconds:g}s. This caller files nothing: an unresolved claim means the "
                f"filing may already be in flight, and a second one is the failure being guarded.",
            )
        time.sleep(_POLL_INTERVAL_SECONDS)


def record_outcome(
    sink: AuditSink,
    outcome: SubmissionOutcome,
    *,
    case_id: str,
    case_version: int,
    at: date,
    actor: str,
) -> AuditEvent:
    """Write what happened, as the one event kind `verify_submissions` grades from.

    `SUBMISSION_MADE` is written **only** for the caller that actually filed. Every refusal and
    every losing caller writes `SUBMISSION_SUPPRESSED`, so a reader can see that fifteen callers
    arrived and were stopped without any of them appearing as a submission. Writing
    `SUBMISSION_MADE` for a duplicate would make kill condition E fail in the log while the system
    had in fact behaved correctly — the log would be reporting the guard's success as a defect.

    `recovery_identity` is in the detail because `verify_submissions` keys its duplicate check on it
    and falls back to the case identifier when it is absent. The fallback is a safety net rather
    than a supported shape: two cases can file the same identity, and the fallback would count those
    as distinct.
    """
    kind = SUBMISSION_MADE if outcome.accepted else SUBMISSION_SUPPRESSED
    return sink.record(
        case_id=case_id,
        case_version=case_version,
        kind=kind,
        node=CaseState.SUBMITTED,
        at=at,
        actor=actor,
        detail={
            "recovery_identity": outcome.recovery_identity,
            "effect_id": outcome.effect_id,
            "duplicate": outcome.duplicate,
            "refusal": None if outcome.refusal is None else outcome.refusal.value,
            "detail": outcome.detail,
        },
    )


def outcome_json(outcome: SubmissionOutcome) -> dict[str, Any]:
    """The outcome as the tool ledger records it, so a resumed case replays the same answer."""
    return {
        "recovery_identity": outcome.recovery_identity,
        "accepted": outcome.accepted,
        "duplicate": outcome.duplicate,
        "effect_id": outcome.effect_id,
        "refusal": None if outcome.refusal is None else outcome.refusal.value,
        "detail": outcome.detail,
    }


def outcome_from_json(payload: Mapping[str, Any]) -> SubmissionOutcome:
    refusal = payload.get("refusal")
    return SubmissionOutcome(
        recovery_identity=str(payload["recovery_identity"]),
        accepted=bool(payload["accepted"]),
        duplicate=bool(payload["duplicate"]),
        effect_id=None if payload.get("effect_id") is None else str(payload["effect_id"]),
        refusal=None if refusal is None else SubmissionRefusal(refusal),
        detail=str(payload["detail"]),
    )


def contract_json() -> str:
    """`PORTAL_CONTRACT`, rendered once, for whatever publishes it.

    Rendered here rather than by each publisher so the console, the docs and any artifact print the
    same bytes. A contract that is serialised twice is a contract that will one day be serialised
    two ways, and the difference will be the field somebody added to one of them.
    """
    return json.dumps(PORTAL_CONTRACT, indent=2, sort_keys=True)
