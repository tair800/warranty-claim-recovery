"""The audit trail, written by the system and verified by something that does not trust it.

`SKILL_MATRIX.md` asks this project for an *independent implementation* of the audit-event contract,
and the word that matters is independent. An audit log the acting code also validates proves that
the acting code is self-consistent, which is not the claim anyone cares about. So this module is
split deliberately in two:

- **`AuditLog`** is what the graph writes to. It appends and it does not delete; there is no update
  and no removal, because an audit record that can be corrected is a record of what someone wanted
  to have happened.
- **`verify_submissions`** is what kill condition D is graded from. It imports nothing from the
  graph, knows nothing about nodes or checkpoints, and answers one question from the event stream
  alone: *did every submission have an approval behind it, for the same case and the same version?*

**Why the version matters.** "Approved" and "approved *this*" are different claims. A case can be
approved, then re-priced because a technician supplied the missing invoice, and then submitted — and
the approval a reader sees in the log would be an approval of a different amount. `case_version` is
bumped whenever the decided content changes, an approval records the version it approved, and a
submission is valid only against an approval for its own version. Without that, kill condition D
would pass on a log that contains an approval for £40 and a submission for £4,000.

**Why the events are ordered by sequence and not by date.** `AuditEvent.at` is a date, because this
system reasons in whole days and a wall-clock timestamp would make every artifact unreproducible.
Two events on the same day are therefore indistinguishable by `at`, so ordering comes from an
explicit monotonic sequence the log assigns. Sorting by date and hoping would make the verifier's
answer depend on dictionary order.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Final

from warranty_claim_recovery.domain import AuditEvent, CaseState

__all__ = [
    "APPROVAL_GRANTED",
    "APPROVAL_REQUESTED",
    "CASE_REPRICED",
    "SUBMISSION_MADE",
    "SUBMISSION_SUPPRESSED",
    "TOOL_COMPLETED",
    "TOOL_STARTED",
    "AuditLog",
    "SubmissionAudit",
    "verify_submissions",
]

#: The kinds this module reasons about. Free strings elsewhere are fine — an audit log that rejects
#: an event it does not recognise would lose exactly the evidence an incident needs — but these five
#: are load-bearing and are constants so a typo in one is a failed import rather than a silent gap
#: in the verifier's view.
APPROVAL_REQUESTED: Final = "approval.requested"
APPROVAL_GRANTED: Final = "approval.granted"
SUBMISSION_MADE: Final = "submission.made"
SUBMISSION_SUPPRESSED: Final = "submission.suppressed"
CASE_REPRICED: Final = "case.repriced"
TOOL_STARTED: Final = "tool.started"
TOOL_COMPLETED: Final = "tool.completed"


@dataclass
class AuditLog:
    """Append-only, in memory, with an explicit sequence.

    In memory because the durable record of a case is the LangGraph checkpoint in PostgreSQL, and a
    second durable store would need its own consistency story with the first. This is the stream the
    verifier reads and the console renders; `store/` persists it alongside the checkpoint.

    There is no `remove`, no `update` and no `__setitem__`. That is the design, not an omission.
    """

    _events: list[AuditEvent] = field(default_factory=list)
    _sequence: int = 0

    def append(
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
        self._sequence += 1
        event = AuditEvent(
            event_id=f"{case_id}:{self._sequence:06d}",
            case_id=case_id,
            case_version=case_version,
            kind=kind,
            node=node,
            at=at,
            actor=actor,
            detail=detail or {},
        )
        self._events.append(event)
        return event

    def extend(self, events: Iterable[AuditEvent]) -> None:
        """Adopt events recorded elsewhere — a resumed case's history, read back from the store."""
        for event in events:
            self._events.append(event)
            self._sequence = max(self._sequence, _sequence_of(event))

    def __iter__(self) -> Iterator[AuditEvent]:
        return iter(self._events)

    def __len__(self) -> int:
        return len(self._events)

    def for_case(self, case_id: str) -> tuple[AuditEvent, ...]:
        return tuple(event for event in self._events if event.case_id == case_id)

    def of_kind(self, kind: str) -> tuple[AuditEvent, ...]:
        return tuple(event for event in self._events if event.kind == kind)


def _sequence_of(event: AuditEvent) -> int:
    """The monotonic part of an event id. Events arrive ordered; ids make that checkable."""
    _, _, tail = event.event_id.rpartition(":")
    try:
        return int(tail)
    except ValueError:
        return 0


@dataclass(frozen=True)
class SubmissionAudit:
    """What the verifier concluded, in the shape `artifacts/submission.json` needs.

    Every count is a count of events, and the denominators are published beside the numerators
    because ADR-001's vacuity guard fails a criterion graded over nothing. A submission audit that
    saw no submissions is not a pass.
    """

    submissions_observed: int
    approvals_observed: int
    submissions_without_approval: int
    submissions_with_stale_version_approval: int
    duplicate_submissions: int
    offending_cases: tuple[str, ...]

    def as_json(self) -> dict[str, Any]:
        return {
            "submissions_observed": self.submissions_observed,
            "approvals_observed": self.approvals_observed,
            "submissions_without_approval": self.submissions_without_approval,
            "submissions_with_stale_version_approval": (
                self.submissions_with_stale_version_approval
            ),
            "duplicate_submissions": self.duplicate_submissions,
            "offending_cases": list(self.offending_cases),
        }


def verify_submissions(events: Sequence[AuditEvent]) -> SubmissionAudit:
    """Answer kill condition D from the event stream alone.

    Deliberately ignorant of the graph. It takes a flat sequence of events and asks three questions
    that a reader of a printed log could ask:

    1. Did anything get submitted that was never approved?
    2. Did anything get submitted whose only approval was for an earlier version of the case — that
       is, approved before the amount changed?
    3. Did any recovery identity get submitted more than once?

    An approval counts for a submission only when it precedes it in sequence. An approval recorded
    *after* a submission is an approval of a fait accompli, and counting it would let a system
    submit first and paper over it afterwards — which is the failure mode a four-eyes control
    exists to prevent, and it would otherwise pass this check silently.
    """
    ordered = sorted(events, key=_sequence_of)

    #: case_id -> the highest version that has been approved so far, walking forward in sequence.
    approved_version: dict[str, int] = {}
    approvals = 0
    submissions = 0
    without_approval = 0
    stale_version = 0
    seen_identities: dict[str, int] = {}
    duplicates = 0
    offenders: list[str] = []

    for event in ordered:
        if event.kind == APPROVAL_GRANTED:
            approvals += 1
            current = approved_version.get(event.case_id, -1)
            approved_version[event.case_id] = max(current, event.case_version)
            continue

        if event.kind != SUBMISSION_MADE:
            continue

        submissions += 1
        approved = approved_version.get(event.case_id)
        if approved is None:
            without_approval += 1
            offenders.append(event.case_id)
        elif approved < event.case_version:
            # The case was approved and then changed. The approval on file approved a different
            # thing from the one that went out.
            stale_version += 1
            offenders.append(event.case_id)

        identity = str(event.detail.get("recovery_identity") or event.case_id)
        seen_identities[identity] = seen_identities.get(identity, 0) + 1
        if seen_identities[identity] > 1:
            duplicates += 1
            offenders.append(event.case_id)

    return SubmissionAudit(
        submissions_observed=submissions,
        approvals_observed=approvals,
        submissions_without_approval=without_approval,
        submissions_with_stale_version_approval=stale_version,
        duplicate_submissions=duplicates,
        offending_cases=tuple(dict.fromkeys(offenders)),
    )
