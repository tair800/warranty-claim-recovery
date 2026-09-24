"""The audit verifier, held to the three failures it exists to catch.

Kill condition D is graded from `verify_submissions`, so a verifier that cannot fail would make the
criterion vacuous no matter how many submissions passed through it. Each test below plants one
specific violation and asserts the verifier reports it; the last asserts a clean stream reports
nothing, because a verifier that always complains is as useless as one that never does.

The stale-version case is the one worth reading. A case approved at £40 and then re-priced to £4,000
before submission has an approval in the log, and a verifier that only asked "is there an approval
for this case?" would call it clean. It is not clean. It is the failure a four-eyes control exists
to prevent, and it is invisible unless the approval is bound to the version it approved.
"""

from __future__ import annotations

import ast
import inspect
from datetime import date

from warranty_claim_recovery import audit as audit_module
from warranty_claim_recovery.audit import (
    APPROVAL_GRANTED,
    APPROVAL_REQUESTED,
    CASE_REPRICED,
    SUBMISSION_MADE,
    AuditLog,
    verify_submissions,
)
from warranty_claim_recovery.domain import CaseState

DAY = date(2026, 3, 2)


def log_with(
    *, approve_version: int | None, submit_version: int, identity: str = "c1:p1:s1"
) -> AuditLog:
    """A one-case stream: request, optionally approve at a version, then submit at another."""
    log = AuditLog()
    log.append(
        case_id="case-1",
        case_version=submit_version,
        kind=APPROVAL_REQUESTED,
        node=CaseState.AWAITING_APPROVAL,
        at=DAY,
        actor="system",
    )
    if approve_version is not None:
        log.append(
            case_id="case-1",
            case_version=approve_version,
            kind=APPROVAL_GRANTED,
            node=CaseState.AWAITING_APPROVAL,
            at=DAY,
            actor="recovery.manager",
        )
    log.append(
        case_id="case-1",
        case_version=submit_version,
        kind=SUBMISSION_MADE,
        node=CaseState.SUBMITTED,
        at=DAY,
        actor="system",
        detail={"recovery_identity": identity},
    )
    return log


def test_a_clean_stream_reports_nothing() -> None:
    audit = verify_submissions(list(log_with(approve_version=0, submit_version=0)))

    assert audit.submissions_observed == 1
    assert audit.approvals_observed == 1
    assert audit.submissions_without_approval == 0
    assert audit.submissions_with_stale_version_approval == 0
    assert audit.duplicate_submissions == 0
    assert audit.offending_cases == ()


def test_a_submission_with_no_approval_at_all_is_caught() -> None:
    audit = verify_submissions(list(log_with(approve_version=None, submit_version=0)))

    assert audit.submissions_without_approval == 1
    assert audit.offending_cases == ("case-1",)


def test_an_approval_for_an_earlier_version_does_not_authorise_a_later_one() -> None:
    """Approved at version 0, re-priced, submitted at version 1. There *is* an approval on file."""
    log = AuditLog()
    log.append(
        case_id="case-1",
        case_version=0,
        kind=APPROVAL_GRANTED,
        node=CaseState.AWAITING_APPROVAL,
        at=DAY,
        actor="recovery.manager",
    )
    log.append(
        case_id="case-1",
        case_version=1,
        kind=CASE_REPRICED,
        node=CaseState.GATE,
        at=DAY,
        actor="system",
        detail={"reason": "the technician supplied the missing invoice"},
    )
    log.append(
        case_id="case-1",
        case_version=1,
        kind=SUBMISSION_MADE,
        node=CaseState.SUBMITTED,
        at=DAY,
        actor="system",
        detail={"recovery_identity": "c1:p1:s1"},
    )

    audit = verify_submissions(list(log))

    assert audit.approvals_observed == 1, "the approval is on file, which is why this is subtle"
    assert audit.submissions_without_approval == 0
    assert audit.submissions_with_stale_version_approval == 1
    assert audit.offending_cases == ("case-1",)


def test_an_approval_recorded_after_the_submission_does_not_authorise_it() -> None:
    """Submitting first and papering over it afterwards must not read as compliant."""
    log = AuditLog()
    log.append(
        case_id="case-1",
        case_version=0,
        kind=SUBMISSION_MADE,
        node=CaseState.SUBMITTED,
        at=DAY,
        actor="system",
        detail={"recovery_identity": "c1:p1:s1"},
    )
    log.append(
        case_id="case-1",
        case_version=0,
        kind=APPROVAL_GRANTED,
        node=CaseState.AWAITING_APPROVAL,
        at=DAY,
        actor="recovery.manager",
    )

    audit = verify_submissions(list(log))

    assert audit.submissions_without_approval == 1


def test_one_recovery_identity_submitted_twice_is_a_duplicate() -> None:
    log = log_with(approve_version=0, submit_version=0)
    log.append(
        case_id="case-2",
        case_version=0,
        kind=APPROVAL_GRANTED,
        node=CaseState.AWAITING_APPROVAL,
        at=DAY,
        actor="recovery.manager",
    )
    log.append(
        case_id="case-2",
        case_version=0,
        kind=SUBMISSION_MADE,
        node=CaseState.SUBMITTED,
        at=DAY,
        actor="system",
        # The same physical recovery reached the portal down a second case.
        detail={"recovery_identity": "c1:p1:s1"},
    )

    audit = verify_submissions(list(log))

    assert audit.submissions_observed == 2
    assert audit.duplicate_submissions == 1
    assert "case-2" in audit.offending_cases


def test_the_log_is_append_only_and_orders_by_sequence_not_by_date() -> None:
    """Every event here shares one date. Ordering must still be total and deterministic."""
    log = AuditLog()
    for index in range(5):
        log.append(
            case_id="case-1",
            case_version=index,
            kind=CASE_REPRICED,
            node=CaseState.GATE,
            at=DAY,
            actor="system",
        )

    ids = [event.event_id for event in log]
    assert ids == sorted(ids), "event ids must order the stream on their own"
    assert len(log) == 5
    assert not hasattr(log, "remove")
    assert not hasattr(log, "update")


def test_the_verifier_imports_nothing_from_the_graph() -> None:
    """The independence claim, asserted rather than asserted-in-prose.

    `SKILL_MATRIX.md` asks for an independent implementation of the audit contract. A verifier that
    imported the graph would be checking that the graph agrees with itself.
    """
    tree = ast.parse(inspect.getsource(audit_module))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
        elif isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)

    assert not [name for name in imported if "graph" in name], (
        f"audit.py imports {imported}; it must answer from the event stream alone"
    )
