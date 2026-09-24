"""Redis: the case work queue, its leases, and the state that makes a submission happen once.

Three things live in Redis in this system and nothing else does. They are here rather than in
PostgreSQL for one reason, and it is worth stating plainly because "we used Redis" is otherwise
decoration:

1. **Which worker may touch a case right now.** A short-lived, expiring claim on a case. It is
   read and written on every poll of every worker, it is worthless the moment it is stale, and it
   must survive the database being unavailable — a worker that cannot reach PostgreSQL still has to
   know it is not duplicating another worker's case.
2. **The order cases are waiting in.** A list, not a table. The queue is a liveness concern and the
   case itself is a durability concern; keeping them in separate components means a failover of one
   does not take out the other.
3. **Whether one recovery identity has already produced a submission effect.** The single
   compare-and-set that ADR-001 §4.2 rests on — one recovery identity, at most one effect, however
   many callers race for it.

**None of the case's own state is here.** The graph's checkpoint is in PostgreSQL and is the
authority on where a case has got to. If every key in Redis vanished, no case would be lost; work
would stop until a queue existed again, which is the correct failure and is what kill condition L
grades. This division is the whole reason Redis can be the volatile component: it holds what may be
rebuilt and never what may not.

**The failure this package exists to prevent.** Two workers on one case both reach the submit node,
both ask the manufacturer's portal for the same recovery, and the distributor's claim is flagged as
a duplicate submission — which is one of the seven rejection codes this system is built to correct.
The system would then be generating the defect it was bought to remove. `leases.CaseQueue` stops
the second worker starting; `idempotency.SubmissionGuard` stops the second effect if one ever does.
Two layers, because the first is a lease and a lease can expire under a worker that is merely slow.

Everything here **fails closed**. No method returns a permissive default when Redis cannot answer.
See `leases.RedisUnavailableError` for the argument, and `failure_injection` for the harness that
takes Redis away for real rather than pretending to.
"""

from __future__ import annotations

from warranty_claim_recovery.queue.idempotency import (
    ConflictingSubmissionEffectError,
    SubmissionClaimExpiredError,
    SubmissionGuard,
    SubmissionGuardError,
)
from warranty_claim_recovery.queue.leases import (
    DEFAULT_LEASE_TTL_SECONDS,
    DEFAULT_NAMESPACE,
    CaseQueue,
    RedisUnavailableError,
)

__all__ = [
    "DEFAULT_LEASE_TTL_SECONDS",
    "DEFAULT_NAMESPACE",
    "CaseQueue",
    "ConflictingSubmissionEffectError",
    "RedisUnavailableError",
    "SubmissionClaimExpiredError",
    "SubmissionGuard",
    "SubmissionGuardError",
]
