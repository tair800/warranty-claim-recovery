"""Kill condition L: with Redis unavailable, no case is leased twice and no submission proceeds.

The criterion is behavioural and is graded behaviourally. Redis is taken away for real — the
connection refused by the operating system, and the client severed mid-flight — and the **real**
`CaseQueue` and `SubmissionGuard` are then asked to do the things a worker would do. Nothing in
this file patches a method on either class. `queue/failure_injection.py` argues at length why that
line matters; the short version is that patching `CaseQueue.lease` to raise and then observing that
it raised measures the patch, and the code that runs during an actual outage never executes.

**The load-bearing claim is proved by contrast, not by assertion.** A `queue_is_load_bearing: true`
written into an artifact by hand proves that somebody typed `true`. Here it is derived: the same
methods, on the same classes, with the same arguments, lease a case and claim a submission when
Redis is up and refuse both when Redis is gone. If Redis were decorative the first half would still
pass, because nothing would depend on it.

**What counts as a failure is deliberately wider than "it handed out a lease".** A `leased_by` that
answers `None` because it could not read tells its caller the case is free; an `effect_for` that
answers `None` tells its caller there is no prior submission. Neither returns a lease or performs a
submission, and both are the sentence immediately before one. They are counted.

This file writes `artifacts/redis.json`, which `tests/test_kill_criteria.py` reads. Every count in
it is produced by the probes below and none is a constant.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest
import redis

from warranty_claim_recovery.queue.failure_injection import (
    INJECTIONS,
    live_redis_url,
    redis_is_reachable,
)
from warranty_claim_recovery.queue.idempotency import SubmissionGuard
from warranty_claim_recovery.queue.leases import CaseQueue, RedisUnavailableError

pytestmark = pytest.mark.infrastructure

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
ARTIFACT_NAME = "redis.json"

#: Long enough that nothing in this file expires underneath its own measurement. The lease and
#: expiry behaviour is graded in `test_leases.py`; here a lapsing lease would only be able to turn
#: a real finding into a false one.
HELD_FOR_SECONDS = 600


@pytest.fixture(scope="session")
def redis_url() -> str:
    url = live_redis_url()
    if not redis_is_reachable(url):
        pytest.fail(
            f"Redis is not reachable at {url}. Start it with `make db`. This file must not skip: "
            f"the second half of kill condition L is that the same code path works when Redis is "
            f"present, and an absent Redis makes that half unobservable rather than true."
        )
    return url


@dataclass
class Ledger:
    """Every count the artifact publishes, accumulated from probes rather than asserted."""

    attempted_without_redis: int = 0
    queue_operations_refused: int = 0
    guard_operations_refused: int = 0
    #: Any non-raising return while Redis was unavailable. Should be none: the modules fail closed.
    answers_without_redis: list[str] = field(default_factory=list)
    #: Returns that would put a second worker on a case another worker holds.
    double_leases: list[str] = field(default_factory=list)
    #: Returns that would let a caller submit, or believe it already had.
    submissions: list[str] = field(default_factory=list)
    attempted_with_redis_up: int = 0
    leases_with_redis_up: int = 0
    submissions_with_redis_up: int = 0
    modes_applied: list[str] = field(default_factory=list)


def _probe(
    ledger: Ledger,
    *,
    family: str,
    name: str,
    call: Callable[[], object],
    enables_breach: Callable[[object], bool],
) -> None:
    """Attempt one operation under an outage and record what it did.

    A refusal is the only correct outcome. Anything else is recorded verbatim, including the value
    returned, because a finding whose report says only "an operation did not raise" cannot be acted
    on and would be argued about rather than fixed.
    """
    ledger.attempted_without_redis += 1
    try:
        returned = call()
    except RedisUnavailableError:
        if family == "queue":
            ledger.queue_operations_refused += 1
        else:
            ledger.guard_operations_refused += 1
        return

    observation = f"{name} returned {returned!r} while Redis was unavailable"
    ledger.answers_without_redis.append(observation)
    if enables_breach(returned):
        if family == "queue":
            ledger.double_leases.append(observation)
        else:
            ledger.submissions.append(observation)


#: What a non-raising return would have meant. Named functions rather than inline lambdas because
#: each one is a judgement about the domain -- "this answer is the sentence before a duplicate
#: submission" -- and a judgement worth making is worth being able to point at in a review.
def _answered_anything(_returned: object) -> bool:
    return True


def _answered_none(returned: object) -> bool:
    return returned is None


def _answered_not_none(returned: object) -> bool:
    return returned is not None


def _answered_true(returned: object) -> bool:
    return returned is True


def _loses_work_rather_than_duplicating_it(_returned: object) -> bool:
    return False


CASE_HELD = "CASE-HELD-0001"
IDENTITY_HELD = "CLM-000900:PN-88-4471:SER-0001"
IDENTITY_FRESH = "CLM-000901:PN-88-4471:SER-0002"
EFFECT_HELD = "SUBMISSION-held-0001"


def _probe_the_queue(ledger: Ledger, *, mode: str, queue: CaseQueue) -> None:
    """Four queue operations a worker performs, all against a case another worker is holding."""
    _probe(
        ledger,
        family="queue",
        name=f"{mode}/CaseQueue.lease",
        call=lambda: queue.lease("worker-blind", ttl_seconds=HELD_FOR_SECONDS),
        # A case id here is a second worker on a case worker-live is holding.
        enables_breach=_answered_not_none,
    )
    _probe(
        ledger,
        family="queue",
        name=f"{mode}/CaseQueue.leased_by",
        call=lambda: queue.leased_by(CASE_HELD),
        # None reads as "nobody holds this", and the case is in fact held.
        enables_breach=_answered_none,
    )
    _probe(
        ledger,
        family="queue",
        name=f"{mode}/CaseQueue.renew",
        call=lambda: queue.renew(CASE_HELD, "worker-blind"),
        # True tells a worker that does not hold the lease that it does.
        enables_breach=_answered_true,
    )
    _probe(
        ledger,
        family="queue",
        name=f"{mode}/CaseQueue.enqueue",
        call=lambda: queue.enqueue("CASE-HELD-0002"),
        # Returning normally claims a case was queued when none was. That loses work rather than
        # duplicating it, so it breaches the fail-closed rule without being a double lease, and
        # counting it as one would overstate the finding.
        enables_breach=_loses_work_rather_than_duplicating_it,
    )


def _probe_the_guard(ledger: Ledger, *, mode: str, guard: SubmissionGuard) -> None:
    """Four guard operations, against an identity that has already produced an effect."""
    _probe(
        ledger,
        family="guard",
        name=f"{mode}/SubmissionGuard.claim_once(claimed)",
        call=lambda: guard.claim_once(IDENTITY_HELD),
        # True authorises a submission for an identity that has already been submitted.
        enables_breach=_answered_true,
    )
    _probe(
        ledger,
        family="guard",
        name=f"{mode}/SubmissionGuard.claim_once(fresh)",
        call=lambda: guard.claim_once(IDENTITY_FRESH),
        # True authorises a submission the guard has not recorded and cannot suppress a repeat of.
        enables_breach=_answered_true,
    )
    _probe(
        ledger,
        family="guard",
        name=f"{mode}/SubmissionGuard.effect_for",
        call=lambda: guard.effect_for(IDENTITY_HELD),
        # None reads as "no prior effect, go ahead"; there is a prior effect.
        enables_breach=_answered_none,
    )
    _probe(
        ledger,
        family="guard",
        name=f"{mode}/SubmissionGuard.record_effect",
        call=lambda: guard.record_effect(IDENTITY_FRESH, "SUBMISSION-blind"),
        # Returning normally tells the caller its effect is written down and will not be repeated,
        # which is precisely the promise the guard can no longer keep.
        enables_breach=_answered_anything,
    )


def observe(redis_url: str) -> Ledger:
    """Run every probe under every injection, then the same operations with Redis present.

    One function rather than one test per probe, so that any subset of the tests below sees the
    whole measurement. A ledger assembled across tests would be partial whenever somebody ran a
    single test by name, and a partial ledger written into `artifacts/redis.json` would grade kill
    condition L over a population nobody chose.
    """
    ledger = Ledger()

    for injection in INJECTIONS:
        ledger.modes_applied.append(injection.name)
        namespace = f"wcr-test-failure-{uuid.uuid4().hex[:12]}"

        # The world as a healthy worker left it: one case leased, one identity claimed and its
        # effect recorded. Everything is probed against this state, so a permissive answer under
        # the outage is a demonstrably wrong answer rather than merely an unchecked one.
        live_queue = CaseQueue(redis_url, namespace=namespace)
        live_guard = SubmissionGuard(redis_url, namespace=namespace)
        try:
            live_queue.enqueue(CASE_HELD)
            assert live_queue.lease("worker-live", ttl_seconds=HELD_FOR_SECONDS) == CASE_HELD
            assert live_guard.claim_once(IDENTITY_HELD, ttl_seconds=HELD_FOR_SECONDS) is True
            live_guard.record_effect(IDENTITY_HELD, EFFECT_HELD)

            with injection.apply(redis_url) as injected_url:
                blind_queue = CaseQueue(injected_url, namespace=namespace)
                blind_guard = SubmissionGuard(injected_url, namespace=namespace)
                try:
                    _probe_the_queue(ledger, mode=injection.name, queue=blind_queue)
                    _probe_the_guard(ledger, mode=injection.name, guard=blind_guard)
                finally:
                    blind_queue.close()
                    blind_guard.close()

            # The outage changed nothing, which is the other half of failing closed: a system that
            # refused correctly but corrupted the state on its way out would still lose the case.
            assert live_queue.leased_by(CASE_HELD) == "worker-live"
            assert live_guard.effect_for(IDENTITY_HELD) == EFFECT_HELD
        finally:
            live_queue.close()
            live_guard.close()
            _purge(redis_url, namespace)

    _observe_with_redis_up(redis_url, ledger)
    return ledger


def _observe_with_redis_up(redis_url: str, ledger: Ledger) -> None:
    """The contrast. Same classes, same methods, Redis present — and now they do the work.

    Without this half the artifact would be satisfied by a queue that refused everything always,
    which is the degenerate way to pass kill condition L and would leave the durability story with
    no queue at all.
    """
    namespace = f"wcr-test-failure-up-{uuid.uuid4().hex[:12]}"
    case_id = "CASE-UP-0001"
    identity = "CLM-000902:PN-88-4471:SER-0003"
    effect_id = "SUBMISSION-up-0001"

    queue = CaseQueue(redis_url, namespace=namespace)
    guard = SubmissionGuard(redis_url, namespace=namespace)
    try:
        queue.enqueue(case_id)
        ledger.attempted_with_redis_up += 1

        leased = queue.lease("worker-up", ttl_seconds=HELD_FOR_SECONDS)
        ledger.attempted_with_redis_up += 1
        if leased == case_id:
            ledger.leases_with_redis_up += 1

        ledger.attempted_with_redis_up += 1
        assert queue.leased_by(case_id) == "worker-up"

        ledger.attempted_with_redis_up += 1
        if guard.claim_once(identity, ttl_seconds=HELD_FOR_SECONDS) is True:
            ledger.submissions_with_redis_up += 1

        guard.record_effect(identity, effect_id)
        ledger.attempted_with_redis_up += 1
        assert guard.effect_for(identity) == effect_id

        queue.release(case_id, "worker-up")
        ledger.attempted_with_redis_up += 1
        assert queue.leased_by(case_id) is None
    finally:
        queue.close()
        guard.close()
        _purge(redis_url, namespace)


def _purge(redis_url: str, namespace: str) -> None:
    client: Any = redis.Redis.from_url(redis_url, decode_responses=True)
    try:
        for key in client.scan_iter(match=f"{namespace}:*", count=500):
            client.delete(key)
    finally:
        client.close()


def build_artifact(ledger: Ledger, redis_url: str) -> dict[str, Any]:
    """Assemble the evidence, deriving the load-bearing flag rather than declaring it."""
    endpoint = urlsplit(redis_url)
    load_bearing = (
        ledger.leases_with_redis_up > 0
        and ledger.submissions_with_redis_up > 0
        and not ledger.double_leases
        and not ledger.submissions
    )
    return {
        "criterion": "L",
        "statement": (
            "with Redis unavailable, a case is leased twice or a submission proceeds -- "
            "0 double-leases, 0 submissions"
        ),
        "injections": ledger.attempted_without_redis,
        "injection_modes": [
            {"name": injection.name, "description": injection.description}
            for injection in INJECTIONS
        ],
        "injection_modes_applied": ledger.modes_applied,
        "operations_probed_per_mode": (
            ledger.attempted_without_redis // len(ledger.modes_applied)
            if ledger.modes_applied
            else 0
        ),
        "queue_operations_refused_without_redis": ledger.queue_operations_refused,
        "guard_operations_refused_without_redis": ledger.guard_operations_refused,
        "answers_without_redis": len(ledger.answers_without_redis),
        "answers_without_redis_detail": ledger.answers_without_redis,
        "double_leases_without_redis": len(ledger.double_leases),
        "double_leases_detail": ledger.double_leases,
        "submissions_without_redis": len(ledger.submissions),
        "submissions_detail": ledger.submissions,
        "operations_with_redis_up": ledger.attempted_with_redis_up,
        "leases_with_redis_up": ledger.leases_with_redis_up,
        "submissions_with_redis_up": ledger.submissions_with_redis_up,
        "queue_is_load_bearing": load_bearing,
        "load_bearing_derivation": (
            "leases_with_redis_up > 0 and submissions_with_redis_up > 0 and "
            "double_leases_without_redis == 0 and submissions_without_redis == 0"
        ),
        "redis_endpoint": f"{endpoint.hostname}:{endpoint.port}",
        "graded_by": "tests/test_redis_failure.py",
        "note": (
            "Every count above comes from calling the shipped CaseQueue and SubmissionGuard while "
            "Redis was genuinely unreachable. No method on either class is patched; see "
            "src/warranty_claim_recovery/queue/failure_injection.py."
        ),
    }


@pytest.fixture(scope="module")
def ledger(redis_url: str) -> Ledger:
    return observe(redis_url)


# ------------------------------------------------------------------------------------------------
# The criterion, in the two halves it is made of.
# ------------------------------------------------------------------------------------------------


def test_every_operation_refuses_while_redis_is_unavailable(ledger: Ledger) -> None:
    assert ledger.attempted_without_redis > 0, "no operation was probed under an outage"
    assert ledger.answers_without_redis == [], (
        "an operation answered while Redis was unavailable. Every answer given without a check is "
        "an answer the caller will act on:\n  " + "\n  ".join(ledger.answers_without_redis)
    )
    refused = ledger.queue_operations_refused + ledger.guard_operations_refused
    assert refused == ledger.attempted_without_redis
    print(
        f"failure injection: {len(ledger.modes_applied)} modes, "
        f"{ledger.attempted_without_redis} operations attempted with Redis gone, "
        f"{refused} refused, {len(ledger.answers_without_redis)} answered"
    )


def test_no_case_is_leased_twice_while_redis_is_unavailable(ledger: Ledger) -> None:
    assert ledger.double_leases == [], "\n  ".join(ledger.double_leases)


def test_no_submission_proceeds_while_redis_is_unavailable(ledger: Ledger) -> None:
    assert ledger.submissions == [], "\n  ".join(ledger.submissions)


def test_the_same_code_path_leases_and_submits_when_redis_is_up(ledger: Ledger) -> None:
    """The half that stops the criterion being satisfied by a queue that refuses everything."""
    assert ledger.leases_with_redis_up > 0
    assert ledger.submissions_with_redis_up > 0
    print(
        f"with Redis up: {ledger.attempted_with_redis_up} operations, "
        f"{ledger.leases_with_redis_up} case leased, "
        f"{ledger.submissions_with_redis_up} submission claimed"
    )


# ------------------------------------------------------------------------------------------------
# The same two refusals again, per injection mode and independently of the ledger. If the loop
# above were ever quietly reduced to one mode, these would still fail.
# ------------------------------------------------------------------------------------------------


@pytest.fixture
def namespace(redis_url: str) -> Iterator[str]:
    prefix = f"wcr-test-failure-direct-{uuid.uuid4().hex[:12]}"
    yield prefix
    _purge(redis_url, prefix)


@pytest.mark.parametrize("injection", INJECTIONS, ids=[i.name for i in INJECTIONS])
def test_lease_raises_under_each_injection(redis_url: str, namespace: str, injection: Any) -> None:
    CaseQueue(redis_url, namespace=namespace).enqueue("CASE-DIRECT-0001")
    with injection.apply(redis_url) as injected_url:
        queue = CaseQueue(injected_url, namespace=namespace)
        with pytest.raises(RedisUnavailableError, match="lease could not reach Redis"):
            queue.lease("worker-blind", ttl_seconds=HELD_FOR_SECONDS)


@pytest.mark.parametrize("injection", INJECTIONS, ids=[i.name for i in INJECTIONS])
def test_claim_once_raises_under_each_injection(
    redis_url: str, namespace: str, injection: Any
) -> None:
    with injection.apply(redis_url) as injected_url:
        guard = SubmissionGuard(injected_url, namespace=namespace)
        with pytest.raises(RedisUnavailableError, match="claim_once could not reach Redis"):
            guard.claim_once("CLM-000903:PN-88-4471:SER-0004")


def test_redis_is_reachable_again_after_every_injection(redis_url: str) -> None:
    """The severed-client injection patches a class attribute; it has to put it back.

    A failure here would mean the injection leaked into the rest of the session, and every later
    test in the run would be measuring an outage nobody asked for.
    """
    assert redis_is_reachable(redis_url) is True


# ------------------------------------------------------------------------------------------------
# The artifact kill condition L is graded from.
# ------------------------------------------------------------------------------------------------


def test_the_evidence_is_written_where_the_kill_test_reads_it(
    ledger: Ledger, redis_url: str
) -> None:
    artifact = build_artifact(ledger, redis_url)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS / ARTIFACT_NAME
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    written = json.loads(path.read_text(encoding="utf-8"))

    # Asserted here as well as in the kill test, because a lane that wrote an artifact the kill
    # test could not read would fail with "redis.json is missing a key" three steps downstream.
    assert written["injections"] >= 1
    assert written["double_leases_without_redis"] == 0
    assert written["submissions_without_redis"] == 0
    assert written["queue_is_load_bearing"] is True
    print(f"wrote {path} with injections={written['injections']}")
