"""The submission guard: one recovery identity, one effect, under real concurrency.

Kill condition E fixes exactly one effect per identity over at least sixteen concurrent attempts.
That number is in the criterion because a duplicate submission is not a rare event in this domain —
it is what happens every time a worker is killed while the manufacturer's portal is thinking, which
is the scenario the whole project is built around. So the race here is a real race: thirty-two
operating-system threads, each with its own client and its own connection pool, released together
by a barrier.

**A loop would pass against a broken implementation.** `for _ in range(32): guard.claim_once(id)`
returns one `True` and thirty-one `False` even if `claim_once` were a `GET` followed by a `SET`,
because nothing ever interleaves. That implementation is the one these tests exist to rule out, and
only genuine concurrency rules it out.

The guard is exercised against the configured Redis and the suite **fails** rather than skips when
that Redis is absent. A skipped idempotency test reads in a summary as a passing one.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable, Iterator
from typing import Any

import pytest
import redis

from warranty_claim_recovery.queue.failure_injection import live_redis_url, redis_is_reachable
from warranty_claim_recovery.queue.idempotency import (
    PENDING_SENTINEL,
    ConflictingSubmissionEffectError,
    SubmissionClaimExpiredError,
    SubmissionGuard,
)

pytestmark = pytest.mark.infrastructure

#: Double the floor kill condition E fixes. The margin costs a few milliseconds and buys a wider
#: interleaving; the floor itself is in `tests/test_kill_criteria.py` and is not this file's to set.
CONTENDING_CALLERS = 32

#: The ceiling on every wait in this file. An assertion rather than a sleep, so a claim that never
#: expires fails a test instead of hanging the suite on a machine that has suspended.
DEADLINE_SECONDS = 30.0


@pytest.fixture(scope="session")
def redis_url() -> str:
    url = live_redis_url()
    if not redis_is_reachable(url):
        pytest.fail(
            f"Redis is not reachable at {url}. Start it with `make db`. These tests fail rather "
            f"than skip: 'nothing leaves twice' is not a claim this project may assume."
        )
    return url


@pytest.fixture
def namespace(redis_url: str) -> Iterator[str]:
    """A key prefix nothing else uses, removed when the test ends.

    The claim TTL is a day by default, so a suite that did not clean up would leave every identity
    it had ever raced for blocked until tomorrow — and the next run, using the same identities,
    would find them all claimed and would measure nothing.
    """
    prefix = f"wcr-test-idem-{uuid.uuid4().hex[:12]}"
    yield prefix
    client: Any = redis.Redis.from_url(redis_url, decode_responses=True)
    try:
        for key in client.scan_iter(match=f"{prefix}:*", count=500):
            client.delete(key)
    finally:
        client.close()


def run_concurrently[T](
    count: int, target: Callable[[int], T]
) -> tuple[list[T | None], list[BaseException]]:
    """Run `target` on `count` threads released together, and return results and failures.

    The barrier is what makes this a race. Failures are collected and returned rather than left to
    die inside their thread: an exception in a worker thread is invisible to pytest, and a test
    whose thirty-one losers all crashed would still see one winner and pass.
    """
    barrier = threading.Barrier(count, timeout=DEADLINE_SECONDS)
    results: list[T | None] = [None] * count
    failures: list[BaseException] = []
    lock = threading.Lock()

    def run(index: int) -> None:
        try:
            barrier.wait()
            results[index] = target(index)
        except BaseException as error:  # re-raised on the main thread; see the docstring
            with lock:
                failures.append(error)

    threads = [
        threading.Thread(target=run, args=(index,), name=f"caller-{index:02d}")
        for index in range(count)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=DEADLINE_SECONDS * 2)
    still_running = [thread.name for thread in threads if thread.is_alive()]
    assert not still_running, f"threads never finished: {still_running}"
    return results, failures


def wait_until(condition: Callable[[], bool], *, what: str) -> float:
    started = time.monotonic()
    while time.monotonic() - started < DEADLINE_SECONDS:
        if condition():
            return time.monotonic() - started
        time.sleep(0.05)
    raise AssertionError(f"{what} did not happen within {DEADLINE_SECONDS}s")


# ------------------------------------------------------------------------------------------------
# The race. Kill condition E, at the level of the guard itself.
# ------------------------------------------------------------------------------------------------


def test_thirty_two_concurrent_callers_produce_exactly_one_claim(
    redis_url: str, namespace: str
) -> None:
    identity = "CLM-000123:PN-88-4471:SER-0099"

    def contend(index: int) -> bool:
        guard = SubmissionGuard(redis_url, namespace=namespace)
        try:
            return guard.claim_once(identity, ttl_seconds=300)
        finally:
            guard.close()

    results, failures = run_concurrently(CONTENDING_CALLERS, contend)
    assert not failures, f"contending callers raised: {failures}"

    winners = [index for index, won in enumerate(results) if won is True]
    losers = [index for index, won in enumerate(results) if won is False]
    print(
        f"submission race: {CONTENDING_CALLERS} threads, {len(winners)} claimed the identity, "
        f"{len(losers)} were refused"
    )
    assert len(winners) == 1, f"{len(winners)} callers each believe they may submit: {winners}"
    assert len(losers) == CONTENDING_CALLERS - 1


def test_the_claim_stays_won_after_the_race_has_finished(redis_url: str, namespace: str) -> None:
    """`False` for every other caller, *forever* within the TTL — not only during the race.

    A guard that only held for the duration of the contention would let the retry that arrives a
    second later through, and a retry a second later is the ordinary case: it is what a worker does
    when the portal's response is slow.
    """
    identity = "CLM-000124:PN-88-4471:SER-0100"
    guard = SubmissionGuard(redis_url, namespace=namespace)
    assert guard.claim_once(identity, ttl_seconds=300) is True

    for _ in range(5):
        assert SubmissionGuard(redis_url, namespace=namespace).claim_once(identity) is False


def test_concurrent_callers_for_different_identities_all_win(
    redis_url: str, namespace: str
) -> None:
    """The guard must block a duplicate, not the throughput.

    A guard keyed too coarsely — on the claim id alone, say — would serialise unrelated recoveries
    and would look identical in the single-identity test above. `Claim.recovery_identity` explains
    why one claim can legitimately produce several recoveries.
    """

    def contend(index: int) -> bool:
        guard = SubmissionGuard(redis_url, namespace=namespace)
        try:
            return guard.claim_once(f"CLM-000200:PN-88-4471:SER-{index:04d}", ttl_seconds=300)
        finally:
            guard.close()

    results, failures = run_concurrently(CONTENDING_CALLERS, contend)
    assert not failures, f"contending callers raised: {failures}"
    assert results.count(True) == CONTENDING_CALLERS


# ------------------------------------------------------------------------------------------------
# The losing caller gets the winner's answer, which is what "idempotent" means.
# ------------------------------------------------------------------------------------------------


def test_a_losing_caller_reads_the_winners_effect_rather_than_an_error(
    redis_url: str, namespace: str
) -> None:
    identity = "CLM-000125:PN-88-4471:SER-0101"
    effect_id = "SUBMISSION-9f3a12"

    winner = SubmissionGuard(redis_url, namespace=namespace)
    assert winner.claim_once(identity, ttl_seconds=300) is True
    winner.record_effect(identity, effect_id)

    loser = SubmissionGuard(redis_url, namespace=namespace)
    assert loser.claim_once(identity, ttl_seconds=300) is False
    assert loser.effect_for(identity) == effect_id


def test_there_is_no_effect_to_report_before_the_winner_records_one(
    redis_url: str, namespace: str
) -> None:
    """In flight and never claimed both read as `None`, and neither authorises a submission.

    The distinction is deliberately not offered: a caller that could tell them apart would be
    tempted to act on "never claimed", and `claim_once` is the only thing entitled to say that.
    """
    identity = "CLM-000126:PN-88-4471:SER-0102"
    guard = SubmissionGuard(redis_url, namespace=namespace)

    assert guard.effect_for(identity) is None
    assert guard.claim_once(identity, ttl_seconds=300) is True
    assert guard.effect_for(identity) is None


def test_the_in_flight_sentinel_is_never_reported_as_an_effect(
    redis_url: str, namespace: str
) -> None:
    identity = "CLM-000127:PN-88-4471:SER-0103"
    guard = SubmissionGuard(redis_url, namespace=namespace)
    assert guard.claim_once(identity, ttl_seconds=300) is True

    client: Any = redis.Redis.from_url(redis_url, decode_responses=True)
    try:
        assert client.get(guard.key_for(identity)) == PENDING_SENTINEL
    finally:
        client.close()

    assert guard.effect_for(identity) is None


# ------------------------------------------------------------------------------------------------
# Recording an effect: the retry, the duplicate and the lapse.
# ------------------------------------------------------------------------------------------------


def test_recording_the_same_effect_twice_is_accepted(redis_url: str, namespace: str) -> None:
    """A caller retrying a failed acknowledgement wants an outcome that is already true."""
    identity = "CLM-000128:PN-88-4471:SER-0104"
    guard = SubmissionGuard(redis_url, namespace=namespace)
    assert guard.claim_once(identity, ttl_seconds=300) is True

    guard.record_effect(identity, "SUBMISSION-aa01")
    guard.record_effect(identity, "SUBMISSION-aa01")

    assert guard.effect_for(identity) == "SUBMISSION-aa01"


def test_a_second_different_effect_is_refused_and_the_first_is_kept(
    redis_url: str, namespace: str
) -> None:
    """Kill condition E, observed at the moment the duplicate tries to write itself down.

    The first effect is kept deliberately: it is the one the manufacturer and the audit log already
    know about, and overwriting it would erase the evidence that a second submission happened.
    """
    identity = "CLM-000129:PN-88-4471:SER-0105"
    guard = SubmissionGuard(redis_url, namespace=namespace)
    assert guard.claim_once(identity, ttl_seconds=300) is True
    guard.record_effect(identity, "SUBMISSION-first")

    with pytest.raises(ConflictingSubmissionEffectError, match="second one"):
        guard.record_effect(identity, "SUBMISSION-second")

    assert guard.effect_for(identity) == "SUBMISSION-first"


def test_recording_an_effect_against_a_lapsed_claim_is_refused(
    redis_url: str, namespace: str
) -> None:
    """Once the claim is gone the guard can no longer say this effect was the only one."""
    identity = "CLM-000130:PN-88-4471:SER-0106"
    guard = SubmissionGuard(redis_url, namespace=namespace)
    assert guard.claim_once(identity, ttl_seconds=1) is True

    wait_until(lambda: _key_is_gone(redis_url, guard.key_for(identity)), what="the claim lapsing")

    with pytest.raises(SubmissionClaimExpiredError, match="expired or was never made"):
        guard.record_effect(identity, "SUBMISSION-late")


def test_recording_an_effect_for_an_identity_nobody_claimed_is_refused(
    redis_url: str, namespace: str
) -> None:
    guard = SubmissionGuard(redis_url, namespace=namespace)
    with pytest.raises(SubmissionClaimExpiredError):
        guard.record_effect("CLM-000131:PN-88-4471:SER-0107", "SUBMISSION-unclaimed")


def test_a_lapsed_claim_may_be_retried_by_the_next_caller(redis_url: str, namespace: str) -> None:
    """The TTL is a safety window, not a permanent record.

    A submission that crashed before recording anything must eventually become retryable, or a
    single crash writes the recovery off — which is the failure the correction window makes
    expensive. The audit log in PostgreSQL is what remembers permanently that a submission
    happened; this key only says "not right now".
    """
    identity = "CLM-000132:PN-88-4471:SER-0108"
    first = SubmissionGuard(redis_url, namespace=namespace)
    assert first.claim_once(identity, ttl_seconds=1) is True

    elapsed = wait_until(
        lambda: _key_is_gone(redis_url, first.key_for(identity)), what="the claim lapsing"
    )
    print(f"claim expiry observed after {elapsed:.2f}s for a 1s claim")

    second = SubmissionGuard(redis_url, namespace=namespace)
    assert second.claim_once(identity, ttl_seconds=300) is True


# ------------------------------------------------------------------------------------------------
# Arguments that would make a key mean nothing.
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("identity", ["", "   ", " CLM-1:PN-1:SER-1", "CLM-1:PN-1:SER-1 "])
def test_a_blank_or_padded_identity_is_refused(
    redis_url: str, namespace: str, identity: str
) -> None:
    with pytest.raises(ValueError, match="not usable as a key component"):
        SubmissionGuard(redis_url, namespace=namespace).claim_once(identity)


def test_the_reserved_sentinel_may_not_be_recorded_as_an_effect(
    redis_url: str, namespace: str
) -> None:
    identity = "CLM-000133:PN-88-4471:SER-0109"
    guard = SubmissionGuard(redis_url, namespace=namespace)
    assert guard.claim_once(identity, ttl_seconds=300) is True

    with pytest.raises(ValueError, match="reserved in-flight sentinel"):
        guard.record_effect(identity, PENDING_SENTINEL)


@pytest.mark.parametrize("ttl", [0, -1, True])
def test_a_claim_with_no_lifetime_is_refused(
    redis_url: str, namespace: str, ttl: int | bool
) -> None:
    with pytest.raises(ValueError, match="positive whole number of seconds"):
        SubmissionGuard(redis_url, namespace=namespace).claim_once(
            "CLM-000134:PN-88-4471:SER-0110", ttl_seconds=ttl
        )


def _key_is_gone(redis_url: str, key: str) -> bool:
    client: Any = redis.Redis.from_url(redis_url, decode_responses=True)
    try:
        return int(client.exists(key)) == 0
    finally:
        client.close()
