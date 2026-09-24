"""The case queue and its leases, exercised against a real Redis.

These tests are not unit tests and are not meant to be. The property under examination — two
workers never hold one case — is a property of Redis executing a script, and a test that replaced
Redis would be asserting that the replacement behaves the way the author expected Redis to. Kill
condition L would then be graded against an assumption. So every test here talks to the Redis the
system is configured to use, and the suite **fails** rather than skips when that Redis is absent:
a skip reads as a pass in a summary line, and the one thing worse than an untested lease is an
untested lease that a green build says is fine.

The concurrency tests use threads and a barrier, not a loop. A loop calling `lease` sixteen times
exercises the same code sixteen times in the same order and would pass against a read-then-write
implementation, which is the implementation these tests exist to rule out. Sixteen is the floor
`tests/test_kill_criteria.py` fixes for kill condition E; thirty-two are used here because the
margin costs nothing and the interleaving is more interesting.
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
from warranty_claim_recovery.queue.leases import CaseQueue

pytestmark = pytest.mark.infrastructure

#: Comfortably above the sixteen that kill condition E fixes as its floor.
CONTENDING_WORKERS = 32

#: How long a test is prepared to wait for a one-second lease to lapse. Generous because the clock
#: that matters is the Redis server's, and a laptop that suspends mid-suite resumes with the lease
#: already long gone rather than with a test that has silently measured nothing.
EXPIRY_DEADLINE_SECONDS = 30.0


@pytest.fixture(scope="session")
def redis_url() -> str:
    url = live_redis_url()
    if not redis_is_reachable(url):
        pytest.fail(
            f"Redis is not reachable at {url}. Start it with `make db`. These tests fail rather "
            f"than skip: the lease guarantee is the one this project is not allowed to assume."
        )
    return url


@pytest.fixture
def namespace(redis_url: str) -> Iterator[str]:
    """A key prefix nothing else in this Redis uses, removed again when the test ends.

    Per test rather than per session so that one test's queue cannot be another's, and removed
    afterwards because a developer's Redis is shared with a console they may have left open — a
    suite that leaves several hundred stale case ids in the ready list makes the console lie.
    """
    prefix = f"wcr-test-leases-{uuid.uuid4().hex[:12]}"
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

    The barrier is the point. Without it the threads start as the loop creates them, the first is
    finished before the last exists, and nothing has contended for anything. Failures are collected
    rather than left to die inside their thread, because an exception in a worker thread is
    invisible to pytest and the test would pass with fifteen of its sixteen attempts missing.
    """
    barrier = threading.Barrier(count, timeout=EXPIRY_DEADLINE_SECONDS)
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
        threading.Thread(target=run, args=(index,), name=f"contender-{index:02d}")
        for index in range(count)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=EXPIRY_DEADLINE_SECONDS * 2)
    still_running = [thread.name for thread in threads if thread.is_alive()]
    assert not still_running, f"threads never finished: {still_running}"
    return results, failures


def wait_until(condition: Callable[[], bool], *, what: str) -> float:
    """Poll until `condition` holds, and return how long it took.

    Polling rather than sleeping for a fixed interval: the lease expires on the Redis server's
    clock, and a fixed sleep either wastes the difference or, on a machine that suspends, wakes up
    after the window it meant to observe. The deadline is an assertion, so a lease that never
    expires fails the test instead of hanging the suite.
    """
    started = time.monotonic()
    while time.monotonic() - started < EXPIRY_DEADLINE_SECONDS:
        if condition():
            return time.monotonic() - started
        time.sleep(0.05)
    raise AssertionError(f"{what} did not happen within {EXPIRY_DEADLINE_SECONDS}s")


# ------------------------------------------------------------------------------------------------
# One case, one holder.
# ------------------------------------------------------------------------------------------------


def test_a_queued_case_is_leased_by_the_first_worker_and_refused_to_the_second(
    redis_url: str, namespace: str
) -> None:
    case_id = "CASE-0001"
    CaseQueue(redis_url, namespace=namespace).enqueue(case_id)

    first = CaseQueue(redis_url, namespace=namespace)
    second = CaseQueue(redis_url, namespace=namespace)

    assert first.lease("worker-a", ttl_seconds=60) == case_id
    assert second.lease("worker-b", ttl_seconds=60) is None
    assert first.leased_by(case_id) == "worker-a"


def test_thirty_two_workers_racing_for_one_case_produce_exactly_one_holder(
    redis_url: str, namespace: str
) -> None:
    """The kill-condition-E shape, applied to the lease rather than to the submission.

    Two workers on one case is how one recovery identity comes to be submitted twice in the first
    place, so the lease is graded with the same rigour as the submission guard.
    """
    case_id = "CASE-RACE-0001"
    CaseQueue(redis_url, namespace=namespace).enqueue(case_id)

    def contend(index: int) -> str | None:
        queue = CaseQueue(redis_url, namespace=namespace)
        try:
            return queue.lease(f"worker-{index:02d}", ttl_seconds=120)
        finally:
            queue.close()

    results, failures = run_concurrently(CONTENDING_WORKERS, contend)
    assert not failures, f"contending workers raised: {failures}"

    winners = [(index, value) for index, value in enumerate(results) if value is not None]
    print(
        f"lease race: {CONTENDING_WORKERS} threads, "
        f"{len(winners)} acquired the case, {CONTENDING_WORKERS - len(winners)} were refused"
    )
    assert len(winners) == 1, f"{len(winners)} workers hold one case: {winners}"
    assert winners[0][1] == case_id

    holder = CaseQueue(redis_url, namespace=namespace).leased_by(case_id)
    assert holder == f"worker-{winners[0][0]:02d}"


def test_cases_are_leased_in_the_order_they_were_enqueued(redis_url: str, namespace: str) -> None:
    queue = CaseQueue(redis_url, namespace=namespace)
    queued = ("CASE-0001", "CASE-0002", "CASE-0003")
    for case_id in queued:
        queue.enqueue(case_id)

    leased = tuple(queue.lease(f"worker-{index}", ttl_seconds=60) for index in range(len(queued)))
    assert leased == queued
    assert queue.lease("worker-late", ttl_seconds=60) is None


def test_enqueueing_the_same_case_twice_queues_it_once(redis_url: str, namespace: str) -> None:
    """A retried enqueue must not put a second copy in the list; see `CaseQueue.enqueue`."""
    queue = CaseQueue(redis_url, namespace=namespace)
    queue.enqueue("CASE-0001")
    queue.enqueue("CASE-0001")
    queue.enqueue("CASE-0002")
    assert queue.pending_case_ids() == ("CASE-0001", "CASE-0002")


def test_an_empty_queue_leases_nothing(redis_url: str, namespace: str) -> None:
    assert CaseQueue(redis_url, namespace=namespace).lease("worker-a", ttl_seconds=60) is None


# ------------------------------------------------------------------------------------------------
# Expiry: the reason this is a lease and not a lock.
# ------------------------------------------------------------------------------------------------


def test_a_dead_workers_lease_expires_and_the_next_worker_takes_the_case(
    redis_url: str, namespace: str
) -> None:
    """The crash-recovery path, with the crash represented by simply never renewing.

    No reaper runs and nothing sweeps: the case stayed in the ready list the whole time and its
    lease key was the only thing excluding anyone. When the key goes, the case is available again.
    """
    case_id = "CASE-0001"
    dead = CaseQueue(redis_url, namespace=namespace)
    dead.enqueue(case_id)
    assert dead.lease("worker-dead", ttl_seconds=1) == case_id

    survivor = CaseQueue(redis_url, namespace=namespace)
    assert survivor.lease("worker-alive", ttl_seconds=60) is None

    elapsed = wait_until(
        lambda: survivor.leased_by(case_id) is None, what="the lease on a dead worker's case"
    )
    print(f"lease expiry observed after {elapsed:.2f}s for a 1s lease")

    assert survivor.lease("worker-alive", ttl_seconds=60) == case_id
    assert survivor.leased_by(case_id) == "worker-alive"


def test_renew_extends_the_holders_lease_and_refuses_everyone_else(
    redis_url: str, namespace: str
) -> None:
    """Asserted against the key's remaining time to live rather than by waiting.

    Waiting would measure the same thing and would take as long as the lease, which is the reason
    people write lease tests with one-second timeouts and then cannot tell a renewal that worked
    from a scheduler that was slow. The key name is public on `CaseQueue` precisely so an operator
    — and this test — can read the remaining time without guessing at it.
    """
    case_id = "CASE-0001"
    holder = CaseQueue(redis_url, namespace=namespace)
    holder.enqueue(case_id)
    assert holder.lease("worker-a", ttl_seconds=30) == case_id

    client: Any = redis.Redis.from_url(redis_url, decode_responses=True)
    try:
        assert 0 < int(client.ttl(holder.lease_key(case_id))) <= 30

        assert holder.renew(case_id, "worker-b") is False
        assert int(client.ttl(holder.lease_key(case_id))) <= 30

        assert holder.renew(case_id, "worker-a", ttl_seconds=300) is True
        assert int(client.ttl(holder.lease_key(case_id))) > 30
    finally:
        client.close()

    assert holder.leased_by(case_id) == "worker-a"


def test_renewing_a_lease_that_has_already_gone_fails_rather_than_recreating_it(
    redis_url: str, namespace: str
) -> None:
    """A renewal is an extension, never a creation.

    If `renew` recreated an expired key the slow worker would silently take back a case another
    worker is already running, which is the double lease arriving through the one method that
    looks harmless.
    """
    case_id = "CASE-0001"
    queue = CaseQueue(redis_url, namespace=namespace)
    queue.enqueue(case_id)
    assert queue.lease("worker-a", ttl_seconds=1) == case_id

    wait_until(lambda: queue.leased_by(case_id) is None, what="the lease lapsing")

    assert queue.renew(case_id, "worker-a", ttl_seconds=300) is False
    assert queue.leased_by(case_id) is None


# ------------------------------------------------------------------------------------------------
# Release: only the holder, and only the holder's case.
# ------------------------------------------------------------------------------------------------


def test_release_by_the_holder_frees_the_case_and_takes_it_out_of_the_queue(
    redis_url: str, namespace: str
) -> None:
    queue = CaseQueue(redis_url, namespace=namespace)
    queue.enqueue("CASE-0001")
    queue.enqueue("CASE-0002")
    assert queue.lease("worker-a", ttl_seconds=60) == "CASE-0001"

    queue.release("CASE-0001", "worker-a")

    assert queue.leased_by("CASE-0001") is None
    assert queue.pending_case_ids() == ("CASE-0002",)


def test_release_by_a_worker_that_does_not_hold_the_lease_changes_nothing(
    redis_url: str, namespace: str
) -> None:
    queue = CaseQueue(redis_url, namespace=namespace)
    queue.enqueue("CASE-0001")
    assert queue.lease("worker-a", ttl_seconds=60) == "CASE-0001"

    queue.release("CASE-0001", "worker-b")

    assert queue.leased_by("CASE-0001") == "worker-a"
    assert queue.pending_case_ids() == ("CASE-0001",)


def test_a_slow_worker_releasing_after_its_lease_lapsed_does_not_disturb_the_new_holder(
    redis_url: str, namespace: str
) -> None:
    """The case `_RELEASE_SCRIPT`'s guard exists for, and the one it is easiest to get wrong.

    An unconditional release here would delete the new holder's lease and remove from the queue a
    case that is actively being worked, so a crash of the new holder would strand it with nothing
    left to pick it up — a case silently written off while its correction window closed.
    """
    case_id = "CASE-0001"
    slow = CaseQueue(redis_url, namespace=namespace)
    slow.enqueue(case_id)
    assert slow.lease("worker-slow", ttl_seconds=1) == case_id

    wait_until(lambda: slow.leased_by(case_id) is None, what="the slow worker's lease lapsing")

    successor = CaseQueue(redis_url, namespace=namespace)
    assert successor.lease("worker-successor", ttl_seconds=300) == case_id

    slow.release(case_id, "worker-slow")

    assert successor.leased_by(case_id) == "worker-successor"
    assert successor.pending_case_ids() == (case_id,)


# ------------------------------------------------------------------------------------------------
# Arguments that would make a key mean nothing.
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("case_id", ["", "   ", " CASE-0001", "CASE-0001 "])
def test_a_blank_or_padded_case_id_is_refused(redis_url: str, namespace: str, case_id: str) -> None:
    with pytest.raises(ValueError, match="not usable as a key component"):
        CaseQueue(redis_url, namespace=namespace).enqueue(case_id)


@pytest.mark.parametrize("ttl", [0, -1, True])
def test_a_lease_with_no_lifetime_is_refused(
    redis_url: str, namespace: str, ttl: int | bool
) -> None:
    with pytest.raises(ValueError, match="positive whole number of seconds"):
        CaseQueue(redis_url, namespace=namespace).lease("worker-a", ttl_seconds=ttl)
