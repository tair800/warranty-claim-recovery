"""The case work queue, the leases that keep two workers off one case, and why it fails closed.

A case in this system is not a request. It is a file that stays open for days while a technician
photographs a serial plate and a customer answers an email, and the durable record of where it has
reached lives in PostgreSQL with the graph's checkpoint. This module owns something smaller and
sharper: **which worker is allowed to touch a case right now**.

Two workers running one case at the same time is not a throughput problem. Both reach the submit
node, both ask the manufacturer's portal for the same recovery, and the guarantee in ADR-001 §4.2 —
one recovery identity, at most one effect — is broken by the queue rather than by the submitter.
`idempotency.SubmissionGuard` is the last line of that defence; this module is the first, and the
project needs both because they fail in different directions.

**A lease, not a lock.** A lock that a crashed worker holds forever is a case that never moves, and
a warranty correction that never moves is money written off when the manufacturer's correction
window shuts — kill condition M, arriving through the queue. A lease expires. A worker that dies
mid-case loses its claim after `ttl_seconds`, the next worker takes the case, and the graph resumes
it from the checkpoint rather than from the beginning. The price of the lease over the lock is that
a worker which is merely *slow* can have its case taken from it, which is why `renew` exists and
why the submit path is idempotent whatever happens here.

**Every mutation is one round trip, in Lua.** Not for speed. A read-then-write — `GET` the lease,
observe that it is free, `SET` it — is a race whose window is a network round trip wide, and two
workers whose reads interleave inside that window both conclude the case is free. That is exactly
the double lease kill condition L forbids, and no amount of care at the call site closes it. `SET
NX EX` and `EVAL` close it inside the server, where the two commands cannot interleave.

**Rejected: `SETNX` followed by `EXPIRE`.** Two commands, and a worker that dies between them
leaves a lease with no expiry — a permanent lock wearing a lease's name, which is the failure the
lease was chosen to avoid. `SET key value NX EX ttl` is one command and is used everywhere here.

**Rejected: `SCAN` over the lease keys to find free work.** `SCAN` offers no ordering and no
snapshot, so the queue's order would be whatever the keyspace happened to look like at the moment
someone asked. The ready list is a Redis list and its order is the order cases were enqueued.

**Rejected: `EVALSHA` via `register_script`.** It saves sending a few hundred bytes per call at a
volume this queue will never reach, and it buys the `NOSCRIPT`-after-a-restart failure mode in
exchange. `EVAL` has neither the saving nor the failure, and a queue whose correctness depends on a
cache being warm is a queue that misbehaves precisely when Redis has just come back.

**Rejected: the queue in PostgreSQL with `SELECT ... FOR UPDATE SKIP LOCKED`.** It would work, and
it would put the durability story and the liveness story in one component, so a database failover
would stop the workers as well as freeze the cases. They are separated deliberately, and
`SKILL_MATRIX.md` gives this project the Redis and Redis-queue cells with no second home.

**Failing closed.** Every method raises `RedisUnavailableError` when Redis cannot answer, and none
returns a permissive default. A `lease` that returned a case id because it could not check would
hand one case to every worker that asked. A `leased_by` that returned `None` because it could not
read would tell its caller the case is free. Both are kill condition L, and both are the class of
defect that only shows up during the incident that caused the outage.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, suppress
from typing import Any, Final

from redis import Redis
from redis.exceptions import BusyLoadingError
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

__all__ = [
    "DEFAULT_LEASE_TTL_SECONDS",
    "DEFAULT_NAMESPACE",
    "SOCKET_TIMEOUT_SECONDS",
    "CaseQueue",
    "RedisUnavailableError",
    "fail_closed",
]

#: Every key this package writes begins with this. Tests override it so that a suite running
#: against the developer's own Redis cannot collide with a console left open in another window —
#: a test that deletes a queue someone is watching is a test nobody trusts afterwards.
DEFAULT_NAMESPACE: Final = "wcr"

#: Long enough that a node doing retrieval and composition finishes inside it, short enough that a
#: killed worker's case is picked up again within the same coffee break. Workers that expect to
#: exceed it call `renew`; they do not raise it, because a longer default would slow every recovery
#: from every crash in order to accommodate the slowest node.
DEFAULT_LEASE_TTL_SECONDS: Final = 30

#: A bounded wait on every socket operation. The default in redis-py is to block indefinitely, and
#: a worker blocked indefinitely on a Redis that has gone away is a worker that never reports the
#: outage and never fails closed — it simply stops, which looks identical to having no work.
SOCKET_TIMEOUT_SECONDS: Final = 5.0


class RedisUnavailableError(RuntimeError):
    """Redis could not answer, so the queue refused to guess.

    Raised in place of every permissive default. The alternative — returning `None` for "nobody
    holds this lease" or `False` for "this has not been submitted" — reads to the caller as a fact
    about the world rather than as an admission that nothing was checked, and the caller then acts
    on it. Under an outage every worker would receive the same encouraging answer at the same
    moment, which is how one case comes to be leased by all of them.

    A distinct type rather than re-raising redis-py's own errors, because the caller's decision is
    not "which network error was this" but "this system may not proceed". The graph's worker loop
    catches this one type, backs off and reports; it never catches it and continues.
    """


#: The errors that mean "Redis did not answer", as opposed to "Redis answered and said no".
#:
#: `redis.exceptions.ResponseError` is **deliberately absent**. A `ResponseError` is Redis rejecting
#: a command — a Lua script with a bug in it, a wrong argument count — and translating that into
#: "Redis is unavailable" would let a defect in this file present itself as an infrastructure
#: outage, which is the report that gets escalated to the wrong team and closed without a fix.
#: Builtin `OSError` is included because a severed socket surfaces as one before redis-py wraps it.
_UNAVAILABLE: Final[tuple[type[BaseException], ...]] = (
    RedisConnectionError,
    RedisTimeoutError,
    BusyLoadingError,
    OSError,
)


@contextmanager
def fail_closed(operation: str) -> Iterator[None]:
    """Translate an unreachable Redis into a refusal, naming the operation that refused.

    Shared with `idempotency`, which imports it from here rather than from a third module. The
    dependency runs one way — idempotency knows about leases, leases knows nothing about
    idempotency — and one shared translation is the point: two modules with their own idea of what
    counts as an outage would eventually disagree, and the one that was more permissive would be
    the one that mattered.
    """
    try:
        yield
    except _UNAVAILABLE as error:
        raise RedisUnavailableError(
            f"{operation} could not reach Redis ({type(error).__name__}: {error}). "
            f"The queue answers from Redis or it does not answer: a lease it cannot check is a "
            f"lease it would hand out twice."
        ) from error


# ------------------------------------------------------------------------------------------------
# The scripts. Each is one round trip and each is atomic; see the module docstring for why that is
# the whole design rather than an optimisation.
# ------------------------------------------------------------------------------------------------

#: Append a case to the ready list unless it is already waiting there. `LPOS` costs a scan of a
#: list that holds open cases rather than events, so it is short; the alternative — a companion set
#: for membership — is a second key that can disagree with the list, and a queue whose two keys
#: disagree loses cases in whichever direction the disagreement runs.
_ENQUEUE_SCRIPT: Final = """
if redis.call('LPOS', KEYS[1], ARGV[1]) then
  return 0
end
redis.call('RPUSH', KEYS[1], ARGV[1])
return 1
"""

#: Walk the ready list from the head, moving each case to the tail as it is examined, and return
#: the first case this worker can actually lock. Cases stay in the ready list while they are
#: leased: their lease key is what excludes other workers, so an expired lease needs no reaper and
#: no in-flight bookkeeping to make a dead worker's case available again. It simply is available
#: again, the next time anyone asks.
#:
#: The loop is bounded by the list length at entry, so a queue in which every case is already
#: leased costs one pass and returns nothing rather than spinning. A full pass is a rotation of the
#: list by its own length, which is the identity — order is preserved when nothing is taken.
_LEASE_SCRIPT: Final = """
local ready = KEYS[1]
local prefix = ARGV[1]
local worker = ARGV[2]
local ttl = tonumber(ARGV[3])
local size = redis.call('LLEN', ready)
for _ = 1, size do
  local case_id = redis.call('LMOVE', ready, ready, 'LEFT', 'RIGHT')
  if not case_id then
    return false
  end
  if redis.call('SET', prefix .. case_id, worker, 'NX', 'EX', ttl) then
    return case_id
  end
end
return false
"""

#: Extend the lease only for the worker that holds it. The comparison and the `EXPIRE` are in one
#: script because a worker that read the holder, was descheduled past its own expiry, and then
#: extended the key would be extending a lease that by then belonged to somebody else.
_RENEW_SCRIPT: Final = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('EXPIRE', KEYS[1], tonumber(ARGV[2]))
end
return 0
"""

#: Release only if this worker still holds the lease, and only then take the case out of the ready
#: list. The guard matters most in the case it looks pedantic in: worker A's lease expired, worker
#: B took the case and is working it, and A — still alive, merely slow — finishes and releases. If
#: the release were unconditional, A would delete B's lease and remove from the queue a case B is
#: still processing, so a second crash of B would strand the case with nothing to pick it up.
_RELEASE_SCRIPT: Final = """
local lease = KEYS[1]
local ready = KEYS[2]
if redis.call('GET', lease) ~= ARGV[1] then
  return 0
end
redis.call('DEL', lease)
redis.call('LREM', ready, 0, ARGV[2])
return 1
"""


def _require_identifier(value: str, field: str) -> None:
    """Refuse an identifier that would make a key or a list entry ambiguous.

    An empty case id becomes an empty list element and a lease key that is nothing but the prefix,
    so every empty case collides with every other. A padded worker id fails the compare-and-set in
    `renew` and `release` against its own unpadded self, and the symptom — "my renewals stopped
    working" — points nowhere near the whitespace that caused it.
    """
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(
            f"{field}={value!r} is not usable as a key component: it must be a non-empty string "
            f"with no leading or trailing whitespace"
        )


def _require_ttl(ttl_seconds: int) -> None:
    """A lease with a non-positive lifetime is not a lease.

    Redis rejects `EX 0` outright, so the error would surface as a `ResponseError` from inside a
    Lua script and read as an infrastructure fault. It is a caller's mistake and it is named here.
    """
    if not isinstance(ttl_seconds, int) or isinstance(ttl_seconds, bool) or ttl_seconds <= 0:
        raise ValueError(
            f"ttl_seconds={ttl_seconds!r} must be a positive whole number of seconds; a lease that "
            f"has already expired excludes nobody"
        )


class CaseQueue:
    """Cases waiting for a worker, and the lease that says which worker has one.

    Construction opens no socket: redis-py connects lazily, and a constructor that connected would
    make every worker's start-up depend on Redis being up at that instant rather than at the moment
    work is attempted. The failure belongs on the operation, where the caller can back off and
    retry, and where kill condition L can observe it.

    `namespace` is keyword-only and defaults to the shared prefix, so the documented one-argument
    construction is exactly the construction the rest of the system uses. It exists for tests and
    for running two isolated queues against one Redis; it is not a multi-tenancy feature and
    nothing in this system authorises anything by reading it.
    """

    __slots__ = ("_client", "_lease_prefix", "_namespace", "_ready_key", "_url")

    def __init__(self, redis_url: str, *, namespace: str = DEFAULT_NAMESPACE) -> None:
        _require_identifier(namespace, "namespace")
        self._url = redis_url
        self._namespace = namespace
        self._ready_key = f"{namespace}:queue:ready"
        self._lease_prefix = f"{namespace}:queue:lease:"
        self._client: Any = Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=SOCKET_TIMEOUT_SECONDS,
            socket_timeout=SOCKET_TIMEOUT_SECONDS,
        )

    # -- keys -------------------------------------------------------------------------------

    @property
    def namespace(self) -> str:
        return self._namespace

    @property
    def ready_key(self) -> str:
        return self._ready_key

    def lease_key(self, case_id: str) -> str:
        """The key a lease lives under. Public so an operator can read one without guessing."""
        return f"{self._lease_prefix}{case_id}"

    # -- the queue --------------------------------------------------------------------------

    def enqueue(self, case_id: str) -> None:
        """Ask for a case to be worked, at most once however many times this is called.

        Enqueueing is idempotent because the callers are retried: an HTTP handler, a scheduled
        sweep for cases whose window is closing, and the graph itself after an interrupt can all
        ask for the same case within a second of each other. Two copies in the list would not
        double-lease — the lease key still excludes the second worker — but the duplicate would
        outlive the release of the first and be leased again after the case was finished, which is
        a worker doing completed work rather than the work that is waiting.
        """
        _require_identifier(case_id, "case_id")
        with fail_closed("enqueue"):
            self._client.eval(_ENQUEUE_SCRIPT, 1, self._ready_key, case_id)

    def lease(self, worker_id: str, ttl_seconds: int = DEFAULT_LEASE_TTL_SECONDS) -> str | None:
        """Take the first waiting case this worker can lock, or `None` if there is none free.

        `None` means "nothing is available", which is a fact this method established by asking
        Redis. It never means "Redis did not answer" — that raises. The distinction is the whole of
        kill condition L: a worker told there is no work idles, and a worker told a case is free
        starts it.
        """
        _require_identifier(worker_id, "worker_id")
        _require_ttl(ttl_seconds)
        with fail_closed("lease"):
            acquired = self._client.eval(
                _LEASE_SCRIPT,
                1,
                self._ready_key,
                self._lease_prefix,
                worker_id,
                str(ttl_seconds),
            )
        return None if acquired is None else str(acquired)

    def renew(
        self, case_id: str, worker_id: str, ttl_seconds: int = DEFAULT_LEASE_TTL_SECONDS
    ) -> bool:
        """Extend this worker's lease. `False` means the worker no longer holds it.

        `False` is not an error and must not be treated as one, but it is also not nothing: a
        worker whose renewal fails has already lost the case to somebody else and must stop
        touching it, because everything it does from here races the worker that now holds it.
        """
        _require_identifier(case_id, "case_id")
        _require_identifier(worker_id, "worker_id")
        _require_ttl(ttl_seconds)
        with fail_closed("renew"):
            renewed = self._client.eval(
                _RENEW_SCRIPT, 1, self.lease_key(case_id), worker_id, str(ttl_seconds)
            )
        return int(renewed) == 1

    def release(self, case_id: str, worker_id: str) -> None:
        """Give the case up, and take it out of the queue — but only if this worker still holds it.

        Releasing removes the case from the ready list as well as dropping the lease, because the
        queue entry is a request for attention rather than the case itself: the case's state is in
        PostgreSQL and survives regardless. A worker that stopped at the approval interrupt, or
        that wants another pass later, calls `enqueue` again. The rejected alternative — a release
        that leaves the case queued — re-leases finished cases forever, and the only way out of
        that is a second concept of "done" living somewhere else.

        A release by a worker whose lease has already expired does nothing at all, quietly. It is
        quiet because it is expected: that worker was slow, not wrong, and the thing it must not do
        is disturb the worker that has since taken over.
        """
        _require_identifier(case_id, "case_id")
        _require_identifier(worker_id, "worker_id")
        with fail_closed("release"):
            self._client.eval(
                _RELEASE_SCRIPT, 2, self.lease_key(case_id), self._ready_key, worker_id, case_id
            )

    def leased_by(self, case_id: str) -> str | None:
        """The worker holding this case, or `None` if nobody does.

        Read-only and for reporting — the console shows it, and the tests assert on it. It is never
        the basis of a decision to start work: between this read and any action on it the lease can
        expire, and that gap is the read-then-write the Lua scripts exist to avoid. `lease` is the
        only way to acquire a case.
        """
        _require_identifier(case_id, "case_id")
        with fail_closed("leased_by"):
            holder = self._client.get(self.lease_key(case_id))
        return None if holder is None else str(holder)

    def pending_case_ids(self) -> tuple[str, ...]:
        """Every case in the queue, in order, leased or not.

        A tuple rather than a set, and in the list's own order rather than sorted, because the
        order is the queue's meaning. A caller that sorted it would be looking at a different
        queue from the one the workers are served from.
        """
        with fail_closed("pending_case_ids"):
            entries = self._client.lrange(self._ready_key, 0, -1)
        return tuple(str(entry) for entry in entries)

    def close(self) -> None:
        """Return the connections to the pool.

        Suppresses everything. A caller closing a client is tidying up, frequently after the very
        outage that makes closing fail, and an exception raised here would replace the error that
        actually matters with one about a socket nobody was using.
        """
        with suppress(Exception):
            self._client.close()
