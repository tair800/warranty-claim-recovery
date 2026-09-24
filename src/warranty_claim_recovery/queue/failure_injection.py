"""Taking Redis away for real, because a mocked outage proves nothing about the code under test.

Kill condition L asks a behavioural question — *with Redis unavailable, is a case leased twice or a
submission made?* — and a behavioural question deserves a behavioural answer. The tempting shortcut
is to patch `CaseQueue.lease` so that it raises, observe that it raised, and record a pass. That
records nothing. It tests the patch. The code that would actually run during an outage, including
every `try` in `leases.py` and the decision about which redis-py exceptions mean "unavailable", is
the code that never executes.

So this module removes Redis from underneath the real objects and leaves the real objects alone.
**Nothing here touches `CaseQueue` or `SubmissionGuard`.** They are constructed normally, their
methods are the ones the workers call, and they meet an outage they cannot tell from a genuine one.

Two injections, because the two outages a queue actually meets fail at different layers and a
system can survive one while mishandling the other:

- **`closed_port`** — the connection is refused by the operating system. Redis is not there: the
  process is down, the container has been stopped, the address is wrong. Nothing in this repository
  fakes it; a socket is bound to obtain a free port, closed again, and the client is pointed at it.
  The refusal comes from Windows or Linux, not from Python.
- **`severed_client`** — the connection is established and then every command fails. This is the
  network partition, the failover mid-flight, the `CLIENT KILL`. It is the more dangerous shape
  because construction succeeds and the code has already got past whatever it does at start-up. It
  is injected by replacing `redis.Redis.from_url` for the duration of the block, so the object the
  queue builds is the object that fails — the queue's own construction path runs unchanged.

**Rejected: `monkeypatch` on the queue's methods.** Stated again because it is the default and it
is wrong: it exercises the test double and reports on it. ADR-001 §3 and this repository's rule 6
say guards must fail from behaviour.

**Rejected: stopping the Redis container.** It is a genuine outage and it is the most faithful
injection available, and it was rejected because the suite would then require Docker control from
inside a test, would leave a developer's Redis stopped when the test was interrupted, and could not
run twice concurrently. `closed_port` produces the identical `ConnectionRefusedError` with none of
that.

**Rejected: a `redis` mock library.** A fake server answers the questions the fake's author thought
of. The two failures here are produced by the operating system and by removing the client's ability
to speak, and neither depends on anyone having anticipated a particular command.
"""

from __future__ import annotations

import socket
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, suppress
from typing import Any, Final, NamedTuple, NoReturn

from redis import Redis
from redis.exceptions import RedisError

from warranty_claim_recovery.config import get_settings

__all__ = [
    "INJECTIONS",
    "Injection",
    "closed_port",
    "live_redis_url",
    "redis_is_reachable",
    "severed_client",
]

#: A short wait for the liveness probe only. The probe's job is to tell a suite whether the
#: infrastructure it needs is present, and a probe that blocks for the full socket timeout turns a
#: missing Redis into a suite that appears to hang rather than one that reports the problem.
_PROBE_TIMEOUT_SECONDS: Final = 2.0


def live_redis_url() -> str:
    """The Redis this deployment actually uses, from configuration rather than from a literal.

    Read through `config.get_settings` so that a test measuring the outage behaviour and the worker
    experiencing it are pointed at the same place. A hard-coded URL here would let the suite pass
    against a Redis the system does not use, which is the shape of a green build that proves
    nothing.
    """
    return get_settings().redis_url


def redis_is_reachable(url: str) -> bool:
    """Whether Redis answers a `PING` at `url`.

    In this module rather than in a test helper because it is the same question the injections
    answer in the negative, and a suite that could not tell "Redis is absent" from "the system
    refused correctly" would record an environment without Redis as a pass for kill condition L.
    """
    client: Any = Redis.from_url(
        url,
        socket_connect_timeout=_PROBE_TIMEOUT_SECONDS,
        socket_timeout=_PROBE_TIMEOUT_SECONDS,
    )
    try:
        return bool(client.ping())
    except (RedisError, OSError):
        return False
    finally:
        with suppress(Exception):
            client.close()


@contextmanager
def closed_port(_live_url: str) -> Iterator[str]:
    """Yield a Redis URL whose port nothing is listening on. The refusal comes from the kernel.

    A free port is obtained by binding one and closing it immediately. There is a window in which
    another process could take that port, and it is accepted deliberately: the alternative is a
    hard-coded port, which fails on the day something else is listening there and fails as a
    confusing `AuthenticationError` rather than as a refusal.

    The live URL is ignored. That is the point of this injection — the client is pointed somewhere
    Redis is not, so no state written before the block is visible inside it, exactly as it would not
    be visible to a worker whose Redis had gone.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    finally:
        probe.close()
    yield f"redis://127.0.0.1:{port}/0"


class _SeveredClient:
    """A Redis client whose every command raises, as though the socket had been cut mid-flight.

    Not a fake Redis. It answers nothing and simulates nothing; it is the absence of a working
    client wearing the right shape, so that code which already holds a client meets the failure at
    the moment it issues a command rather than at the moment it connects.

    `close` is the one exception and is a no-op, because tidying up after a failure must not raise
    a second failure on top of the first.
    """

    __slots__ = ("_url",)

    def __init__(self, url: str) -> None:
        self._url = url

    def close(self) -> None:
        return None

    def __getattr__(self, name: str) -> Callable[..., NoReturn]:
        url = self._url

        def refuse(*_args: object, **_kwargs: object) -> NoReturn:
            raise ConnectionError(
                f"redis command {name!r} against {url} failed: the connection has been severed "
                f"by failure injection"
            )

        return refuse


@contextmanager
def severed_client(live_url: str) -> Iterator[str]:
    """Make every client built inside this block unable to issue a command.

    `redis.Redis.from_url` is replaced for the duration, so `CaseQueue` and `SubmissionGuard` build
    their clients exactly as they always do and receive one that cannot speak. The seam is in the
    redis library, not in the classes under test, which is what keeps this an observation of their
    behaviour rather than of a substitute for it.

    The original attribute is recovered from `Redis.__dict__` and restored there. Reading it with
    `getattr` would yield the bound classmethod, and putting *that* back would leave the class with
    a plain bound method where a `classmethod` descriptor used to be — subclasses would then bind
    to `Redis` instead of to themselves, long after this block had exited and in a place nobody
    would connect back to a test.
    """
    original = Redis.__dict__["from_url"]

    def severed(url: str, **_kwargs: object) -> _SeveredClient:
        return _SeveredClient(url)

    Redis.from_url = severed  # type: ignore[method-assign, assignment]
    try:
        yield live_url
    finally:
        Redis.from_url = original  # type: ignore[method-assign]


class Injection(NamedTuple):
    """One way of taking Redis away, named so an artifact can say which ones were applied.

    `apply` takes the live URL and yields the URL the code under test should use: `closed_port`
    substitutes a dead one, `severed_client` hands back the live one and breaks the client instead.
    A uniform shape so the evidence loop treats both identically and cannot accidentally grade one
    of them differently from the other.
    """

    name: str
    description: str
    apply: Callable[[str], AbstractContextManager[str]]


#: Both injections, in a fixed order. A tuple rather than a set: the evidence artifact lists the
#: modes it applied, and a listing whose order changed between runs would make two identical runs
#: produce two different artifacts.
INJECTIONS: Final[tuple[Injection, ...]] = (
    Injection(
        name="closed_port",
        description=(
            "the client is pointed at a TCP port nothing is listening on; the connection is "
            "refused by the operating system"
        ),
        apply=closed_port,
    ),
    Injection(
        name="severed_client",
        description=(
            "redis.Redis.from_url yields a client whose every command raises ConnectionError; "
            "construction succeeds and the first command does not"
        ),
        apply=severed_client,
    ),
)
