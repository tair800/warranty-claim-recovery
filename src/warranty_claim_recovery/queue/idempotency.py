"""One recovery identity, one effect — and why the caller that loses is told the winner's answer.

ADR-001 §4.2 makes two promises about what leaves this system. Nothing leaves without a recorded
human approval, and **nothing leaves twice**. This module is the whole of the second promise.

The failure it prevents is concrete and expensive. A worker submits a corrected warranty claim to
the manufacturer, the response is slow, the worker is killed or the operator refreshes the console,
and the case is picked up again. Without a guard the correction is filed a second time, and
`DUPLICATE_CLAIM` is one of the seven rejection codes in `domain.RejectionCode` — the system would
be manufacturing the defect it was bought to remove, against a claim window that is already running
down. Worse, a distributor that files duplicates loses standing with the manufacturer's
adjudicators in a way no individual claim shows.

**`claim_once` is `SET NX`, and nothing else would do.** Reading the key and then writing it is a
race whose window is a round trip wide, and sixteen workers whose reads interleave inside it all
see an unclaimed identity. `SET key value NX EX ttl` is decided inside Redis, where the sixteen
requests are serialised whether they like it or not, so exactly one of them is told it won. That is
kill condition E, and it is a property of the command rather than of the care taken at the call
site.

**The loser is given the winner's effect, not an error.** This is what "idempotent" means and it is
the part most implementations skip. A duplicate attempt that raises pushes the retry logic onto
every caller: each one must now distinguish "already done, here is the result" from "failed, try
again", and the first caller to get that wrong retries a submission that succeeded. So the guard
stores the winner's effect identifier against the identity, and `effect_for` hands it to whoever
asks. The second caller returns the first caller's answer and no second effect exists.

**Why the effect identifier and the claim share one key.** Two keys — a claim flag and an effect
value — can disagree: the claim can expire while the effect remains, or an effect can be written
against a claim that has lapsed and been re-won by somebody else. One key holds a sentinel while
the submission is in flight and the effect identifier once there is one, so "claimed" and "produced
this effect" are the same fact and cannot drift apart. The cost is that `effect_for` cannot
distinguish "never claimed" from "claimed, still in flight"; both are `None`. That is acceptable
because neither authorises a caller to submit — only `claim_once` does, and it will refuse.

**Rejected: the effect identity as the key's value with no sentinel.** It would need the winner to
know its effect identifier before it has performed the effect, which inverts the order: the whole
point is to claim *before* acting, so that a crash between the claim and the act cannot be replayed
as a fresh claim by the next worker. The TTL is what limits the damage of that crash, and it is
long — a day — because a submission stuck in flight must not be retried by a sweep an hour later
while the manufacturer's portal is still processing the first one.

**Rejected: an idempotency table in PostgreSQL.** It would work, and it would also be the component
the submit node already depends on for the checkpoint, so a database incident would take out the
duplicate guard and the case state together, at the moment every worker is retrying. `SKILL_MATRIX`
gives this project the idempotency-key cell; splitting the two concerns across two stores is the
engineering reason as well as the portfolio one.

Everything here **raises** when Redis cannot answer. See `leases.RedisUnavailableError`: a guard
that returned `True` from `claim_once` because it could not check has not guarded anything, and a
guard that returned `None` from `effect_for` under an outage tells its caller there is no prior
effect — which is the sentence that precedes a duplicate submission.
"""

from __future__ import annotations

from contextlib import suppress
from typing import Any, Final

from redis import Redis

from warranty_claim_recovery.queue.leases import (
    DEFAULT_NAMESPACE,
    SOCKET_TIMEOUT_SECONDS,
    fail_closed,
)

__all__ = [
    "DEFAULT_CLAIM_TTL_SECONDS",
    "PENDING_SENTINEL",
    "ConflictingSubmissionEffectError",
    "SubmissionClaimExpiredError",
    "SubmissionGuard",
    "SubmissionGuardError",
]

#: A day. Long enough that a submission which is genuinely still in flight at a slow manufacturer
#: portal is not retried underneath itself; short enough that the keyspace does not grow without
#: bound. It is a safety window rather than a record — the audit log in PostgreSQL is what says
#: permanently that a submission happened, and this key only says "do not do it again *now*".
DEFAULT_CLAIM_TTL_SECONDS: Final = 86_400

#: What the key holds between winning the claim and recording the effect. Reserved: `record_effect`
#: refuses to store it as an effect identifier, because an effect that was named this would be
#: indistinguishable from a submission still in flight and would be silently retried forever.
PENDING_SENTINEL: Final = "__wcr_submission_pending__"


class SubmissionGuardError(RuntimeError):
    """The guard was asked to record something it cannot honestly record.

    One base type so a caller can decide "do not submit" without enumerating the reasons, and two
    subclasses because the two reasons want different human responses: an expired claim is an
    operational timeout, a conflicting effect is a defect in the submit path.
    """


class SubmissionClaimExpiredError(SubmissionGuardError):
    """An effect was recorded against a claim that no longer exists.

    Either the submission took longer than the TTL, or it was never claimed at all. Both mean the
    identity is currently unguarded, so another caller may already have claimed it and produced its
    own effect. Raising is the only honest answer: writing the effect anyway would create a key
    that asserts this caller's submission was the only one, which is the claim that has just been
    shown to be unsupportable.
    """


class ConflictingSubmissionEffectError(SubmissionGuardError):
    """A second, different effect was recorded for one recovery identity.

    This is kill condition E having already happened, observed at the moment the second effect
    tries to write itself down. It cannot be repaired here — the second submission has been made —
    so the guard refuses the write and keeps the first effect, which is the one the audit log and
    the manufacturer both already know about. Overwriting would erase the evidence of the very
    duplicate this exception exists to report.
    """


#: Store the effect against a live claim, without disturbing the claim's expiry.
#:
#: `KEEPTTL` rather than a fresh `EX`: the window is measured from the moment the identity was
#: claimed, and refreshing it at the end of the submission would extend it by the duration of every
#: submission, so a slow portal would quietly stretch the guard's memory.
#:
#: Returning distinguishable strings rather than booleans because the three outcomes call for three
#: different responses, and a boolean would flatten "already recorded, exactly this" into the same
#: answer as "recorded something else" — which is the difference between a harmless retry and a
#: duplicate submission.
_RECORD_EFFECT_SCRIPT: Final = """
local current = redis.call('GET', KEYS[1])
if current == false then
  return 'EXPIRED'
end
if current == ARGV[2] then
  return 'ALREADY_RECORDED'
end
if current == ARGV[1] then
  redis.call('SET', KEYS[1], ARGV[2], 'KEEPTTL')
  return 'RECORDED'
end
return 'CONFLICT'
"""


def _require_identity(value: str, field: str) -> None:
    """Refuse an identity or effect identifier that would make the key ambiguous.

    `Claim.recovery_identity` is built by joining three fields with colons, so an empty component
    already produces a distinguishable string. An empty *whole* identity does not: it would key
    every unidentified submission to one guard entry, and the first one through would suppress all
    the rest. Padding is refused for the same reason `leases` refuses it — the value is compared
    verbatim inside Lua, and a space is invisible in the failure report.
    """
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(
            f"{field}={value!r} is not usable as a key component: it must be a non-empty string "
            f"with no leading or trailing whitespace"
        )


class SubmissionGuard:
    """At most one submission effect per recovery identity, under any amount of concurrency.

    Construction opens no socket, for the same reason `CaseQueue`'s does not: the refusal belongs
    on the operation the caller is about to rely on, not on start-up.

    The identity this is keyed by is `Claim.recovery_identity` — claim, part and serial — and that
    module explains why it is not the claim id alone. This module takes whatever string it is given
    and makes exactly one of them win.
    """

    __slots__ = ("_client", "_namespace", "_prefix", "_url")

    def __init__(self, redis_url: str, *, namespace: str = DEFAULT_NAMESPACE) -> None:
        _require_identity(namespace, "namespace")
        self._url = redis_url
        self._namespace = namespace
        self._prefix = f"{namespace}:submission:"
        self._client: Any = Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=SOCKET_TIMEOUT_SECONDS,
            socket_timeout=SOCKET_TIMEOUT_SECONDS,
        )

    @property
    def namespace(self) -> str:
        return self._namespace

    def key_for(self, recovery_identity: str) -> str:
        """The key a claim lives under. Public so an operator can read one without guessing."""
        return f"{self._prefix}{recovery_identity}"

    def claim_once(
        self, recovery_identity: str, *, ttl_seconds: int = DEFAULT_CLAIM_TTL_SECONDS
    ) -> bool:
        """`True` for exactly one caller; `False` for every other, for as long as the claim lives.

        The winner must go on to perform the effect and call `record_effect`. A winner that does
        neither leaves the identity blocked until the TTL lapses, which is the deliberate trade:
        blocking a retry for a day is recoverable by a human, and filing a second warranty
        correction is not.

        Losing is not an error and does not raise. The loser asks `effect_for` and returns what the
        winner produced; see the module docstring for why an exception here would push retry logic
        into every caller.
        """
        _require_identity(recovery_identity, "recovery_identity")
        if not isinstance(ttl_seconds, int) or isinstance(ttl_seconds, bool) or ttl_seconds <= 0:
            raise ValueError(
                f"ttl_seconds={ttl_seconds!r} must be a positive whole number of seconds; a claim "
                f"that has already expired guards nothing"
            )
        with fail_closed("claim_once"):
            won = self._client.set(
                self.key_for(recovery_identity), PENDING_SENTINEL, nx=True, ex=ttl_seconds
            )
        return won is True

    def record_effect(self, recovery_identity: str, effect_id: str) -> None:
        """Bind the effect this submission produced to the identity that authorised it.

        Recording the same effect twice is accepted in silence, because the caller retrying a
        failed acknowledgement is doing the right thing and the outcome it wants is already true.
        Recording a *different* effect raises: two effects for one identity is kill condition E,
        and the guard reports it rather than quietly keeping the newer one.
        """
        _require_identity(recovery_identity, "recovery_identity")
        _require_identity(effect_id, "effect_id")
        if effect_id == PENDING_SENTINEL:
            raise ValueError(
                f"{effect_id!r} is the reserved in-flight sentinel. An effect recorded under it "
                f"would be indistinguishable from a submission that never completed, and would be "
                f"retried for as long as the claim lived."
            )
        with fail_closed("record_effect"):
            outcome = str(
                self._client.eval(
                    _RECORD_EFFECT_SCRIPT,
                    1,
                    self.key_for(recovery_identity),
                    PENDING_SENTINEL,
                    effect_id,
                )
            )
        if outcome == "EXPIRED":
            raise SubmissionClaimExpiredError(
                f"the claim on {recovery_identity!r} has expired or was never made, so this "
                f"system cannot say that effect {effect_id!r} was the only one. Another caller may "
                f"hold the identity now."
            )
        if outcome == "CONFLICT":
            raise ConflictingSubmissionEffectError(
                f"{recovery_identity!r} already produced a different submission effect, and "
                f"{effect_id!r} is a second one. The first effect is kept: overwriting it would "
                f"erase the evidence of the duplicate."
            )

    def effect_for(self, recovery_identity: str) -> str | None:
        """What the winning caller produced, or `None` if there is nothing to report yet.

        `None` covers two situations the caller does not need to tell apart — the identity was
        never claimed, and it is claimed but still in flight — because neither authorises a
        submission. Only `claim_once` does that, and under both it answers correctly.

        It never means "Redis did not answer". That raises, because a caller reading `None` from an
        outage concludes there is no prior effect, and that conclusion is the last step before a
        duplicate correction reaches the manufacturer.
        """
        _require_identity(recovery_identity, "recovery_identity")
        with fail_closed("effect_for"):
            stored = self._client.get(self.key_for(recovery_identity))
        if stored is None:
            return None
        recorded = str(stored)
        return None if recorded == PENDING_SENTINEL else recorded

    def close(self) -> None:
        """Return the connections to the pool; see `CaseQueue.close` for why it cannot raise."""
        with suppress(Exception):
            self._client.close()
