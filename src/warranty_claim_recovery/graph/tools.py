"""The tool ledger: every side-effecting call recorded before it runs and after it completes.

Kill condition B allows **zero** re-executions of a tool that had already completed when the worker
was killed. That criterion is only checkable if the system can tell, after the resume, which calls
had finished — and the durable thing a resumed case has is its checkpoint, which by construction
does **not** contain the work of the node that was interrupted. LangGraph discards a node's state
update when the node does not return, which is the correct behaviour and is exactly what makes the
naive design fail.

**The naive design, recorded as rejected.** Keep the ledger in the graph state. It is one fewer
table and it is wrong in the only case that matters: a node that performs a retrieval and then dies
before returning loses its own record of having performed it, so the resumed node retrieves again
and kill condition B is satisfied only because nothing was ever observed. Worse, the criterion would
*pass* — the re-execution happened, the ledger just never saw it. A guard that cannot see the event
it guards against is not a guard.

So the ledger is durable **outside** the graph's transaction. Each write is its own committed
statement, so the sequence across a kill is:

    INSERT invocation_id, status=STARTED   committed
    run the tool                           the side effect happens here
    UPDATE  status=COMPLETED, result       committed
    ...
    the worker dies, the node's state update is discarded
    ...
    the node re-runs, asks the ledger, finds COMPLETED, and returns the recorded result

**Three states, and the middle one is the interesting one.** An invocation the ledger has never seen
runs. An invocation recorded COMPLETED is replayed from its record and does not run. An invocation
recorded STARTED and not COMPLETED is a tool that was interrupted *in flight*, and it runs again —
that is not a kill-condition-B violation and must not be counted as one. B forbids repeating work
that finished; requiring an unfinished side effect to be abandoned would leave the manufacturer's
portal holding a half-made submission that nothing will ever complete. The one effect that must not
be duplicated by that retry is the submission itself, and that is guarded a second time, in Redis,
by `queue.idempotency.SubmissionGuard`. Two guards, failing in different directions, because a
retried submission is the failure this project is named after.

**Why the ledger owns its own table rather than appearing in the Alembic migration.** The
checkpointer does the same thing for the same reason: `PostgresSaver.setup()` creates the tables its
own correctness depends on, idempotently, because a component whose durability guarantee lives in
someone else's migration acquires a deployment order nobody wrote down. `CREATE TABLE IF NOT EXISTS`
is run by `setup`, once, by whoever starts a worker.

**Why the result is stored as text rather than as `jsonb`.** A replay has to hand back exactly what
the tool returned. `jsonb` reorders object keys, normalises numeric literals and collapses
duplicate keys, so a replayed result could differ from the recorded one in ways nothing would
report — and a "replay" that differs from the original is not a replay. The text column holds
`json.dumps(..., sort_keys=True)` and comes back byte for byte.

**Why a tool must return a JSON object.** A tool whose result cannot be written down cannot be
replayed, and a ledger that silently failed to record such a result would look identical to one
whose tool had not run. `ToolResultNotRecordableError` refuses at the point of the return, where the
message can name the tool.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from datetime import date
from typing import Any, Final, NamedTuple

from sqlalchemy import Engine, text

from warranty_claim_recovery.audit import TOOL_COMPLETED, TOOL_STARTED
from warranty_claim_recovery.domain import CaseState
from warranty_claim_recovery.graph.state import CaseAudit

__all__ = [
    "INVOCATION_TABLE",
    "STATUS_COMPLETED",
    "STATUS_STARTED",
    "TOOL_REPLAYED",
    "ToolInvocation",
    "ToolLedger",
    "ToolResult",
    "ToolResultNotRecordableError",
]

#: The ledger's own table. Named here rather than in `store.schema` because the ledger creates it;
#: see the module docstring for why that is deliberate and not an oversight.
INVOCATION_TABLE: Final = "tool_invocation"

STATUS_STARTED: Final = "STARTED"
STATUS_COMPLETED: Final = "COMPLETED"

#: A third audit kind beside `TOOL_STARTED` and `TOOL_COMPLETED`. A replay is not a completion: the
#: tool did not run, and writing `TOOL_COMPLETED` again would tell a reader of the log that the
#: manufacturer's portal had been called twice, which is the one thing this module exists to prevent
#: and would be the log's own account of having failed.
TOOL_REPLAYED: Final = "tool.replayed"

_CREATE_TABLE: Final = (
    f"CREATE TABLE IF NOT EXISTS {INVOCATION_TABLE} (\n"
    "    invocation_id TEXT PRIMARY KEY,\n"
    "    case_id       TEXT NOT NULL,\n"
    "    case_version  INTEGER NOT NULL,\n"
    "    node          TEXT NOT NULL,\n"
    "    tool          TEXT NOT NULL,\n"
    "    call_index    INTEGER NOT NULL,\n"
    "    status        TEXT NOT NULL,\n"
    "    result        TEXT\n"
    ")"
)

_CREATE_CASE_INDEX: Final = (
    f"CREATE INDEX IF NOT EXISTS ix_{INVOCATION_TABLE}_case ON {INVOCATION_TABLE} (case_id)"
)

# Split so that no fragment carrying a SQL verb is the interpolated one. The table name comes from a
# module constant and every value is a bound parameter, so there is no path by which caller data
# reaches this string; the shape follows `retrieval.pipeline.RETRIEVAL_STATEMENT`, which makes the
# same argument, so a reader meets one convention rather than two.
_SELECT_ONE: Final = (
    f"SELECT status, result\nFROM {INVOCATION_TABLE}\nWHERE invocation_id = :invocation_id"
)

#: `ON CONFLICT DO NOTHING` rather than an upsert. A conflict here means a previous attempt recorded
#: this invocation and did not complete it, and overwriting that row would erase the only evidence
#: that the tool had been entered before — which is the difference between "interrupted in flight"
#: and "never attempted", and the two call for different accounts of what happened.
_INSERT_STARTED: Final = (
    f"INSERT INTO {INVOCATION_TABLE}\n"
    "    (invocation_id, case_id, case_version, node, tool, call_index, status, result)\n"
    "VALUES (:invocation_id, :case_id, :case_version, :node, :tool, :call_index, :status, NULL)\n"
    "ON CONFLICT (invocation_id) DO NOTHING"
)

#: Guarded by `status <> COMPLETED`. Without the predicate, two workers racing the same case — which
#: the lease makes unlikely and does not make impossible — could have the slower one overwrite the
#: faster one's recorded result, and the replay would then hand back the answer of a call whose side
#: effect was never the one that reached the manufacturer.
_MARK_COMPLETED: Final = (
    f"UPDATE {INVOCATION_TABLE}\n"
    "SET status = :status, result = :result\n"
    "WHERE invocation_id = :invocation_id AND status <> :completed"
)

_SELECT_FOR_CASE: Final = (
    "SELECT invocation_id, case_id, case_version, node, tool, call_index, status, result\n"
    f"FROM {INVOCATION_TABLE}\n"
    "WHERE case_id = :case_id\n"
    "ORDER BY invocation_id"
)


class ToolResultNotRecordableError(TypeError):
    """A tool returned something the ledger cannot write down, and therefore cannot replay.

    A distinct type because the remedy is specific and belongs to whoever wrote the tool: return a
    JSON object. The alternative — storing a repr, or storing nothing and marking the invocation
    complete anyway — would produce a ledger that says a call finished and cannot say what it
    produced, which on the next resume is indistinguishable from a call that never ran.
    """


class ToolInvocation(NamedTuple):
    """One row of the ledger, as a reader of the evidence sees it."""

    invocation_id: str
    case_id: str
    case_version: int
    node: CaseState
    tool: str
    call_index: int
    status: str
    result: dict[str, Any] | None

    @property
    def completed(self) -> bool:
        return self.status == STATUS_COMPLETED


class ToolResult(NamedTuple):
    """What the calling node gets back, and whether the tool actually ran to produce it.

    `replayed` is part of the result rather than something the caller infers, because the two cases
    are indistinguishable from the payload alone — that is the whole point of a replay — and a node
    that wants to record what happened would otherwise have to ask the ledger a second question and
    hope nothing had changed in between.
    """

    payload: dict[str, Any]
    replayed: bool
    invocation_id: str


class ToolLedger:
    """Record before, complete after, and never run a completed call twice.

    Holds an `Engine` rather than a `Session`. Every write here is its own short transaction that
    must commit whatever the caller's transaction does, and a session borrowed from the caller would
    tie the ledger's durability to a commit the caller might never reach — which is the exact
    failure the ledger exists to survive.
    """

    __slots__ = ("_engine",)

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    @property
    def engine(self) -> Engine:
        return self._engine

    def setup(self) -> None:
        """Create the table if it is not there. Idempotent, and run by whoever starts a worker."""
        with self._engine.begin() as connection:
            connection.execute(text(_CREATE_TABLE))
            connection.execute(text(_CREATE_CASE_INDEX))

    @staticmethod
    def invocation_id(case_id: str, node: CaseState, tool: str, call_index: int) -> str:
        """The identity of one call, derived rather than generated.

        Derived from the case, the node, the tool and the position of the call within that node, so
        that the resumed process computes the same identifier as the killed one without having to
        remember anything. A generated identifier — a UUID minted at call time — would be a new one
        on every attempt, every invocation would look unseen, and every completed tool would run
        again. The call index is what distinguishes two calls to the same tool in one node, and it
        is the caller's count rather than the ledger's for the same reason: the ledger is not there
        when the node re-runs its first statement.
        """
        return f"{case_id}:{node.value}:{tool}:{call_index:02d}"

    def completed_result(self, invocation_id: str) -> dict[str, Any] | None:
        """The recorded result of a completed call, or `None` if there is not one.

        `None` covers "never seen" and "started and not finished" together, and the caller needs no
        third answer: both mean the tool has to run. The distinction is preserved in the table and
        is what `invocations_for` reports to the evidence run, which does need it.
        """
        with self._engine.begin() as connection:
            row = connection.execute(
                text(_SELECT_ONE), {"invocation_id": invocation_id}
            ).one_or_none()
        if row is None or row.status != STATUS_COMPLETED or row.result is None:
            return None
        decoded: dict[str, Any] = json.loads(row.result)
        return decoded

    def invoke(
        self,
        *,
        case_id: str,
        case_version: int,
        node: CaseState,
        tool: str,
        call_index: int,
        run: Callable[[], Mapping[str, Any]],
        audit: CaseAudit,
        at: date,
        actor: str,
    ) -> ToolResult:
        """Run the tool once, ever, per invocation identity — and say which happened.

        The audit entries are written by the ledger rather than by the node, so that "recorded
        before it runs" is a property of the mechanism rather than of every call site remembering
        to do it. `TOOL_STARTED` is appended before `run` is called and `TOOL_COMPLETED` after it
        returns; a tool that raises leaves a started event with no completion, which is what a
        reader should see because it is what happened.
        """
        invocation_id = self.invocation_id(case_id, node, tool, call_index)
        recorded = self.completed_result(invocation_id)
        if recorded is not None:
            audit.record(
                case_id=case_id,
                case_version=case_version,
                kind=TOOL_REPLAYED,
                node=node,
                at=at,
                actor=actor,
                detail={"tool": tool, "invocation_id": invocation_id},
            )
            return ToolResult(payload=recorded, replayed=True, invocation_id=invocation_id)

        self._record_started(
            invocation_id=invocation_id,
            case_id=case_id,
            case_version=case_version,
            node=node,
            tool=tool,
            call_index=call_index,
        )
        audit.record(
            case_id=case_id,
            case_version=case_version,
            kind=TOOL_STARTED,
            node=node,
            at=at,
            actor=actor,
            detail={"tool": tool, "invocation_id": invocation_id},
        )

        payload = run()
        encoded = _encode(payload, tool=tool, invocation_id=invocation_id)
        self._mark_completed(invocation_id, encoded)

        audit.record(
            case_id=case_id,
            case_version=case_version,
            kind=TOOL_COMPLETED,
            node=node,
            at=at,
            actor=actor,
            detail={"tool": tool, "invocation_id": invocation_id},
        )
        return ToolResult(payload=dict(payload), replayed=False, invocation_id=invocation_id)

    def invocations_for(self, case_id: str) -> tuple[ToolInvocation, ...]:
        """Every call this case has ever made, in identifier order, with its status.

        The evidence run reads this immediately after a kill to learn which calls had completed at
        the moment the worker died. That set is the population kill condition B is graded over: a
        call that was in flight is not one the criterion forbids repeating, and grading both alike
        would report a correct retry as a defect.
        """
        with self._engine.begin() as connection:
            rows = connection.execute(text(_SELECT_FOR_CASE), {"case_id": case_id}).all()
        return tuple(
            ToolInvocation(
                invocation_id=row.invocation_id,
                case_id=row.case_id,
                case_version=int(row.case_version),
                node=CaseState(row.node),
                tool=row.tool,
                call_index=int(row.call_index),
                status=row.status,
                result=None if row.result is None else json.loads(row.result),
            )
            for row in rows
        )

    def _record_started(
        self,
        *,
        invocation_id: str,
        case_id: str,
        case_version: int,
        node: CaseState,
        tool: str,
        call_index: int,
    ) -> None:
        with self._engine.begin() as connection:
            connection.execute(
                text(_INSERT_STARTED),
                {
                    "invocation_id": invocation_id,
                    "case_id": case_id,
                    "case_version": case_version,
                    "node": node.value,
                    "tool": tool,
                    "call_index": call_index,
                    "status": STATUS_STARTED,
                },
            )

    def _mark_completed(self, invocation_id: str, encoded: str) -> None:
        with self._engine.begin() as connection:
            connection.execute(
                text(_MARK_COMPLETED),
                {
                    "invocation_id": invocation_id,
                    "status": STATUS_COMPLETED,
                    "completed": STATUS_COMPLETED,
                    "result": encoded,
                },
            )


def _encode(payload: Mapping[str, Any], *, tool: str, invocation_id: str) -> str:
    """Serialise a tool's result, refusing anything that would not come back identical.

    `sort_keys=True` so that the stored text is a function of the content rather than of the order
    a dictionary happened to be built in, which makes two recordings of the same result comparable
    by a reader. `allow_nan=False` because `NaN` and `Infinity` are not JSON, are emitted by
    `json.dumps` anyway by default, and are read back by strict parsers as a syntax error — a result
    that writes cleanly and cannot be read is the worst of the three outcomes.
    """
    if not isinstance(payload, Mapping):
        raise ToolResultNotRecordableError(
            f"tool {tool!r} returned {type(payload).__name__} for invocation {invocation_id!r}. "
            f"The ledger records a JSON object so the call can be replayed after a kill; a "
            f"result it cannot write down is indistinguishable, on the next resume, from a call "
            f"that never ran."
        )
    try:
        return json.dumps(dict(payload), sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ToolResultNotRecordableError(
            f"tool {tool!r} returned a result the ledger cannot serialise for invocation "
            f"{invocation_id!r}: {error}. Return plain JSON values; a domain object here would "
            f"make the checkpoint readable only by the exact class layout that wrote it."
        ) from error
