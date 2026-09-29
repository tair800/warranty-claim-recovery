"""The graph: ten nodes, one conditional edge, and a checkpoint in PostgreSQL.

Every edge is declared. Nothing routes on a model output, nothing routes on a string a node
produced, and the one branch in the whole machine — submit or write off — reads two booleans the
deterministic core computed. ADR-001 §8 records that a multi-agent architecture is out of scope; the
reason it is out of scope is visible here, in that the entire control flow fits on one screen and a
reviewer can check it against `CLAUDE.md` §2.1 line by line.

### Why the checkpointer connection string is not the application's

`config.Settings.database_url` is a SQLAlchemy URL — `postgresql+psycopg://…` — and the `+psycopg`
part is a SQLAlchemy dialect instruction that libpq has never heard of. `PostgresSaver` opens the
connection itself, with psycopg, and would be handed a host named after a driver. `checkpoint_url`
strips exactly that fragment and touches nothing else, so a query string carrying `?sslmode=require`
survives — `store.engine` records at length why rebuilding a URL from parsed components is how an
encrypted connection quietly becomes an unencrypted one.

### Why `pending_nodes` reads `tasks` and only then `next`

Kill condition A is graded by comparing the node the checkpoint recorded as next against the node
that actually ran after the resume, so the reading of "what the checkpoint recorded" has to be the
one the runtime itself acts on. `StateSnapshot.tasks` is that reading: it is the list of tasks the
next superstep will execute, and it is what carries the interrupt state of a case waiting on a
person. `next` is derived from the same set and is read only when a snapshot does not expose tasks
at all.

An earlier version of this module claimed the two diverge at the approval node — that a case killed
after a human's answer had been recorded would report an empty `next` while `tasks` still named
`approval`. That was measured against this checkpointer and it is false; the two agreed in every
killed case the evidence run produced, including that one. It is recorded as rejected rather than
quietly deleted, because the reading it argued for is the one this function still uses and the next
reader deserves to know the reason is "tasks is the runtime's own list", not "the two differ".

### Why `setup` is called by the caller rather than on every build

`PostgresSaver.setup()` and `ToolLedger.setup()` each issue DDL. A graph that ran them on every
construction would issue DDL on every worker start, against a free-tier PostgreSQL, for tables that
exist — and the first symptom of a lock contention there is a worker that will not start. They are
idempotent and they are cheap; they are still not something to do in a constructor.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from itertools import pairwise
from typing import Any, Final

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph

from warranty_claim_recovery.graph.nodes import (
    NODE_SEQUENCE,
    CaseDependencies,
    CaseNodes,
    route_after_approval,
)
from warranty_claim_recovery.graph.state import GraphState
from warranty_claim_recovery.store.engine import normalise_database_url

__all__ = [
    "ENTRY_NODE",
    "NODE_NAMES",
    "TERMINAL_NODES",
    "build_machine",
    "case_config",
    "checkpoint_url",
    "checkpointer",
    "pending_nodes",
]

#: The node names in pipeline order, derived from `NODE_SEQUENCE` rather than retyped. A second
#: list would be a second opinion about the pipeline, and the durability evidence asserts that it
#: killed every node in this tuple at least once.
NODE_NAMES: Final[tuple[str, ...]] = tuple(name for name, _ in NODE_SEQUENCE)

ENTRY_NODE: Final = "intake"

#: The two ways a case ends. Both are ends: a write-off is a decision, not a failure, and treating
#: it as an error path would make "the case finished" mean "the case filed something".
TERMINAL_NODES: Final[tuple[str, str]] = ("submit", "write_off")

#: The linear spine. `approval` is absent because it is the one node with a choice after it, and
#: listing it here as well would give the graph two opinions about where a case goes next.
_LINEAR: Final[tuple[str, ...]] = (
    "intake",
    "requirements",
    "deadline",
    "eligibility",
    "retrieval",
    "compose",
    "gate",
    "approval",
)


def checkpoint_url(database_url: str) -> str:
    """The libpq URL for the same database, with the SQLAlchemy driver fragment removed.

    Normalised through `store.engine.normalise_database_url` first, so that a provider's
    `postgres://` or a bare `postgresql://` reaches the checkpointer as the same string the rest of
    the system connects with. Only the scheme is rewritten; the authority, the path and the query
    string cross unchanged, which is the property that keeps `?sslmode=require` attached.
    """
    normalised = normalise_database_url(database_url)
    return normalised.replace("postgresql+psycopg://", "postgresql://", 1)


@contextmanager
def checkpointer(database_url: str, *, setup: bool = False) -> Iterator[PostgresSaver]:
    """A `PostgresSaver` over this project's database, closed when the caller is done with it.

    A context manager because the saver owns a connection and a worker that leaked one per resumed
    case would exhaust a free-tier connection limit long before anybody noticed — and the symptom,
    a case that cannot be resumed, would look exactly like the durability bug this project exists to
    disprove.

    `setup` defaults to false. See the module docstring: the DDL is idempotent and is still not
    something to run on every worker start.
    """
    with PostgresSaver.from_conn_string(checkpoint_url(database_url)) as saver:
        if setup:
            saver.setup()
        yield saver


def build_machine(deps: CaseDependencies, checkpoint: Any) -> Any:
    """Compile the case machine over a checkpointer.

    The checkpointer is a parameter rather than something this function builds, because the whole
    durability claim is the ability to construct **a second machine over the same checkpointer in a
    different process** and have it resume the first one's work. A function that owned the
    checkpointer would make that the hard case instead of the ordinary one.

    Typed `Any` on both sides. LangGraph's compiled graph is generic over the state type and its
    checkpointer protocol is not re-exported at a stable path; annotating them precisely would pin
    this module to internals that move between minor versions, and the alternative — a cast that
    asserts a shape nothing checks — would be a type that is wrong rather than absent.
    """
    graph: Any = StateGraph(GraphState)
    for name, function in CaseNodes(deps).as_mapping().items():
        graph.add_node(name, function)

    graph.add_edge(START, ENTRY_NODE)
    for source, target in pairwise(_LINEAR):
        graph.add_edge(source, target)
    graph.add_conditional_edges(
        "approval",
        route_after_approval,
        {"submit": "submit", "write_off": "write_off"},
    )
    for terminal in TERMINAL_NODES:
        graph.add_edge(terminal, END)

    return graph.compile(checkpointer=checkpoint)


def case_config(case_id: str) -> dict[str, Any]:
    """The configuration that binds a run to one case's checkpoint thread.

    The thread identifier is the case identifier and nothing else — no worker, no attempt number, no
    timestamp. A thread keyed by anything that varies between attempts would give the resumed run a
    fresh, empty history, and every kill would present as a case that started again from intake
    while reporting success.
    """
    return {"configurable": {"thread_id": case_id}}


def pending_nodes(snapshot: Any) -> tuple[str, ...]:
    """What the checkpoint says has to run next, read the way the runtime reads it.

    `tasks` first and `next` as the fallback; the module docstring argues why, and records the
    reason that was tested and found false so it is not reinstated.
    """
    tasks = tuple(str(task.name) for task in getattr(snapshot, "tasks", ()) or ())
    return tasks or tuple(str(name) for name in getattr(snapshot, "next", ()) or ())
