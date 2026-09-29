"""The case machine: one LangGraph graph, a PostgreSQL checkpoint, and a durable tool ledger.

This package is the project's headline. `CLAUDE.md` §2.1 fixes the nodes and their order, ADR-001
§4.1 fixes what killing a worker mid-case must not cost, and the three modules here divide that
claim into parts that can each be checked on their own:

- `state` is what the checkpoint holds, and holds nothing that is not a JSON primitive;
- `tools` is the ledger that records a side-effecting call **before** it runs and **after** it
  completes, in PostgreSQL, so that a resume can tell the two apart;
- `nodes` is the work, one function per `domain.CaseState`, none of which knows it can be killed;
- `machine` wires them into a graph with explicit edges and no dynamic routing.

Nothing is re-exported from here. A package `__init__` that imports its submodules would make
`import warranty_claim_recovery.graph` pull in LangGraph, psycopg and the checkpointer for a caller
that only wanted a type name — and in this repository the caller that only wants a type name is the
console, which must start without a database.
"""

from __future__ import annotations

__all__: list[str] = []
