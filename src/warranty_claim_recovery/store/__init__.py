"""The PostgreSQL side of the system: the engine, the schema, and the loader that fills them.

This package is deliberately three separate modules rather than one, and this file deliberately
re-exports nothing.

The reason is a real failure and not a taste in packaging. `engine.normalise_database_url` is pure
string arithmetic over a connection URL, and it is the first thing a deployment needs: a Render
instance that emits `postgres://` will fail at boot, before any table exists, and the error a
reviewer sees should name the URL scheme. If this file imported `schema`, then importing
`warranty_claim_recovery.store.engine` would drag in SQLAlchemy's ORM registry and the `pgvector`
dialect extension, and a missing or mismatched `pgvector` wheel would turn a URL-scheme bug into an
import error about a vector type. The diagnosis would then start in the wrong place.

Rejected: a package `__init__` that re-exports `build_engine`, `Retriever` and the ORM rows for
convenience at the call site. The convenience is one import line per caller. The cost is that the
cheapest module in the package can no longer be imported cheaply, which is precisely what the
configuration tests and the deployment smoke check want to do.
"""

from __future__ import annotations

__all__: list[str] = []
