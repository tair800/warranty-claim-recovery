"""The Warranty Recovery Lab — the console a reviewer reads the evidence on.

One module, `app`, exporting one ASGI application. Nothing is re-exported here on purpose: the
console is an edge, and an edge that other modules import from is no longer an edge. The
deterministic core, the retrieval pipeline, the audit contract and the corpus all know nothing
about this package, and the day one of them imports from it is the day the console can change a
number that is graded somewhere else.

`warranty_claim_recovery.api.app:app` is the entry point the Makefile and the container both name.
"""

from __future__ import annotations

__all__: list[str] = []
