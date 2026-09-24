"""Retrieval: the encoder's policy, the pipeline that uses it, and the citation check.

Nothing is re-exported here, for the same reason as `store/__init__.py`: `citations.verify_citation`
is pure string arithmetic with no dependency beyond the domain models, and it is imported by the
graders that produce `artifacts/groundedness.json`. A package `__init__` that pulled in `pipeline`
would make that import load SQLAlchemy and, on first use, a 285MB ONNX session — so a grader whose
whole job is to slice a string and compare it would fail on a machine with no database.
"""

from __future__ import annotations

__all__: list[str] = []
