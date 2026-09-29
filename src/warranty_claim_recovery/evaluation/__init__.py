"""The evaluation: the numbers that decide whether this project ships, and nothing else.

Four modules, and the split between them is the point rather than tidiness.

`metrics` is arithmetic over hand-buildable inputs. It imports the domain and the standard library
and nothing else — no file reading, no database, no corpus — so every figure that decides whether
this project ships can be checked against inputs a reader wrote by hand. A metric that can only be
exercised by running the whole system is a metric whose behaviour at its boundaries nobody has ever
seen, and the boundaries are where a rate quietly becomes a division by zero or a confusion matrix
quietly stops counting a class.

`pipeline` runs the deterministic path over the committed corpus and produces one assessment per
claim. It owns the join between the corpus's evidence vocabulary and the requirement matrix's, which
is the one place those two tables are allowed to meet.

`baselines` implements the four retrieval systems ADR-001 §6 predeclared. Each removes or replaces a
component of the retrieval stage itself, so beating one is a real question rather than a comparison
of a system with itself.

`artifacts` assembles the three JSON payloads `tests/test_kill_criteria.py` grades. It takes data
and returns dictionaries, so the key contract can be tested without a database.

Nothing here reads a wall clock. A score that differs between two runs over the same corpus is a
score nobody can reproduce, and an evaluation nobody can reproduce is a description.
"""

from __future__ import annotations

__all__: tuple[str, ...] = ()
