"""clause currency: a withdrawn bulletin is excluded in the same statement that ranks

Revision ID: 0003_clause_is_current
Revises: 0002_submission_record

### The defect this revision closes

Until this revision the retrieval statement filtered on `program_id` and `policy_version` and on
nothing else. Whether a document is still in force lives on `document.is_current`, and the ranking
statement never joined to it, so a clause of a withdrawn service bulletin was as retrievable as the
clause that replaced it — and, the corpus having been written that way on purpose, it usually ranked
beside it. Measured over the development split before this revision: 507 of 520 queries returned at
least one withdrawn clause in their top five, and 69 of them returned one at rank one. A correction
that quotes a withdrawn bulletin is the one a manufacturer rejects on sight, because the bulletin
says in its own text that it has been replaced.

### Why the fact is copied onto the clause rather than joined

`store.schema` records why `policy_version` is already denormalised onto the clause: the metadata
filter must run in the same statement as the vector search, and a join in that statement gives the
planner a reason to rank first and join afterwards, which is the post-filter this project refuses.
Currency is the same kind of fact and gets the same treatment. The backfill below copies it from
the document once, in the migration, so that no row is ever left without a value to filter on.

### Why a composite foreign key carries it, and not the loader's good behaviour

A denormalised copy is a second place for the truth to live, and the second place is always the
one that goes stale. The dangerous staleness here is specific: a bulletin withdrawn **after** its
clauses were loaded — by the loader's document upsert, or by an operator marking a bulletin
withdrawn with an `UPDATE` — would leave its clauses marked current, and the filter this revision
adds would go on admitting them. That is the defect this revision exists to close, reintroduced by
the mechanism that closes it.

So the copy is made the database's responsibility. The clause's foreign key now covers
`(document_id, is_current)` and references a unique key on the same pair, with `ON UPDATE CASCADE`.
A withdrawal propagates to every clause of the document in the statement that performs it, without
re-embedding a character of text that did not change, and a clause row that disagrees with its
document about currency is refused as a foreign-key violation rather than trusted to the code that
wrote it. The unique key is trivially satisfied — `document_id` is already the primary key — and
exists only because a foreign key may reference nothing less.

Rejected: a trigger on `document`. It would do the same work invisibly, a later reader would find
behaviour in no table definition, and a trigger is precisely what a restore script or a bulk load
disables for speed. Rejected: keeping the single-column foreign key and adding the composite one
beside it. The composite key already proves the document exists, so the pair would be two
constraints asserting one fact.

### Why the index gains a column rather than becoming partial

The metadata-filter index is rebuilt over `(program_id, policy_version, is_current)`. A partial
index `WHERE is_current` would be smaller and would match today's predicate exactly, and it was
rejected for that reason: it restates the retrieval rule in DDL, and the day the rule changes the
index silently stops matching and the planner silently stops choosing it — a plan change nobody sees
until the corpus is large enough for it to cost something. The composite index serves the same
predicate without holding a second copy of it.

The literals here are written out rather than imported from `store.schema`, for the reason
`0001_initial_schema` records: a migration that reads a live constant replays differently after the
constant changes. `tests/test_retrieval.py` reconciles the two by reading the catalogue back.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_clause_is_current"
down_revision: str | None = "0002_submission_record"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Nullable first, so the column can exist before a value has been decided for any row.
    op.add_column("clause", sa.Column("is_current", sa.Boolean(), nullable=True))

    # The backfill. Every clause takes its document's currency; the join is on the primary key, so
    # no row can take two values and none can take a value that belongs to another document.
    op.execute(
        "UPDATE clause SET is_current = document.is_current "
        "FROM document WHERE clause.document_id = document.document_id"
    )

    # Only now NOT NULL. A row the backfill did not reach would be a clause with no document, which
    # the existing foreign key already makes impossible; if one existed this statement would fail
    # and say so, rather than a default quietly declaring it current.
    op.alter_column("clause", "is_current", existing_type=sa.Boolean(), nullable=False)

    op.create_unique_constraint(
        "uq_document_document_id_is_current", "document", ["document_id", "is_current"]
    )
    op.drop_constraint("fk_clause_document_id_document", "clause", type_="foreignkey")
    op.create_foreign_key(
        "fk_clause_document_id_is_current_document",
        "clause",
        "document",
        ["document_id", "is_current"],
        ["document_id", "is_current"],
        ondelete="CASCADE",
        onupdate="CASCADE",
    )

    op.drop_index("ix_clause_program_policy", table_name="clause")
    op.create_index(
        "ix_clause_program_policy_current",
        "clause",
        ["program_id", "policy_version", "is_current"],
    )


def downgrade() -> None:
    op.drop_index("ix_clause_program_policy_current", table_name="clause")
    op.create_index("ix_clause_program_policy", "clause", ["program_id", "policy_version"])

    op.drop_constraint("fk_clause_document_id_is_current_document", "clause", type_="foreignkey")
    op.create_foreign_key(
        "fk_clause_document_id_document",
        "clause",
        "document",
        ["document_id"],
        ["document_id"],
        ondelete="CASCADE",
    )
    op.drop_constraint("uq_document_document_id_is_current", "document", type_="unique")

    # Dropping the column drops the fact. A database downgraded to 0002 is back to retrieving
    # withdrawn bulletins, and that is what 0002 did; a downgrade that pretended otherwise would
    # leave a column the 0002 code never reads, which is a filter nobody applies.
    op.drop_column("clause", "is_current")
