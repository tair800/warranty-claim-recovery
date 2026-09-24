"""initial schema: policy documents, their clauses, and the pgvector index

Revision ID: 0001_initial_schema
Revises:

The literals in this file are deliberate, and the alternative was considered and rejected.

`store.schema` exports `HNSW_BUILD_PARAMETERS`, and the obvious move is to import it here so that
the index and the constant can never disagree. That coupling is wrong for a migration. A migration
is a frozen record of what a particular upgrade did; one that reads a live constant replays
*differently* after that constant changes, so a database built today and one rebuilt from scratch
next year would end up with different indexes while `alembic_version` claimed they were identical.
That is a worse failure than drift, because nothing reveals it.

So the values are written out, and drift is caught by **observation** instead of prevented by
coupling: `tests/test_retrieval.py` migrates a throwaway database, reads the index definition back
out of `pg_indexes`, and asserts that every value in `HNSW_BUILD_PARAMETERS` appears in what the
server actually built. Two independent statements, reconciled against the database rather than
against each other — which is also what `CLAUDE.md` §3 rule 6 asks of every guard here.

The extension statement runs first and uses `IF NOT EXISTS`, because a managed provider may have
installed pgvector already and a migration that failed on a database that was *more* ready than
expected would be an obstacle rather than a check.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "0001_initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

#: 384 is the output width of BAAI/bge-small-en-v1.5, which `retrieval.embeddings.EMBEDDING_POLICY`
#: declares. Written out here for the reason in the module docstring; the test reconciles the two.
EMBEDDING_DIMENSIONS = 384


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "document",
        sa.Column("document_id", sa.String(length=64), nullable=False),
        sa.Column("program_id", sa.String(length=64), nullable=False),
        sa.Column("policy_version", sa.String(length=32), nullable=False),
        sa.Column("title", sa.String(length=256), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("is_current", sa.Boolean(), nullable=False),
        sa.Column("superseded_by", sa.String(length=64), nullable=True),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.PrimaryKeyConstraint("document_id", name="pk_document"),
    )
    op.create_index("ix_document_program_id", "document", ["program_id"])

    op.create_table(
        "clause",
        sa.Column("clause_id", sa.String(length=64), nullable=False),
        sa.Column("program_id", sa.String(length=64), nullable=False),
        sa.Column("policy_version", sa.String(length=32), nullable=False),
        sa.Column("document_id", sa.String(length=64), nullable=False),
        sa.Column("section", sa.String(length=128), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("start_offset", sa.Integer(), nullable=False),
        sa.Column("end_offset", sa.Integer(), nullable=False),
        sa.Column("governs", sa.ARRAY(sa.Text()), nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIMENSIONS), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["document.document_id"],
            name="fk_clause_document_id_document",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("clause_id", name="pk_clause"),
    )

    # The metadata filter's index. It exists so that the plan which filters before ranking stays the
    # cheap one as the corpus grows: retrieval's correctness does not depend on the planner choosing
    # it, because the predicate is in the same statement either way, but its cost does.
    op.create_index("ix_clause_program_policy", "clause", ["program_id", "policy_version"])

    # The vector index. `vector_cosine_ops` must match the `<=>` operator in the retrieval
    # statement; an index built with a different operator class is simply never used, silently.
    op.create_index(
        "ix_clause_embedding_hnsw",
        "clause",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 200},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    # The extension is not dropped. It is a database-wide object that this migration found absent
    # and created, but other schemas in the same database may now depend on it, and `DROP EXTENSION`
    # cascades to every vector column anywhere. A downgrade that can destroy data outside the tables
    # it created is not a downgrade.
    op.drop_index("ix_clause_embedding_hnsw", table_name="clause")
    op.drop_index("ix_clause_program_policy", table_name="clause")
    op.drop_table("clause")
    op.drop_index("ix_document_program_id", table_name="document")
    op.drop_table("document")
