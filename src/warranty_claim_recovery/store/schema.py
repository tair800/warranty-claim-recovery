"""The two tables retrieval needs, and why each column is there rather than derived.

There are exactly two: `document` holds a policy or service bulletin as one uninterrupted string,
and `clause` holds the retrievable chunks of it together with their vectors. The split is not
normalisation for its own sake — it is what makes kill condition I checkable.

**Why the full document is stored even though every clause already carries its own text.** A
citation in this system is a quote plus a pair of offsets into a named document version. Kill
condition I verifies a citation by slicing that document at those offsets and comparing, and it
allows zero failures. If the only text in the database were the clause's own copy, the check would
degenerate into comparing the clause text with itself, which passes for every citation including the
ones whose coordinates have drifted. The document row is the independent copy that makes the
comparison mean something, and storing it is therefore the cheap half of the guarantee rather than
duplication.

**Why `policy_version` is on the clause as well as on the document.** It is derivable by a join, and
it is denormalised on purpose, because the metadata filter must run inside the same statement as the
vector search. A join in that statement would give the planner a reason to materialise the ranking
first and join afterwards, and the ordering of those two operations is the entire argument of the
retrieval stage. A redundant column costs one string per clause; a post-filter silently shortens
every result list.

**Why the HNSW parameters are exported as a constant.** `artifacts/pgvector.json` publishes the
index definition, and a published number that is typed out a second time in prose is a number that
will eventually disagree with the server. `HNSW_BUILD_PARAMETERS` is the single source, and the
evidence the artifact carries is read back from `pg_indexes` rather than asserted from here.

**Why cosine and not L2.** The encoder returns L2-normalised vectors — measured, not assumed; see
`artifacts/encoder_memory.json` and the norm assertion in `tests/test_retrieval.py`. For unit
vectors, cosine distance and squared Euclidean distance rank identically, so the choice does not
change today's results. It matters because the operator class of the index and the operator in the
`ORDER BY` must agree or PostgreSQL cannot use the index at all, and `vector_cosine_ops` with `<=>`
is the pair that keeps working if a later encoder stops normalising. Rejected: `vector_ip_ops` with
`<#>`, which is marginally cheaper and is only equivalent to cosine while the vectors stay
normalised — a condition no part of this system enforces.
"""

from __future__ import annotations

from typing import Any, Final

from pgvector.sqlalchemy import Vector
from sqlalchemy import ARRAY, Boolean, ForeignKey, Index, Integer, MetaData, String, Text, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from warranty_claim_recovery.retrieval.embeddings import EMBEDDING_POLICY

__all__ = [
    "CLAUSE_TABLE",
    "DOCUMENT_TABLE",
    "HNSW_BUILD_PARAMETERS",
    "METADATA",
    "VECTOR_EXTENSION_STATEMENT",
    "Base",
    "ClauseRow",
    "DocumentRow",
    "create_schema",
]

DOCUMENT_TABLE: Final = "document"
CLAUSE_TABLE: Final = "clause"

#: Run before anything else touches a `vector` column. `IF NOT EXISTS` because both Alembic and
#: `create_schema` execute it and neither may assume it is first.
VECTOR_EXTENSION_STATEMENT: Final = "CREATE EXTENSION IF NOT EXISTS vector"

# The HNSW build parameters, one value each, so that the index, the migration's expectation and the
# published artifact cannot drift apart. m=16 and ef_construction=200 are pgvector's documented
# defaults for m and a doubled ef_construction: this corpus is small enough that a longer build is
# free, and recall at k=5 is a graded criterion where the index's own approximation error would be
# indistinguishable from a retrieval failure. Rejected: tuning these against the hold-out. ADR-001
# §7 forbids tuning anything against the hold-out, and an index parameter is not an exception.
HNSW_INDEX_NAME: Final = "ix_clause_embedding_hnsw"
HNSW_METHOD: Final = "hnsw"
HNSW_OPCLASS: Final = "vector_cosine_ops"
HNSW_M: Final = 16
HNSW_EF_CONSTRUCTION: Final = 200

#: The exported view of the five constants above. `artifacts/pgvector.json` reports these next to
#: the definition read back from the server, so a reader can see that the claim and the database
#: agree without trusting either one alone.
HNSW_BUILD_PARAMETERS: Final[dict[str, Any]] = {
    "index_name": HNSW_INDEX_NAME,
    "method": HNSW_METHOD,
    "opclass": HNSW_OPCLASS,
    "m": HNSW_M,
    "ef_construction": HNSW_EF_CONSTRUCTION,
}

#: The btree that supports the metadata filter. Named and declared rather than left to the planner:
#: the filter runs before ranking by construction, and an index that makes the filtered plan cheap
#: is what stops a planner from deciding otherwise on a corpus that later grows.
METADATA_FILTER_INDEX_NAME: Final = "ix_clause_program_policy"

# Explicit naming so that a constraint's name is a property of the schema rather than of whichever
# PostgreSQL generated it. An unnamed constraint cannot be dropped by a later migration without
# first querying the catalogue for whatever name the server invented.
_NAMING_CONVENTION: Final[dict[str, str]] = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

METADATA: Final = MetaData(naming_convention=_NAMING_CONVENTION)


class Base(DeclarativeBase):
    """The declarative base, carrying the shared `MetaData` the migration is generated against."""

    metadata = METADATA


class DocumentRow(Base):
    """A policy or service bulletin, stored whole.

    `text` is the exact string the offsets in `clause` and in every `Citation` index into. It is
    never normalised, re-wrapped or stripped on the way in: a whitespace change applied at load time
    shifts every offset after it, and a citation that was faithful when it was written becomes
    unfaithful without anything in the system having changed its mind.
    """

    __tablename__ = DOCUMENT_TABLE

    document_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    program_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    policy_version: Mapped[str] = mapped_column(String(32), nullable=False)
    title: Mapped[str] = mapped_column(String(256), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    #: `policy` or a service bulletin. Stored because a bulletin and the policy it amends are
    #: different kinds of authority and a correction that cites one as the other is wrong in a way
    #: an adjudicator notices.
    kind: Mapped[str] = mapped_column(String(32), nullable=False, default="policy")
    #: False when this document has been withdrawn or replaced. **Nothing in this package filters on
    #: it**, deliberately: the metadata filter is programme and policy version, as ADR-001's
    #: retrieval order fixes it, and widening a predeclared stage is not a decision one module makes
    #: on its own. It is stored so that the stage which composes a correction can refuse to cite a
    #: withdrawn bulletin, and so that the question can be asked at all. In the committed corpus
    #: every withdrawn bulletin shares a programme and a policy version with a current one and
    #: governs no rejection code, so it is a distractor rather than authority.
    is_current: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    superseded_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    #: The loader's idempotency key. A rerun that finds this value unchanged writes nothing, which
    #: is what makes `scripts/seed_index.py` safe to run from a crashed position.
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)


class ClauseRow(Base):
    """One retrievable chunk of a document, with its coordinates and its vector.

    `start_offset` and `end_offset` are the chunk's position in `DocumentRow.text` and they are the
    reason this table is not merely a cache of embeddings. A retrieval result that cannot be located
    in a named document version is an assertion, and this project's third claim is that it never
    makes one.
    """

    __tablename__ = CLAUSE_TABLE

    clause_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    program_id: Mapped[str] = mapped_column(String(64), nullable=False)
    #: Denormalised from the document on purpose; see the module docstring.
    policy_version: Mapped[str] = mapped_column(String(32), nullable=False)
    document_id: Mapped[str] = mapped_column(
        String(64), ForeignKey(f"{DOCUMENT_TABLE}.document_id", ondelete="CASCADE"), nullable=False
    )
    section: Mapped[str] = mapped_column(String(128), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    start_offset: Mapped[int] = mapped_column(Integer, nullable=False)
    end_offset: Mapped[int] = mapped_column(Integer, nullable=False)
    #: The rejection codes this clause has authority over. A PostgreSQL array rather than a join
    #: table: it is read as a whole on every row that is returned and never queried by element, so
    #: a join table would add a second statement to the retrieval path and buy nothing.
    governs: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, default=list)
    # Annotated `Any` because no single static type is true in both directions: the bind side takes
    # a sequence of floats, and the result side is a string from psycopg that pgvector's result
    # processor turns into a `numpy.ndarray`. Claiming `list[float]` here would be a type that is
    # wrong on read. The column is never read back through the ORM — retrieval selects the distance,
    # not the vector — so nothing depends on narrowing it.
    embedding: Mapped[Any] = mapped_column(Vector(EMBEDDING_POLICY.dimensions), nullable=False)
    #: Covers the clause content **and** the embedding policy; see `loader.content_fingerprint` for
    #: why the model identity has to be inside the idempotency key.
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)


Index(
    METADATA_FILTER_INDEX_NAME,
    ClauseRow.program_id,
    ClauseRow.policy_version,
)

Index(
    HNSW_INDEX_NAME,
    ClauseRow.embedding,
    postgresql_using=HNSW_METHOD,
    postgresql_with={"m": HNSW_M, "ef_construction": HNSW_EF_CONSTRUCTION},
    postgresql_ops={"embedding": HNSW_OPCLASS},
)


def create_schema(engine: Engine) -> None:
    """Build the whole schema from `METADATA`, for a throwaway database only.

    Alembic owns the shipped schema. This exists so that a test can stand up an isolated database
    without the migration, and it builds from the same `MetaData` the migration was generated
    against, so the two cannot describe different tables.

    It is **not** an alternative to migrating. A deployment that called this would have a database
    with no `alembic_version` row, and the next `alembic upgrade head` would try to create tables
    that already exist and fail in a way that reads like a corrupt migration history. The tests that
    exercise the shipped index run the real migration for exactly this reason; this helper is for
    the cases where only the table shapes are under test.
    """
    with engine.begin() as connection:
        connection.execute(text(VECTOR_EXTENSION_STATEMENT))
    METADATA.create_all(engine)
