"""Retrieval against a real PostgreSQL with a real pgvector index.

These tests need a server and they run against a **throwaway database** that they create, migrate
with the shipped Alembic revision, and drop. That costs a second or two and buys three things a test
against the development database cannot have: it cannot leave rows behind that a later corpus
evaluation would score, it cannot pass because of something a previous run happened to insert, and
it exercises `alembic upgrade head` on every run rather than assuming somebody ran it.

The test this file exists for is
`test_the_metadata_filter_runs_in_the_same_statement_as_the_ranking`. It builds a corpus in which
the five globally nearest clauses all belong to a programme and policy version that was **not**
asked for, then asserts two things: that the retriever returns the three clauses of the programme
that was asked for, and — by running the rank-first statement itself — that a system which ranked
first and filtered in Python would have returned none of them. The failure it catches is silent: a
post-filtered result list is shorter than `k` and nothing in the response says so, so kill condition
J measures it as a retrieval failure and the diagnosis leads to the encoder, which is innocent.

The vectors for that test are scripted rather than produced by the encoder. The question being asked
is about SQL — about which of two operations happens first — and answering it with a real encoder
would make the assertion depend on whether a language model happens to rank one sentence above
another, which is a different question with a different answer on a different day. The encoder gets
its own tests at the bottom of this file, where what is being checked really is the encoder.
"""

from __future__ import annotations

import json
import math
from argparse import Namespace
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import psycopg
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from warranty_claim_recovery.config import get_settings
from warranty_claim_recovery.domain import PolicyClause, RejectionCode
from warranty_claim_recovery.retrieval.embeddings import (
    EMBEDDING_POLICY,
    EmbeddingPolicy,
    FastEmbedEncoder,
)
from warranty_claim_recovery.retrieval.pipeline import RETRIEVAL_STATEMENT, Retriever, compose_query
from warranty_claim_recovery.store import loader as loader_module
from warranty_claim_recovery.store.engine import (
    build_engine,
    normalise_database_url,
    session_scope,
)
from warranty_claim_recovery.store.loader import (
    CorpusShapeError,
    DocumentRecord,
    LoadableClause,
    load_corpus,
    read_clauses,
    read_documents,
)
from warranty_claim_recovery.store.schema import (
    CLAUSE_TABLE,
    HNSW_BUILD_PARAMETERS,
    METADATA,
    create_schema,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

SYNTHETIC_NOTICE = (
    "SYNTHETIC DOCUMENT. Generated for the warranty-claim-recovery test corpus. "
    "Not a manufacturer publication.\n\n"
)


def _base_url() -> Any:
    return make_url(normalise_database_url(get_settings().database_url))


def _server_reachable() -> bool:
    """Ask the server once, at import time, instead of failing sixteen times at assertion time."""
    url = _base_url()
    try:
        with psycopg.connect(
            host=url.host,
            port=url.port,
            user=url.username,
            password=url.password,
            dbname="postgres",
            connect_timeout=2,
        ):
            return True
    except Exception:  # any failure to reach the server means the same thing here
        return False


pytestmark = pytest.mark.skipif(
    not _server_reachable(),
    reason=(
        "PostgreSQL is not reachable at the configured WCR_DATABASE_URL. Run `make db`. These "
        "tests are skipped only when the server is absent; they are never skipped when it is there."
    ),
)


# ------------------------------------------------------------------------------------------------
# Throwaway databases.
# ------------------------------------------------------------------------------------------------


def _admin_engine() -> Engine:
    """AUTOCOMMIT, because CREATE DATABASE cannot run inside a transaction block."""
    return create_engine(_base_url().set(database="postgres"), isolation_level="AUTOCOMMIT")


def _recreate_database(name: str) -> str:
    engine = _admin_engine()
    try:
        with engine.connect() as connection:
            # WITH (FORCE) terminates any connection left behind by an interrupted earlier run.
            # Without it, a suspended machine's stale backend makes every later run fail at setup.
            connection.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
            connection.execute(text(f'CREATE DATABASE "{name}"'))
    finally:
        engine.dispose()
    return _base_url().set(database=name).render_as_string(hide_password=False)


def _drop_database(name: str) -> None:
    engine = _admin_engine()
    try:
        with engine.connect() as connection:
            connection.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
    finally:
        engine.dispose()


def _migrate(url: str) -> None:
    """Run the shipped revision, in process, against one database.

    `Config()` is built without an ini file so that `alembic/env.py` skips `fileConfig`, which would
    otherwise reconfigure logging for the whole pytest session and disable existing loggers. The URL
    arrives through `-x url=`, the same mechanism `env.py` documents, rather than by mutating
    `os.environ` — `get_settings` is `lru_cache`d, so an environment mutation would take effect or
    not depending on which test ran first.
    """
    config = Config()
    config.set_main_option("script_location", str(REPO_ROOT / "alembic"))
    config.cmd_opts = Namespace(x=[f"url={url}"])
    command.upgrade(config, "head")


@pytest.fixture(scope="session")
def migrated_engine() -> Iterator[Engine]:
    """One migrated database for every read-only test in this file."""
    name = "wcr_test_retrieval"
    url = _recreate_database(name)
    _migrate(url)
    engine = build_engine(url)
    try:
        yield engine
    finally:
        engine.dispose()
        _drop_database(name)


@pytest.fixture
def scratch_engine() -> Iterator[Engine]:
    """A fresh migrated database per test, for the tests that write."""
    name = "wcr_test_scratch"
    url = _recreate_database(name)
    _migrate(url)
    engine = build_engine(url)
    try:
        yield engine
    finally:
        engine.dispose()
        _drop_database(name)


# ------------------------------------------------------------------------------------------------
# A scripted corpus with known geometry.
# ------------------------------------------------------------------------------------------------


def _unit(angle: float) -> list[float]:
    """A unit vector at `angle` radians from the query axis.

    Cosine distance is then `1 - cos(angle)`, which is strictly increasing on [0, pi]. Writing the
    geometry down like this means every expected ordering in this file is arithmetic rather than a
    guess about what an encoder will do.
    """
    vector = [0.0] * EMBEDDING_POLICY.dimensions
    vector[0] = math.cos(angle)
    vector[1] = math.sin(angle)
    return vector


class ScriptedEncoder:
    """A `TextEncoder` whose geometry the test chooses, and which records what it was asked.

    It raises `KeyError` on a passage it was not given an angle for. A default would let a test
    silently embed something it did not intend at a distance it did not choose, and the assertion
    that then failed would be about ranking rather than about the fixture.
    """

    def __init__(self, angles: Mapping[str, float]) -> None:
        self._angles = dict(angles)
        self.queries: list[str] = []
        self.passages: list[str] = []

    @property
    def dimensions(self) -> int:
        return EMBEDDING_POLICY.dimensions

    def encode_passages(self, texts: Sequence[str]) -> list[list[float]]:
        self.passages.extend(texts)
        return [_unit(self._angles[text]) for text in texts]

    def encode_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return _unit(0.0)


def _clause_text(document_id: str, index: int) -> str:
    return (
        f"Section {index + 1:02d} of {document_id}: the distributor shall retain the failed "
        f"component for ninety days and record its serial number on the repair order."
    )


def _make_document(
    *,
    document_id: str,
    program_id: str,
    policy_version: str,
    count: int,
    first_angle: float,
    step: float,
) -> tuple[DocumentRecord, list[LoadableClause], dict[str, float]]:
    """A document whose clause offsets are computed, not asserted.

    The offsets are derived from the text as it is assembled, so the fixture cannot drift from the
    document the way a hand-written offset would. `read_clauses` verifies them again on the way in,
    which is the check that matters; building them correctly here just means a fixture bug reads as
    a fixture bug.
    """
    header = f"{SYNTHETIC_NOTICE}{document_id} — policy version {policy_version}\n\n"
    bodies: list[str] = []
    clauses: list[LoadableClause] = []
    angles: dict[str, float] = {}
    offset = len(header)

    for index in range(count):
        body = _clause_text(document_id, index)
        clause = PolicyClause(
            clause_id=f"{document_id}-C{index:03d}",
            program_id=program_id,
            document_id=document_id,
            section=f"Section {index + 1:02d}",
            text=body,
            start_offset=offset,
            end_offset=offset + len(body),
            governs=(RejectionCode.MISSING_SERIAL,)
            if index % 2 == 0
            else (RejectionCode.LABOUR_RATE_EXCEEDED, RejectionCode.PART_NOT_COVERED),
        )
        clauses.append(LoadableClause(clause=clause, policy_version=policy_version))
        angles[body] = first_angle + index * step
        bodies.append(body)
        offset += len(body) + 2

    record = DocumentRecord(
        document_id=document_id,
        program_id=program_id,
        policy_version=policy_version,
        title=f"{document_id} warranty policy",
        text=header + "\n\n".join(bodies),
    )
    return record, clauses, angles


def _scripted_corpus() -> tuple[dict[str, DocumentRecord], list[LoadableClause], ScriptedEncoder]:
    """Three documents, arranged so that the nearest clauses are the ones nobody asked for.

    * `PGM-A` version `v1` — three clauses, far from the query.
    * `PGM-A` version `v2` — six clauses, the nearest in the corpus. Another version of the same
      manufacturer's policy is the realistic near neighbour, and the one a version filter must
      exclude.
    * `PGM-B` version `v1` — sixty clauses, nearer than `PGM-A/v1` and further than `PGM-A/v2`. A
      second manufacturer whose policy is written from the same template, which is why a
      manufacturer filter cannot be left to the ranking.
    """
    specifications = [
        ("DOC-A-V1", "PGM-A", "v1", 3, 1.00, 0.05),
        ("DOC-A-V2", "PGM-A", "v2", 6, 0.02, 0.01),
        ("DOC-B-V1", "PGM-B", "v1", 60, 0.10, 0.005),
    ]
    documents: dict[str, DocumentRecord] = {}
    clauses: list[LoadableClause] = []
    angles: dict[str, float] = {}
    for document_id, program_id, policy_version, count, first, step in specifications:
        record, loadable, document_angles = _make_document(
            document_id=document_id,
            program_id=program_id,
            policy_version=policy_version,
            count=count,
            first_angle=first,
            step=step,
        )
        documents[document_id] = record
        clauses.extend(loadable)
        angles.update(document_angles)
    return documents, clauses, ScriptedEncoder(angles)


@pytest.fixture(scope="session")
def seeded(migrated_engine: Engine) -> Iterator[tuple[Engine, ScriptedEncoder]]:
    documents, clauses, encoder = _scripted_corpus()
    with session_scope(migrated_engine) as session:
        load_corpus(session, documents=documents, clauses=clauses, encoder=encoder)
    yield migrated_engine, encoder


@pytest.fixture
def session(seeded: tuple[Engine, ScriptedEncoder]) -> Iterator[Session]:
    engine, _ = seeded
    with session_scope(engine) as open_session:
        yield open_session


@pytest.fixture
def retriever(seeded: tuple[Engine, ScriptedEncoder]) -> Retriever:
    _, encoder = seeded
    return Retriever(encoder)


# ------------------------------------------------------------------------------------------------
# The shipped schema, read back from the server that built it.
# ------------------------------------------------------------------------------------------------


def test_the_shipped_migration_builds_the_index_the_constant_declares(
    migrated_engine: Engine,
) -> None:
    """`HNSW_BUILD_PARAMETERS` and the migration are two independent statements, reconciled here.

    The migration writes its values out rather than importing the constant, because a migration that
    reads a live constant replays differently after that constant changes — see the revision's own
    docstring. Drift is therefore caught by observation: this reads what PostgreSQL actually built
    and checks it against what the artifact will publish.
    """
    with migrated_engine.connect() as connection:
        definition = connection.execute(
            text("SELECT indexdef FROM pg_indexes WHERE indexname = :name"),
            {"name": HNSW_BUILD_PARAMETERS["index_name"]},
        ).scalar_one()

    # PostgreSQL prints storage parameters as m='16'; compare against the unquoted form.
    unquoted = definition.replace("'", "")
    assert f"USING {HNSW_BUILD_PARAMETERS['method']}" in unquoted
    assert HNSW_BUILD_PARAMETERS["opclass"] in unquoted
    assert f"m={HNSW_BUILD_PARAMETERS['m']}" in unquoted
    assert f"ef_construction={HNSW_BUILD_PARAMETERS['ef_construction']}" in unquoted


def test_the_embedding_column_is_a_vector_of_the_declared_width(migrated_engine: Engine) -> None:
    """A `vector` column, not an array of doubles that a distance operator could never use."""
    with migrated_engine.connect() as connection:
        type_name, dimensions = connection.execute(
            text(
                "SELECT a.atttypid::regtype::text, a.atttypmod FROM pg_attribute a "
                "JOIN pg_class c ON c.oid = a.attrelid "
                "WHERE c.relname = :table AND a.attname = 'embedding'"
            ),
            {"table": CLAUSE_TABLE},
        ).one()

    assert type_name == "vector"
    assert dimensions == EMBEDDING_POLICY.dimensions


def test_create_schema_and_the_migration_describe_the_same_tables(scratch_engine: Engine) -> None:
    """`create_schema` is for throwaway databases and must not be a second, divergent definition.

    It builds from `METADATA`, which is what Alembic's autogenerate compares against, so this is the
    check that the claim in `store.schema`'s docstring is true rather than aspirational.
    """
    migrated = inspect(scratch_engine)
    migrated_tables = {
        name: {
            (column["name"], str(column["type"]), column["nullable"])
            for column in migrated.get_columns(name)
        }
        for name in migrated.get_table_names()
        if name != "alembic_version"
    }

    name = "wcr_test_create_schema"
    url = _recreate_database(name)
    engine = build_engine(url)
    try:
        create_schema(engine)
        built = inspect(engine)
        built_tables = {
            table: {
                (column["name"], str(column["type"]), column["nullable"])
                for column in built.get_columns(table)
            }
            for table in built.get_table_names()
        }
    finally:
        engine.dispose()
        _drop_database(name)

    assert set(built_tables) == set(migrated_tables) == set(METADATA.tables)
    assert built_tables == migrated_tables


# ------------------------------------------------------------------------------------------------
# The ordering that is the argument.
# ------------------------------------------------------------------------------------------------


def test_the_metadata_filter_runs_in_the_same_statement_as_the_ranking(
    session: Session, retriever: Retriever
) -> None:
    """The point of this file. See the module docstring.

    Three assertions in one test, deliberately, because separately none of them means what they mean
    together: the retriever returns every clause of the requested programme and version; the five
    globally nearest clauses contain none of them; therefore no amount of post-filtering a ranked
    list could have produced this answer.
    """
    result = retriever.retrieve(
        session,
        query="which clause governs a missing serial number",
        program_id="PGM-A",
        policy_version="v1",
        rejection_code=RejectionCode.MISSING_SERIAL,
        k=5,
    )

    assert len(result.clauses) == 3
    assert {scored.clause.program_id for scored in result.clauses} == {"PGM-A"}
    assert {scored.clause.document_id for scored in result.clauses} == {"DOC-A-V1"}

    globally_nearest = session.execute(
        text(
            "SELECT program_id, policy_version FROM clause "
            "ORDER BY embedding <=> CAST(:query_vector AS vector) LIMIT 5"
        ),
        {"query_vector": "[" + ",".join(str(value) for value in _unit(0.0)) + "]"},
    ).all()

    survivors = [row for row in globally_nearest if (row[0], row[1]) == ("PGM-A", "v1")]
    assert survivors == [], (
        "the fixture no longer reproduces the situation this test exists for: the globally nearest "
        "clauses must belong to a different programme or version, or post-filtering would "
        "accidentally give the right answer"
    )


def test_a_superseded_policy_version_of_the_same_manufacturer_is_excluded(
    session: Session, retriever: Retriever
) -> None:
    """A correct quote from the wrong policy version is the wrong authority. See `Citation`."""
    result = retriever.retrieve(
        session,
        query="which clause governs a missing serial number",
        program_id="PGM-A",
        policy_version="v2",
        rejection_code=RejectionCode.MISSING_SERIAL,
        k=5,
    )
    assert len(result.clauses) == 5
    assert {scored.clause.document_id for scored in result.clauses} == {"DOC-A-V2"}


def test_the_result_is_ordered_by_distance_and_the_ranks_are_dense(
    session: Session, retriever: Retriever
) -> None:
    result = retriever.retrieve(
        session,
        query="labour rate above the published schedule",
        program_id="PGM-B",
        policy_version="v1",
        rejection_code=RejectionCode.LABOUR_RATE_EXCEEDED,
        k=5,
    )
    distances = [scored.distance for scored in result.clauses]
    assert distances == sorted(distances)
    assert [scored.rank for scored in result.clauses] == [1, 2, 3, 4, 5]

    # The geometry is known, so the distance is arithmetic rather than an observation: the nearest
    # PGM-B clause sits at 0.10 radians from the query axis.
    assert result.clauses[0].distance == pytest.approx(1 - math.cos(0.10), abs=1e-6)


def test_the_executed_statement_carries_the_operator_and_the_metadata_predicate(
    session: Session, retriever: Retriever
) -> None:
    """Kill condition K greps this string out of an artifact, so it must be the one that ran."""
    result = retriever.retrieve(
        session,
        query="serial number",
        program_id="PGM-A",
        policy_version="v1",
        rejection_code=RejectionCode.MISSING_SERIAL,
        k=5,
    )
    statement = result.executed_statement
    assert statement == RETRIEVAL_STATEMENT
    assert "<=>" in statement
    assert "WHERE program_id = :program_id" in statement
    assert "AND policy_version = :policy_version" in statement
    # The predicate precedes the ranking in the text as well as in the plan. A statement that
    # ordered first and filtered in a subquery would still contain both fragments.
    assert statement.index("WHERE program_id") < statement.index("ORDER BY")


def test_the_servers_own_plan_performs_the_distance_arithmetic(
    session: Session, retriever: Retriever
) -> None:
    """Kill condition K, from `EXPLAIN ANALYZE` and not from this repository's own SQL text."""
    plan = retriever.explain_plan(
        session,
        query="serial number",
        program_id="PGM-A",
        policy_version="v1",
        rejection_code=RejectionCode.MISSING_SERIAL,
        k=5,
    )
    assert "<=>" in plan
    assert "::vector" in plan
    # ADR-001 §5: the criterion asks whether the extension does the arithmetic, not which plan the
    # planner chose. Asserting a vector index scan here would fail on a filter selective enough to
    # make a sequential scan correctly cheaper, which is exactly what happened to project 7.
    assert "actual time" in plan, "ANALYZE must have executed the statement, not merely planned it"


def test_k_below_one_is_refused_before_anything_is_embedded(
    session: Session, retriever: Retriever
) -> None:
    with pytest.raises(ValueError, match="asks for no results"):
        retriever.retrieve(
            session,
            query="serial number",
            program_id="PGM-A",
            policy_version="v1",
            rejection_code=RejectionCode.MISSING_SERIAL,
            k=0,
        )


def test_the_rejection_code_reaches_the_encoder_and_never_the_where_clause(
    session: Session, retriever: Retriever
) -> None:
    """Filtering on `governs` would make this retriever identical to a predeclared baseline.

    ADR-001 §6 predeclares `exact_code_lookup` — resolving the governing clause by a code-to-clause
    lookup with no text search. Kill condition J requires the system to beat every baseline, and a
    system that *is* one of them cannot. So the code enriches the embedded query instead, and the
    statement's only predicate is the programme and the policy version.
    """
    encoder = retriever.embedder
    assert isinstance(encoder, ScriptedEncoder)
    before = len(encoder.queries)

    retriever.retrieve(
        session,
        query="which clause governs a missing serial number",
        program_id="PGM-A",
        policy_version="v1",
        rejection_code=RejectionCode.MISSING_SERIAL,
        k=5,
    )

    assert encoder.queries[before:] == [
        compose_query("which clause governs a missing serial number", RejectionCode.MISSING_SERIAL)
    ]
    assert RejectionCode.MISSING_SERIAL.value in encoder.queries[-1]
    assert "governs" not in RETRIEVAL_STATEMENT.split("WHERE", 1)[1].split("ORDER BY", 1)[0]


def test_governs_survives_the_round_trip_as_the_closed_enum(
    session: Session, retriever: Retriever
) -> None:
    result = retriever.retrieve(
        session,
        query="serial number",
        program_id="PGM-A",
        policy_version="v1",
        rejection_code=RejectionCode.MISSING_SERIAL,
        k=5,
    )
    by_id = {scored.clause.clause_id: scored.clause for scored in result.clauses}
    assert by_id["DOC-A-V1-C000"].governs == (RejectionCode.MISSING_SERIAL,)
    assert by_id["DOC-A-V1-C001"].governs == (
        RejectionCode.LABOUR_RATE_EXCEEDED,
        RejectionCode.PART_NOT_COVERED,
    )


def test_a_stored_code_the_system_does_not_define_is_refused_on_read(
    scratch_engine: Engine,
) -> None:
    """A corpus that has drifted from `RejectionCode` must raise, not return a plain string.

    The requirement matrix is a total function over the enum. An unknown code that flowed through as
    text would reach that lookup, find nothing, and read as "this rejection demands no evidence" —
    which gates a resubmission that nothing supports.
    """
    documents, clauses, encoder = _scripted_corpus()
    with session_scope(scratch_engine) as session:
        load_corpus(session, documents=documents, clauses=clauses[:3], encoder=encoder)
        session.execute(
            text("UPDATE clause SET governs = ARRAY['NOT_A_REAL_CODE'] WHERE clause_id = :id"),
            {"id": "DOC-A-V1-C000"},
        )
        session.commit()

        with pytest.raises(ValueError, match="does not define"):
            Retriever(encoder).retrieve(
                session,
                query="serial number",
                program_id="PGM-A",
                policy_version="v1",
                rejection_code=RejectionCode.MISSING_SERIAL,
                k=5,
            )


# ------------------------------------------------------------------------------------------------
# The loader: idempotent, resumable, and invalidated by a policy change.
# ------------------------------------------------------------------------------------------------


def test_a_second_run_over_an_unchanged_corpus_embeds_nothing(scratch_engine: Engine) -> None:
    documents, clauses, encoder = _scripted_corpus()
    with session_scope(scratch_engine) as session:
        first = load_corpus(session, documents=documents, clauses=clauses, encoder=encoder)
        embedded_after_first = len(encoder.passages)
        second = load_corpus(session, documents=documents, clauses=clauses, encoder=encoder)

    assert first.clauses_written == len(clauses)
    assert first.clauses_unchanged == 0
    assert second.clauses_written == 0
    assert second.clauses_embedded == 0
    assert second.clauses_unchanged == len(clauses)
    assert second.documents_written == 0
    assert len(encoder.passages) == embedded_after_first, "the rerun called the encoder again"


def test_a_changed_embedding_policy_invalidates_every_clause(
    scratch_engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The failure `clause_fingerprint` exists to prevent, planted by behaviour rather than grepped.

    A key built from the clause text alone reports every clause unchanged after a model swap, and
    the table keeps the old model's vectors while the retriever asks in the new model's space.
    Nothing raises; recall falls by an amount nobody can attribute. Replacing the policy the loader
    reads is the smallest change that reproduces a model swap.
    """
    documents, clauses, encoder = _scripted_corpus()
    with session_scope(scratch_engine) as session:
        load_corpus(session, documents=documents, clauses=clauses, encoder=encoder)

        monkeypatch.setattr(
            loader_module,
            "EMBEDDING_POLICY",
            EmbeddingPolicy(
                model_name="a-different-encoder",
                dimensions=EMBEDDING_POLICY.dimensions,
                normalised=True,
                query_instruction="",
                passage_instruction="",
            ),
        )
        after_swap = load_corpus(session, documents=documents, clauses=clauses, encoder=encoder)

    assert after_swap.clauses_unchanged == 0
    assert after_swap.clauses_embedded == len(clauses)
    # The documents are not embedded, so a model swap must not rewrite them.
    assert after_swap.documents_written == 0


def test_the_loader_refuses_a_clause_that_is_not_at_its_own_offsets(tmp_path: Path) -> None:
    """Kill condition I, enforced at the boundary where the error can still name a file."""
    document, clauses, _ = _make_document(
        document_id="DOC-X",
        program_id="PGM-X",
        policy_version="v1",
        count=3,
        first_angle=0.1,
        step=0.1,
    )
    documents_path = tmp_path / "documents.json"
    clauses_path = tmp_path / "clauses.json"
    documents_path.write_text(json.dumps([document._asdict()]), encoding="utf-8")

    payload = []
    for index, loadable in enumerate(clauses):
        entry = loadable.clause.model_dump(mode="json")
        if index == 1:
            entry["start_offset"] += 3
            entry["end_offset"] += 3
        payload.append(entry)
    clauses_path.write_text(json.dumps({"clauses": payload}), encoding="utf-8")

    parsed = read_documents(documents_path)
    with pytest.raises(CorpusShapeError, match="holds different text"):
        read_clauses(clauses_path, parsed)


def test_a_clause_whose_document_is_missing_is_refused(tmp_path: Path) -> None:
    document, clauses, _ = _make_document(
        document_id="DOC-X",
        program_id="PGM-X",
        policy_version="v1",
        count=2,
        first_angle=0.1,
        step=0.1,
    )
    (tmp_path / "documents.json").write_text(json.dumps([document._asdict()]), encoding="utf-8")
    entry = clauses[0].clause.model_dump(mode="json")
    entry["document_id"] = "DOC-ABSENT"
    (tmp_path / "clauses.json").write_text(json.dumps([entry]), encoding="utf-8")

    with pytest.raises(CorpusShapeError, match="not in the document corpus"):
        read_clauses(tmp_path / "clauses.json", read_documents(tmp_path / "documents.json"))


def test_a_clause_declaring_a_policy_version_its_document_does_not_have_is_refused(
    tmp_path: Path,
) -> None:
    document, clauses, _ = _make_document(
        document_id="DOC-X",
        program_id="PGM-X",
        policy_version="v1",
        count=2,
        first_angle=0.1,
        step=0.1,
    )
    (tmp_path / "documents.json").write_text(json.dumps([document._asdict()]), encoding="utf-8")
    entry = clauses[0].clause.model_dump(mode="json")
    entry["policy_version"] = "v9"
    (tmp_path / "clauses.json").write_text(json.dumps([entry]), encoding="utf-8")

    with pytest.raises(CorpusShapeError, match="policy version"):
        read_clauses(tmp_path / "clauses.json", read_documents(tmp_path / "documents.json"))


def test_an_envelope_without_the_expected_key_is_refused_rather_than_guessed(
    tmp_path: Path,
) -> None:
    """Guessing which value is the record list loads nothing and reports success."""
    path = tmp_path / "documents.json"
    path.write_text(json.dumps({"note": "synthetic", "records": []}), encoding="utf-8")
    with pytest.raises(CorpusShapeError, match="will not guess"):
        read_documents(path)


def test_a_synthetic_notice_in_the_envelope_does_not_stop_the_corpus_loading(
    tmp_path: Path,
) -> None:
    """The generator puts its synthetic-corpus notice beside the records; that must still parse."""
    document, _, _ = _make_document(
        document_id="DOC-X",
        program_id="PGM-X",
        policy_version="v1",
        count=1,
        first_angle=0.1,
        step=0.1,
    )
    path = tmp_path / "documents.json"
    path.write_text(
        json.dumps({"is_synthetic": True, "documents": [document._asdict()]}), encoding="utf-8"
    )
    assert set(read_documents(path)) == {"DOC-X"}


# ------------------------------------------------------------------------------------------------
# The shipped encoder. These load the real ONNX session.
# ------------------------------------------------------------------------------------------------


def test_the_shipped_encoder_returns_unit_vectors_of_the_declared_width() -> None:
    """`EMBEDDING_POLICY.normalised` is a claim about the model, so it is measured.

    `store.schema` chooses `vector_cosine_ops` partly on the strength of it, and
    `artifacts/pgvector.json` publishes it. A published property that nothing checks is a property
    that becomes false when the model changes.
    """
    encoder = FastEmbedEncoder()
    vectors = encoder.encode_passages(
        [
            "The warranty period runs from the in-service date recorded on the delivery note.",
            "Labour is reimbursed at the published schedule rate and not at the invoiced rate.",
        ]
    )
    assert encoder.dimensions == EMBEDDING_POLICY.dimensions
    for vector in vectors:
        assert len(vector) == EMBEDDING_POLICY.dimensions
        norm = math.sqrt(sum(component * component for component in vector))
        assert norm == pytest.approx(1.0, abs=1e-3)
        assert EMBEDDING_POLICY.normalised is True


def test_the_query_instruction_is_applied_to_queries_and_never_to_passages() -> None:
    """The asymmetry BGE is trained for, checked on the text that reaches the model.

    Prepending the instruction to passages as well would shift every stored vector by the same
    direction and degrade ranking uniformly, which is the hardest kind of degradation to attribute.
    The model is never loaded here: `_embed` is replaced, so what is under test is the preparation
    rather than the inference.
    """
    captured: list[list[str]] = []

    class Capturing(FastEmbedEncoder):
        def _embed(self, prepared: Sequence[str]) -> list[list[float]]:
            captured.append(list(prepared))
            return [[0.0] * EMBEDDING_POLICY.dimensions for _ in prepared]

    encoder = Capturing()
    encoder.encode_query("which clause governs a missing serial number")
    encoder.encode_passages(["the distributor shall record the serial number on the repair order"])

    assert captured[0] == [
        EMBEDDING_POLICY.query_instruction + "which clause governs a missing serial number"
    ]
    assert captured[1] == ["the distributor shall record the serial number on the repair order"]
    assert EMBEDDING_POLICY.query_instruction != ""
    assert EMBEDDING_POLICY.passage_instruction == ""


def test_the_shipped_encoder_retrieves_the_governing_clause(scratch_engine: Engine) -> None:
    """The whole stage, end to end, on the real model and the real index.

    Small and deliberately unambiguous: five clauses about five different obligations, and a query
    that names one of them. This is not a measurement of retrieval quality — ADR-001 §6 fixes four
    baselines and `artifacts/retrieval.json` carries that number over the hold-out. It is the check
    that the parts fit together: the instruction policy, the vector column, the operator and the
    filter, with nothing scripted.
    """
    topics = [
        "The serial number of the failed component must be recorded on the repair order and on "
        "the claim, and a claim without it is rejected under code MISSING_SERIAL.",
        "Labour is reimbursed at the published schedule rate for the model concerned, and any "
        "amount invoiced above that rate is excluded from the recovery.",
        "Corrosion arising from the operating environment is excluded from cover unless the "
        "protective coating was applied at the factory.",
        "Damage occurring in transit is a carrier claim and is not a warranty claim, whatever the "
        "condition of the component on arrival.",
        "Software updates published after the in-service date are supplied without charge and are "
        "not reimbursable as a repair.",
    ]
    header = f"{SYNTHETIC_NOTICE}DOC-REAL — policy version v1\n\n"
    offset = len(header)
    clauses: list[LoadableClause] = []
    for index, body in enumerate(topics):
        clauses.append(
            LoadableClause(
                clause=PolicyClause(
                    clause_id=f"DOC-REAL-C{index:03d}",
                    program_id="PGM-REAL",
                    document_id="DOC-REAL",
                    section=f"Section {index + 1}",
                    text=body,
                    start_offset=offset,
                    end_offset=offset + len(body),
                    governs=(RejectionCode.MISSING_SERIAL,) if index == 0 else (),
                ),
                policy_version="v1",
            )
        )
        offset += len(body) + 2

    document = DocumentRecord(
        document_id="DOC-REAL",
        program_id="PGM-REAL",
        policy_version="v1",
        title="DOC-REAL warranty policy",
        text=header + "\n\n".join(topics),
    )

    with session_scope(scratch_engine) as session:
        load_corpus(
            session,
            documents={"DOC-REAL": document},
            clauses=clauses,
            encoder=FastEmbedEncoder(),
        )
        result = Retriever().retrieve(
            session,
            query="the claim was rejected because the serial number of the part was not supplied",
            program_id="PGM-REAL",
            policy_version="v1",
            rejection_code=RejectionCode.MISSING_SERIAL,
            k=5,
        )

    assert len(result.clauses) == 5
    assert result.clauses[0].clause.clause_id == "DOC-REAL-C000"
    assert result.clauses[0].distance < result.clauses[1].distance
    assert set(result.timings_ms) == {"embed", "query", "total"}


# ------------------------------------------------------------------------------------------------
# The committed corpus, as the generator actually writes it.
# ------------------------------------------------------------------------------------------------

CORPUS_DIR = REPO_ROOT / "data" / "generated"
corpus_present = pytest.mark.skipif(
    not (CORPUS_DIR / "clauses.json").is_file(),
    reason="the synthetic corpus has not been generated; run `make corpus`",
)


def test_the_generators_nested_clause_record_is_read(tmp_path: Path) -> None:
    """The generator wraps each clause in a record that also carries corpus bookkeeping.

    The wrapper holds the policy version, the document's kind and currency, and the hold-out split
    the clause fell in. That is the right shape for a corpus file and the wrong shape for a table:
    `split` in particular must never reach the database, because a queryable hold-out membership is
    a channel by which a system could behave differently on the data it is scored on. This pins that
    the loader reads through the wrapper and carries none of it across.
    """
    document, clauses, _ = _make_document(
        document_id="DOC-W",
        program_id="PGM-W",
        policy_version="v3",
        count=2,
        first_angle=0.1,
        step=0.1,
    )
    (tmp_path / "documents.json").write_text(
        json.dumps({"is_synthetic": True, "documents": [document._asdict()]}), encoding="utf-8"
    )
    wrapped = [
        {
            "clause": loadable.clause.model_dump(mode="json"),
            "policy_version": "v3",
            "document_kind": "policy",
            "document_is_current": True,
            "split": "holdout",
            "characters": len(loadable.clause.text),
        }
        for loadable in clauses
    ]
    (tmp_path / "clauses.json").write_text(
        json.dumps({"is_synthetic": True, "clauses": wrapped}), encoding="utf-8"
    )

    parsed = read_clauses(tmp_path / "clauses.json", read_documents(tmp_path / "documents.json"))
    assert [item.clause.clause_id for item in parsed] == [item.clause.clause_id for item in clauses]
    assert {item.policy_version for item in parsed} == {"v3"}
    assert not hasattr(parsed[0].clause, "split")


def test_a_wrapper_declaring_the_wrong_policy_version_is_still_refused(tmp_path: Path) -> None:
    """The cross-check has to look in the wrapper, which is where the generator puts the version."""
    document, clauses, _ = _make_document(
        document_id="DOC-W",
        program_id="PGM-W",
        policy_version="v3",
        count=1,
        first_angle=0.1,
        step=0.1,
    )
    (tmp_path / "documents.json").write_text(json.dumps([document._asdict()]), encoding="utf-8")
    (tmp_path / "clauses.json").write_text(
        json.dumps([{"clause": clauses[0].clause.model_dump(mode="json"), "policy_version": "v9"}]),
        encoding="utf-8",
    )
    with pytest.raises(CorpusShapeError, match="policy version"):
        read_clauses(tmp_path / "clauses.json", read_documents(tmp_path / "documents.json"))


@corpus_present
def test_every_committed_clause_is_verbatim_at_its_own_offsets() -> None:
    """Kill condition I over the whole committed corpus, enforced at the boundary.

    `read_clauses` slices each document at each clause's offsets and compares, so this passing means
    no clause in the corpus can produce an unfaithful citation. It reads the files rather than the
    database on purpose: the property belongs to the corpus, and checking it through a load would
    make it conditional on a server being up.
    """
    documents = read_documents(CORPUS_DIR / "documents.json")
    clauses = read_clauses(CORPUS_DIR / "clauses.json", documents)
    assert len(documents) > 0
    assert len(clauses) > 0
    assert {item.policy_version for item in clauses} <= {
        document.policy_version for document in documents.values()
    }


@corpus_present
def test_no_withdrawn_document_governs_a_rejection_code() -> None:
    """The fact the shipped metadata filter relies on, asserted rather than assumed.

    `retrieval.pipeline` records that the filter is programme and policy version only, and that this
    is safe although the corpus contains withdrawn bulletins sharing both: no clause of a withdrawn
    document governs a rejection code, so a withdrawn clause competes for rank but can never be the
    authority behind a satisfied requirement. If a regenerated corpus ever put a governing clause in
    a withdrawn document, that reasoning would stop holding and the filter would have to be
    widened — a change to a predeclared stage, and therefore a decision to record rather than one
    to make silently.
    """
    documents = read_documents(CORPUS_DIR / "documents.json")
    clauses = read_clauses(CORPUS_DIR / "clauses.json", documents)

    withdrawn = {
        item.clause.clause_id
        for item in clauses
        if not documents[item.clause.document_id].is_current and item.clause.governs
    }
    assert withdrawn == set(), (
        "clauses in withdrawn documents now govern rejection codes: "
        f"{sorted(withdrawn)}. The metadata filter in retrieval.pipeline is programme and policy "
        "version only, which no longer excludes them from being cited as authority."
    )
    assert any(not document.is_current for document in documents.values()), (
        "the corpus no longer contains a withdrawn document, so this test proves nothing; "
        "check whether the distractor was removed deliberately"
    )
