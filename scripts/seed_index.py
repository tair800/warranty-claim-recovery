"""Embed the clause corpus into PostgreSQL, and write the evidence for kill condition K.

Two jobs in one command, deliberately, because they have to be about the same database. The
alternative — a seeding script and a separate evidence script — lets the evidence be produced
against a database that the seeder never touched, and an artifact that describes a different
database from the one the system queries is worse than no artifact.

**Seeding is idempotent and resumable.** `store.loader` keys every clause on a content fingerprint
that includes the embedding policy, so a rerun over an unchanged corpus embeds nothing and reports
every clause unchanged, and a run killed halfway leaves a committed prefix that the next run skips.
Embedding is the one slow step in this repository and the machine it runs on suspends; a seeder that
had to start from the beginning after every interruption would be a seeder nobody ran.

**The evidence comes from the server, not from this file.** `artifacts/pgvector.json` reports the
extension version, the column's type as PostgreSQL reports it, the index definition as `pg_indexes`
prints it, and whether the distance operator appears in the plan **the server produced for the
statement that actually ran**. ADR-001 §5 is explicit that kill condition K asks whether the
extension is doing the arithmetic, not whether a particular index was scanned: project 7's version
of this criterion demanded a vector index scan and failed because its own metadata filter left so
few rows that a sequential scan was correctly cheaper. A criterion that asks for the wrong plan is a
criterion that punishes a correct system.

`EXPLAIN` is run with `ANALYZE`, so the statement is executed and the plan is the plan that ran
rather than one the planner would consider. Grepping this repository's own SQL string would prove
only that this repository contains a string.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Final, NamedTuple

from sqlalchemy import func, inspect, select, text
from sqlalchemy.orm import Session

from warranty_claim_recovery.config import get_settings
from warranty_claim_recovery.domain import RejectionCode
from warranty_claim_recovery.retrieval.embeddings import EMBEDDING_POLICY, FastEmbedEncoder
from warranty_claim_recovery.retrieval.pipeline import DEFAULT_K, Retriever
from warranty_claim_recovery.store.engine import build_engine, session_scope
from warranty_claim_recovery.store.loader import (
    DEFAULT_BATCH_SIZE,
    LoadReport,
    load_corpus,
    read_clauses,
    read_documents,
)
from warranty_claim_recovery.store.schema import (
    CLAUSE_TABLE,
    HNSW_BUILD_PARAMETERS,
    HNSW_INDEX_NAME,
    ClauseRow,
    DocumentRow,
)

REPO_ROOT: Final = Path(__file__).resolve().parents[1]

#: The operator kill condition K looks for. Imported by nobody: the kill test states it
#: independently, which is the point of a predeclared grader.
DISTANCE_OPERATOR: Final = "<=>"

#: How many (programme, policy version) pairs to explain. Every pair is a statement the system will
#: really run, so the denominator is a real population rather than a padded one — but a corpus of
#: forty programmes does not need forty plans in an artifact a person is expected to read.
MAX_EXPLAINED_PROGRAMS: Final = 8

#: A vector literal in a plan is 384 numbers and appears three times per plan. The stored plan
#: abbreviates them so the artifact stays readable. The boolean is computed from the text **before**
#: abbreviation, and this pattern matches only a bracketed list of numbers, so it can neither
#: introduce nor remove an operator.
_VECTOR_LITERAL = re.compile(r"'\[[-0-9eE.,+ ]{40,}\]'")


class Probe(NamedTuple):
    """One real query: a programme, a policy version, a question and a rejection code."""

    program_id: str
    policy_version: str
    query: str
    rejection_code: RejectionCode


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=None, help="defaults to WCR_DATABASE_URL")
    parser.add_argument(
        "--corpus-dir",
        default=None,
        help="directory holding documents.json and clauses.json; defaults to WCR_CORPUS_DIR",
    )
    parser.add_argument("--artifacts-dir", default=None, help="defaults to WCR_ARTIFACTS_DIR")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--allow-empty-corpus",
        action="store_true",
        help=(
            "write the pgvector evidence even when the generated corpus is absent. Off by default: "
            "indexing nothing and reporting success is how a retrieval stage comes to be measured "
            "over an empty table."
        ),
    )
    parser.add_argument(
        "--no-evidence",
        action="store_true",
        help="load only, and do not rewrite artifacts/pgvector.json",
    )
    return parser.parse_args(argv)


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def load(session: Session, corpus_dir: Path, *, batch_size: int) -> LoadReport:
    documents = read_documents(corpus_dir / "documents.json")
    clauses = read_clauses(corpus_dir / "clauses.json", documents)
    return load_corpus(
        session,
        documents=documents,
        clauses=clauses,
        encoder=FastEmbedEncoder(),
        batch_size=batch_size,
    )


def probes(session: Session) -> list[Probe]:
    """Real queries drawn from the indexed corpus, in a fixed order.

    Ordered by clause identifier and deduplicated by (programme, version), so two runs over the same
    corpus explain the same statements. A sample drawn from a set's iteration order would make the
    artifact differ between runs for no reason, and a diff that changes every time is a diff nobody
    reads.
    """
    rows = session.execute(
        select(
            ClauseRow.program_id,
            ClauseRow.policy_version,
            ClauseRow.section,
            ClauseRow.governs,
        ).order_by(ClauseRow.clause_id)
    ).all()

    chosen: dict[tuple[str, str], Probe] = {}
    for program_id, policy_version, section, governs in rows:
        key = (program_id, policy_version)
        if key in chosen:
            continue
        codes = [RejectionCode(value) for value in governs or ()]
        chosen[key] = Probe(
            program_id=program_id,
            policy_version=policy_version,
            query=f"Which clause governs {section}?",
            # A clause that governs nothing is context rather than authority, so the probe falls
            # back to a fixed code rather than inventing one. The code only enriches the query text;
            # it is never a filter. See `retrieval.pipeline`.
            rejection_code=codes[0] if codes else RejectionCode.MISSING_SERIAL,
        )
        if len(chosen) >= MAX_EXPLAINED_PROGRAMS:
            break
    return list(chosen.values())


def _scalar(session: Session, statement: str, **parameters: Any) -> Any:
    """One value from a catalogue query, with every name passed as a bound parameter.

    The table and index names are module constants and could be interpolated safely, and they are
    bound anyway. A catalogue query with an interpolated identifier is the shape a reviewer has to
    read twice, and the shape that someone later extends with a value that did not come from a
    constant. Binding costs nothing here: `pg_class.relname` and `pg_indexes.indexname` are compared
    as values, not as identifiers.
    """
    return session.execute(text(statement), parameters).scalar()


def _count(session: Session, row_type: type[Any]) -> int:
    """`SELECT count(*)` through SQLAlchemy Core, so no table name is ever formatted into SQL."""
    return int(session.execute(select(func.count()).select_from(row_type)).scalar_one())


def evidence(session: Session) -> dict[str, Any]:
    """Everything kill condition K reads, every value of it read back from PostgreSQL."""
    extension_version = _scalar(
        session, "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
    )
    column_type = _scalar(
        session,
        "SELECT a.atttypid::regtype::text FROM pg_attribute a "
        "JOIN pg_class c ON c.oid = a.attrelid "
        "WHERE c.relname = :table AND a.attname = 'embedding'",
        table=CLAUSE_TABLE,
    )
    column_dimensions = _scalar(
        session,
        "SELECT a.atttypmod FROM pg_attribute a JOIN pg_class c ON c.oid = a.attrelid "
        "WHERE c.relname = :table AND a.attname = 'embedding'",
        table=CLAUSE_TABLE,
    )
    index_definition = _scalar(
        session,
        "SELECT indexdef FROM pg_indexes WHERE indexname = :index_name",
        index_name=HNSW_INDEX_NAME,
    )
    clauses_indexed = _count(session, ClauseRow)
    documents_indexed = _count(session, DocumentRow)

    retriever = Retriever()
    sample = probes(session)
    if not sample:
        # An empty corpus still has a real table with a real vector column and a real index, and the
        # server still plans a real statement over it. Recording that honestly — with
        # `clauses_indexed` at zero — is better than refusing to write the artifact, because the
        # criterion this feeds is about where the arithmetic happens and not about how much of it
        # there was. `--allow-empty-corpus` is what makes this path reachable, and it is off by
        # default.
        sample = [
            Probe(
                program_id="__no-corpus__",
                policy_version="__no-corpus__",
                query="Which clause governs the correction window?",
                rejection_code=RejectionCode.MISSING_SERIAL,
            )
        ]

    plans: list[dict[str, Any]] = []
    executed_statement = ""
    for probe in sample:
        result = retriever.retrieve(
            session,
            query=probe.query,
            program_id=probe.program_id,
            policy_version=probe.policy_version,
            rejection_code=probe.rejection_code,
            k=DEFAULT_K,
        )
        executed_statement = result.executed_statement
        plan = retriever.explain_plan(
            session,
            query=probe.query,
            program_id=probe.program_id,
            policy_version=probe.policy_version,
            rejection_code=probe.rejection_code,
            k=DEFAULT_K,
        )
        plans.append(
            {
                "program_id": probe.program_id,
                "policy_version": probe.policy_version,
                "rejection_code": probe.rejection_code.value,
                "rows_returned": len(result.clauses),
                "uses_distance_operator": DISTANCE_OPERATOR in plan,
                "plan": _VECTOR_LITERAL.sub(
                    f"'[<{EMBEDDING_POLICY.dimensions} dimensions abbreviated>]'", plan
                ),
            }
        )

    using_operator = sum(1 for plan in plans if plan["uses_distance_operator"])
    return {
        "is_synthetic_corpus": True,
        "what_this_is": (
            "kill condition K: proof that the dense retrieval stage's distance arithmetic is "
            "performed by pgvector against a vector column, taken from the server's own EXPLAIN "
            "ANALYZE of the statement that ran"
        ),
        "why": (
            "a retrieval layer that computes distances in Python over rows it fetched is not a "
            "pgvector deployment, and nothing in a response distinguishes the two. ADR-001 section "
            "5 records why this criterion asks for the operator rather than for a vector index "
            "scan: project 7 demanded an index scan and failed because its metadata filter left "
            "so few candidate rows that a sequential scan was correctly cheaper."
        ),
        "statements_explained": len(plans),
        "explains_using_distance_operator": using_operator,
        "explain_uses_distance_operator": bool(plans) and using_operator == len(plans),
        "extension_installed": extension_version is not None,
        "extension_version": extension_version,
        "column_is_vector_type": column_type == "vector",
        "column_type": column_type,
        "column_dimensions": column_dimensions,
        "declared_dimensions": EMBEDDING_POLICY.dimensions,
        "distance_operator": DISTANCE_OPERATOR,
        "executed_statement": executed_statement,
        "index_definition": index_definition,
        "declared_build_parameters": dict(HNSW_BUILD_PARAMETERS),
        "clauses_indexed": clauses_indexed,
        "documents_indexed": documents_indexed,
        # Computed rather than written, so it cannot be left behind saying the wrong thing after a
        # real seed. An artifact that reports a passing criterion over an empty table has to say so
        # in its own body: ADR-001's vacuity guard exists because project 7 published a pass whose
        # numerator was empty by construction and nobody could tell from the file.
        "corpus_state": (
            "indexed"
            if clauses_indexed
            else (
                "empty: this evidence was produced against the migrated but unpopulated table, "
                "before the synthetic corpus existed. Rerun `make index` once `make corpus` has "
                "run, and these counts and plans will describe real rows."
            )
        ),
        "embedding_policy": {
            "model": EMBEDDING_POLICY.model_name,
            "dimensions": EMBEDDING_POLICY.dimensions,
            "normalised": EMBEDDING_POLICY.normalised,
            "query_instruction": EMBEDDING_POLICY.query_instruction,
            "fingerprint": EMBEDDING_POLICY.fingerprint,
        },
        "server_version": _scalar(session, "SELECT version()"),
        "database": _scalar(session, "SELECT current_database()"),
        "plans": plans,
        "plan_note": (
            "the stored plans abbreviate the 384-number vector literal, which appears three times "
            "in each; the operator and every plan node are verbatim, and the booleans above were "
            "computed from the unabbreviated text"
        ),
        "no_credentials_recorded": (
            "the connection URL is not written here; the database is identified by the server's "
            "own report of its version and current database name"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = get_settings()
    corpus_dir = _resolve(args.corpus_dir or settings.corpus_dir)
    artifacts_dir = _resolve(args.artifacts_dir or settings.artifacts_dir)

    engine = build_engine(args.database_url or settings.database_url)
    if not inspect(engine).has_table(CLAUSE_TABLE):
        print(
            f"the {CLAUSE_TABLE!r} table does not exist. Run `alembic upgrade head` first: this "
            f"script deliberately does not create the schema, because a seeder that also migrates "
            f"is a seeder that can invent a schema the migration history does not describe.",
            file=sys.stderr,
        )
        return 2

    missing = [
        name for name in ("documents.json", "clauses.json") if not (corpus_dir / name).is_file()
    ]
    if missing and not args.allow_empty_corpus:
        print(
            f"{corpus_dir} has no {', '.join(missing)}. Run `make corpus` first, or pass "
            f"--allow-empty-corpus to write the pgvector evidence against the empty table.",
            file=sys.stderr,
        )
        return 2

    with session_scope(engine) as session:
        if missing:
            print(f"corpus absent in {corpus_dir}; loading nothing")
        else:
            report = load(session, corpus_dir, batch_size=args.batch_size)
            print(
                f"documents: written {report.documents_written}, "
                f"unchanged {report.documents_unchanged}"
            )
            print(f"clauses: written {report.clauses_written}, embedded {report.clauses_embedded}")
            print(f"unchanged {report.clauses_unchanged}")

        if not args.no_evidence:
            payload = evidence(session)
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            destination = artifacts_dir / "pgvector.json"
            destination.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            print(
                f"{destination}: {payload['statements_explained']} statement(s) explained, "
                f"operator present in {payload['explains_using_distance_operator']}"
            )

    engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
