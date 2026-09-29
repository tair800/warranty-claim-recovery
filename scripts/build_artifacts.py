"""Run the deterministic path over the whole corpus and write the evidence the kill test grades.

Three files come out of this command: `artifacts/recovery.json`, `artifacts/retrieval.json` and
`artifacts/groundedness.json`. `durability.json`, `submission.json`, `pgvector.json`, `redis.json`
and `holdout.json` are written by the commands that can actually observe what they claim — the
kill-and-resume harness, the idempotency race, the seeding script, the failure injector and the
freeze. Nothing here fabricates any of them, and a missing one is reported by
`scripts/release_gate.py` as a criterion that was not graded rather than silently omitted.

**Every number here comes from the shipped modules, called in the shipped order.** The loop in
`evaluation.pipeline.assess_case` calls `requirements`, `deadline`, `eligibility`, `recovery`, the
real `Retriever` against the real PostgreSQL, `citation_for`, and `gate.decide`. What this script
adds is a loop, four baselines and a comparison against `data/generated/truth.json`. An evaluation
that scored a reimplementation of the pipeline would be scoring the reimplementation, and the first
time the two drifted the artifact would go on reporting the answer the evaluation liked.

**One encoder, memoised, shared by the system and by `dense_no_metadata`.** The two have to embed
the *same* vector for the same case or the comparison is between a ranker and a paraphrase.
`evaluation.pipeline.MemoisingEncoder` guarantees that; halving the slowest step of the run is a
side effect rather than the reason.

**The run is ordered and reads no clock.** Cases are assessed in the generator's order, the
baselines are built in `BASELINE_ORDER`, and nothing written here carries a timestamp, a hostname
or a duration. Two runs over the same corpus and the same index produce the same bytes, so a diff
of an artifact is a change in the system rather than a change in the weather.

Rejected: a `--limit` flag for a quick smoke run. It would write a real artifact filename over a
partial population, and the kill test reads those filenames without asking how many cases were in
them. A smaller corpus is a corpus, and it belongs in the generator rather than in the grader.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

from warranty_claim_recovery.config import get_settings
from warranty_claim_recovery.corpus.holdout import DEVELOPMENT, HOLDOUT
from warranty_claim_recovery.evaluation.artifacts import (
    GradedCase,
    Measurement,
    graded,
    groundedness_artifact,
    recovery_artifact,
    retrieval_artifact,
    summary_lines,
)
from warranty_claim_recovery.evaluation.baselines import (
    BASELINE_ORDER,
    Baseline,
    Bm25Only,
    ClauseIndex,
    DenseNoMetadata,
    ExactCodeLookup,
    FirstClauseOfPolicy,
    RetrievalQuery,
    code_lookup_ceiling,
    index_clauses,
    pgvector_no_metadata_search,
    rank_of,
)
from warranty_claim_recovery.evaluation.metrics import Rank
from warranty_claim_recovery.evaluation.pipeline import (
    CorpusCase,
    MemoisingEncoder,
    assess_case,
    candidate_counts,
    case_question,
    load_cases,
    pgvector_search,
)
from warranty_claim_recovery.retrieval.embeddings import FastEmbedEncoder
from warranty_claim_recovery.retrieval.pipeline import DEFAULT_K, Retriever
from warranty_claim_recovery.store.engine import build_engine, session_scope
from warranty_claim_recovery.store.loader import read_clauses, read_documents
from warranty_claim_recovery.store.schema import CLAUSE_TABLE

REPO_ROOT: Final = Path(__file__).resolve().parents[1]

#: How often to report progress. The embedding pass is the slow step and this machine suspends
#: mid-run; a command that prints nothing for four minutes is a command a person kills.
PROGRESS_EVERY: Final = 60

#: The description published beside the system's own row in `retrieval.json`. Written here rather
#: than in `retrieval/pipeline.py` because it describes the system *as a measured entrant*, in the
#: same vocabulary as the four baselines, which is a property of the comparison rather than of the
#: retriever.
SYSTEM_DESCRIPTION: Final = (
    "dense retrieval in pgvector with the manufacturer and policy-version filter applied in the "
    "same statement, before ranking"
)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=None, help="defaults to WCR_DATABASE_URL")
    parser.add_argument("--corpus-dir", default=None, help="defaults to WCR_CORPUS_DIR")
    parser.add_argument("--artifacts-dir", default=None, help="defaults to WCR_ARTIFACTS_DIR")
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help=(
            "the result-list length every system is measured at. ADR-001 §5 fixes kill condition "
            "J at five; a larger k here measures something the criterion does not grade."
        ),
    )
    return parser.parse_args(argv)


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def _query_of(case: CorpusCase) -> RetrievalQuery:
    """The baselines' view of one case, carrying the same question the system was asked.

    `case_question` is called once here and the text is handed to every baseline, so a difference
    between two rows of `retrieval.json` is a difference between rankers. A baseline that composed
    its own version of the same idea would be searching for a slightly different string, and the
    direction of that error would be whichever the author happened to choose.
    """
    return RetrievalQuery(
        claim_id=case.claim.claim_id,
        program_id=case.claim.program_id,
        policy_version=case.program.policy_version,
        rejection_code=case.claim.rejection_code,
        part_number=case.claim.part_number,
        question=case_question(case.claim, case.program),
        gold_clause_id=case.governing_clause_id,
        split=case.split,
    )


def _ranks(
    baseline: Baseline, queries: Sequence[RetrievalQuery], k: int, split: str
) -> tuple[Rank, ...]:
    return tuple(
        rank_of(baseline.rank(query, k), query.gold_clause_id)
        for query in queries
        if query.split == split
    )


def _measure(baseline: Baseline, queries: Sequence[RetrievalQuery], k: int) -> Measurement:
    return Measurement(
        name=baseline.name,
        description=baseline.description,
        holdout_ranks=_ranks(baseline, queries, k, HOLDOUT),
        development_ranks=_ranks(baseline, queries, k, DEVELOPMENT),
    )


class _CodeLookupCeiling:
    """The diagnostic beside `exact_code_lookup`: every clause governing the code, not one.

    A class rather than a closure so that it satisfies the same `Baseline` protocol the four
    predeclared systems do and is measured by exactly the same code path. It is published under
    `diagnostics` and never under `baselines`: ADR-001 §6 fixed four baselines before any score
    existed, and a fifth entry added afterwards — however honest — would become a criterion the
    system has to beat, which is the predeclaration undone from the inside.
    """

    __slots__ = ("_index",)

    def __init__(self, index: ClauseIndex) -> None:
        self._index = index

    @property
    def name(self) -> str:
        return "code_lookup_ceiling"

    @property
    def description(self) -> str:
        return (
            "DIAGNOSTIC, not one of ADR-001 §6's four: every clause of the programme governing the "
            "rejection code, which is the ceiling a code-keyed set lookup would reach and shows "
            "how much of exact_code_lookup's gap is the service bulletin's amendment"
        )

    def rank(self, query: RetrievalQuery, k: int) -> tuple[str, ...]:
        return code_lookup_ceiling(self._index, query, k)


def _write(destination: Path, payload: dict[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    # One long function, and the length is argued rather than tolerated. It is a build script: the
    # order of its steps *is* the thing a reader needs to check against `CLAUDE.md` §2.1, and
    # splitting it into six helpers that each take the session, the encoder and the corpus would
    # hide that order behind a call graph. Every step below is three to six lines and does one
    # thing; none of them is reused anywhere else.
    args = parse_args(argv)
    settings = get_settings()
    corpus_dir = _resolve(args.corpus_dir or settings.corpus_dir)
    artifacts_dir = _resolve(args.artifacts_dir or settings.artifacts_dir)
    k = int(args.k)

    try:
        cases = load_cases(corpus_dir, artifacts_dir)
    except (FileNotFoundError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2
    if not cases:
        print(
            f"{corpus_dir} holds no claims. An evaluation over nothing produces artifacts whose "
            f"denominators are zero, which ADR-001's vacuity guard fails rather than passes; run "
            f"`make corpus` first.",
            file=sys.stderr,
        )
        return 2

    documents = read_documents(corpus_dir / "documents.json")
    clauses = read_clauses(corpus_dir / "clauses.json", documents)
    index = index_clauses(documents, clauses)
    queries = tuple(_query_of(case) for case in cases)
    print(f"{len(cases)} case(s), {len(clauses)} clause(s), k={k}")

    encoder = MemoisingEncoder(FastEmbedEncoder())
    retriever = Retriever(encoder)
    engine = build_engine(args.database_url or settings.database_url)

    graded_cases: list[GradedCase] = []
    try:
        with session_scope(engine) as session:
            search = pgvector_search(session, retriever)
            for position, case in enumerate(cases, start=1):
                graded_cases.append(graded(assess_case(case, search, k=k)))
                if position % PROGRESS_EVERY == 0 or position == len(cases):
                    print(f"  assessed {position}/{len(cases)}", flush=True)

            pools = candidate_counts(session)
            baselines = (
                ExactCodeLookup(index),
                Bm25Only(index),
                DenseNoMetadata(encoder, pgvector_no_metadata_search(session)),
                FirstClauseOfPolicy(index),
            )
            measured = {baseline.name: _measure(baseline, queries, k) for baseline in baselines}
            ceiling = _measure(_CodeLookupCeiling(index), queries, k)
    except Exception as error:
        # Reported with the connection's shape and not its credentials, and the command stops. A
        # build script that caught a database failure and wrote the artifacts it had would publish
        # a recall measured over the cases that happened to run before the connection dropped, and
        # nothing in the file would say so.
        print(
            f"the evaluation could not complete against {CLAUSE_TABLE!r}: "
            f"{type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 3
    finally:
        engine.dispose()

    if sorted(measured) != sorted(BASELINE_ORDER):
        # A predeclared baseline that silently stopped being measured would leave kill condition J
        # comparing the system against a shorter list, which it would pass more easily. The check
        # is here rather than in the artifact module because this is the only place that decides
        # which four objects get constructed.
        print(
            f"the baselines measured {sorted(measured)} are not ADR-001 §6's "
            f"{sorted(BASELINE_ORDER)}",
            file=sys.stderr,
        )
        return 4

    system = Measurement(
        name="warranty-claim-recovery",
        description=SYSTEM_DESCRIPTION,
        holdout_ranks=tuple(
            case.governing_clause_rank for case in graded_cases if case.split == HOLDOUT
        ),
        development_ranks=tuple(
            case.governing_clause_rank for case in graded_cases if case.split == DEVELOPMENT
        ),
    )
    candidates = {
        split: [
            pools.get((query.program_id, query.policy_version), 0)
            for query in queries
            if query.split == split
        ]
        for split in (HOLDOUT, DEVELOPMENT)
    }

    recovery = recovery_artifact(graded_cases)
    retrieval = retrieval_artifact(
        k=k,
        system=system,
        baselines=[measured[name] for name in BASELINE_ORDER],
        diagnostics=[ceiling],
        holdout_candidates=candidates[HOLDOUT],
        development_candidates=candidates[DEVELOPMENT],
    )
    groundedness = groundedness_artifact(
        graded_cases, {document_id: record.text for document_id, record in documents.items()}
    )

    _write(artifacts_dir / "recovery.json", recovery)
    _write(artifacts_dir / "retrieval.json", retrieval)
    _write(artifacts_dir / "groundedness.json", groundedness)

    print()
    for line in summary_lines(recovery, retrieval):
        print(line)
    print(retrieval["candidate_profile_verdict"])
    print(
        f"wrote {artifacts_dir / 'recovery.json'}, {artifacts_dir / 'retrieval.json'} and "
        f"{artifacts_dir / 'groundedness.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
