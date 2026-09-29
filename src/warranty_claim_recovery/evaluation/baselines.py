"""The four retrieval systems ADR-001 §6 predeclared, each different in the retrieval stage itself.

A baseline is only worth measuring if beating it is a real question. ADR-001 §5 records why this
project's baselines are shaped the way they are: project 7's kill condition F required its system to
beat a baseline that removed only a downstream component and therefore ran the identical retriever,
so the comparison was of a system with itself and the criterion could never be met. Each baseline
here removes or replaces a component of the **retrieval stage**:

| baseline | what it removes or replaces |
|---|---|
| `exact_code_lookup` | the search, entirely: a rejection code resolves to a clause identifier |
| `bm25_only` | the embeddings: the same metadata filter, a lexical ranker over the survivors |
| `dense_no_metadata` | the filter: the same embeddings, ranked over every manufacturer at once |
| `first_clause_of_policy` | the ranking: a fixed answer, and the floor under the whole task |

**Every baseline is given the same query text as the system**, composed by
`retrieval.pipeline.compose_query` from `evaluation.pipeline.case_question`. A baseline handed a
different question measures the question rather than the ranker, and the direction of that error is
whichever the author chose.

**No baseline is weakened to be beatable.** `bm25_only` is a plain BM25 over the clause texts with a
plain tokeniser, given the same metadata filter the system enjoys; it is the strongest honest form
of "the same pipeline without embeddings". If it wins, that is the measurement, and the measurement
is what goes in the artifact.

### The one place a baseline needs a stated convention

`exact_code_lookup` is a table keyed by a rejection code, and a table has one row per key. In this
corpus a rejection code can have two governing clauses within one programme — the policy's, and the
current service bulletin's where it amends that code for one named part family — so the table has to
choose, and it chooses the policy clause. That is not a handicap imposed on the baseline; it is the
baseline's defining limitation. A code-keyed table cannot express a rule that is conditional on the
part, which is exactly why resolving the governing clause by table lookup is a different retrieval
system rather than a cheaper spelling of this one. The ceiling a code-keyed *set* lookup would reach
is published beside it in the artifact as a diagnostic, so a reader can see how much of the gap is
the amendment, but the predeclared baseline is the lookup ADR-001 named and the diagnostic is
labelled as not being one of the four.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final, NamedTuple, Protocol

from pgvector.sqlalchemy import Vector
from rank_bm25 import BM25Okapi
from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session

from warranty_claim_recovery.corpus.clauses import POLICY
from warranty_claim_recovery.domain import RejectionCode
from warranty_claim_recovery.retrieval.embeddings import EMBEDDING_POLICY, TextEncoder
from warranty_claim_recovery.retrieval.pipeline import compose_query
from warranty_claim_recovery.store.loader import DocumentRecord, LoadableClause
from warranty_claim_recovery.store.schema import CLAUSE_TABLE

__all__ = [
    "BASELINE_ORDER",
    "NO_METADATA_STATEMENT",
    "Baseline",
    "Bm25Only",
    "ClauseIndex",
    "DenseNoMetadata",
    "ExactCodeLookup",
    "FirstClauseOfPolicy",
    "IndexedClause",
    "RetrievalQuery",
    "code_lookup_ceiling",
    "index_clauses",
    "pgvector_no_metadata_search",
    "rank_of",
]

#: The order the baselines are built and written in. Fixed so that two runs produce the same
#: artifact and a diff of it means something. It is also the order of ADR-001 §6's table, so a
#: reader can check the file against the decision record without holding a mapping in their head.
BASELINE_ORDER: Final[tuple[str, ...]] = (
    "exact_code_lookup",
    "bm25_only",
    "dense_no_metadata",
    "first_clause_of_policy",
)

#: Lower-cased runs of letters and digits. The plainest tokeniser there is, applied to the clause
#: texts and to the query alike.
#:
#: It splits `MISSING_SERIAL` into `missing` and `serial`, which is the honest consequence of a
#: plain tokeniser rather than a decision about this corpus: keeping the underscore would make the
#: code a single rare token and hand BM25 a stronger signal, and dropping the digits would take the
#: part family's number away from it. Either change would move a predeclared baseline's score, so
#: the tokeniser is fixed here and the artifact records what it is.
_TOKEN: Final = re.compile(r"[a-z0-9]+")

#: The `dense_no_metadata` statement: the same table, the same operator, no `WHERE` clause. Written
#: out here rather than derived from `retrieval.pipeline.RETRIEVAL_STATEMENT` by deleting a line,
#: because a baseline assembled by editing the system's own SQL at run time is a baseline that
#: changes whenever the system's SQL does — and the point of a predeclared baseline is that it does
#: not.
NO_METADATA_STATEMENT: Final = (
    "SELECT clause_id, embedding <=> CAST(:query_vector AS vector) AS distance\n"
    f"FROM {CLAUSE_TABLE}\n"
    "ORDER BY embedding <=> CAST(:query_vector AS vector)\n"
    "LIMIT :k"
)


def _tokens(value: str) -> list[str]:
    return _TOKEN.findall(value.lower())


class RetrievalQuery(NamedTuple):
    """One graded retrieval question: what was asked, of which programme, and what the answer is.

    `question` is the uncomposed text from `evaluation.pipeline.case_question`; `composed` appends
    the rejection code through the shipped `compose_query`. Both are carried because the system
    composes internally and the baselines do not, and a baseline that composed its own version of
    the same idea would be searching for a slightly different string.
    """

    claim_id: str
    program_id: str
    policy_version: str
    rejection_code: RejectionCode
    part_number: str
    question: str
    gold_clause_id: str
    split: str

    @property
    def composed(self) -> str:
        return compose_query(self.question, self.rejection_code)


class IndexedClause(NamedTuple):
    """A clause with the two facts the baselines need beyond its text: its version and its document.

    `document_kind` is carried because `exact_code_lookup` and `first_clause_of_policy` are both
    defined in terms of the policy document specifically, and resolving that by pattern-matching the
    clause identifier would make both baselines depend on a naming convention rather than on the
    corpus's own record of what each document is.
    """

    clause_id: str
    program_id: str
    policy_version: str
    document_id: str
    document_kind: str
    text: str
    governs: tuple[RejectionCode, ...]


class ClauseIndex:
    """The clause corpus, grouped the way the baselines need it and in a fixed order.

    Built once and shared by every baseline, so each of them ranks over exactly the same clauses.
    Three baselines reading three separately assembled corpora would be three measurements of three
    different things, and nothing in their outputs would say so.
    """

    __slots__ = ("_clauses", "_pools")

    def __init__(self, clauses: Sequence[IndexedClause]) -> None:
        self._clauses = tuple(clauses)
        pools: dict[tuple[str, str], list[IndexedClause]] = {}
        for clause in self._clauses:
            pools.setdefault((clause.program_id, clause.policy_version), []).append(clause)
        # Sorted by identifier inside each pool. The corpus writes them in document order already;
        # sorting makes the tie-break in `bm25_only` reproducible without depending on that.
        self._pools = {
            key: tuple(sorted(group, key=lambda clause: clause.clause_id))
            for key, group in sorted(pools.items())
        }

    @property
    def clauses(self) -> tuple[IndexedClause, ...]:
        return self._clauses

    @property
    def pools(self) -> Mapping[tuple[str, str], tuple[IndexedClause, ...]]:
        return self._pools

    def pool(self, program_id: str, policy_version: str) -> tuple[IndexedClause, ...]:
        return self._pools.get((program_id, policy_version), ())


def index_clauses(
    documents: Mapping[str, DocumentRecord], clauses: Sequence[LoadableClause]
) -> ClauseIndex:
    """Join the clause corpus to its documents so each clause knows which kind of document it is in.

    Takes the same two structures `store.loader` produces for the seeding script, so the baselines
    rank over the corpus that was actually indexed rather than over a second reading of the files.
    """
    indexed: list[IndexedClause] = []
    for loadable in clauses:
        document = documents[loadable.clause.document_id]
        indexed.append(
            IndexedClause(
                clause_id=loadable.clause.clause_id,
                program_id=loadable.clause.program_id,
                policy_version=loadable.policy_version,
                document_id=loadable.clause.document_id,
                document_kind=document.kind,
                text=loadable.clause.text,
                governs=loadable.clause.governs,
            )
        )
    return ClauseIndex(indexed)


class Baseline(Protocol):
    """What every baseline has to be able to do, and nothing more.

    A protocol rather than a base class so that a baseline is defined by what it answers rather than
    by what it inherits, and so that a test can supply a fixed ranking without constructing an
    index, a session or an encoder.
    """

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    def rank(self, query: RetrievalQuery, k: int) -> tuple[str, ...]: ...


def rank_of(clause_ids: Sequence[str], gold: str) -> int | None:
    """Where the governing clause came in a result list, one-based, or `None` if it is absent.

    One function for every system measured here, the shipped one included, so that a rank means the
    same thing in every row of the artifact. Two implementations of "where did it come" is how one
    of them ends up zero-based and a comparison ends up off by one in a single column.
    """
    for position, clause_id in enumerate(clause_ids, start=1):
        if clause_id == gold:
            return position
    return None


class ExactCodeLookup:
    """Resolve the governing clause by a rejection-code table. No text search at all.

    The table is built from the corpus itself: for each programme and policy version, the clause of
    the **policy** document that takes authority over the code, resolved by clause identifier when a
    policy document somehow declares two. The service bulletin's amendment is deliberately not in
    the table — see the module docstring — because a table keyed by a code cannot hold a rule that
    is conditional on the part family, and pretending otherwise would make this baseline a different
    system from the one ADR-001 §6 predeclared.
    """

    __slots__ = ("_table",)

    def __init__(self, index: ClauseIndex) -> None:
        table: dict[tuple[str, str, RejectionCode], str] = {}
        for clause in sorted(index.clauses, key=lambda clause: clause.clause_id):
            if clause.document_kind != POLICY:
                continue
            for code in clause.governs:
                table.setdefault((clause.program_id, clause.policy_version, code), clause.clause_id)
        self._table = table

    @property
    def name(self) -> str:
        return "exact_code_lookup"

    @property
    def description(self) -> str:
        return (
            "resolve the governing clause by an exact rejection-code to clause-id table lookup "
            "over the policy document, with no text search at all"
        )

    # `k` is unused and stays in the signature. It is the `Baseline` protocol's parameter, and a
    # table lookup answering with one row is the defining shape of this baseline rather than an
    # oversight: dropping the parameter would put this class outside the protocol and would let a
    # reader think the four systems are called differently from one another.
    def rank(self, query: RetrievalQuery, k: int) -> tuple[str, ...]:  # noqa: ARG002
        found = self._table.get((query.program_id, query.policy_version, query.rejection_code))
        return () if found is None else (found,)


def code_lookup_ceiling(index: ClauseIndex, query: RetrievalQuery, k: int) -> tuple[str, ...]:
    """Every clause governing the code, not one. A diagnostic, and **not** one of the baselines.

    Published beside `exact_code_lookup` so a reader can see how much of that baseline's gap is the
    bulletin's amendment and how much is anything else. It is not a baseline because it is not a
    table lookup: a table maps a key to a value, and a structure that returns every clause governing
    a code is an index over the `governs` column — which is the filter `retrieval.pipeline`
    deliberately does not apply, and which would make the comparison one of a system against a
    component of itself.
    """
    pool = index.pool(query.program_id, query.policy_version)
    return tuple(clause.clause_id for clause in pool if query.rejection_code in clause.governs)[:k]


class Bm25Only:
    """Lexical BM25 over the same clause corpus, behind the same metadata filter. No embeddings.

    The filter stays because this baseline answers "what do the embeddings buy?", and a baseline
    that dropped the filter as well would answer two questions at once and attribute the difference
    to whichever one the reader assumed. `dense_no_metadata` is the one that removes the filter, and
    keeping each baseline to a single removal is what makes the four of them diagnostic rather than
    merely different.

    One `BM25Okapi` per pool, built once in the constructor. Rebuilding it per query would compute
    the same idf table seven hundred times and would leave a reader wondering whether the corpus
    statistics had changed between queries; they cannot, and building once makes that visible.
    """

    __slots__ = ("_index", "_models")

    def __init__(self, index: ClauseIndex) -> None:
        self._index = index
        self._models: dict[tuple[str, str], Any] = {
            key: BM25Okapi([_tokens(clause.text) for clause in pool])
            for key, pool in index.pools.items()
        }

    @property
    def name(self) -> str:
        return "bm25_only"

    @property
    def description(self) -> str:
        return (
            "lexical BM25 over the same clause corpus behind the same manufacturer and "
            "policy-version filter, with no embeddings"
        )

    def rank(self, query: RetrievalQuery, k: int) -> tuple[str, ...]:
        key = (query.program_id, query.policy_version)
        pool = self._index.pool(*key)
        model = self._models.get(key)
        if model is None or not pool:
            return ()
        scores = [float(score) for score in model.get_scores(_tokens(query.composed))]
        # Ties broken by clause identifier, not by pool order. BM25 over short clauses of similar
        # length produces exact ties often enough to matter, and a tie broken by whatever order the
        # pool happened to be built in is a score that depends on the corpus writer.
        order = sorted(
            range(len(pool)),
            key=lambda position: (-scores[position], pool[position].clause_id),
        )
        return tuple(pool[position].clause_id for position in order[:k])


class DenseNoMetadata:
    """The same embeddings, ranked over every manufacturer at once. No metadata filter.

    This is the baseline that makes `retrieval.pipeline`'s central argument checkable. That module
    argues that filtering before ranking is a correctness property rather than an optimisation,
    because a global ranking over a corpus of many manufacturers returns clauses from the wrong
    policy and silently shortens the result list for the right one. The argument is only worth
    anything if the difference was measured, and this is the measurement.
    """

    __slots__ = ("_encoder", "_search")

    def __init__(
        self, encoder: TextEncoder, search: Callable[[Sequence[float], int], tuple[str, ...]]
    ) -> None:
        self._encoder = encoder
        self._search = search

    @property
    def name(self) -> str:
        return "dense_no_metadata"

    @property
    def description(self) -> str:
        return (
            "the same embeddings over the same corpus with no manufacturer or policy-version "
            "filter, so every manufacturer's clauses compete for the top k"
        )

    def rank(self, query: RetrievalQuery, k: int) -> tuple[str, ...]:
        return self._search(self._encoder.encode_query(query.composed), k)


def pgvector_no_metadata_search(
    session: Session,
) -> Callable[[Sequence[float], int], tuple[str, ...]]:
    """The unfiltered search, as a callable, so `DenseNoMetadata` needs no session of its own.

    A closure for the same reason `evaluation.pipeline.pgvector_search` is one: the baseline should
    not be able to reach a database and do something other than what its description says.
    """
    statement = text(NO_METADATA_STATEMENT).bindparams(
        bindparam("query_vector", type_=Vector(EMBEDDING_POLICY.dimensions))
    )

    def search(vector: Sequence[float], k: int) -> tuple[str, ...]:
        rows = session.execute(statement, {"query_vector": list(vector), "k": k}).all()
        return tuple(str(row[0]) for row in rows)

    return search


class FirstClauseOfPolicy:
    """Always the policy's first clause. The floor that says whether the task is trivial.

    Its score is the share of claims whose governing clause happens to be the first one, and a task
    where that share is high is a task no retrieval system needed to be built for. It is in ADR-001
    §6 for exactly that reason: a criterion with no floor cannot distinguish a working retriever
    from a corpus that answers itself.
    """

    __slots__ = ("_first",)

    def __init__(self, index: ClauseIndex) -> None:
        first: dict[tuple[str, str], str] = {}
        for key, pool in index.pools.items():
            policy = [clause for clause in pool if clause.document_kind == POLICY]
            if policy:
                first[key] = policy[0].clause_id
        self._first = first

    @property
    def name(self) -> str:
        return "first_clause_of_policy"

    @property
    def description(self) -> str:
        return "always the first clause of the programme's policy document, whatever was asked"

    # `k` unused, for the reason recorded on `ExactCodeLookup.rank`: a baseline that always answers
    # with one clause has nothing to truncate, and the parameter belongs to the protocol.
    def rank(self, query: RetrievalQuery, k: int) -> tuple[str, ...]:  # noqa: ARG002
        found = self._first.get((query.program_id, query.policy_version))
        return () if found is None else (found,)
