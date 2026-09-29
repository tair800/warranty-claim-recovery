"""Dense retrieval over the clause corpus, and the ordering that is the whole argument.

The order is fixed and it is the claim:

    metadata filter, in SQL          program_id AND policy_version AND is_current
      -> pgvector distance search    over the survivors, with the `<=>` operator
      -> the top k, with distances

**Both stages are one statement.** That is not an optimisation; it is the correctness property.

The tempting alternative is to rank first and filter afterwards in Python: fetch the nearest fifty
clauses, keep the ones belonging to the right manufacturer and policy version, and return the first
five. It looks equivalent and it is not, in two ways that both cost money. First, it **silently
shrinks the result**: a corpus with forty manufacturers means the global top fifty may contain one
clause from the programme actually being adjudicated, or none, and `k=5` quietly becomes `k=1` with
nothing in the response indicating that it did. Kill condition J grades recall at five; a stage that
sometimes returns one is measured as a retrieval failure, and the diagnosis leads to the encoder,
which is innocent. Second, it makes the shrinkage **depend on the corpus**: adding a manufacturer
degrades retrieval for every existing manufacturer, so a system that measured well at build time
degrades in production without a single line of it having changed.

Putting the predicate in the `WHERE` clause of the same statement removes the question. The server
ranks the survivors of the filter, so `LIMIT k` is `k` of the right programme's clauses or all of
them, and the number of manufacturers in the corpus is irrelevant to every one of them.

**Why `rejection_code` enriches the query text instead of filtering on `governs`.** A filter on the
`governs` array would be the strongest possible signal and it would also make this retriever
identical to the `exact_code_lookup` baseline that ADR-001 §6 predeclares — resolving the governing
clause by a code-to-clause table lookup with no text search at all. Kill condition J requires the
system to beat every baseline, and a system that *is* one of its baselines cannot beat it; the
comparison would be a system measured against itself, which is the exact defect ADR-001 §5 records
about project 7's kill condition F. The rejection code is real signal, so it is folded into the text
that gets embedded, where it competes with everything else in the clause corpus rather than
short-circuiting the search.

**Why withdrawn documents are excluded here, and why the first version did not exclude them.**
The committed corpus contains withdrawn service bulletins that share a programme **and** a policy
version with the current ones, written to read almost exactly like the clauses that replaced them.
They are excluded in the same statement that ranks, by `is_current`, which migration
`0003_clause_is_current` copies onto the clause row under a composite foreign key so the copy cannot
drift from the document it came from.

The first version of this module left them in, deliberately, and argued it. The argument is kept
because it was half right, and the half that was wrong is the instructive part:

- **Right:** no clause of a withdrawn bulletin governs any rejection code — 0 of 36, measured — so a
  withdrawn clause can never be the *authority* for a satisfied requirement under the rule
  `domain.PolicyClause` states.
- **Wrong:** "a distractor that competes for rank rather than an authority" assumed nothing
  downstream would cite a clause merely for ranking first. Something did. The graph's authority rule
  fell back to the top-ranked clause whenever no retrieved clause governed the code, and on the
  development split 69 of 520 queries had a withdrawn clause at rank one. A resubmission could
  therefore quote a bulletin whose own text says it has been withdrawn, as the basis of a
  requirement it had marked satisfied — the correction a manufacturer rejects on sight, after the
  window has run.

So the rule is enforced in two layers that do not trust each other: a withdrawn document cannot be
retrieved, and a clause that governs nothing cannot be cited as authority
(`graph.nodes._governing_entry`, which now agrees with `evaluation.pipeline._authority`).

**Why this is a current-state filter and not project 7's as-of filter.** Project 7 answered
historical questions — what was the approved procedure on the day of an incident — and needed
validity time. A resubmission is different: it is filed **now**, against the manufacturer's policy
**as it stands now**, and the only bulletin that can support it is one currently in force. A
withdrawn bulletin being correct on the day of the repair does not make it citable today.

**Why the index is not asserted to be used.** ADR-001 §5 explains it: project 7 demanded a vector
index scan in the plan and failed because its own metadata filter left so few candidate rows that
PostgreSQL correctly preferred a sequential scan. The criterion asked for the wrong plan. What is
claimed here, and what `explain_plan` proves from the server's own output, is that the extension is
doing the distance arithmetic — which is the claim actually being made.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any, Final, NamedTuple

from pgvector.sqlalchemy import Vector
from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session

from warranty_claim_recovery.domain import PolicyClause, RejectionCode
from warranty_claim_recovery.retrieval.embeddings import (
    EMBEDDING_POLICY,
    FastEmbedEncoder,
    TextEncoder,
)
from warranty_claim_recovery.store.schema import CLAUSE_TABLE

__all__ = [
    "DEFAULT_K",
    "RETRIEVAL_STATEMENT",
    "RetrievalResult",
    "Retriever",
    "ScoredClause",
    "compose_query",
]

DEFAULT_K: Final = 5

# The table name is interpolated from `store.schema.CLAUSE_TABLE` and every value that comes from a
# caller is a bound parameter — there is no path by which caller data reaches this string. Writing
# "clause" as a literal would remove the interpolation and reintroduce the drift the constant exists
# to prevent: a renamed table would then fail at query time rather than at import.
RETRIEVAL_STATEMENT: Final = (
    "SELECT clause_id, program_id, policy_version, document_id, section, text,\n"
    "       start_offset, end_offset, governs,\n"
    "       embedding <=> CAST(:query_vector AS vector) AS distance\n"
    f"FROM {CLAUSE_TABLE}\n"
    "WHERE program_id = :program_id\n"
    "  AND policy_version = :policy_version\n"
    "  AND is_current\n"
    "ORDER BY embedding <=> CAST(:query_vector AS vector)\n"
    "LIMIT :k"
)


class ScoredClause(NamedTuple):
    """One retrieved clause with the distance that ranked it, and its position.

    `rank` is carried explicitly rather than left to the caller's enumeration because it ends up in
    `artifacts/retrieval.json` as the position at which the governing clause was found, and a rank
    recomputed by whichever grader happens to iterate the tuple is a rank that can be recomputed
    differently by the next one.
    """

    clause: PolicyClause
    distance: float
    rank: int


class RetrievalResult(NamedTuple):
    """What came back, the statement that produced it, and how long each stage took.

    `executed_statement` is the literal SQL that ran, parameter placeholders and all. It is part of
    the result rather than something a caller reconstructs because kill condition K reads it out of
    an artifact and greps it for the distance operator, and a statement rebuilt for the artifact is
    a statement that can differ from the one the system ran.

    `timings_ms` is diagnostic and nothing scores it. It is here because the two stages fail
    differently — a slow embed is a cold ONNX session, a slow query is a plan — and a single total
    cannot tell them apart. No test asserts on these numbers: they read a clock, and ADR-001 and
    `CLAUDE.md` §4 both require that nothing graded depends on one.
    """

    clauses: tuple[ScoredClause, ...]
    executed_statement: str
    timings_ms: dict[str, float]


def compose_query(query: str, rejection_code: RejectionCode) -> str:
    """The text that is actually embedded: the case's question plus the rejection code.

    One function, so that the evaluation harness, the graph's retrieval node and the baselines all
    ask the same question. A query assembled at three call sites is three queries, and the recall
    they measure is not the recall the system will have.

    The code is appended in a fixed, human-readable form rather than interpolated into a sentence.
    An engineered prompt-like phrasing would be a tuning knob, and a tuning knob adjusted after the
    hold-out has been scored is exactly what ADR-001 §7 forbids. Fixing the form now makes the
    question of whether a better phrasing exists a new experiment rather than an edit.
    """
    return f"{query.strip()}\nManufacturer rejection code: {rejection_code.value}"


class Retriever:
    """The dense stage: embed the query, filter in SQL, rank in pgvector, return the top k."""

    __slots__ = ("_embedder",)

    def __init__(self, embedder: TextEncoder | None = None) -> None:
        """Take an encoder, or build the shipped one.

        The default is constructed rather than injected so that the ordinary caller — the graph's
        retrieval node — gets the policy in `EMBEDDING_POLICY` without having to know it exists. The
        parameter is there because the SQL is worth testing without a 285MB ONNX session; see
        `TextEncoder` for why that matters. `FastEmbedEncoder` loads its model lazily, so
        constructing a `Retriever` still costs nothing.
        """
        self._embedder: TextEncoder = embedder if embedder is not None else FastEmbedEncoder()

    @property
    def embedder(self) -> TextEncoder:
        return self._embedder

    def _parameters(
        self,
        *,
        query: str,
        program_id: str,
        policy_version: str,
        rejection_code: RejectionCode,
        k: int,
    ) -> tuple[dict[str, Any], float]:
        if k < 1:
            raise ValueError(
                f"k={k} asks for no results. A retrieval stage that returns nothing is not a "
                f"cheaper retrieval stage; it is a composer with no evidence to cite."
            )
        started = time.perf_counter()
        vector = self._embedder.encode_query(compose_query(query, rejection_code))
        embed_ms = (time.perf_counter() - started) * 1000.0
        if len(vector) != EMBEDDING_POLICY.dimensions:
            raise ValueError(
                f"the encoder returned {len(vector)} dimensions and the clause column is "
                f"{EMBEDDING_POLICY.dimensions} wide; these vectors are not in the same space as "
                f"the stored ones and any ranking over them would be meaningless rather than wrong"
            )
        return (
            {
                "query_vector": vector,
                "program_id": program_id,
                "policy_version": policy_version,
                "k": k,
            },
            embed_ms,
        )

    def retrieve(
        self,
        session: Session,
        *,
        query: str,
        program_id: str,
        policy_version: str,
        rejection_code: RejectionCode,
        k: int = DEFAULT_K,
    ) -> RetrievalResult:
        """Retrieve the k nearest clauses **within** the named programme and policy version."""
        parameters, embed_ms = self._parameters(
            query=query,
            program_id=program_id,
            policy_version=policy_version,
            rejection_code=rejection_code,
            k=k,
        )

        started = time.perf_counter()
        rows = session.execute(_statement(), parameters).mappings().all()
        query_ms = (time.perf_counter() - started) * 1000.0

        scored = tuple(
            ScoredClause(clause=_clause_from_row(row), distance=float(row["distance"]), rank=rank)
            for rank, row in enumerate(rows, start=1)
        )
        return RetrievalResult(
            clauses=scored,
            executed_statement=RETRIEVAL_STATEMENT,
            timings_ms={
                "embed": round(embed_ms, 3),
                "query": round(query_ms, 3),
                "total": round(embed_ms + query_ms, 3),
            },
        )

    def explain_plan(
        self,
        session: Session,
        *,
        query: str,
        program_id: str,
        policy_version: str,
        rejection_code: RejectionCode,
        k: int = DEFAULT_K,
        analyse: bool = True,
    ) -> str:
        """The server's own plan for the statement `retrieve` runs, as the server prints it.

        `ANALYZE` by default, so the statement is genuinely executed and the returned plan is the
        plan that ran rather than one the planner would consider. Kill condition K is graded from
        this text, and a plan obtained by explaining a statement assembled separately for the
        artifact would prove something about that second statement. The SQL comes from the same
        constant `retrieve` uses, prefixed and not rewritten.
        """
        parameters, _ = self._parameters(
            query=query,
            program_id=program_id,
            policy_version=policy_version,
            rejection_code=rejection_code,
            k=k,
        )
        options = "ANALYZE true, VERBOSE true" if analyse else "VERBOSE true"
        plan = session.execute(_statement(prefix=f"EXPLAIN ({options})\n"), parameters).all()
        return "\n".join(str(row[0]) for row in plan)


def _statement(prefix: str = "") -> Any:
    """`RETRIEVAL_STATEMENT`, with the vector parameter typed so pgvector serialises it.

    The type is attached here rather than at the call site so that `retrieve` and `explain_plan`
    cannot bind the same parameter two different ways — which would make the explained statement
    differ from the executed one in the one detail kill condition K is about.
    """
    return text(prefix + RETRIEVAL_STATEMENT).bindparams(
        bindparam("query_vector", type_=Vector(EMBEDDING_POLICY.dimensions))
    )


def _clause_from_row(row: Any) -> PolicyClause:
    """Rebuild the domain object, validating on the way out of the database.

    `PolicyClause` re-checks that the offsets span the text. Doing that on read looks redundant —
    the loader checked it on write — and it is the cheap half of a real guarantee: a row edited by
    hand, restored from a backup taken against a different document version, or written by a future
    loader that skipped the check would otherwise flow straight into a citation. The check costs one
    comparison per row and closes the gap between "was correct when written" and "is correct".
    """
    return PolicyClause(
        clause_id=row["clause_id"],
        program_id=row["program_id"],
        document_id=row["document_id"],
        section=row["section"],
        text=row["text"],
        start_offset=int(row["start_offset"]),
        end_offset=int(row["end_offset"]),
        governs=_governed_codes(row["governs"], row["clause_id"]),
    )


def _governed_codes(raw: Sequence[str] | None, clause_id: str) -> tuple[RejectionCode, ...]:
    """Coerce the stored array to the closed enum, refusing a code the system does not know.

    `RejectionCode` is closed because `requirements.REQUIREMENT_MATRIX` is a total function over it.
    A stored string that is not a member is therefore not a clause with an unusual authority; it is
    a corpus that no longer matches the code, and letting it through as a plain string would push
    the failure into a matrix lookup that returns nothing — which reads as "this rejection demands
    no evidence" and gates a resubmission that nothing supports.
    """
    if not raw:
        return ()
    try:
        return tuple(RejectionCode(value) for value in raw)
    except ValueError as error:
        raise ValueError(
            f"{clause_id} claims authority over a rejection code this system does not define: "
            f"{list(raw)}. The corpus and RejectionCode have diverged, and the requirement matrix "
            f"would silently return no requirements for it."
        ) from error
