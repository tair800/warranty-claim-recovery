"""Reading the generated corpus into PostgreSQL, once, and refusing to do it twice.

Two properties, and each of them is here because of a specific way the seeding step goes wrong.

**It is idempotent, and the key includes the embedding policy.** The naive idempotency key is the
clause text: if the text has not changed, skip the clause. It is wrong in a way that is invisible.
Swap the encoder — a better model, a different dimensionality, a changed instruction prefix — rerun
the seeding script, and every clause reports as unchanged. The table then holds vectors from the old
model while the retriever asks its questions in the new model's space. Nothing raises. Recall falls
by an amount nobody can attribute, and the obvious suspect is the new model, which is innocent. So
`content_fingerprint` folds `EMBEDDING_POLICY.fingerprint` into the hash, and a policy change
invalidates every row by construction.

**It is resumable, which is a stronger property than restartable.** Embedding a corpus is the one
slow step in this repository, and the machine it runs on suspends. A loader that opened one
transaction and committed at the end would lose the whole batch to an interrupt and would re-embed
everything on the next attempt. This one commits per batch, so a kill leaves a committed prefix and
the rerun embeds only the remainder — which is the same durability argument the case machine makes,
applied to the step that builds the thing the case machine reads.

**It refuses a clause that is not where it says it is.** Every clause is verified against its
document with `retrieval.citations.verify_citation` before it is written — the same function kill
condition I is graded with, not a second implementation of the same idea. A clause whose offsets do
not address its own text cannot produce a faithful citation later no matter what the retriever does,
so it is rejected at the boundary, where the error can name the clause and the document. Kill
condition I is then a confirmation that the boundary held rather than a discovery that it did not.

**It writes every clause with its document's currency, and it does not fingerprint that currency.**
The retrieval statement filters on `clause.is_current`, so a clause written with the wrong value is
a withdrawn bulletin made citable or a current one made invisible. The value is taken from the
document record at write time, and the database's composite foreign key refuses a row whose value
disagrees with its document. It is deliberately **not** folded into `clause_fingerprint`: a
withdrawal changes no text, and a fingerprint that moved with it would re-embed every clause of a
withdrawn bulletin to produce vectors identical to the ones already stored. The document's own
fingerprint does include it, the document row is rewritten, and `ON UPDATE CASCADE` carries the
change to the clauses in the same statement.

Rejected: `ON CONFLICT DO NOTHING` for the clause upsert. It is the shorter statement and it turns
the model-swap case into a silent no-op — the row exists, so nothing is written, so the stale vector
survives a rerun that was specifically intended to replace it.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from hashlib import blake2b
from pathlib import Path
from typing import Any, Final, NamedTuple

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from warranty_claim_recovery.domain import PolicyClause
from warranty_claim_recovery.retrieval.citations import citation_failure, citation_for
from warranty_claim_recovery.retrieval.embeddings import EMBEDDING_POLICY, TextEncoder
from warranty_claim_recovery.store.schema import ClauseRow, DocumentRow

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "CorpusShapeError",
    "DocumentRecord",
    "LoadReport",
    "LoadableClause",
    "clause_fingerprint",
    "document_fingerprint",
    "load_corpus",
    "read_clauses",
    "read_documents",
]

#: Small enough that an interrupt costs at most this many embeddings, large enough that the commit
#: overhead disappears against the ONNX call. Not tuned: the corpus is a few hundred clauses and any
#: value in this range is indistinguishable, so a measured number here would be a measurement of
#: noise published as if it meant something.
DEFAULT_BATCH_SIZE: Final = 32

_UNIT_SEPARATOR: Final = b"\x1f"


class CorpusShapeError(ValueError):
    """A generated corpus file that this loader cannot read, named as such.

    Distinct from a validation error on a well-formed record: this one means the *file* is not the
    shape the generator is supposed to produce, so the remedy is to rerun `make corpus` rather than
    to correct a clause.
    """


class DocumentRecord(NamedTuple):
    """A policy document or service bulletin as the generator writes it.

    `text` is the exact string every offset in the corpus indexes into. It is never stripped,
    re-wrapped or normalised anywhere on this path; see `store.schema.DocumentRow`.

    `kind` and `superseded_by` are provenance the store carries through. `is_current` is no longer
    provenance: the retrieval statement excludes every clause of a document that is not current, in
    the same statement that ranks, so this flag decides what can be cited. The corpus contains a
    withdrawn service bulletin per programme that shares a programme and a policy version with the
    current one and says almost the same thing, and this flag is the only thing that tells them
    apart.

    The fields keep their defaults for a record built in code, where the call site is a line a
    reviewer reads. A record read from a **file** is different, and `read_documents` requires the
    flag by name and as a JSON boolean: a corpus exported without it, or with the string `"false"`
    — which `bool()` reads as true — would otherwise publish every withdrawn bulletin as current,
    and nothing downstream could tell.

    `split` is deliberately **not** read. The generator marks each document with its hold-out
    membership, and a `split` column in the deployment database would make that membership readable
    at query time — a channel by which a system could behave differently on the data it is scored
    on. ADR-001 §7 fixes the hold-out before it is scored; it does not need to be queryable.
    """

    document_id: str
    program_id: str
    policy_version: str
    title: str
    text: str
    kind: str = "policy"
    is_current: bool = True
    superseded_by: str | None = None


class LoadableClause(NamedTuple):
    """A clause that has been checked against its document, with the version it belongs to.

    The pair exists because `PolicyClause` deliberately has no `policy_version` field — the version
    is a property of the document, not of the chunk — and the clause table needs it denormalised so
    the metadata filter can run in SQL. Carrying it alongside rather than adding it to the domain
    model keeps `retrieval.citations.citation_for` honest about where a version comes from.
    """

    clause: PolicyClause
    policy_version: str


class LoadReport(NamedTuple):
    """What the run did, in the terms a rerun should be judged by.

    `clauses_unchanged` is the number the operator reads: a second run of an unchanged corpus should
    report every clause unchanged and embed nothing. A report that only counted rows written would
    make an idempotent rerun and a silently skipped one look identical.
    """

    documents_written: int
    documents_unchanged: int
    clauses_written: int
    clauses_unchanged: int
    clauses_embedded: int


def _entries(payload: Any, key: str, path: Path) -> list[dict[str, Any]]:
    """Accept either a bare list or a `{key: [...]}` envelope, and nothing else.

    Both shapes are accepted because the generator carries a synthetic-corpus notice in the body of
    every file it writes, and a notice needs somewhere to live that is not inside a record. What is
    not accepted is a dict without the expected key: guessing which of its values is the list would
    make a renamed field load zero records and report success.
    """
    if isinstance(payload, list):
        raw: list[Any] = payload
    elif isinstance(payload, dict):
        if key not in payload:
            raise CorpusShapeError(
                f"{path} is an object with no {key!r} key; it has {sorted(payload)}. This loader "
                f"will not guess which value is the record list, because guessing wrongly loads "
                f"nothing and reports success."
            )
        raw = payload[key]
        if not isinstance(raw, list):
            raise CorpusShapeError(f"{path}: {key!r} is {type(raw).__name__}, not a list")
    else:
        raise CorpusShapeError(f"{path} holds {type(payload).__name__}, not a list or an object")

    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise CorpusShapeError(
                f"{path}: entry {index} is {type(entry).__name__}, not an object"
            )
    return list(raw)


def _required(entry: Mapping[str, Any], field: str, path: Path, index: int) -> Any:
    if field not in entry:
        raise CorpusShapeError(
            f"{path}: entry {index} has no {field!r}; it has {sorted(entry)}. The fields this "
            f"loader needs are required by name rather than inferred, because a renamed field that "
            f"defaulted quietly would index a corpus missing exactly the part nobody checked."
        )
    return entry[field]


def _clause_fields(entry: Mapping[str, Any]) -> tuple[Mapping[str, Any], Any]:
    """The clause's own fields, and the policy version the file declares beside them.

    The generator writes each clause wrapped in a record that also carries corpus bookkeeping — the
    policy version, the document's kind, whether the document is current, and which hold-out split
    it fell in. That is the right shape for a corpus file and the wrong shape for a table: the
    bookkeeping belongs to the evaluation, not to the store, and a `split` column in the deployment
    database would make hold-out membership queryable at run time.

    So this reaches into the wrapper rather than asking the generator to flatten it, and it also
    accepts a bare clause record, because a hand-written fixture and a future generator version have
    no reason to carry bookkeeping they do not have. Both shapes are read the same way afterwards;
    nothing downstream can tell which one the file used.
    """
    nested = entry.get("clause")
    if isinstance(nested, dict):
        return nested, entry.get("policy_version", nested.get("policy_version"))
    return entry, entry.get("policy_version")


def _currency(entry: Mapping[str, Any], path: Path, index: int) -> bool:
    """The document's `is_current`, required by name and refused unless it is a JSON boolean.

    Required, because the retrieval statement now decides on it and a default is a decision taken
    for every document that did not say. Refused unless boolean, because the permissive reading is
    `bool(value)`, and `bool("false")` is `True`: a corpus whose flags were serialised as strings
    would load every withdrawn bulletin as current, and the only symptom would be a correction that
    quotes a withdrawn authority.
    """
    value = _required(entry, "is_current", path, index)
    if not isinstance(value, bool):
        raise CorpusShapeError(
            f"{path}: entry {index} has is_current={value!r} ({type(value).__name__}). It must be "
            f"a JSON true or false; a string or a number here would be read by truthiness, and the "
            f"string 'false' is truthy."
        )
    return value


def read_documents(path: Path) -> dict[str, DocumentRecord]:
    """Read `documents.json`, keyed by document id.

    A dict rather than a tuple because every consumer looks documents up by id — the clause reader
    to resolve a policy version, the citation grader to slice the text — and a linear scan per
    clause over a corpus of a few hundred is a quadratic nobody would notice until it mattered.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    documents: dict[str, DocumentRecord] = {}
    for index, entry in enumerate(_entries(payload, "documents", path)):
        record = DocumentRecord(
            document_id=str(_required(entry, "document_id", path, index)),
            program_id=str(_required(entry, "program_id", path, index)),
            policy_version=str(_required(entry, "policy_version", path, index)),
            title=str(entry.get("title") or _required(entry, "document_id", path, index)),
            text=str(_required(entry, "text", path, index)),
            kind=str(entry.get("kind") or "policy"),
            is_current=_currency(entry, path, index),
            superseded_by=(
                str(entry["superseded_by"]) if entry.get("superseded_by") is not None else None
            ),
        )
        if record.document_id in documents:
            raise CorpusShapeError(
                f"{path}: {record.document_id} appears twice. Two documents with one identifier "
                f"means every citation naming it is ambiguous about which text it was checked "
                f"against."
            )
        documents[record.document_id] = record
    return documents


def read_clauses(path: Path, documents: Mapping[str, DocumentRecord]) -> tuple[LoadableClause, ...]:
    """Read `clauses.json`, resolve each clause's version, and verify it against its document.

    Verification happens here rather than at retrieval time because this is the last point at which
    the error can name a file and a line's worth of context. A clause whose offsets do not address
    its own text in its own document is not a clause with a minor defect — it is a citation that is
    unfaithful by construction, and kill condition I allows none.

    Each entry may be a clause record or the generator's wrapper around one; `_clause_fields` says
    why both are read. A `policy_version` declared beside the clause — which is where the generator
    puts it — is accepted and cross-checked rather than ignored. Ignoring it would let the two
    disagree indefinitely; trusting it over the document would let a clause claim a version its own
    text was never in.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    loadable: list[LoadableClause] = []
    seen: set[str] = set()

    for index, entry in enumerate(_entries(payload, "clauses", path)):
        record, declared = _clause_fields(entry)
        clause = PolicyClause(
            clause_id=str(_required(record, "clause_id", path, index)),
            program_id=str(_required(record, "program_id", path, index)),
            document_id=str(_required(record, "document_id", path, index)),
            section=str(_required(record, "section", path, index)),
            text=str(_required(record, "text", path, index)),
            start_offset=int(_required(record, "start_offset", path, index)),
            end_offset=int(_required(record, "end_offset", path, index)),
            governs=tuple(record.get("governs") or ()),
        )
        if clause.clause_id in seen:
            raise CorpusShapeError(f"{path}: {clause.clause_id} appears twice")
        seen.add(clause.clause_id)

        document = documents.get(clause.document_id)
        if document is None:
            raise CorpusShapeError(
                f"{path}: {clause.clause_id} cites document {clause.document_id}, which is not in "
                f"the document corpus. Its citation could never be verified against anything."
            )
        if document.program_id != clause.program_id:
            raise CorpusShapeError(
                f"{path}: {clause.clause_id} belongs to programme {clause.program_id} and its "
                f"document {clause.document_id} to {document.program_id}. The metadata filter "
                f"would then exclude the clause from the programme whose policy actually contains "
                f"it."
            )
        if declared is not None and str(declared) != document.policy_version:
            raise CorpusShapeError(
                f"{path}: {clause.clause_id} declares policy version {declared!r} and its document "
                f"is version {document.policy_version!r}. A citation would then name a version the "
                f"quoted text was never in."
            )

        failure = citation_failure(citation_for(clause, document.policy_version), document.text)
        if failure is not None:
            raise CorpusShapeError(
                f"{path}: {failure}. A clause that is not at its own offsets cannot produce a "
                f"faithful citation however good the retrieval is, so it is refused here rather "
                f"than counted later."
            )

        loadable.append(LoadableClause(clause=clause, policy_version=document.policy_version))

    return tuple(loadable)


def _digest(*parts: str) -> str:
    """blake2b over the parts, separated so two different tuples cannot hash the same.

    The unit separator is the point. `("ab", "c")` and `("a", "bc")` concatenate identically, so a
    plain join makes two different clauses share a fingerprint — and a shared fingerprint here means
    one of them is reported unchanged and keeps the other's vector. The separator is a control
    character that cannot appear in an identifier, a section heading or policy prose.
    """
    digest = blake2b(digest_size=16)
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(_UNIT_SEPARATOR)
    return digest.hexdigest()


def document_fingerprint(record: DocumentRecord) -> str:
    """Hash the document's identity and its whole text. The text is what offsets address.

    The embedding policy is deliberately **not** folded in: documents are not embedded, so a model
    change leaves every document row valid and rewriting them all would be work with no effect.

    The provenance fields are folded in, because a bulletin that has been withdrawn since the last
    load is a changed document even when not one character of its text moved, and a rerun that
    reported it unchanged would leave the database saying it is still in force.
    """
    return _digest(
        record.document_id,
        record.program_id,
        record.policy_version,
        record.title,
        record.kind,
        str(record.is_current),
        record.superseded_by or "",
        record.text,
    )


def clause_fingerprint(loadable: LoadableClause) -> str:
    """Hash the clause's content **and** the embedding policy. See the module docstring.

    `governs` is sorted before hashing. The set of governed codes is what matters; a generator that
    emitted them in a different order between runs would otherwise invalidate every clause and force
    a full re-embed that changed no vector.

    The document's currency is absent on purpose. The module docstring gives the reason: it is
    carried to the clause by the foreign key, and fingerprinting it would re-embed unchanged text.
    """
    clause = loadable.clause
    return _digest(
        clause.clause_id,
        clause.program_id,
        loadable.policy_version,
        clause.document_id,
        clause.section,
        clause.text,
        str(clause.start_offset),
        str(clause.end_offset),
        "|".join(sorted(code.value for code in clause.governs)),
        EMBEDDING_POLICY.fingerprint,
    )


def _batched(items: Sequence[LoadableClause], size: int) -> Iterator[Sequence[LoadableClause]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def load_corpus(
    session: Session,
    *,
    documents: Mapping[str, DocumentRecord],
    clauses: Sequence[LoadableClause],
    encoder: TextEncoder,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> LoadReport:
    """Write the corpus, embedding only what has actually changed, committing as it goes.

    The session is committed by this function rather than by a surrounding scope, because
    per-batch commits are the resumability property and a caller holding the transaction open would
    silently remove it. A caller that wants all-or-nothing wants a different function, and should
    say so rather than get it by accident.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size={batch_size} would make no progress")
    orphans = sorted(
        {item.clause.document_id for item in clauses if item.clause.document_id not in documents}
    )
    if orphans:
        # Refused before anything is written. The clause row needs its document's currency, and a
        # clause whose document was not handed over has no currency this function could state
        # without guessing — and a guess of "current" is the defect migration 0003 closed.
        raise CorpusShapeError(
            f"{len(orphans)} document(s) are cited by clauses and were not supplied: "
            f"{orphans[:5]}. A clause cannot be written without its document's currency."
        )

    documents_written = _write_documents(session, documents)

    existing: dict[str, str] = dict(
        session.execute(select(ClauseRow.clause_id, ClauseRow.content_hash)).tuples().all()
    )
    stale = [
        loadable
        for loadable in clauses
        if existing.get(loadable.clause.clause_id) != clause_fingerprint(loadable)
    ]

    embedded = 0
    for batch in _batched(stale, batch_size):
        vectors = encoder.encode_passages([loadable.clause.text for loadable in batch])
        for loadable, vector in zip(batch, vectors, strict=True):
            is_current = documents[loadable.clause.document_id].is_current
            session.execute(_clause_upsert(loadable, vector, is_current=is_current))
        session.commit()
        embedded += len(batch)

    return LoadReport(
        documents_written=documents_written,
        documents_unchanged=len(documents) - documents_written,
        clauses_written=len(stale),
        clauses_unchanged=len(clauses) - len(stale),
        clauses_embedded=embedded,
    )


def _write_documents(session: Session, documents: Mapping[str, DocumentRecord]) -> int:
    existing: dict[str, str] = dict(
        session.execute(select(DocumentRow.document_id, DocumentRow.content_hash)).tuples().all()
    )
    written = 0
    for record in documents.values():
        fingerprint = document_fingerprint(record)
        if existing.get(record.document_id) == fingerprint:
            continue
        statement = insert(DocumentRow).values(
            document_id=record.document_id,
            program_id=record.program_id,
            policy_version=record.policy_version,
            title=record.title,
            text=record.text,
            kind=record.kind,
            is_current=record.is_current,
            superseded_by=record.superseded_by,
            content_hash=fingerprint,
        )
        session.execute(
            statement.on_conflict_do_update(
                index_elements=[DocumentRow.document_id], set_=_excluded(statement)
            )
        )
        written += 1
    session.commit()
    return written


def _clause_upsert(loadable: LoadableClause, vector: Sequence[float], *, is_current: bool) -> Any:
    """One clause's upsert, carrying its document's currency.

    `is_current` is keyword-only and has no default. A positional boolean at the end of a call is
    the argument a later edit transposes, and a default would be the guess `ClauseRow.is_current`
    refuses to make. The composite foreign key checks the value against the document row this run
    has just written, so a wrong one fails here, loudly, rather than at citation time.
    """
    clause = loadable.clause
    statement = insert(ClauseRow).values(
        clause_id=clause.clause_id,
        program_id=clause.program_id,
        policy_version=loadable.policy_version,
        document_id=clause.document_id,
        is_current=is_current,
        section=clause.section,
        text=clause.text,
        start_offset=clause.start_offset,
        end_offset=clause.end_offset,
        governs=[code.value for code in clause.governs],
        embedding=list(vector),
        content_hash=clause_fingerprint(loadable),
    )
    return statement.on_conflict_do_update(
        index_elements=[ClauseRow.clause_id], set_=_excluded(statement)
    )


def _excluded(statement: Any) -> dict[str, Any]:
    """Every non-key column, taken from the row PostgreSQL rejected.

    Built from the statement rather than listed by hand so that adding a column to the table cannot
    produce an upsert that quietly stops updating it — which would leave a stale value on exactly
    the rows a rerun was meant to correct.
    """
    return {
        column.name: statement.excluded[column.name]
        for column in statement.table.columns
        if not column.primary_key
    }
