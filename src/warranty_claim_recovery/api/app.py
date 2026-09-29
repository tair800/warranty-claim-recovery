"""The Warranty Recovery Lab: a console that shows the evidence and grades nothing.

A reviewer arrives with ten minutes and a reasonable suspicion that the README is generous. This
console exists to let them check it. Every screen answers one question — what is this case, what is
the evidence behind it, how was the money arrived at, what was measured, and what happened — and
each of them is reproducible from files that are committed beside the code.

### The console never grades

Not one verdict on these pages is computed here. `artifacts/release_gate.json` carries the
release-gate verdict and `artifacts/*.json` carry the thirteen criteria's counts, and this module
reads them. The rejected alternative was to recompute a criterion in the console so that a screen
would work before the artifacts existed. It was rejected because two graders is one grader too
many: the day the console and `scripts/release_gate.py` disagree, the number a reader quotes is
whichever one they happened to look at, and the project's central claim — that the evidence is
measured rather than asserted — is gone. A console that reports and does not grade cannot drift
from the grader, because it has no opinion to drift with.

That is also why an absent artifact renders **`NOT RUN`** and never an empty cell. A blank in a
verdict column reads as a pass to everyone who has ever skimmed a table, and a criterion that has
never been evaluated is the one case where a reader must not be allowed to assume anything.

### Nothing from `Settings` reaches a template

`Settings` holds the database URL — credentials and all — and the model key. Putting the object in
a template context makes every one of those fields one `{{ settings.database_url }}` away from a
page, and the mistake is invisible in review because the template that leaks it looks exactly like
the template that does not. So the header is built as `PageHeader`, a two-field value object, and
that is the only thing derived from configuration that any template can see. `tests/test_api.py`
plants a sentinel DSN and a sentinel approver token in the environment, requests every screen and
both health endpoints, and asserts that neither string appears in the bytes that come back.

The same rule governs error text. A failed connection raises an exception whose message routinely
contains the host, the port and the user, and a console that renders `str(error)` publishes the
DSN the first time PostgreSQL is asleep. Every diagnostic on these pages is the exception's **type
name** and nothing else. It is less helpful than the full message, and the operator reads the full
message in the process log where it belongs.

### The screens work without a database

`/evaluation` and `/audit` read `artifacts/*.json` and touch no connection at all. `/`, `/evidence`
and `/recovery` read the generated corpus from disk and compute with the deterministic core, which
was built to run on hand-made fixtures precisely so that it would never need a store. Only the
dense retrieval on `/evidence` needs PostgreSQL, it runs behind a button rather than on page load,
and it degrades to a stated error rather than to a stack trace.

`/healthz` is the endpoint where this matters most. A readiness check that raises instead of
answering is worse than one that answers "no": the platform cannot tell a broken database from a
broken health check, and the operator debugging the outage starts in the wrong place. The database
probe is therefore a dependency that catches everything and returns a finding, never one that
raises while FastAPI is still resolving dependencies. `/livez` takes no dependency whatsoever.

### Read-only by construction

There is exactly one route in this file that mutates anything, and three separate things have to be
true before it does: the deployment must not be read-only, an approver token must be configured,
and the caller must present it. `WCR_READ_ONLY` defaults to **true** and `WCR_APPROVER_TOKEN` has
no default, so a deployment that forgot both locks has both locks. The read-only check runs first,
so a correct token in a read-only deployment still cannot mutate — the reverse order would make the
token the senior control, and the token is the one that can leak.

### What this console is not

It is not the case machine. The graph owns requirement adjudication, checkpointing, approval and
submission; this module reads what the graph and the evaluation harness wrote. Where a case fact
has no committed source yet, the screen says so in that many words rather than inventing one. The
one place the console is obliged to form its own view — whether a claim's evidence bundle is
complete, because `gate.decide` cannot be called without a requirement list — is confined to
`adjudicate` below, is deliberately conservative in the direction that withholds money, and is
labelled on the screen as the console's own reading.
"""

from __future__ import annotations

import hmac
import json
from collections.abc import Mapping, Sequence
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Final, NamedTuple

from fastapi import FastAPI, Header, Query, Request, Response
from fastapi import Path as PathParam
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import text as sql_text
from sqlalchemy.engine import Engine

from warranty_claim_recovery import deadline, eligibility, gate, recovery
from warranty_claim_recovery.audit import (
    APPROVAL_GRANTED,
    APPROVAL_REQUESTED,
    CASE_REPRICED,
    SUBMISSION_MADE,
    SUBMISSION_SUPPRESSED,
    TOOL_COMPLETED,
    TOOL_STARTED,
    AuditLog,
)
from warranty_claim_recovery.config import APPROVER_TOKEN_ENV, Settings, get_settings
from warranty_claim_recovery.corpus.claims import EVIDENCE_KEYS, EvidenceStatus
from warranty_claim_recovery.domain import (
    AuditEvent,
    CaseState,
    Citation,
    Claim,
    ClaimWindow,
    GateDecision,
    PolicyClause,
    RecoveryComputation,
    RejectionCode,
    Requirement,
    RequirementStatus,
    WarrantyProgram,
)
from warranty_claim_recovery.eligibility import Eligibility, PartCoverage
from warranty_claim_recovery.money import Currency, CurrencyMismatchError, Money, parse_money
from warranty_claim_recovery.requirements import RequirementSpec, required_for
from warranty_claim_recovery.retrieval.citations import citation_for, verify_citation
from warranty_claim_recovery.retrieval.pipeline import DEFAULT_K, Retriever
from warranty_claim_recovery.store.engine import build_engine, session_scope
from warranty_claim_recovery.store.loader import DocumentRecord, read_clauses, read_documents

__all__ = ["app"]

# ------------------------------------------------------------------------------------------------
# Locations and vocabulary.
# ------------------------------------------------------------------------------------------------

TEMPLATES_DIR: Final = Path(__file__).resolve().parent / "templates"

#: `src/warranty_claim_recovery/api/app.py` -> the repository root. Relative paths in `Settings`
#: are resolved against this rather than against the working directory. A console started from
#: somewhere else would otherwise find no corpus and render an empty picker, which reads as a
#: broken build rather than as a misconfigured one — and the reader has no way to tell them apart.
REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[3]

#: ADR-001 §1 discharges the blueprint's fixture-realism risk by making the corpus synthetic and
#: saying so everywhere. `CLAUDE.md` §3.8 makes "everywhere" include every screen, so this string
#: is rendered in the footer of every page from one constant rather than typed into six templates.
SYNTHETIC_NOTICE: Final = (
    "The corpus behind this console is SYNTHETIC. It is generated from a committed seed, it is not "
    "a manufacturer publication, it describes no real warranty programme, and no clause shown here "
    "has legal effect. No figure on these pages means anything outside this corpus."
)

#: What a verdict column says when nothing has graded it. Never an empty string: a blank cell in a
#: verdict column is read as a pass by every reader who has ever skimmed a table.
NOT_RUN: Final = "NOT RUN"

#: What a verdict column says when the artifact exists and carries no verdict this console knows
#: how to read. Distinct from `NOT_RUN` because the remedies differ — one is "run the gate", the
#: other is "the gate wrote a shape this console does not recognise" — and a reader who cannot tell
#: them apart reruns the build and gets the same screen.
UNREADABLE: Final = "UNREADABLE"

#: What a metric cell says when no committed artifact publishes it. Also never blank.
NOT_MEASURED: Final = "not measured"

#: The header the one mutating route reads the approver token from. A header rather than a query
#: parameter because query strings are written to access logs and browser history by default, and a
#: credential in a URL is a credential in a log file.
APPROVER_HEADER_NAME: Final = "X-WCR-Approver-Token"

#: Seconds libpq is given to establish a connection before the probe gives up. Short, because the
#: only caller that waits on it is a page a person is looking at, and a database that has not
#: answered in five seconds is not going to render this request either way. `bounded` argues it.
CONNECT_TIMEOUT_SECONDS: Final = 5

_APPROVAL_ACTOR: Final = "console"

templates: Final = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _resolve(configured: str) -> Path:
    """A configured directory, anchored to the repository when it is relative. See above."""
    path = Path(configured)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _fault(error: BaseException) -> str:
    """The exception's type name, and never its message.

    A connection failure's message carries the host, the port and the user, and a retrieval failure
    carries the statement. Rendering either would publish the DSN on a public console the first
    time the database was asleep. The operator reads the full exception in the process log, where
    the credentials are already as exposed as the process is; the page reads a class name.
    """
    return type(error).__name__


# ------------------------------------------------------------------------------------------------
# The artifacts. Read, never graded.
# ------------------------------------------------------------------------------------------------


def read_artifact(name: str) -> dict[str, Any] | None:
    """One artifact, or `None` when it is absent or unreadable.

    Read on every request rather than cached at import. The console is the thing a reviewer leaves
    open in a second window while `make artifacts` runs in the first, and a cached read would show
    them the state of the world at the moment uvicorn started. The files are tens of kilobytes and
    the parse is invisible against the request.

    A malformed artifact is the same answer as an absent one — `None`, which renders `NOT RUN` —
    because the console has no way to partially trust a file it could not parse, and guessing at
    the half it could read is how a screen comes to publish a number nothing produced.
    """
    path = _resolve(get_settings().artifacts_dir) / name
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


class Banner(NamedTuple):
    """The release-gate verdict, carried into the header of every screen."""

    verdict: str
    detail: str
    tone: str


def _tone(verdict: str) -> str:
    if verdict == "PASS":
        return "pass"
    return "fail" if verdict == "FAIL" else "unknown"


def release_gate_banner() -> Banner:
    """The verdict `scripts/release_gate.py` recorded, read out of its artifact.

    Two shapes are accepted, `verdict` as a string and `passed` as a boolean, because the gate
    script is a separate lane and pinning its output format from here would be this module
    legislating for one it does not own. What is **not** accepted is inference: an artifact that
    carries neither key renders `UNREADABLE` rather than a verdict derived from, say, counting the
    criteria that look green. Deriving a verdict here would make the console a second grader, which
    the module docstring rejects at length.
    """
    payload = read_artifact("release_gate.json")
    if payload is None:
        return Banner(
            verdict=NOT_RUN,
            detail=(
                "artifacts/release_gate.json is absent, so nothing has graded the thirteen "
                "predeclared kill conditions in this working tree. Run `make release-gate`."
            ),
            tone=_tone(NOT_RUN),
        )

    raw_verdict = payload.get("verdict")
    if isinstance(raw_verdict, str) and raw_verdict.strip():
        verdict = raw_verdict.strip().upper()
    elif isinstance(payload.get("passed"), bool):
        verdict = "PASS" if payload["passed"] else "FAIL"
    else:
        return Banner(
            verdict=UNREADABLE,
            detail=(
                "artifacts/release_gate.json exists and carries neither a 'verdict' string nor a "
                "'passed' boolean. This console reports a verdict and never infers one."
            ),
            tone=_tone(UNREADABLE),
        )

    detail = payload.get("summary") or payload.get("detail") or ""
    return Banner(verdict=verdict, detail=str(detail), tone=_tone(verdict))


class Criterion(NamedTuple):
    """One of ADR-001 §5's thirteen, with the artifact that grades it."""

    letter: str
    fails_if: str
    threshold: str
    artifact: str


#: The thirteen, in ADR-001 §5's order. These are labels: the verdicts beside them on `/evaluation`
#: come from `artifacts/release_gate.json` and from nowhere else. The thresholds are quoted from
#: the ADR rather than imported from `tests/test_kill_criteria.py`, because that file is
#: predeclared and importing it into `src` would let a change here reach the grader.
CRITERIA: Final[tuple[Criterion, ...]] = (
    Criterion(
        letter="A",
        fails_if="a resumed case continues at a different node from the checkpoint's",
        threshold="0 divergences",
        artifact="durability.json",
    ),
    Criterion(
        letter="B",
        fails_if="a tool that completed before the kill runs again after the resume",
        threshold="0 re-executions",
        artifact="durability.json",
    ),
    Criterion(
        letter="C",
        fails_if="a human decision recorded before the kill is absent after the resume",
        threshold="0 losses",
        artifact="durability.json",
    ),
    Criterion(
        letter="D",
        fails_if="a submission appears with no approval for the same case and case version",
        threshold="0 unapproved",
        artifact="submission.json",
    ),
    Criterion(
        letter="E",
        fails_if="concurrent identical submissions produce more than one submission effect",
        threshold="exactly 1 effect, 0 duplicates",
        artifact="submission.json",
    ),
    Criterion(
        letter="F",
        fails_if="a computed recoverable amount differs from the generator's ground truth",
        threshold="0 mismatches, exact Decimal equality",
        artifact="recovery.json",
    ),
    Criterion(
        letter="G",
        fails_if="a case is gated recoverable whose ground truth is NOT_RECOVERABLE",
        threshold="0 false recoveries",
        artifact="recovery.json",
    ),
    Criterion(
        letter="H",
        fails_if="a requirement is reported satisfied with no citation",
        threshold="0",
        artifact="groundedness.json",
    ),
    Criterion(
        letter="I",
        fails_if="a cited span is not verbatim at its recorded offsets in the version it names",
        threshold="0",
        artifact="groundedness.json",
    ),
    Criterion(
        letter="J",
        fails_if="hold-out recall@5 is below the floor, or not above every predeclared baseline",
        threshold=">= 0.85 and > max(baselines)",
        artifact="retrieval.json",
    ),
    Criterion(
        letter="K",
        fails_if="the dense stage does not use the pgvector operator, by the server's own plan",
        threshold="operator present",
        artifact="pgvector.json",
    ),
    Criterion(
        letter="L",
        fails_if="with Redis unavailable a case is leased twice or a submission proceeds",
        threshold="0 double-leases, 0 submissions",
        artifact="redis.json",
    ),
    Criterion(
        letter="M",
        fails_if="a case is resubmitted after the manufacturer's claim window closed",
        threshold="0",
        artifact="recovery.json",
    ),
)


def criterion_verdicts() -> dict[str, str]:
    """Letter -> verdict, from the release-gate artifact, defaulting to `NOT_RUN`.

    Both shapes the gate might reasonably write are read: a mapping keyed by letter, and a list of
    records each naming its own letter. A criterion the artifact does not mention is `NOT_RUN`
    rather than absent from the table — the table is the thirteen, always, because a criterion that
    quietly stops being listed is a criterion nobody notices stopped being graded.
    """
    verdicts = {criterion.letter: NOT_RUN for criterion in CRITERIA}
    payload = read_artifact("release_gate.json")
    if payload is None:
        return verdicts

    entries: list[tuple[str, Any]] = []
    raw = payload.get("criteria")
    if isinstance(raw, dict):
        entries = [(str(key), value) for key, value in raw.items()]
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                letter = item.get("letter") or item.get("id") or item.get("criterion")
                if letter is not None:
                    entries.append((str(letter), item))

    for letter, value in entries:
        key = letter.strip().upper()[:1]
        if key not in verdicts:
            continue
        verdicts[key] = _verdict_of(value)
    return verdicts


def _verdict_of(value: Any) -> str:
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if isinstance(value, str) and value.strip():
        return value.strip().upper()
    if isinstance(value, dict):
        for key in ("verdict", "status", "result"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip().upper()
        if isinstance(value.get("passed"), bool):
            return "PASS" if value["passed"] else "FAIL"
    return UNREADABLE


def metric(payload: Mapping[str, Any] | None, *keys: str) -> str:
    """One value out of an artifact, by a path of keys, or `NOT_MEASURED`.

    Returns a string because the only thing the template does with it is print it, and a template
    that has to distinguish `0` from `None` to decide whether to print a dash is a template making
    an editorial decision. `0` is a measurement and prints as `0`; an absent key is not a
    measurement and prints as `not measured`. Conflating those two is how a criterion that was
    never evaluated comes to be read as a criterion that found nothing.
    """
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return NOT_MEASURED
        current = current[key]
    if isinstance(current, bool):
        return "yes" if current else "no"
    if current is None:
        return NOT_MEASURED
    return str(current)


# ------------------------------------------------------------------------------------------------
# The corpus, read from the generated files rather than from the database.
# ------------------------------------------------------------------------------------------------


class ClaimRecord(NamedTuple):
    """One generated claim, with the corpus bookkeeping the console shows beside it.

    `as_of` is carried because every deadline function in this project takes the date it is
    reasoning as of and none of them reads a clock. The corpus fixed that date when it built the
    claim, so the window this console renders is the window the evaluation graded, on any day.
    """

    claim: Claim
    as_of: date
    split: str
    construction: str
    adversarial_kind: str
    evidence: Mapping[str, str]


class TruthRecord(NamedTuple):
    """The generator's construction metadata for one claim.

    Shown on the case screen beside the computed outcome, and used for nothing else. It is the
    answer key: reading it to *decide* anything would make every figure on these pages a figure the
    console was told rather than one it worked out.
    """

    outcome: str
    recoverable_amount: str
    currency: str
    governing_clause_id: str
    missing_requirements: tuple[str, ...]


class Corpus(NamedTuple):
    """Everything the case screens read, indexed the way they read it."""

    programs: dict[str, WarrantyProgram]
    claims: dict[str, ClaimRecord]
    claim_order: tuple[str, ...]
    coverage: dict[tuple[str, str], PartCoverage]
    clauses: dict[str, PolicyClause]
    clause_version: dict[str, str]
    documents: dict[str, DocumentRecord]
    truth: dict[str, TruthRecord]
    #: program_id -> the clause identifiers of its current documents, in corpus order. The order is
    #: the file's, not a set's, so "the first clause that governs this code" is the same clause on
    #: every machine and in every process.
    current_clause_ids: dict[str, tuple[str, ...]]


class CorpusStatus(NamedTuple):
    corpus: Corpus | None
    problem: str | None


def _money(raw: Any) -> Money:
    return parse_money(raw["amount"], raw["currency"])


def _entries(path: Path, key: str) -> list[dict[str, Any]]:
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return [item for item in payload[key] if isinstance(item, dict)]


def _program(record: Mapping[str, Any]) -> WarrantyProgram:
    return WarrantyProgram(
        program_id=str(record["program_id"]),
        manufacturer=str(record["manufacturer"]),
        policy_version=str(record["policy_version"]),
        currency=Currency(record["currency"]),
        correction_window_days=int(record["correction_window_days"]),
        warranty_months=int(record["warranty_months"]),
        labour_rate_cap_per_hour=_money(record["labour_rate_cap_per_hour"]),
        deductible=_money(record["deductible"]),
        claim_cap=_money(record["claim_cap"]),
    )


def _claim(record: Mapping[str, Any]) -> Claim:
    prior = record.get("previously_recovered")
    return Claim(
        claim_id=str(record["claim_id"]),
        program_id=str(record["program_id"]),
        part_number=str(record["part_number"]),
        serial_number=None if record.get("serial_number") is None else str(record["serial_number"]),
        in_service_date=date.fromisoformat(str(record["in_service_date"])),
        failure_date=date.fromisoformat(str(record["failure_date"])),
        repair_invoice_date=date.fromisoformat(str(record["repair_invoice_date"])),
        rejection_code=RejectionCode(record["rejection_code"]),
        rejected_on=date.fromisoformat(str(record["rejected_on"])),
        claimed_parts=_money(record["claimed_parts"]),
        claimed_labour_hours=int(record["claimed_labour_hours"]),
        claimed_labour_rate=_money(record["claimed_labour_rate"]),
        previously_recovered=None if prior is None else _money(prior),
    )


def _read_corpus(corpus_dir: Path) -> Corpus:
    """Build the whole index once. Every reader below is a dictionary lookup afterwards."""
    programs = {
        str(entry["program"]["program_id"]): _program(entry["program"])
        for entry in _entries(corpus_dir / "programs.json", "programs")
    }

    claims: dict[str, ClaimRecord] = {}
    claim_order: list[str] = []
    for entry in _entries(corpus_dir / "claims.json", "claims"):
        claim = _claim(entry["claim"])
        claims[claim.claim_id] = ClaimRecord(
            claim=claim,
            as_of=date.fromisoformat(str(entry["as_of"])),
            split=str(entry.get("split", "")),
            construction=str(entry.get("construction", "")),
            adversarial_kind=str(entry.get("adversarial_kind", "")),
            evidence={str(k): str(v) for k, v in dict(entry.get("evidence", {})).items()},
        )
        claim_order.append(claim.claim_id)

    coverage = {
        (str(entry["program_id"]), str(entry["coverage"]["part_number"])): PartCoverage(
            part_number=str(entry["coverage"]["part_number"]),
            covered=bool(entry["coverage"]["covered"]),
            serial_first=(
                None
                if entry["coverage"].get("serial_first") is None
                else int(entry["coverage"]["serial_first"])
            ),
            serial_last=(
                None
                if entry["coverage"].get("serial_last") is None
                else int(entry["coverage"]["serial_last"])
            ),
        )
        for entry in _entries(corpus_dir / "coverage.json", "coverage")
    }

    # `read_documents` and `read_clauses` are the store's readers, reused rather than reimplemented.
    # They also re-verify every clause against its own document's offsets on the way in, which is
    # kill condition I checked at console start-up for free; a second parser written here would be a
    # second opinion about what a clause is, and the two would differ the first time one changed.
    documents = read_documents(corpus_dir / "documents.json")
    loadable = read_clauses(corpus_dir / "clauses.json", documents)
    clauses = {item.clause.clause_id: item.clause for item in loadable}
    clause_version = {item.clause.clause_id: item.policy_version for item in loadable}

    current: dict[str, list[str]] = {}
    for item in loadable:
        document = documents[item.clause.document_id]
        if document.is_current:
            current.setdefault(item.clause.program_id, []).append(item.clause.clause_id)

    truth_payload: Any = json.loads((corpus_dir / "truth.json").read_text(encoding="utf-8"))
    truth = {
        str(claim_id): TruthRecord(
            outcome=str(entry["outcome"]),
            recoverable_amount=str(entry["recoverable_amount"]),
            currency=str(entry["currency"]),
            governing_clause_id=str(entry["governing_clause_id"]),
            missing_requirements=tuple(str(item) for item in entry.get("missing_requirements", ())),
        )
        for claim_id, entry in truth_payload["truth"].items()
    }

    return Corpus(
        programs=programs,
        claims=claims,
        claim_order=tuple(claim_order),
        coverage=coverage,
        clauses=clauses,
        clause_version=clause_version,
        documents=documents,
        truth=truth,
        current_clause_ids={key: tuple(value) for key, value in current.items()},
    )


@lru_cache(maxsize=4)
def load_corpus(corpus_dir: str) -> CorpusStatus:
    """The corpus for one directory, cached, and never raising into a request.

    Cached on the directory rather than on nothing, so that a test pointing `WCR_CORPUS_DIR` at a
    fixture does not have to reach into this module's cache to be believed. Cached at all because
    parsing eighteen programmes, seven hundred claims and two hundred and seventy clauses — and
    verifying every clause's offsets against its document — is work no page view should repeat.

    Every failure becomes a `problem` string. A console whose corpus is missing has something
    truthful and specific to render; one that raises has a traceback, and a reviewer reading a
    traceback cannot tell a missing `make corpus` from a broken parser.
    """
    try:
        return CorpusStatus(corpus=_read_corpus(Path(corpus_dir)), problem=None)
    except (OSError, ValueError, KeyError, TypeError) as error:
        return CorpusStatus(corpus=None, problem=_fault(error))


def current_corpus() -> CorpusStatus:
    return load_corpus(str(_resolve(get_settings().corpus_dir)))


# ------------------------------------------------------------------------------------------------
# The case. Computed by the deterministic core, not by this module.
# ------------------------------------------------------------------------------------------------


#: Evidence statuses that mean a requirement is not evidenced, in the order the gate weighs them.
#: `CONFLICTING` is checked before `ABSENT` here for the same reason `gate._RULES` checks conflict
#: before absence: a contradiction is a decision for a person and will still be a contradiction
#: after the missing document arrives, so discovering it on a second pass costs another day against
#: a window that is already running.
_WITHHOLDING_STATUSES: Final = (EvidenceStatus.CONFLICTING, EvidenceStatus.ABSENT)

_STATUS_FOR_EVIDENCE: Final[dict[EvidenceStatus, RequirementStatus]] = {
    EvidenceStatus.CONFLICTING: RequirementStatus.CONFLICTING,
    EvidenceStatus.ABSENT: RequirementStatus.MISSING,
}


def adjudicate(
    record: ClaimRecord,
    specs: Sequence[RequirementSpec],
    citation: Citation,
) -> tuple[Requirement, ...]:
    """Each requirement the rejection code demands, with the console's reading of the evidence.

    **This is the one judgement the console makes, and it is provisional.** The case machine's
    `requirements` node owns requirement adjudication and is a separate lane; until it writes case
    records this module has to produce a requirement list anyway, because `gate.decide` cannot be
    called without one and a case screen with no outcome is a case screen with no point.

    So the reading is deliberately coarse and deliberately conservative. It asks one question of the
    claim's whole evidence bundle — does it carry anything absent or contradictory — and applies the
    answer to every requirement the code raised, rather than mapping a particular missing artefact
    onto a particular requirement. It is coarse because that mapping does not exist in committed
    code: `requirements.RequirementSpec.evidence_key` and `corpus.claims.EVIDENCE_KEYS` are two
    vocabularies that barely intersect, and `corpus.claims.EVIDENCE_RELEVANT_TO` says in its own
    docstring that it is not the requirement matrix and must not be used as one. Inventing the join
    here would be writing the `requirements` node in the console, in a file nobody grades, and the
    two implementations would disagree the first time either changed.

    It is conservative because the errors are not symmetrical. Marking a requirement unmet that was
    in fact met sends a case to a person who did not need to see it, which costs somebody an hour.
    Marking one met that was not authorises a resubmission on evidence that does not exist, which is
    kill condition G and is budgeted at zero. So the bundle's worst finding governs, and a
    requirement is satisfied only when the bundle carries nothing absent and nothing contradictory.

    A satisfied requirement always carries `citation`, because `domain.Requirement` refuses to be
    constructed without one — kill condition H made impossible rather than measured.
    """
    worst: EvidenceStatus | None = None
    offending: tuple[str, ...] = ()
    for status in _WITHHOLDING_STATUSES:
        named = tuple(key for key in EVIDENCE_KEYS if record.evidence.get(key) == status.value)
        if named:
            worst, offending = status, named
            break

    if worst is None:
        return tuple(
            Requirement(
                requirement_id=spec.requirement_id,
                description=spec.description,
                status=RequirementStatus.SATISFIED,
                citation=citation,
                detail=(
                    f"the claim's evidence bundle carries every artefact it declares, and "
                    f"{citation.clause_id} governs {record.claim.rejection_code.value}"
                ),
            )
            for spec in specs
        )

    withheld = _STATUS_FOR_EVIDENCE[worst]
    detail = (
        f"the claim's evidence bundle records {', '.join(offending)} as {worst.value}; the case "
        f"machine's requirements node will attribute it to a specific requirement, and this "
        f"console applies it to all of them rather than guessing which"
    )
    return tuple(
        Requirement(
            requirement_id=spec.requirement_id,
            description=spec.description,
            status=withheld,
            citation=None,
            detail=detail,
        )
        for spec in specs
    )


def authority_clause(corpus: Corpus, claim: Claim) -> PolicyClause | None:
    """The first clause of the programme's **current** policy that governs this rejection code.

    Deterministic, offline, and deliberately not read out of `truth.json`. The generator records a
    `governing_clause_id` per claim and using it here would let the console cite the answer key —
    every citation would verify, the evidence screen would look immaculate, and it would be showing
    a lookup rather than a retrieval. Scanning the programme's own clauses for one whose `governs`
    list names the code is a question the corpus answers the same way for every reader.

    Withdrawn bulletins are excluded. The corpus contains superseded documents that share a
    programme and a policy version with the current ones, and a correct quote from a withdrawn
    bulletin is the wrong authority — the manufacturer will say so, and `domain.Citation` records
    the version precisely because of it.

    The order is the corpus file's order, which is fixed by the generator's written sequence rather
    than by a set. "The first clause that governs this code" therefore names the same clause on
    every machine and in every process.
    """
    for clause_id in corpus.current_clause_ids.get(claim.program_id, ()):
        clause = corpus.clauses[clause_id]
        if claim.rejection_code in clause.governs:
            return clause
    return None


class MoneyRow(NamedTuple):
    """One line of the recovery column, as a person adding up an invoice would write it."""

    label: str
    operator: str
    amount: Money
    note: str


class CaseView(NamedTuple):
    """One case, fully computed, or the reason it could not be.

    `problem` is not an error page. A cross-currency claim genuinely has no recoverable amount this
    system is entitled to state — `eligibility.refuse_cross_currency_claim` raises rather than
    converting, and ADR-001 records why — so the honest screen is the claim, the programme, the two
    currencies and a refusal, not a 500 and not a zero.
    """

    record: ClaimRecord
    program: WarrantyProgram
    truth: TruthRecord | None
    window: ClaimWindow | None
    assessment: Eligibility | None
    computation: RecoveryComputation | None
    decision: GateDecision | None
    requirements: tuple[Requirement, ...]
    clause: PolicyClause | None
    citation: Citation | None
    rows: tuple[MoneyRow, ...]
    problem: str | None


def _money_rows(computation: RecoveryComputation) -> tuple[MoneyRow, ...]:
    """The arithmetic as a column, ending on a figure that is the sum of the ones above it.

    Built from `RecoveryComputation`'s own fields and from nothing else. The one line that is not a
    stored field is the floor, and it is computed as a **residual** — the difference between the
    recorded recoverable amount and what the subtractions above it come to — rather than by
    re-deriving `recovery.compute`'s floors here. That distinction is the point of the row: a
    console that re-ran the formula would print a column that always adds up, including on the day
    the formula and the record disagreed, which is exactly the day a reader needs to be told.

    `capped_amount` is what the cap removed rather than the cap itself. `recovery.py` argues why: an
    adjudicator who cannot see the subtraction cannot check the total and refuses the package.
    """
    running = (
        computation.eligible_amount
        - computation.deductible
        - computation.capped_amount
        - computation.already_recovered
    )
    floor = computation.recoverable_amount - running
    return (
        MoneyRow("claimed total", "", computation.claimed_total, "parts plus labour, as invoiced"),
        MoneyRow(
            "labour excess",
            "-",
            computation.labour_excess,
            "the hours charged, at the amount by which the rate beat the programme's cap",
        ),
        MoneyRow(
            "uncovered parts",
            "-",
            computation.uncovered_parts,
            "parts the covered-parts schedule does not list for this build",
        ),
        MoneyRow(
            "eligible amount",
            "=",
            computation.eligible_amount,
            "what the policy ever covered",
        ),
        MoneyRow(
            "deductible",
            "-",
            computation.deductible,
            "the programme's excess, applied to the eligible amount and before the cap",
        ),
        MoneyRow(
            "removed by the claim cap",
            "-",
            computation.capped_amount,
            "what the ceiling took out, after the deductible; zero when the cap did not bind",
        ),
        MoneyRow(
            "already recovered",
            "-",
            computation.already_recovered,
            "what this recovery identity has already been paid",
        ),
        MoneyRow(
            "floor applied",
            "+",
            floor,
            "a recovery is never negative; zero unless a floor bound",
        ),
        MoneyRow(
            "recoverable amount",
            "=",
            computation.recoverable_amount,
            "what this system would file for",
        ),
    )


def build_case(corpus: Corpus, claim_id: str) -> CaseView | None:
    """Assemble one case by calling the committed core, and never by reimplementing it.

    Every figure on the case, evidence and recovery screens comes out of `deadline.claim_window`,
    `eligibility.assess`, `recovery.compute` and `gate.decide`, called in the order `CLAUDE.md`
    §2.1 fixes for the graph's nodes. The console holds no formula of its own: if it did, a reader
    checking the screen against the artifacts would be comparing two implementations rather than
    checking one.
    """
    record = corpus.claims.get(claim_id)
    if record is None:
        return None
    claim = record.claim
    program = corpus.programs.get(claim.program_id)
    if program is None:
        return None

    truth = corpus.truth.get(claim_id)
    window = deadline.claim_window(claim, program, record.as_of)
    clause = authority_clause(corpus, claim)
    citation = (
        None
        if clause is None
        else citation_for(
            clause, corpus.clause_version.get(clause.clause_id, program.policy_version)
        )
    )

    partial = CaseView(
        record=record,
        program=program,
        truth=truth,
        window=window,
        assessment=None,
        computation=None,
        decision=None,
        requirements=(),
        clause=clause,
        citation=citation,
        rows=(),
        problem=None,
    )

    if citation is None:
        return partial._replace(
            problem=(
                f"no clause in {program.program_id}'s current policy governs rejection code "
                f"{claim.rejection_code.value}, so this console cannot cite an authority and will "
                f"not report a requirement as satisfied by nothing"
            )
        )

    coverage = corpus.coverage.get((claim.program_id, claim.part_number))
    if coverage is None:
        return partial._replace(
            problem=(
                f"the covered-parts schedule for {program.program_id} has no entry for "
                f"{claim.part_number}; eligibility.assess refuses a coverage record for a "
                f"different part rather than answering about the wrong one"
            )
        )

    try:
        assessment = eligibility.assess(claim, program, coverage)
        computation = recovery.compute(claim, program, assessment)
    except CurrencyMismatchError:
        return partial._replace(
            problem=(
                f"{claim.claim_id} is claimed in {claim.claimed_parts.currency.value} against a "
                f"programme denominated in {program.currency.value}. This system holds no exchange "
                f"rate it is entitled to apply, so it states no recoverable amount at all rather "
                f"than a converted one that would look exact and already be wrong."
            )
        )

    requirements = adjudicate(record, required_for(claim.rejection_code), citation)
    decision = gate.decide(claim, program, window, assessment, computation, requirements)

    return partial._replace(
        assessment=assessment,
        computation=computation,
        decision=decision,
        requirements=requirements,
        rows=_money_rows(computation),
    )


# ------------------------------------------------------------------------------------------------
# The database, and the two things it is asked for.
# ------------------------------------------------------------------------------------------------


class DatabaseHealth(NamedTuple):
    reachable: bool
    detail: str


def bounded(url: str) -> str:
    """The DSN with a connection timeout, unless it already carries one.

    Measured rather than assumed: against an address that neither answers nor refuses, libpq's
    default behaviour here took over a minute to give up, and a readiness probe that takes a minute
    is a probe the platform gives up on first. The platform then reports the console as unreachable
    rather than the database as not ready, which is precisely the confusion `/healthz` exists to
    prevent — and the operator starts debugging the wrong process.

    Appended to the URL rather than passed as `connect_args`, because `build_engine` is where
    `pool_pre_ping` and the rest of this project's engine policy live and a second `create_engine`
    call in the console would be a second policy. `normalise_database_url` copies the query string
    verbatim, so an existing `?sslmode=require` survives and a password is never re-encoded.

    An explicit `connect_timeout` in the configured URL wins. Overriding one somebody set
    deliberately would be this console legislating for a deployment it knows nothing about.
    """
    if "connect_timeout=" in url:
        return url
    return f"{url}{'&' if '?' in url else '?'}connect_timeout={CONNECT_TIMEOUT_SECONDS}"


@lru_cache(maxsize=4)
def _engine(url: str) -> Engine:
    """One engine per URL. `build_engine` turns on `pool_pre_ping` and `store/engine.py` says why.

    Cached because a new engine per request is a new connection pool per request, and a console
    that opens a pool on every page view exhausts a free-tier PostgreSQL's connection limit long
    before anybody notices it is the console doing it.
    """
    return build_engine(bounded(url))


def database_health() -> DatabaseHealth:
    """Whether PostgreSQL answers, as a finding rather than as an exception.

    The broad catch is the entire purpose of this function. A readiness probe that raises is the one
    endpoint that must never do it: FastAPI turns an exception raised while resolving a dependency
    into a 500, and a 500 from `/healthz` tells the platform that the health check is broken rather
    than that the database is. The operator then debugs the console while the database is the thing
    that is down. Everything from a malformed URL through a refused connection to a driver that is
    not installed is the same answer here — not ready — and `_fault` keeps the DSN out of it.
    """
    settings = get_settings()
    try:
        engine = _engine(settings.database_url)
        with engine.connect() as connection:
            connection.execute(sql_text("SELECT 1"))
    except Exception as error:
        return DatabaseHealth(reachable=False, detail=_fault(error))
    return DatabaseHealth(reachable=True, detail="SELECT 1 answered")


@lru_cache(maxsize=1)
def _retriever() -> Retriever:
    """One retriever for the process. The encoder loads its ONNX session lazily and once.

    A retriever per request would reload a 285MB model on every page view — `artifacts/
    encoder_memory.json` has the measurement — and the console would present as a memory leak.
    """
    return Retriever()


class RetrievedClause(NamedTuple):
    rank: int
    clause: PolicyClause
    distance: float
    citation: Citation
    verified: bool
    document_title: str


class RetrievalView(NamedTuple):
    """What the dense stage returned, or why it did not run."""

    attempted: bool
    problem: str | None
    query: str
    clauses: tuple[RetrievedClause, ...]
    executed_statement: str
    timings_ms: Mapping[str, float]


def console_query(claim: Claim) -> str:
    """The question this console asks the index about a case.

    Assembled from the requirement matrix's own descriptions for the rejection code, plus the part
    number, because that is precisely what the case needs the policy to speak to and both halves
    come from committed tables rather than from a phrasing somebody tuned. `compose_query` then
    appends the rejection code in its fixed form; `retrieval.pipeline` argues why that form is
    fixed rather than engineered.

    This is **not** the query the evaluation harness scores. The recall figures on `/evaluation`
    come from `artifacts/retrieval.json`, and the screen says so. Two call sites asking two
    questions is a real risk — the pipeline's own docstring names it — and the honest handling is to
    label which question produced what is on the screen rather than to have the console quietly
    publish a recall of its own.
    """
    demands = " ".join(spec.description for spec in required_for(claim.rejection_code))
    return f"{claim.part_number} — {demands}"


def retrieve(corpus: Corpus, case: CaseView) -> RetrievalView:
    """Run the dense stage for one case, or report why it could not run.

    Behind a button rather than on page load, and the caller decides. Embedding a query costs an
    ONNX session on the first request of a process, and a screen whose first paint waits on that is
    a screen a reviewer closes. The deterministic half of `/evidence` — the requirement list, the
    authority clause and its offsets — renders with no database at all.

    Every citation that comes back is verified against the document's own text with
    `verify_citation`, the same function kill condition I is graded with. A retrieval screen that
    printed offsets without checking them would be showing coordinates nobody had tested.
    """
    query = console_query(case.record.claim)
    empty = RetrievalView(
        attempted=True,
        problem=None,
        query=query,
        clauses=(),
        executed_statement="",
        timings_ms={},
    )
    try:
        engine = _engine(get_settings().database_url)
        with session_scope(engine) as session:
            result = _retriever().retrieve(
                session,
                query=query,
                program_id=case.program.program_id,
                policy_version=case.program.policy_version,
                rejection_code=case.record.claim.rejection_code,
                k=DEFAULT_K,
            )
    except Exception as error:
        return empty._replace(problem=_fault(error))

    clauses: list[RetrievedClause] = []
    for scored in result.clauses:
        document = corpus.documents.get(scored.clause.document_id)
        version = corpus.clause_version.get(scored.clause.clause_id, case.program.policy_version)
        citation = citation_for(scored.clause, version)
        clauses.append(
            RetrievedClause(
                rank=scored.rank,
                clause=scored.clause,
                distance=scored.distance,
                citation=citation,
                verified=document is not None and verify_citation(citation, document.text),
                document_title=(
                    document.title if document is not None else scored.clause.document_id
                ),
            )
        )
    return empty._replace(
        clauses=tuple(clauses),
        executed_statement=result.executed_statement,
        timings_ms=result.timings_ms,
    )


# ------------------------------------------------------------------------------------------------
# The header, and the one thing configuration is allowed to tell a template.
# ------------------------------------------------------------------------------------------------


class PageHeader(NamedTuple):
    """The only two configuration values any template sees, carried by value.

    Not a `Settings`. `Settings` holds `database_url` and `llm_api_key`, and a template that has the
    object has the credentials — one `{{ settings.database_url }}` away, in a file that looks
    exactly like the file that does not do it. Copying out the two fields the header actually
    renders makes the leak impossible rather than merely absent, and `tests/test_api.py` proves it
    with sentinels planted in the environment.

    `approvals_possible` is deliberately **not** here although it is only a boolean. A boolean
    derived from a secret is still a channel, the header has no use for it, and the 403 from the
    approval route already tells a caller everything they are entitled to know.
    """

    environment: str
    read_only: bool


def page_header(settings: Settings) -> PageHeader:
    return PageHeader(environment=settings.environment, read_only=settings.read_only)


def _page(request: Request, active: str, **extra: Any) -> dict[str, Any]:
    """The context every screen shares.

    One place, so that no screen can forget the banner or the synthetic-corpus notice. A context
    assembled per route is a route that will one day be added without them.
    """
    context: dict[str, Any] = {
        "request": request,
        "active": active,
        "header": page_header(get_settings()),
        "banner": release_gate_banner(),
        "synthetic_notice": SYNTHETIC_NOTICE,
        "not_run": NOT_RUN,
        "not_measured": NOT_MEASURED,
    }
    context.update(extra)
    return context


# ------------------------------------------------------------------------------------------------
# The application.
# ------------------------------------------------------------------------------------------------

app: Final = FastAPI(
    title="Warranty Recovery Lab",
    description=(
        "A read-only console over a synthetic warranty-claim corpus. It reports measurements "
        "recorded in artifacts/*.json and computes no verdict of its own."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url=None,
)

#: The console's own audit log. In memory, bounded by the process, and never presented as durable:
#: `CLAUDE.md` §2.1 puts the durable record in the LangGraph checkpoint, and a second durable store
#: would need its own consistency story with the first. What it does prove is the approval contract
#: — that a granted approval is recorded against a case **and a case version** before anything
#: could act on it — and `/audit` renders it under a heading that says where it lives.
_console_log: Final = AuditLog()


class ApprovalRequest(BaseModel):
    """The body of the one mutating route.

    Frozen and `extra="forbid"` like every model in this project, and `case_version` is required
    with no default. A default would produce an approval bound to a version nobody stated, which is
    the failure kill condition D exists for: "approved" and "approved *this*" are different claims,
    and an approval that names no version cannot be checked against the thing that went out.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_version: int = Field(ge=0)
    note: str = Field(default="", max_length=500)


def _selected(corpus: Corpus, claim: str | None) -> str | None:
    if claim and claim in corpus.claims:
        return claim
    return corpus.claim_order[0] if corpus.claim_order else None


def _picker_groups(corpus: Corpus, *, program_id: str | None) -> list[dict[str, Any]]:
    """Claim identifiers grouped by programme, in corpus order.

    Restricted to one programme when the caller names one. The full list is seven hundred and
    twenty options and belongs on the case screen where a reader is choosing; repeating it on every
    screen would put sixty kilobytes of `<option>` in front of the evidence on each of them.
    """
    groups: dict[str, list[str]] = {}
    for claim_id in corpus.claim_order:
        owner = corpus.claims[claim_id].claim.program_id
        if program_id is not None and owner != program_id:
            continue
        groups.setdefault(owner, []).append(claim_id)
    return [
        {
            "program_id": key,
            "label": (
                f"{corpus.programs[key].manufacturer} {corpus.programs[key].policy_version}"
                if key in corpus.programs
                else key
            ),
            "claim_ids": value,
        }
        for key, value in groups.items()
    ]


def _case_context(request: Request, active: str, claim: str | None) -> dict[str, Any]:
    """The corpus, the selected case and the picker, or the empty state explaining their absence.

    Returning a context rather than a 404 for an unknown claim is deliberate: a reviewer who
    mistypes an identifier is better served by the first case and a working picker than by a status
    code, and nothing here is an API contract. `/api/case/{claim_id}` does return 404, because that
    one is.
    """
    status = current_corpus()
    if status.corpus is None:
        return _page(request, active, corpus_problem=status.problem, case=None, groups=[])
    corpus = status.corpus
    claim_id = _selected(corpus, claim)
    case = None if claim_id is None else build_case(corpus, claim_id)
    program_id = None if case is None or active == "case" else case.program.program_id
    return _page(
        request,
        active,
        corpus_problem=None,
        case=case,
        claim_id=claim_id,
        groups=_picker_groups(corpus, program_id=program_id),
        corpus_size=len(corpus.claim_order),
    )


@app.get("/", response_class=HTMLResponse)
def case_screen(
    request: Request,
    claim: Annotated[str | None, Query(max_length=120)] = None,
) -> Response:
    """CASE — the claim, the part, the failure date, the rejection, the window and the outcome."""
    return templates.TemplateResponse(request, "case.html", _case_context(request, "case", claim))


@app.get("/evidence", response_class=HTMLResponse)
def evidence_screen(
    request: Request,
    claim: Annotated[str | None, Query(max_length=120)] = None,
    retrieve_now: Annotated[bool, Query(alias="retrieve")] = False,
) -> Response:
    """EVIDENCE — the requirement list, the authority clause, and the dense stage on request."""
    context = _case_context(request, "evidence", claim)
    case = context.get("case")
    status = current_corpus()
    retrieval: RetrievalView | None = None
    if retrieve_now and case is not None and status.corpus is not None:
        retrieval = retrieve(status.corpus, case)
    context["retrieval"] = retrieval
    context["retrieval_requested"] = retrieve_now
    context["k"] = DEFAULT_K
    return templates.TemplateResponse(request, "evidence.html", context)


@app.get("/recovery", response_class=HTMLResponse)
def recovery_screen(
    request: Request,
    claim: Annotated[str | None, Query(max_length=120)] = None,
) -> Response:
    """RECOVERY — the arithmetic as a column a person can add up, not a total."""
    return templates.TemplateResponse(
        request, "recovery.html", _case_context(request, "recovery", claim)
    )


@app.get("/evaluation", response_class=HTMLResponse)
def evaluation_screen(request: Request) -> Response:
    """EVALUATION — what was measured, by whom, and every kill condition with its verdict.

    Reads `artifacts/*.json` and opens no connection. False recoveries get their own number and are
    never folded into a combined score; the template says why, and it is the reason the whole
    screen is laid out around counts rather than around rates.
    """
    context = _page(
        request,
        "evaluation",
        criteria=CRITERIA,
        verdicts=criterion_verdicts(),
        recovery_artifact=read_artifact("recovery.json"),
        retrieval_artifact=read_artifact("retrieval.json"),
        groundedness_artifact=read_artifact("groundedness.json"),
        holdout_artifact=read_artifact("holdout.json"),
        corpus_artifact=read_artifact("corpus.json"),
        pgvector_artifact=read_artifact("pgvector.json"),
        metric=metric,
    )
    return templates.TemplateResponse(request, "evaluation.html", context)


@app.get("/audit", response_class=HTMLResponse)
def audit_screen(request: Request) -> Response:
    """AUDIT — the decision path, the events, the approval binding and the idempotency evidence."""
    context = _page(
        request,
        "audit",
        path=tuple(CaseState),
        kinds=(
            APPROVAL_REQUESTED,
            APPROVAL_GRANTED,
            SUBMISSION_MADE,
            SUBMISSION_SUPPRESSED,
            CASE_REPRICED,
            TOOL_STARTED,
            TOOL_COMPLETED,
        ),
        submission_artifact=read_artifact("submission.json"),
        durability_artifact=read_artifact("durability.json"),
        redis_artifact=read_artifact("redis.json"),
        console_events=tuple(_console_log),
        approver_header=APPROVER_HEADER_NAME,
        approver_env=APPROVER_TOKEN_ENV,
        metric=metric,
    )
    return templates.TemplateResponse(request, "audit.html", context)


# ------------------------------------------------------------------------------------------------
# Health. Two endpoints, because they answer two questions.
# ------------------------------------------------------------------------------------------------


@app.get("/livez")
def livez() -> JSONResponse:
    """Liveness. Always 200, and takes no dependency of any kind.

    `docs/architecture.md` §6 points the platform's health check here rather than at `/healthz`. The
    evidence screens work perfectly without PostgreSQL, so a platform check wired to readiness would
    refuse to route traffic to a console that is serving them — a database still waking up would
    present as a service that is down.
    """
    return JSONResponse({"status": "alive"}, status_code=200)


@app.get("/healthz")
def healthz() -> JSONResponse:
    """Readiness. 200 with a database, 503 without, and never 500.

    503 is the truthful answer and it stays 503 for as long as the database is unreachable. The
    detail is the exception's type name; `_fault` explains why it is never the message.
    """
    health = database_health()
    return JSONResponse(
        {
            "status": "ready" if health.reachable else "degraded",
            "database": "reachable" if health.reachable else "unreachable",
            "detail": health.detail,
            "screens_available_without_a_database": ["/evaluation", "/audit"],
        },
        status_code=200 if health.reachable else 503,
    )


# ------------------------------------------------------------------------------------------------
# JSON.
# ------------------------------------------------------------------------------------------------


def _requirement_json(requirement: Requirement) -> dict[str, Any]:
    citation = requirement.citation
    return {
        "requirement_id": requirement.requirement_id,
        "description": requirement.description,
        "status": requirement.status.value,
        "detail": requirement.detail,
        "citation": None if citation is None else citation.model_dump(mode="json"),
    }


@app.get("/api/case/{claim_id}")
def case_json(claim_id: Annotated[str, PathParam(max_length=120)]) -> JSONResponse:
    """One case as JSON, with every amount a string.

    `Money.as_json` and never a float. `money.py` argues it: `json.dumps` on a float reintroduces
    binary floating point at the one boundary where this system's numbers are read by something
    else, which is the boundary that matters.
    """
    status = current_corpus()
    if status.corpus is None:
        return JSONResponse(
            {"error": "corpus_unavailable", "detail": status.problem or "unknown"},
            status_code=503,
        )
    case = build_case(status.corpus, claim_id)
    if case is None:
        return JSONResponse({"error": "unknown_claim", "claim_id": claim_id}, status_code=404)

    claim = case.record.claim
    window = case.window
    computation = case.computation
    decision = case.decision
    return JSONResponse(
        {
            "is_synthetic": True,
            "notice": SYNTHETIC_NOTICE,
            "claim": {
                "claim_id": claim.claim_id,
                "program_id": claim.program_id,
                "part_number": claim.part_number,
                "serial_number": claim.serial_number,
                "in_service_date": claim.in_service_date.isoformat(),
                "failure_date": claim.failure_date.isoformat(),
                "repair_invoice_date": claim.repair_invoice_date.isoformat(),
                "rejection_code": claim.rejection_code.value,
                "rejected_on": claim.rejected_on.isoformat(),
                "claimed_parts": claim.claimed_parts.as_json(),
                "claimed_labour_hours": claim.claimed_labour_hours,
                "claimed_labour_rate": claim.claimed_labour_rate.as_json(),
                "recovery_identity": claim.recovery_identity,
            },
            "program": {
                "program_id": case.program.program_id,
                "manufacturer": case.program.manufacturer,
                "policy_version": case.program.policy_version,
                "currency": case.program.currency.value,
                "correction_window_days": case.program.correction_window_days,
                "warranty_months": case.program.warranty_months,
                "labour_rate_cap_per_hour": case.program.labour_rate_cap_per_hour.as_json(),
                "deductible": case.program.deductible.as_json(),
                "claim_cap": case.program.claim_cap.as_json(),
            },
            "window": None
            if window is None
            else {
                "rejected_on": window.rejected_on.isoformat(),
                "closes_on": window.closes_on.isoformat(),
                "as_of": window.as_of.isoformat(),
                "is_open": window.is_open,
                "days_remaining": window.days_remaining,
            },
            "computation": None
            if computation is None
            else {
                "currency": computation.currency.value,
                "claimed_total": computation.claimed_total.as_json(),
                "labour_excess": computation.labour_excess.as_json(),
                "uncovered_parts": computation.uncovered_parts.as_json(),
                "eligible_amount": computation.eligible_amount.as_json(),
                "deductible": computation.deductible.as_json(),
                "capped_amount": computation.capped_amount.as_json(),
                "already_recovered": computation.already_recovered.as_json(),
                "recoverable_amount": computation.recoverable_amount.as_json(),
                "excluded_amount": computation.excluded_amount.as_json(),
            },
            "decision": None
            if decision is None
            else {
                "outcome": decision.outcome.value,
                "reason": decision.reason,
                "authorises_recovery": decision.authorises_recovery,
                "withholds_recovery": gate.withholds_recovery(decision.outcome),
                "signals": decision.signals.model_dump(mode="json"),
            },
            "requirements": [_requirement_json(item) for item in case.requirements],
            "requirement_adjudication": (
                "provisional, by this console: the case machine's requirements node is the "
                "authority and its output is not yet committed"
            ),
            "authority_citation": None
            if case.citation is None
            else case.citation.model_dump(mode="json"),
            "ground_truth": None
            if case.truth is None
            else {
                "outcome": case.truth.outcome,
                "recoverable_amount": case.truth.recoverable_amount,
                "currency": case.truth.currency,
                "governing_clause_id": case.truth.governing_clause_id,
                "missing_requirements": list(case.truth.missing_requirements),
                "vocabulary": "evidence keys, as the generator records them",
            },
            "problem": case.problem,
        },
        status_code=200,
    )


# ------------------------------------------------------------------------------------------------
# The one mutating route.
# ------------------------------------------------------------------------------------------------


def _refusal(reason: str, detail: str) -> JSONResponse:
    return JSONResponse({"error": reason, "detail": detail}, status_code=403)


@app.post("/api/case/{claim_id}/approval")
def grant_approval(
    claim_id: Annotated[str, PathParam(max_length=120)],
    body: ApprovalRequest,
    presented: Annotated[str | None, Header(alias=APPROVER_HEADER_NAME)] = None,
) -> JSONResponse:
    """Record a human approval against a case and a case version. The only mutation in this file.

    Three locks, checked in this order and for this reason.

    **Read-only first.** `WCR_READ_ONLY` defaults to true, so a deployment that configured nothing
    refuses everything. Checking the token first would make a correct token sufficient in a
    read-only deployment, which inverts the two controls: read-only is a property of the deployment
    and the token is a credential, and a credential is the one of the two that can leak.

    **Then the token's existence.** `WCR_APPROVER_TOKEN` has no default, so an instance that was
    never given one can approve nothing. `config.py` calls this failing closed, and it is the second
    lock on the public demo: `docs/architecture.md` §6 records that the deployed instance sets
    neither, so nothing can be approved there and — since submission is reachable only through an
    approval — nothing can be submitted either.

    **Then the presented value**, compared with `hmac.compare_digest` rather than `==`. The timing
    difference on a short string over the public internet is not a realistic attack and the
    constant-time comparison costs nothing, so the argument for `==` is that it reads more
    naturally, which is not an argument.

    The comparison is made over **encoded bytes** and not over the two strings. `compare_digest`
    raises `TypeError` when either `str` carries a character above U+007F, and Starlette decodes a
    request header as latin-1, so any request presenting a byte above 127 in this header would
    produce a non-ASCII `str` and turn the refusal into a 500. A 500 on the one mutating route is
    the worst possible answer to a bad credential: it reports the console as broken rather than the
    credential as wrong, and it is trivially reachable by anyone who can send a header. Encoding
    both sides first makes every wrong value the same wrong value — a 403. The rejected alternative
    was to validate the header against an ASCII pattern before comparing, which adds a second place
    that decides what a token may look like and would reject a legitimately configured non-ASCII
    token outright rather than compare it.

    All three refusals are 403 and not 401. 401 invites a `WWW-Authenticate` negotiation this
    console does not implement, and a browser prompting for credentials against a demo is a worse
    outcome than a plain refusal that says what is wrong.

    What it records is an audit event, not a submission. The event carries the case **and** the
    version, which is what kill condition D grades; the durable binding lives in the case machine's
    checkpoint, and `/audit` labels these events as this process's own.
    """
    settings = get_settings()
    if settings.read_only:
        return _refusal(
            "read_only",
            "WCR_READ_ONLY is true, so this instance refuses every mutation whatever credential "
            "is presented. A misconfigured public console that accepts approvals is an incident; "
            "one that refuses them is an inconvenience.",
        )

    expected = settings.approver_token
    if not expected:
        return _refusal(
            "no_approver_configured",
            f"{APPROVER_TOKEN_ENV} has no value, so the human-in-the-loop gate can approve "
            f"nothing. It has no default on purpose: a deployment that forgot it fails closed.",
        )

    if presented is None or not hmac.compare_digest(
        presented.encode("utf-8"), expected.encode("utf-8")
    ):
        return _refusal(
            "approver_token_rejected",
            f"the {APPROVER_HEADER_NAME} header was absent or did not match the configured "
            f"approver token.",
        )

    status = current_corpus()
    if status.corpus is None or claim_id not in status.corpus.claims:
        return JSONResponse({"error": "unknown_claim", "claim_id": claim_id}, status_code=404)

    record = status.corpus.claims[claim_id]
    event = _console_log.append(
        case_id=claim_id,
        case_version=body.case_version,
        kind=APPROVAL_GRANTED,
        node=CaseState.AWAITING_APPROVAL,
        # The corpus's own as-of date, never a wall-clock reading. `CLAUDE.md` §4 requires
        # deterministic output, and an audit event stamped with the hour it was granted would make
        # every artifact that quoted it unreproducible.
        at=record.as_of,
        actor=_APPROVAL_ACTOR,
        detail={
            "recovery_identity": record.claim.recovery_identity,
            "note": body.note,
            "recorded_in": "the console's in-process audit log, not a durable store",
        },
    )
    return JSONResponse(
        {
            "recorded": _event_json(event),
            "durability": (
                "this event lives in the console process. The durable approval binding is the case "
                "machine's checkpoint, and kill condition D is graded from the audit stream rather "
                "than from this response."
            ),
        },
        status_code=201,
    )


def _event_json(event: AuditEvent) -> dict[str, Any]:
    return {
        "event_id": event.event_id,
        "case_id": event.case_id,
        "case_version": event.case_version,
        "kind": event.kind,
        "node": event.node.value,
        "at": event.at.isoformat(),
        "actor": event.actor,
        "detail": event.detail,
    }
