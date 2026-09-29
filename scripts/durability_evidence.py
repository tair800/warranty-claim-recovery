"""Kill a worker at every node, resume it from a different process, and write down what it cost.

    python scripts/durability_evidence.py            # write artifacts/durability.json and
                                                     # artifacts/submission.json

ADR-001 §4.1 and §4.2 are the two claims this script has to make checkable rather than plausible:
killing a worker mid-case loses no node, no completed tool and no human decision; and nothing leaves
the system without a recorded approval or leaves it twice. Kill conditions A, B, C, D and E are
graded from the two files written here, by `tests/test_kill_criteria.py`, which imports nothing from
this package.

### A kill is a process death, and nothing else counts

Every kill below is `os._exit` inside a node, in a child process the parent spawned, and the parent
asserts the child exited with exactly that status. The rejected alternative — raise an exception
inside the node, catch it, and call the graph again in the same interpreter — is the version that
proves nothing, because the thing being claimed is precisely that **no in-process state is
load-bearing**. A simulated kill leaves the engine's pool, the encoder's ONNX session, every
module-level cache and the ledger's own connection exactly where the killed run left them, so a
system that secretly depended on any of them would pass. `os._exit` runs no `finally`, flushes no
buffer, commits no transaction and closes no socket, which is what a `SIGKILL` on a container does.

The resume is then performed by a **second graph built over the same checkpointer**, in the parent,
which never executed any part of the killed run. Two graph objects, two processes, one PostgreSQL
thread.

### Where the kills are placed, and why not by a timer

`CaseDependencies.kill_switch` is offered the node and the number of tool calls that have completed,
so a kill can be placed at a point that can be named in the evidence: *after the retrieval tool
completed and before the node returned* is a different claim from *before it ran*, and kill
condition B is a statement about exactly that difference. A timer-placed kill lands somewhere nobody
can name, and the resulting artifact could not say which node was interrupted or whether a tool had
finished. A kill that cannot be placed is a kill that cannot be graded.

`KILL_PLAN` places one kill at every node in `machine.NODE_NAMES`, which is what makes
`cases_killed >= len(NODE_NAMES)` a statement about coverage rather than about volume. Two of the
eleven are the ones the project exists for: `SUBMITTED` after the filing completed, and
`AWAITING_APPROVAL` after a person's decision was recorded.

### Why the re-execution counter lives in Redis

Kill condition B asks whether a tool that had completed ran again. Asking the ledger would be asking
the component under test to grade itself: the ledger decides whether to replay, so a count derived
from its rows is a count of what it *believes* it did. `ExecutionCounter` increments a Redis key
inside the closure the ledger calls, so it counts the tool actually starting, and `ToolLedger` has
no Redis client and no code path that reads one. The instrument and the subject are in different
stores.

### Why the concurrency race is staged by killing three cases

The race needs a genuine case: a real claim, a real gate decision, a real recorded human approval,
and nothing filed yet. Three cases are carried by the machine to the submit node and killed there
**before** the filing, which produces exactly that state without any part of the pipeline being
reimplemented for the benefit of the test. Twenty-four threads then enter `submission.submit` on a
barrier. The alternative — hand-building a `GateDecision` and an approval event — would race a
fixture rather than the system, and the approval it recorded would be one nobody had granted.

### What this script does not decide

Every count in `artifacts/submission.json` that kill conditions D and E read comes from
`audit.verify_submissions`, which is fed the event stream and imports nothing from the graph. This
script counts threads, portal calls and identities — the things `verify_submissions` cannot see —
and takes the approval and duplicate findings from it verbatim. A script that counted its own
approvals would be grading its own opinion of whether it had behaved.

### Determinism

The cases are chosen from the committed corpus in corpus order, the case identifiers are derived
from claim identifiers, and no wall clock is read: the date a case is reasoned as of is the one the
corpus recorded. The run therefore clears its own rows before it starts — the checkpoints, the
ledger invocations and the Redis keys for its own case identifiers, and nothing else — so a second
run writes the same two files rather than reporting a duplicate submission it made itself.

**Only development-split programmes are used.** Nothing here is scored against the hold-out and
nothing here would be improved by touching it, so ADR-001 §7 is honoured by not reading it at all
rather than by promising not to look.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable, Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any, Final, NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from langgraph.types import Command  # noqa: E402
from redis import Redis  # noqa: E402
from sqlalchemy import text as sql_text  # noqa: E402
from sqlalchemy.engine import Engine  # noqa: E402

from warranty_claim_recovery.audit import (  # noqa: E402
    APPROVAL_GRANTED,
    SUBMISSION_SUPPRESSED,
    verify_submissions,
)
from warranty_claim_recovery.config import APPROVER_TOKEN_ENV, Settings, get_settings  # noqa: E402
from warranty_claim_recovery.corpus.claims import EvidenceStatus  # noqa: E402
from warranty_claim_recovery.corpus.holdout import is_holdout  # noqa: E402
from warranty_claim_recovery.domain import AuditEvent, CaseState, RejectionCode  # noqa: E402
from warranty_claim_recovery.evaluation.pipeline import EVIDENCE_JOIN  # noqa: E402
from warranty_claim_recovery.graph.machine import (  # noqa: E402
    NODE_NAMES,
    build_machine,
    case_config,
    checkpointer,
    pending_nodes,
)
from warranty_claim_recovery.graph.nodes import (  # noqa: E402
    APPROVAL_REFUSED,
    NODE_SEQUENCE,
    SYSTEM_ACTOR,
    TOOL_RECORD_DECISION,
    CaseDependencies,
    KillSwitch,
)
from warranty_claim_recovery.graph.state import (  # noqa: E402
    CaseAudit,
    GraphState,
    claim_from_json,
    decision_from_json,
    event_from_json,
    initial_state,
    window_from_json,
)
from warranty_claim_recovery.graph.tools import (  # noqa: E402
    INVOCATION_TABLE,
    ToolLedger,
    ToolResult,
)
from warranty_claim_recovery.queue.idempotency import SubmissionGuard  # noqa: E402
from warranty_claim_recovery.requirements import required_for  # noqa: E402
from warranty_claim_recovery.retrieval.pipeline import Retriever  # noqa: E402
from warranty_claim_recovery.store.engine import build_engine, session_scope  # noqa: E402
from warranty_claim_recovery.store.schema import DOCUMENT_TABLE  # noqa: E402
from warranty_claim_recovery.submission import (  # noqa: E402
    ManufacturerPortal,
    SubmissionOutcome,
    record_outcome,
    submit,
)

DEFAULT_CORPUS = REPO_ROOT / "data" / "generated"
DEFAULT_ARTIFACTS = REPO_ROOT / "artifacts"

#: The status a killed child exits with. A distinct value rather than 1, so that a child which died
#: of a genuine error cannot be read as a kill that landed where the plan put it — which would turn
#: a broken run into a passing artifact.
KILLED_EXIT: Final = 97

#: Threads per raced identity. Kill condition E requires at least sixteen; twenty-four is used so
#: that a run which lost a thread to a scheduling accident still clears the floor, and so that the
#: number in the artifact is one the criterion did not choose.
RACE_THREADS: Final = 24

#: Identities raced. Three rather than one, because "exactly one effect" measured over a single
#: identity is a result that a coin could produce.
RACE_IDENTITIES: Final = 3

#: The Redis namespace this run claims submissions under, and the prefix its execution counters
#: live behind. Both are specific to the evidence run so that clearing them cannot touch a lease or
#: a claim belonging to a worker.
GUARD_NAMESPACE: Final = "wcr-evidence"
COUNTER_PREFIX: Final = "wcr:evidence:executions:"

#: Who approves, in the evidence run. A fixed name so the artifact is reproducible; the token comes
#: from the environment and has no default, which is the whole of `CLAUDE.md` §3 rule 9.
APPROVER: Final = "d.okonkwo"
APPROVAL_NOTE: Final = "evidence run: approved against the gate's own reason"

# Both follow `retrieval.pipeline.RETRIEVAL_STATEMENT` and `graph.tools`: the table name comes from
# a module constant, every value is a bound parameter, and the verb sits on its own line so that no
# interpolated fragment carries one. There is no path by which caller data reaches either string.
_DOCUMENT_TEXT: Final = f"SELECT text\nFROM {DOCUMENT_TABLE}\nWHERE document_id = :document_id"
_DELETE_INVOCATIONS: Final = f"DELETE\nFROM {INVOCATION_TABLE}\nWHERE case_id = :case_id"

#: `CaseState` to the LangGraph node name, derived from the pipeline rather than retyped. A second
#: mapping would be a second opinion about which node a kill landed in, and kill condition A is
#: graded by comparing exactly that against what the checkpoint recorded.
_NODE_OF: Final[dict[CaseState, str]] = {state: name for name, state in NODE_SEQUENCE}


class EvidenceError(RuntimeError):
    """The run could not produce evidence, as distinct from producing evidence of a failure.

    A separate type because the two must never be confused. A kill that did not land, a case that
    did not reach the node it was supposed to be killed at, or a corpus too small to cover every
    node all mean the artifact would describe something other than what it claims to describe. The
    honest outcome is no artifact and a non-zero exit, not a file with small numbers in it.
    """


# ------------------------------------------------------------------------------- the instruments


class ExecutionCounter:
    """What actually ran, counted in a store the component under test cannot read.

    Kill condition B asks whether a tool that had already completed ran again after the resume.
    Answering it from `ToolLedger`'s own rows would be asking the ledger whether it did the thing it
    decides: the rows say what it recorded, and a ledger that replayed when it should have run, or
    ran when it should have replayed, would write the same rows either way.

    So the count is taken inside the closure the ledger calls, and it is kept in Redis.
    `ToolLedger` holds a SQLAlchemy `Engine` and nothing else, has no Redis client and no code path
    that could acquire one, so there is no arrangement under which its decision could be informed by
    this number. The instrument and the subject are in different stores, which is the only version
    of "the wrapper cannot see it" that survives somebody refactoring the wrapper.

    The counter increments **before** the tool runs. A tool killed halfway through has run, and
    counting only completions would report the interrupted attempt as though it had never happened —
    which is the one case where an honest retry must not be graded as a re-execution and where the
    evidence therefore has to be able to see both numbers.
    """

    __slots__ = ("_client",)

    def __init__(self, redis_url: str) -> None:
        self._client: Any = Redis.from_url(redis_url, decode_responses=True)

    @staticmethod
    def key(invocation_id: str) -> str:
        return f"{COUNTER_PREFIX}{invocation_id}"

    def record(self, invocation_id: str) -> None:
        self._client.incr(self.key(invocation_id))

    def count(self, invocation_id: str) -> int:
        raw = self._client.get(self.key(invocation_id))
        return 0 if raw is None else int(raw)

    def forget(self) -> int:
        """Clear this run's counters. Matches only the evidence prefix; see the module docstring."""
        keys = [str(key) for key in self._client.scan_iter(match=f"{COUNTER_PREFIX}*", count=500)]
        if keys:
            self._client.delete(*keys)
        return len(keys)

    def close(self) -> None:
        self._client.close()


class CountingLedger(ToolLedger):
    """The shipped ledger with an observer wrapped around the callable it invokes.

    It overrides nothing about the decision. `invoke` wraps `run` in a closure that increments the
    counter and then calls the original, and hands that to `ToolLedger.invoke`, which replays or
    runs exactly as it would have. A subclass that reimplemented the replay logic would be grading a
    copy of the thing that ships; this one cannot diverge from it, because it does not contain it.
    """

    __slots__ = ("_counter",)

    def __init__(self, engine: Engine, counter: ExecutionCounter) -> None:
        super().__init__(engine)
        self._counter = counter

    def invoke(
        self,
        *,
        case_id: str,
        case_version: int,
        node: CaseState,
        tool: str,
        call_index: int,
        run: Callable[[], Mapping[str, Any]],
        audit: CaseAudit,
        at: date,
        actor: str,
    ) -> ToolResult:
        invocation_id = self.invocation_id(case_id, node, tool, call_index)

        def counted() -> Mapping[str, Any]:
            self._counter.record(invocation_id)
            return run()

        return super().invoke(
            case_id=case_id,
            case_version=case_version,
            node=node,
            tool=tool,
            call_index=call_index,
            run=counted,
            audit=audit,
            at=at,
            actor=actor,
        )


class PgVectorEvidence:
    """The graph's `EvidenceSource`, backed by the real index rather than by a fixture.

    The durability claim does not depend on which retriever sits behind the tool, and the evidence
    is worth more if the tool is the expensive one the system actually runs: an embedding call and a
    pgvector search are what a reader pictures when they read "no completed tool re-executed", and a
    dictionary lookup is not. So this is `retrieval.pipeline.Retriever` and the seeded clause index,
    with no shortcut.

    Document bodies are cached per process. They are immutable for the length of a run and the
    citation check reads one per retrieved clause; the cache saves five round trips per case and
    changes no answer. It is deliberately per process rather than shared, so a killed child takes
    its cache with it and the resuming parent reads the store.
    """

    __slots__ = ("_documents", "_engine", "_retriever")

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._retriever = Retriever()
        self._documents: dict[str, str | None] = {}

    def clauses(
        self,
        *,
        program_id: str,
        policy_version: str,
        rejection_code: RejectionCode,
        query: str,
        k: int,
    ) -> list[dict[str, Any]]:
        with session_scope(self._engine) as session:
            found = self._retriever.retrieve(
                session,
                query=query,
                program_id=program_id,
                policy_version=policy_version,
                rejection_code=rejection_code,
                k=k,
            )
        return [
            {
                "clause_id": scored.clause.clause_id,
                "document_id": scored.clause.document_id,
                "section": scored.clause.section,
                "text": scored.clause.text,
                "start_offset": scored.clause.start_offset,
                "end_offset": scored.clause.end_offset,
                "governs": [code.value for code in scored.clause.governs],
                "rank": scored.rank,
            }
            for scored in found.clauses
        ]

    def document_text(self, document_id: str) -> str | None:
        if document_id in self._documents:
            return self._documents[document_id]
        with session_scope(self._engine) as session:
            row = session.execute(
                sql_text(_DOCUMENT_TEXT), {"document_id": document_id}
            ).one_or_none()
        body = None if row is None else str(row.text)
        self._documents[document_id] = body
        return body


class LockedAudit:
    """`CaseAudit` behind a lock, because twenty-four threads write to it at once.

    `submission.py` records why the submitter does not write its own events: `AuditLog` assigns
    sequence numbers from a mutable counter, and a lock held across the Redis round trip and the
    filing would serialise the race being measured. The lock here covers the append and nothing
    else, so the twenty-four callers still contend for the identity exactly as they would in
    production and only the bookkeeping is ordered.
    """

    __slots__ = ("_audit", "_lock")

    def __init__(self, existing: Sequence[Mapping[str, Any]]) -> None:
        self._audit = CaseAudit(existing)
        self._lock = threading.Lock()

    def record(
        self,
        *,
        case_id: str,
        case_version: int,
        kind: str,
        node: CaseState,
        at: date,
        actor: str,
        detail: dict[str, Any] | None = None,
    ) -> AuditEvent:
        with self._lock:
            return self._audit.record(
                case_id=case_id,
                case_version=case_version,
                kind=kind,
                node=node,
                at=at,
                actor=actor,
                detail=detail,
            )

    @property
    def events(self) -> tuple[AuditEvent, ...]:
        with self._lock:
            return self._audit.events


# ------------------------------------------------------------------------------------ the corpus


class CorpusCase(NamedTuple):
    """One claim from the committed corpus, in the shapes the graph's state already accepts.

    The three payloads are the corpus's own JSON objects, handed to `state.initial_state` unchanged.
    `graph.state`'s rebuild functions read exactly this shape — amounts as `{"amount", "currency"}`
    strings, dates as ISO — so there is nothing to convert and therefore nothing that could convert
    it differently from the way the rest of the system does.
    """

    claim_id: str
    program_id: str
    claim: dict[str, Any]
    program: dict[str, Any]
    coverage: dict[str, Any]
    evidence: dict[str, str]
    as_of: date
    construction: str


def _payload(path: Path, key: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise EvidenceError(
            f"{path} is missing. The corpus is rebuilt from a committed seed rather than kept in "
            f"git; run `make corpus` before running the evidence."
        )
    entries = json.loads(path.read_text(encoding="utf-8"))[key]
    return [entry for entry in entries if isinstance(entry, dict)]


def requirement_evidence(carried: Mapping[str, str], code: RejectionCode) -> dict[str, str]:
    """Translate a claim's document bundle into the vocabulary the requirement matrix asks in.

    The two vocabularies are different on purpose and `evaluation.pipeline` argues it at length: the
    matrix names what a **rejection code demands**, the corpus names what **artefacts a claim
    carries**, and renaming either to match the other would make one of them a description of the
    other. `EVIDENCE_JOIN` is the single place they meet, and in a deployment this function is the
    intake adapter between a distributor's document store and the matrix.

    A requirement is present only when every artefact behind it is present, and one contradictory
    artefact makes the whole requirement contradictory. That asymmetry is the safe direction:
    listing more artefacts against a requirement can only make it harder to satisfy, and the failure
    being avoided is a claim whose withheld document no requirement consults reaching the gate with
    everything apparently in order.
    """
    bundle: dict[str, str] = {}
    for spec in required_for(code):
        states = {carried[artefact] for artefact in EVIDENCE_JOIN[spec.evidence_key]}
        if EvidenceStatus.CONFLICTING.value in states:
            bundle[spec.evidence_key] = EvidenceStatus.CONFLICTING.value
        elif states == {EvidenceStatus.PRESENT.value}:
            bundle[spec.evidence_key] = EvidenceStatus.PRESENT.value
        else:
            bundle[spec.evidence_key] = EvidenceStatus.ABSENT.value
    return bundle


def load_corpus(corpus_dir: Path) -> tuple[CorpusCase, ...]:
    """Every development-split claim, in corpus order, with its programme and coverage attached.

    Hold-out programmes are excluded here rather than filtered later. ADR-001 §7 forbids tuning
    against the hold-out after it has been scored, and the cheapest way to honour that is for this
    run to have no access to it: a case list that cannot contain a hold-out programme cannot be
    quietly widened to include one when a number comes out wrong.
    """
    programs = {
        str(record["program"]["program_id"]): dict(record["program"])
        for record in _payload(corpus_dir / "programs.json", "programs")
    }
    coverage: dict[tuple[str, str], dict[str, Any]] = {}
    for record in _payload(corpus_dir / "coverage.json", "coverage"):
        row = dict(record["coverage"])
        coverage[(str(record["program_id"]), str(row["part_number"]))] = row

    cases: list[CorpusCase] = []
    for record in _payload(corpus_dir / "claims.json", "claims"):
        claim = dict(record["claim"])
        program_id = str(claim["program_id"])
        if is_holdout(program_id):
            continue
        cases.append(
            CorpusCase(
                claim_id=str(claim["claim_id"]),
                program_id=program_id,
                claim=claim,
                program=programs[program_id],
                coverage=coverage[(program_id, str(claim["part_number"]))],
                evidence=requirement_evidence(
                    {str(k): str(v) for k, v in record["evidence"].items()},
                    RejectionCode(str(claim["rejection_code"])),
                ),
                as_of=date.fromisoformat(str(record["as_of"])),
                construction=str(record["construction"]),
            )
        )
    return tuple(cases)


# -------------------------------------------------------------------------------- the kill plan


class KillPoint(NamedTuple):
    """A node, and how many of its tool calls had completed when the process died.

    The pair is the whole point. "Killed at `retrieval`" is ambiguous between a case that had
    searched and one that had not, and kill condition B is a statement about the difference: the
    first must not search again, the second must.
    """

    node: CaseState
    after_tool_calls: int

    @property
    def node_name(self) -> str:
        return _NODE_OF[self.node]

    @property
    def label(self) -> str:
        return f"{self.node_name}+{self.after_tool_calls}"


#: One kill per node of the pipeline, in pipeline order. `_check_plan` asserts the coverage rather
#: than trusting this list to stay complete, because a node added to `NODE_SEQUENCE` and not to this
#: table would silently stop being evidence for anything.
#:
#: Two entries are the ones ADR-001 §4 is about. `SUBMITTED` after one completed tool call is a
#: worker killed with the correction already filed and nothing written down about it; the resume
#: must replay the filing from the ledger and must not call the portal again. `AWAITING_APPROVAL`
#: after one completed call is a worker killed between a person deciding and the case moving on; the
#: resume must find the decision and must not ask them a second time.
KILL_PLAN: Final[tuple[KillPoint, ...]] = (
    KillPoint(CaseState.INTAKE, 0),
    KillPoint(CaseState.REQUIREMENTS, 0),
    KillPoint(CaseState.DEADLINE, 0),
    KillPoint(CaseState.ELIGIBILITY, 0),
    KillPoint(CaseState.RETRIEVAL, 1),
    KillPoint(CaseState.RETRIEVAL, 2),
    KillPoint(CaseState.COMPOSE, 0),
    KillPoint(CaseState.GATE, 0),
    KillPoint(CaseState.AWAITING_APPROVAL, 1),
    KillPoint(CaseState.SUBMITTED, 0),
    KillPoint(CaseState.SUBMITTED, 1),
)

#: The refused case's kill. Separate from `KILL_PLAN` because it needs a claim the gate turns down
#: and the other eleven need claims it authorises, and a single list would hide that the two
#: populations are different.
WRITE_OFF_KILL: Final = KillPoint(CaseState.WRITTEN_OFF, 0)


def _check_plan() -> None:
    """Refuse a plan that leaves a node unkilled, before anything has been run.

    Checked here rather than asserted in the artifact, so that a pipeline which grew a node fails
    at the start of a five-minute run rather than at the end of it with a file already written.
    """
    covered = {point.node_name for point in (*KILL_PLAN, WRITE_OFF_KILL)}
    absent = [name for name in NODE_NAMES if name not in covered]
    if absent:
        raise EvidenceError(
            f"the kill plan never kills {absent}. A node that is never interrupted contributes "
            f"nothing to kill condition A, and the artifact would report coverage it does not have."
        )


# -------------------------------------------------------------------------------- the child run


def _approval_answer(case_version: int, token: str) -> dict[str, Any]:
    return {
        "decision": "APPROVED",
        "actor": APPROVER,
        "token": token,
        "case_version": case_version,
        "note": APPROVAL_NOTE,
    }


def _kill_switch(point: KillPoint) -> KillSwitch:
    """A switch that abandons the process at the named node, after the named number of tool calls.

    `os._exit` and not `sys.exit`. `sys.exit` raises `SystemExit`, which LangGraph's task runner and
    `store.engine.session_scope` both handle — the transaction rolls back, the pool is returned, the
    checkpointer's connection closes cleanly — and a graph that only survived *that* would be a
    graph that had never been killed. `os._exit` runs no handler and commits nothing, which is what
    a container being terminated does to a worker.
    """

    def switch(node: CaseState, completed_tool_calls: int) -> None:
        if node is point.node and completed_tool_calls == point.after_tool_calls:
            os._exit(KILLED_EXIT)

    return switch


def build_dependencies(
    settings: Settings, *, kill_switch: KillSwitch | None
) -> tuple[CaseDependencies, ExecutionCounter, Engine]:
    """Everything a worker reaches outside itself, built the same way in the child and the parent.

    One function for both, because the claim being made is that the two are interchangeable. If the
    parent's dependencies were assembled differently from the child's — a different ledger, a
    different guard namespace, a different retriever — then "a different process resumed it" would
    be true and would not mean what it says.
    """
    engine = build_engine(settings.database_url)
    counter = ExecutionCounter(settings.redis_url)
    ledger = CountingLedger(engine, counter)
    ledger.setup()
    deps = CaseDependencies(
        ledger=ledger,
        evidence=PgVectorEvidence(engine),
        guard=SubmissionGuard(settings.redis_url, namespace=GUARD_NAMESPACE),
        portal=ManufacturerPortal(),
        approver_token=settings.approver_token,
        kill_switch=kill_switch,
    )
    return deps, counter, engine


def _run_child(job: Mapping[str, Any]) -> int:
    """Carry one case until the kill switch fires. Returning at all means it did not.

    A child that completes its case has not produced evidence: the parent has nothing to resume and
    the artifact would describe a case that was never interrupted. It returns a distinct status, the
    parent refuses it, and the run stops.
    """
    settings = get_settings()
    point = KillPoint(CaseState(job["kill_node"]), int(job["kill_after"]))
    deps, counter, _ = build_dependencies(settings, kill_switch=_kill_switch(point))
    config = case_config(str(job["case_id"]))
    answer = job.get("answer")
    try:
        with checkpointer(settings.database_url, setup=True) as saver:
            machine = build_machine(deps, saver)
            result = machine.invoke(job["state"], config)
            if "__interrupt__" in result and answer is not None:
                machine.invoke(Command(resume=answer), config)
    finally:
        counter.close()
    return 0


def _spawn(job: Mapping[str, Any], scratch: Path) -> int:
    """Run one case in a child process and report how it died.

    The job crosses as a file rather than as an argument list because it carries the case's whole
    initial state, and a Windows command line has a length limit that a warranty policy's clause
    text would reach. The file is written by the parent and read once.
    """
    scratch.mkdir(parents=True, exist_ok=True)
    job_path = scratch / f"{job['case_id']}.json"
    job_path.write_text(json.dumps(job, sort_keys=True), encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker", str(job_path)],
        cwd=str(REPO_ROOT),
        check=False,
    )
    return completed.returncode


# ------------------------------------------------------------------------------ the parent's run


class CaseEvidence(NamedTuple):
    """What one killed case proved, in the terms A, B and C are graded in."""

    case_id: str
    killed_at: str
    recorded_next: tuple[str, ...]
    resumed_at: str | None
    diverged: bool
    completed_before_kill: tuple[str, ...]
    reexecuted: tuple[str, ...]
    invocations_after_resume: int
    decision_before_kill: bool
    decision_lost: bool
    asked_the_person_again: bool


def _decision_invocation(case_id: str) -> str:
    return ToolLedger.invocation_id(case_id, CaseState.AWAITING_APPROVAL, TOOL_RECORD_DECISION, 0)


def _history(snapshot: Any) -> tuple[str, ...]:
    values = getattr(snapshot, "values", None) or {}
    return tuple(str(name) for name in values.get("node_history") or ())


def _decision_survived(recorded: Mapping[str, Any], values: Mapping[str, Any]) -> bool:
    """Whether the decision the ledger holds is the decision the resumed case ended up with.

    Three things have to agree, and the third is the one that matters. The state's approval record
    must exist, it must say what the person said, and the audit stream must carry the approval event
    — because the state is the graph's own account of itself and the audit is what
    `verify_submissions` reads. A resumed case that carried the decision in its state and never
    wrote the event would satisfy kill condition C and fail kill condition D, and the two failures
    would look unrelated.
    """
    approval = values.get("approval")
    if not isinstance(approval, Mapping):
        return False
    if bool(approval.get("granted")) != bool(recorded.get("granted")):
        return False
    if str(approval.get("actor")) != str(recorded.get("actor")):
        return False
    kinds = {str(event.get("kind")) for event in values.get("audit") or ()}
    return bool(kinds & {APPROVAL_GRANTED, APPROVAL_REFUSED})


def kill_and_resume(
    machine: Any,
    ledger: ToolLedger,
    counter: ExecutionCounter,
    *,
    case: CorpusCase,
    case_id: str,
    point: KillPoint,
    approve: bool,
    scratch: Path,
    token: str,
    finish: bool,
) -> CaseEvidence:
    """Kill one case in a child process, then resume it here, and record what changed.

    `finish` is false for the cases staged for the concurrency race: they are carried to the submit
    node, killed before the filing, and deliberately left there, because what the race needs is a
    real approved case with nothing filed against it yet.
    """
    answer = _approval_answer(1, token) if approve else None
    job = {
        "case_id": case_id,
        "kill_node": point.node.value,
        "kill_after": point.after_tool_calls,
        "answer": answer,
        "state": _initial(case, case_id),
    }
    status = _spawn(job, scratch)
    if status != KILLED_EXIT:
        raise EvidenceError(
            f"{case_id} was to be killed at {point.label} and its worker exited {status} instead. "
            f"Either the case never reached that node or it finished, and a case that was not "
            f"interrupted is evidence of nothing."
        )

    config = case_config(case_id)
    snapshot = machine.get_state(config)
    recorded_next = pending_nodes(snapshot)
    history_before = _history(snapshot)
    completed = tuple(
        invocation.invocation_id
        for invocation in ledger.invocations_for(case_id)
        if invocation.completed
    )
    counts_before = {invocation_id: counter.count(invocation_id) for invocation_id in completed}
    recorded_decision = ledger.completed_result(_decision_invocation(case_id))

    if finish:
        result = machine.invoke(None, config)
        asked_again = "__interrupt__" in result
        if asked_again and answer is not None:
            machine.invoke(Command(resume=answer), config)
    else:
        result = {}
        asked_again = False

    final = machine.get_state(config)
    history_after = _history(final)
    resumed_at = (
        history_after[len(history_before)] if len(history_after) > len(history_before) else None
    )
    values: Mapping[str, Any] = getattr(final, "values", None) or {}

    decision_lost = recorded_decision is not None and (
        (asked_again and finish) or (finish and not _decision_survived(recorded_decision, values))
    )
    return CaseEvidence(
        case_id=case_id,
        killed_at=point.label,
        recorded_next=recorded_next,
        resumed_at=resumed_at,
        diverged=bool(recorded_next) and resumed_at != recorded_next[0],
        completed_before_kill=completed,
        reexecuted=tuple(
            invocation_id
            for invocation_id in completed
            if counter.count(invocation_id) > counts_before[invocation_id]
        ),
        invocations_after_resume=len(ledger.invocations_for(case_id)),
        decision_before_kill=recorded_decision is not None,
        decision_lost=decision_lost,
        asked_the_person_again=asked_again and recorded_decision is not None,
    )


def _initial(case: CorpusCase, case_id: str) -> GraphState:
    return initial_state(
        case_id=case_id,
        claim=case.claim,
        program=case.program,
        coverage=case.coverage,
        evidence=case.evidence,
        as_of=case.as_of,
    )


# ------------------------------------------------------------------------------------- the race


class RaceEvidence(NamedTuple):
    """One identity, twenty-four callers, and what the portal was actually asked to do."""

    recovery_identity: str
    winners: int
    duplicates: int
    refusals: int
    portal_calls: int
    events: tuple[AuditEvent, ...]


def race_one_identity(
    machine: Any, *, case_id: str, portal: ManufacturerPortal, guard: SubmissionGuard
) -> RaceEvidence:
    """Twenty-four threads enter `submit` together for one recovery identity.

    Everything they are given comes out of the checkpoint the machine wrote: the claim, the gate's
    decision, the correction window, the composed narrative and the audit stream that holds the
    person's approval. Nothing is assembled for the race, so what is being raced is the system.

    `threading.Barrier` rather than starting the threads and hoping. Twenty-four threads started in
    a loop reach `claim_once` several milliseconds apart, and the first one is usually finished
    before the last one has been created — which measures a sequence and calls it a race. The
    barrier makes them contend.
    """
    values: Mapping[str, Any] = machine.get_state(case_config(case_id)).values
    claim = claim_from_json(values["claim"])
    decision = decision_from_json(values["gate"])
    window = window_from_json(values["window"])
    proposal = values["proposal"]
    case_version = int(values["case_version"])
    at = date.fromisoformat(str(values["as_of"]))

    sink = LockedAudit(values["audit"] or ())
    approvals = sink.events
    barrier = threading.Barrier(RACE_THREADS)
    outcomes: list[SubmissionOutcome] = []
    lock = threading.Lock()

    def attempt() -> None:
        barrier.wait()
        outcome = submit(
            claim=claim,
            decision=decision,
            window=window,
            case_id=case_id,
            case_version=case_version,
            approvals=approvals,
            guard=guard,
            portal=portal,
            filed_on=at,
            narrative=str(proposal["narrative"]),
            clause_ids=[str(item["clause_id"]) for item in proposal["citations"]],
        )
        with lock:
            outcomes.append(outcome)
        record_outcome(
            sink, outcome, case_id=case_id, case_version=case_version, at=at, actor=SYSTEM_ACTOR
        )

    threads = [
        threading.Thread(target=attempt, name=f"racer-{index}") for index in range(RACE_THREADS)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    identity = claim.recovery_identity
    return RaceEvidence(
        recovery_identity=identity,
        winners=sum(1 for outcome in outcomes if outcome.accepted),
        duplicates=sum(1 for outcome in outcomes if outcome.duplicate),
        refusals=sum(1 for outcome in outcomes if outcome.refusal is not None),
        portal_calls=len(portal.filings_for(identity)),
        events=sink.events,
    )


# ------------------------------------------------------------------------------------- the reset


def reset(settings: Settings, case_ids: Sequence[str], identities: Sequence[str]) -> None:
    """Clear this run's own rows, and only its own.

    The evidence run owns the case identifiers it uses, so it may delete them; it names each one and
    touches nothing else, because a sweep by prefix over a checkpoint table is how somebody's
    in-flight case gets deleted by a build. Without this a second run would find the first run's
    submission claims still held in Redis, every filing would be suppressed as a duplicate, and the
    artifact would report the guard working perfectly against a race that never happened.
    """
    engine = build_engine(settings.database_url)
    CountingLedger(engine, ExecutionCounter(settings.redis_url)).setup()
    with engine.begin() as connection:
        for case_id in case_ids:
            connection.execute(sql_text(_DELETE_INVOCATIONS), {"case_id": case_id})

    with checkpointer(settings.database_url, setup=True) as saver:
        for case_id in case_ids:
            saver.delete_thread(case_id)

    guard = SubmissionGuard(settings.redis_url, namespace=GUARD_NAMESPACE)
    client: Any = Redis.from_url(settings.redis_url, decode_responses=True)
    keys = [guard.key_for(identity) for identity in identities]
    if keys:
        client.delete(*keys)
    client.close()
    guard.close()
    counter = ExecutionCounter(settings.redis_url)
    counter.forget()
    counter.close()


# -------------------------------------------------------------------------------- the artifacts


def _synthetic_notice(subject: str) -> dict[str, Any]:
    return {
        "is_synthetic_corpus": True,
        "notice": (
            f"{subject} measured over the committed synthetic corpus. The claims, policies and "
            f"manufacturers do not exist and the manufacturer portal is a local mock; ADR-001 §8 "
            f"records that the idempotency guarantee is about this system's own effects."
        ),
    }


def durability_artifact(evidence: Sequence[CaseEvidence]) -> dict[str, Any]:
    """The five numbers kill conditions A, B and C read, and the populations behind them."""
    return {
        **_synthetic_notice("Durability"),
        "cases_killed": len(evidence),
        "node_count": len(NODE_NAMES),
        "nodes_killed": sorted({item.killed_at.split("+")[0] for item in evidence}),
        "kill_points": [item.killed_at for item in evidence],
        "kill_exit_status": KILLED_EXIT,
        "tool_invocations_observed": sum(item.invocations_after_resume for item in evidence),
        "tool_invocations_completed_before_kill": sum(
            len(item.completed_before_kill) for item in evidence
        ),
        "human_decisions_before_kill": sum(1 for item in evidence if item.decision_before_kill),
        "node_divergences": sum(1 for item in evidence if item.diverged),
        "tool_reexecutions": sum(len(item.reexecuted) for item in evidence),
        "human_decisions_lost": sum(1 for item in evidence if item.decision_lost),
        "people_asked_twice": sum(1 for item in evidence if item.asked_the_person_again),
        "resumed_at": {item.killed_at: item.resumed_at for item in evidence},
        "recorded_next": {item.killed_at: list(item.recorded_next) for item in evidence},
    }


def submission_artifact(
    events: Sequence[AuditEvent], races: Sequence[RaceEvidence]
) -> dict[str, Any]:
    """Kill conditions D and E, with every approval and duplicate finding taken from the verifier.

    `verify_submissions` is handed the whole event stream and answers on its own terms. This
    function adds only what the verifier cannot see — how many threads raced, how many identities,
    and how many times the portal was actually called — and asserts nothing about approvals itself.
    """
    audited = verify_submissions(events)
    return {
        **_synthetic_notice("Submission control"),
        "submissions_observed": audited.submissions_observed,
        "approvals_observed": audited.approvals_observed,
        "submissions_without_approval": audited.submissions_without_approval,
        "submissions_with_stale_version_approval": (
            audited.submissions_with_stale_version_approval
        ),
        "duplicate_submissions": audited.duplicate_submissions,
        "identities_raced": len(races),
        "concurrent_attempts_per_identity": RACE_THREADS,
        "max_effects_for_one_identity": max((race.portal_calls for race in races), default=0),
        "offending_cases": list(audited.offending_cases),
        "race_winners": sum(race.winners for race in races),
        "race_callers_given_the_winners_effect": sum(race.duplicates for race in races),
        "race_callers_refused": sum(race.refusals for race in races),
        "submissions_suppressed": sum(1 for event in events if event.kind == SUBMISSION_SUPPRESSED),
        "events_examined": len(events),
        "graded_by": "warranty_claim_recovery.audit.verify_submissions",
    }


# --------------------------------------------------------------------------------------- the run


def _require_token(settings: Settings) -> str:
    if not settings.approver_token:
        raise EvidenceError(
            f"{APPROVER_TOKEN_ENV} is not set. It has no default so that a deployment without one "
            f"can approve nothing, and an evidence run that approved without it would be measuring "
            f"a gate that was not there. Set it and run again."
        )
    return settings.approver_token


Selection = tuple[tuple[CorpusCase, ...], CorpusCase, tuple[CorpusCase, ...]]


def _select(cases: Sequence[CorpusCase]) -> Selection:
    """The claims this run uses: eleven the gate authorises, one it refuses, three for the race.

    Taken in corpus order from the constructions whose ground truth the generator fixed, so the
    selection is a property of the committed corpus rather than of this script. `CLEAN` claims are
    the ones built to be recoverable and `WINDOW_CLOSED` is built to be refused on the first gate
    rule, which is the outcome that routes a case to `write_off` — the one node no authorised case
    ever reaches.
    """
    clean = tuple(case for case in cases if case.construction == "CLEAN")
    refused = tuple(case for case in cases if case.construction == "WINDOW_CLOSED")
    needed = len(KILL_PLAN) + RACE_IDENTITIES
    if len(clean) < needed or not refused:
        raise EvidenceError(
            f"the corpus offers {len(clean)} clean and {len(refused)} window-closed development "
            f"claims; this run needs {needed} and one. Regenerate the corpus rather than shrinking "
            f"the plan: a kill plan trimmed to fit its inputs stops covering the nodes it names."
        )
    return clean[: len(KILL_PLAN)], refused[0], clean[len(KILL_PLAN) : needed]


def run(corpus_dir: Path, artifacts_dir: Path, scratch: Path) -> dict[str, dict[str, Any]]:
    _check_plan()
    settings = get_settings()
    token = _require_token(settings)
    cases = load_corpus(corpus_dir)
    authorised, refused, raced = _select(cases)

    plan: list[tuple[CorpusCase, KillPoint, bool, bool]] = [
        (case, point, True, True) for case, point in zip(authorised, KILL_PLAN, strict=True)
    ]
    plan.append((refused, WRITE_OFF_KILL, False, True))
    plan.extend((case, KillPoint(CaseState.SUBMITTED, 0), True, False) for case in raced)

    case_ids = [f"dur-{case.claim_id}" for case, _, _, _ in plan]
    identities = [claim_from_json(case.claim).recovery_identity for case, _, _, _ in plan]
    reset(settings, case_ids, identities)

    deps, counter, _ = build_dependencies(settings, kill_switch=None)
    ledger = deps.ledger
    killed: list[CaseEvidence] = []
    with checkpointer(settings.database_url, setup=True) as saver:
        machine = build_machine(deps, saver)
        for case_id, (case, point, approve, finish) in zip(case_ids, plan, strict=True):
            print(f"  {case_id}: killing at {point.label}")
            killed.append(
                kill_and_resume(
                    machine,
                    ledger,
                    counter,
                    case=case,
                    case_id=case_id,
                    point=point,
                    approve=approve,
                    scratch=scratch,
                    token=token,
                    finish=finish,
                )
            )

        graded = [item for item in killed if item.case_id not in set(case_ids[-RACE_IDENTITIES:])]
        portal = ManufacturerPortal()
        guard = SubmissionGuard(settings.redis_url, namespace=GUARD_NAMESPACE)
        races = [
            race_one_identity(machine, case_id=case_id, portal=portal, guard=guard)
            for case_id in case_ids[-RACE_IDENTITIES:]
        ]

        events: list[AuditEvent] = []
        for case_id in case_ids[:-RACE_IDENTITIES]:
            values: Mapping[str, Any] = machine.get_state(case_config(case_id)).values
            events.extend(event_from_json(item) for item in values.get("audit") or ())
    for race in races:
        events.extend(race.events)

    counter.close()
    guard.close()

    artifacts = {
        "durability.json": durability_artifact(graded),
        "submission.json": submission_artifact(events, races),
    }
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in artifacts.items():
        (artifacts_dir / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return artifacts


def _report(artifacts: Mapping[str, Mapping[str, Any]]) -> None:
    durability = artifacts["durability.json"]
    submission = artifacts["submission.json"]
    print()
    print("artifacts/durability.json")
    print(
        f"  cases killed                {durability['cases_killed']} over "
        f"{durability['node_count']} nodes"
    )
    print(
        f"  tool invocations observed   {durability['tool_invocations_observed']} "
        f"({durability['tool_invocations_completed_before_kill']} completed before a kill)"
    )
    print(f"  human decisions before kill {durability['human_decisions_before_kill']}")
    print(f"  node divergences            {durability['node_divergences']}")
    print(f"  tool re-executions          {durability['tool_reexecutions']}")
    print(f"  human decisions lost        {durability['human_decisions_lost']}")
    print()
    print("artifacts/submission.json")
    print(f"  submissions observed        {submission['submissions_observed']}")
    print(f"  approvals observed          {submission['approvals_observed']}")
    print(f"  without approval            {submission['submissions_without_approval']}")
    print(f"  stale-version approval      {submission['submissions_with_stale_version_approval']}")
    print(
        f"  identities raced            {submission['identities_raced']} x "
        f"{submission['concurrent_attempts_per_identity']} threads"
    )
    print(f"  effects for one identity    {submission['max_effects_for_one_identity']}")
    print(f"  duplicate submissions       {submission['duplicate_submissions']}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--worker",
        type=Path,
        default=None,
        help="internal: carry one case in this process until its kill switch fires",
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument(
        "--scratch",
        type=Path,
        default=None,
        help="where the per-case job files are written; a temporary directory by default",
    )
    args = parser.parse_args(argv)

    if args.worker is not None:
        job = json.loads(args.worker.read_text(encoding="utf-8"))
        return _run_child(job)

    # A temporary directory rather than a folder in the repository. The job files are a transport
    # detail between the parent and its children, they carry a case's whole initial state, and a
    # build that left fifteen of them beside the source is a build that will eventually commit one.
    with tempfile.TemporaryDirectory(prefix="wcr-evidence-") as temporary:
        scratch = args.scratch if args.scratch is not None else Path(temporary)
        try:
            artifacts = run(args.corpus, args.artifacts, scratch)
        except EvidenceError as error:
            print(str(error), file=sys.stderr)
            return 1
    _report(artifacts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
