"""The correction text, assembled from cited spans — and the model arm that refuses to pretend.

ADR-001 §3 records the decision this module implements: **the shipped composer is extractive, and
the abstractive arm is a port whose `propose` raises.** No model credential exists in this
environment, and a stub, an echo or a canned string would enter the evaluation as a number
attributed to a model that was never called. That is the defect the ADR forbids by name, and the
only honest way to build the arm is to build everything except the call — the typed request, the
permission check, the exact payload, the post-validation — and let the call itself fail loudly.

### Why the grounding property is string arithmetic and not a measured rate

A generated correction is usually scored for faithfulness: sample it, judge it, publish a
percentage. That is the right instrument for a system whose text is generated, and it is the wrong
one here, because the shipped composer does not generate. `ExtractiveComposer` assembles the
narrative from the citations it was given, so the set of clause identifiers the narrative mentions
is a **subset of the set it cites by construction** — a property of concatenation rather than a
property that held on the sample somebody checked. `grounding_report` therefore counts, it does not
estimate, and `validated` refuses the proposal outright when the count is not zero.

The check is run against the whole **retrieved** set rather than against the cited set. Checking the
cited identifiers against themselves is checking a set against itself and passes for everything. The
failure worth catching is a narrative that names a clause the case saw and the gate did not approve
as evidence — the retrieved-but-uncited distractor, which in this corpus includes clauses of
withdrawn service bulletins. That is a real mistake a composer can make, and it is the one this
check can actually fail on.

### Why the composer is only ever asked about an authorised case

`correction_request` raises for a `GateDecision` that does not authorise. A refusal has no
correction to file: the thing a handler needs is the gate's reason and the list of what is missing,
both of which are already on the decision. Composing prose for a refusal would produce a document
that reads like a resubmission for a claim that is not going anywhere, and the failure mode is
somebody filing it. The rejected alternative — compose for every case and let the caller decide
whether to use it — puts the decision about whether money leaves the system in the caller, which is
where `CLAUDE.md` §3 says it may not be.

### Why the request carries the gate's own citations rather than the retriever's

`correction_request` reads the citations off `decision.requirements` and takes none from its caller.
`domain.Requirement` refuses to be `SATISFIED` without one and `domain.GateDecision` refuses to
authorise with an unsatisfied requirement, so the evidence in the request is exactly the evidence
the gate approved, and there is no parameter through which a fourth clause could be added on the way
to the model. A citation the composer could be handed separately is a citation nothing checked.

### The disclosure

EU AI Act Article 50 asks that a person be told the text was produced by an AI system. The sentence
lives on `domain.CorrectionProposal` as a field default — a property of the response, not of the
page it is drawn on — and `validated` checks that it survived to the boundary rather than trusting
that nobody overwrote it. The expected text is read from the model's own field default rather than
retyped here, so the claim and the implementation are the same string.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Any, Final, NamedTuple, Protocol

from warranty_claim_recovery.domain import (
    Citation,
    Claim,
    ClaimWindow,
    CorrectionProposal,
    GateDecision,
    RecoveryOutcome,
    RejectionCode,
    Requirement,
    RequirementStatus,
    WarrantyProgram,
)
from warranty_claim_recovery.money import Money

__all__ = [
    "ARTICLE_50_DISCLOSURE",
    "COMPOSER_ABSTRACTIVE",
    "COMPOSER_EXTRACTIVE",
    "PROMPT_PAYLOAD_KEYS",
    "AbstractiveComposer",
    "Composer",
    "CorrectionRequest",
    "DisclosureMissingError",
    "ExtractiveComposer",
    "GroundingReport",
    "LiveModelUnavailableError",
    "UngroundedCorrectionError",
    "correction_request",
    "grounding_report",
    "mentioned_clause_ids",
    "validated",
]

#: Read from the model rather than retyped. `CLAUDE.md` §4 requires that a constant which appears in
#: an artifact be imported from the code that implements it; a disclosure asserted in one file and
#: defaulted in another is a disclosure that will one day be checked against the wrong sentence.
ARTICLE_50_DISCLOSURE: Final[str] = str(CorrectionProposal.model_fields["ai_disclosure"].default)

#: The value carried in `CorrectionProposal.composer`. Versioned because a change to the assembly is
#: a change to every proposal produced afterwards, and an audit record that says only "extractive"
#: cannot tell two of them apart.
COMPOSER_EXTRACTIVE: Final = "extractive-v1"
COMPOSER_ABSTRACTIVE: Final = "abstractive-port"

#: Exactly what may be sent to a model, and nothing else. A frozen set rather than a comment,
#: because `tests/test_composer.py` asserts equality against it: a payload that quietly grew a key
#: would otherwise carry whatever the new key held to a third party, and the review that would have
#: caught it is the one nobody runs on a dictionary literal.
PROMPT_PAYLOAD_KEYS: Final[frozenset[str]] = frozenset(
    {
        "claim_id",
        "program_id",
        "policy_version",
        "rejection_code",
        "outcome",
        "claimed_total",
        "recoverable_amount",
        "currency",
        "window_closes_on",
        "days_remaining",
        "approved_evidence",
        "instruction",
        "disclosure",
    }
)

#: The one instruction the port would send. Fixed here rather than assembled per case, because a
#: prompt that varies with the case is a tuning knob, and ADR-001 §7 forbids tuning after the
#: hold-out has been scored. Fixing it now makes a better phrasing a new experiment rather than an
#: edit nobody records.
_INSTRUCTION: Final = (
    "Rewrite the approved evidence below as a correction letter to the manufacturer. Use only the "
    "quoted passages supplied. Do not name a clause, a document or an amount that does not appear "
    "here, and do not assert that any requirement is met beyond those listed."
)


class LiveModelUnavailableError(RuntimeError):
    """The abstractive arm was asked to compose and no model credential exists.

    Raised rather than substituted. A stub that returned the extractive narrative would be scored as
    a model output, published as a model's groundedness figure, and attributed to a system that was
    never called — the exact defect ADR-001 §3 records as the reason this arm is a port. The error
    names what would make it work, so the absence is a configuration fact rather than a bug.
    """


class UngroundedCorrectionError(ValueError):
    """A correction named a clause it does not cite.

    This is the composer having produced an assertion instead of a proposal. It is refused at the
    boundary rather than counted, because the document in question would go to a manufacturer's
    adjudicator who would look the clause up, find that it was never cited as evidence, and refuse
    the package — after the correction window had spent another day.
    """


class DisclosureMissingError(ValueError):
    """A proposal reached the boundary without the Article 50 disclosure it was constructed with.

    Distinct from `UngroundedCorrectionError` because the two have different readers: an ungrounded
    citation is an engineering defect, and a stripped disclosure is a compliance one. Collapsing
    them into one type would send both to whoever happens to catch it first.
    """


class CorrectionRequest(NamedTuple):
    """Everything a composer is allowed to see, and nothing the gate did not approve.

    A `NamedTuple` rather than a Pydantic model: it is built in exactly one place, by
    `correction_request`, from values that have already been validated by the models they came off,
    and a second validation pass would re-check `domain`'s work rather than check anything new.
    """

    claim_id: str
    program_id: str
    policy_version: str
    rejection_code: RejectionCode
    outcome: RecoveryOutcome
    claimed_total: Money
    recoverable_amount: Money
    closes_on: date
    days_remaining: int
    #: In the requirement matrix's order, every one of them `SATISFIED` and every one cited.
    requirements: tuple[Requirement, ...]

    @property
    def citations(self) -> tuple[Citation, ...]:
        """The cited spans, deduplicated, in requirement order.

        Deduplicated because one clause commonly satisfies two requirements of the same rejection
        code — a single paragraph states both what a serial must show and where it must come from —
        and a proposal that cited it twice would read as two separate authorities. The order is the
        matrix's rather than sorted, because that is the order a technician was asked to work in and
        the order the correction should read in.
        """
        seen: dict[str, Citation] = {}
        for requirement in self.requirements:
            citation = requirement.citation
            if citation is not None and citation.clause_id not in seen:
                seen[citation.clause_id] = citation
        return tuple(seen.values())


class Composer(Protocol):
    """What the graph's compose node depends on.

    A protocol rather than a base class so that the extractive arm, the abstractive port and a test
    double that deliberately misbehaves are all the same shape without inheriting anything. The
    misbehaving double matters: `CLAUDE.md` §3 rule 6 requires a guard to be shown failing from
    behaviour, and the only way to plant an ungrounded narrative is to substitute a composer the
    running system actually calls.
    """

    @property
    def name(self) -> str: ...

    def propose(self, request: CorrectionRequest) -> CorrectionProposal: ...


def correction_request(
    *,
    claim: Claim,
    program: WarrantyProgram,
    window: ClaimWindow,
    decision: GateDecision,
) -> CorrectionRequest:
    """The approved evidence bundle for a case the gate has authorised.

    Refuses a decision that does not authorise, and refuses one whose requirements are not all
    satisfied and cited. The second check duplicates a validator on `GateDecision`, and the
    duplication is deliberate: that validator runs at construction, this function can be handed a
    decision built by a test or read back from a checkpoint written by an older version, and the
    thing being protected — that no uncited text reaches a composer — is worth one comprehension.
    """
    if not decision.authorises_recovery:
        raise ValueError(
            f"{claim.claim_id} was gated {decision.outcome.value} and has no correction to file. "
            f"The handler needs the gate's reason and the requirements it refused over, both of "
            f"which are on the decision; prose written for a refusal is a document somebody files."
        )
    unsupported = [
        requirement.requirement_id
        for requirement in decision.requirements
        if requirement.status is not RequirementStatus.SATISFIED or requirement.citation is None
    ]
    if unsupported:
        raise ValueError(
            f"{claim.claim_id} authorises a recovery with {unsupported} unsatisfied or uncited. A "
            f"composer handed one of those would write a sentence nothing supports."
        )
    return CorrectionRequest(
        claim_id=claim.claim_id,
        program_id=program.program_id,
        policy_version=program.policy_version,
        rejection_code=claim.rejection_code,
        outcome=decision.outcome,
        claimed_total=decision.computation.claimed_total,
        recoverable_amount=decision.computation.recoverable_amount,
        closes_on=window.closes_on,
        days_remaining=window.days_remaining,
        requirements=decision.requirements,
    )


class ExtractiveComposer:
    """The shipped arm. Assembles the correction from the cited spans and writes nothing else.

    Every sentence is a fixed template with values interpolated from the request, and the only free
    text is the quoted passage, which is the clause verbatim. That is what makes the grounding
    property arithmetic: the narrative cannot mention a clause identifier the request did not carry,
    because there is no path by which one could arrive.
    """

    __slots__ = ()

    @property
    def name(self) -> str:
        return COMPOSER_EXTRACTIVE

    def propose(self, request: CorrectionRequest) -> CorrectionProposal:
        return CorrectionProposal(
            claim_id=request.claim_id,
            narrative=self.narrative(request),
            citations=request.citations,
            composer=self.name,
        )

    def narrative(self, request: CorrectionRequest) -> str:
        """The letter, in the order an adjudicator reads one.

        The amount comes before the evidence and the deadline comes last. An adjudicator opens with
        "what is being asked for", checks the evidence against it, and needs the deadline only to
        decide how quickly to answer; leading with the deadline reads as pressure and is the first
        thing a disputed package is criticised for.
        """
        lines = [
            f"Correction for claim {request.claim_id} under warranty programme "
            f"{request.program_id}, policy version {request.policy_version}.",
            "",
            f"The manufacturer returned this claim with rejection code "
            f"{request.rejection_code.value}. The corrected claim answers that rejection with the "
            f"evidence set out below and asks for {request.recoverable_amount} of "
            f"{request.claimed_total} claimed. The gate assessed this case as "
            f"{request.outcome.value} on the deterministic evidence; no part of that assessment "
            f"was made by a language model.",
            "",
            "Evidence, quoted from the policy version in force:",
        ]
        for requirement in request.requirements:
            citation = requirement.citation
            if citation is None:  # pragma: no cover - refused by `correction_request`
                continue
            lines.extend(
                [
                    "",
                    f"  {requirement.requirement_id} — {requirement.description}",
                    f"  Clause {citation.clause_id}, {citation.section}, of "
                    f"{citation.document_id} at policy version {citation.policy_version}, "
                    f"characters {citation.start_offset} to {citation.end_offset}:",
                    f'    "{citation.quote}"',
                ]
            )
        lines.extend(
            [
                "",
                f"The correction window for this rejection closes on "
                f"{request.closes_on.isoformat()}, {request.days_remaining} day(s) after the date "
                f"this case was assessed.",
            ]
        )
        return "\n".join(lines)


class AbstractiveComposer:
    """The model arm, built to the boundary and stopping there.

    Everything except the call exists: the typed request, the permission check, the exact payload
    and the post-validation the caller is expected to apply. `propose` raises, and
    `prompt_payload` is public so that a reviewer can read precisely what would leave this system
    if a credential were supplied — which is a stronger statement than a paragraph in a README
    promising that it would be safe.

    `api_key` is a constructor argument rather than a read of the environment, because a component
    that reads configuration at the point of use cannot be tested for what it does when the
    configuration is absent without editing the environment of the test process.
    """

    __slots__ = ("_api_key", "_model")

    def __init__(self, *, api_key: str | None = None, model: str = "unconfigured") -> None:
        self._api_key = api_key
        self._model = model

    @property
    def name(self) -> str:
        return COMPOSER_ABSTRACTIVE

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    def propose(self, request: CorrectionRequest) -> CorrectionProposal:
        """Always raises in this deployment. See the module docstring and ADR-001 §3.

        It raises even when an API key *is* present, because no provider client is wired to it and
        returning something would mean returning something invented. The key changes the message, so
        that a deployment which has supplied one is told the remaining work rather than told its
        credential was ignored.
        """
        if self.available:
            raise LiveModelUnavailableError(
                f"{request.claim_id}: a credential is configured for model {self._model!r} and no "
                f"provider client is wired to it. This arm is a port: `prompt_payload` shows "
                f"exactly what would be sent. Wire a client, and record in DECISIONS.md that a "
                f"cost, latency or quality figure may now be published for it."
            )
        raise LiveModelUnavailableError(
            f"{request.claim_id}: no model credential is configured, so this arm cannot compose. "
            f"Returning a stub here would put a number in the evaluation attributed to a model "
            f"that was never called, which ADR-001 §3 forbids by name. The shipped composer is "
            f"{COMPOSER_EXTRACTIVE}."
        )

    def prompt_payload(self, request: CorrectionRequest) -> dict[str, Any]:
        """The exact request that would be sent, with the gate's approved evidence and nothing else.

        Amounts are decimal strings. A float here would be the one place in this system where a
        monetary value crossed a boundary in binary floating point, and it would cross it on its way
        to a third party — the boundary where being wrong is least recoverable.

        The evidence entries carry the requirement, the clause identity and the verbatim quote, and
        carry no `detail` from the requirement. `Requirement.detail` is written for a technician
        chasing a document; sending it would invite a model to repeat an internal instruction back
        to a manufacturer as though it were part of the claim.
        """
        return {
            "claim_id": request.claim_id,
            "program_id": request.program_id,
            "policy_version": request.policy_version,
            "rejection_code": request.rejection_code.value,
            "outcome": request.outcome.value,
            "claimed_total": str(request.claimed_total.quantize().amount),
            "recoverable_amount": str(request.recoverable_amount.quantize().amount),
            "currency": request.recoverable_amount.currency.value,
            "window_closes_on": request.closes_on.isoformat(),
            "days_remaining": request.days_remaining,
            "approved_evidence": [
                {
                    "requirement_id": requirement.requirement_id,
                    "description": requirement.description,
                    "clause_id": citation.clause_id,
                    "document_id": citation.document_id,
                    "section": citation.section,
                    "quote": citation.quote,
                }
                for requirement in request.requirements
                if (citation := requirement.citation) is not None
            ],
            "instruction": _INSTRUCTION,
            "disclosure": ARTICLE_50_DISCLOSURE,
        }


class GroundingReport(NamedTuple):
    """Which clause identifiers a narrative names, which it cites, and the difference.

    `ungrounded` is the graded number and the other two are its denominator and its context. A
    report that published only the count would leave anyone investigating a non-zero to re-derive
    both sets by hand, and ADR-001's vacuity guard exists because a numerator without a denominator
    is not evidence.
    """

    mentioned: tuple[str, ...]
    cited: tuple[str, ...]
    ungrounded: tuple[str, ...]

    @property
    def grounded(self) -> bool:
        return not self.ungrounded


def mentioned_clause_ids(narrative: str, vocabulary: Sequence[str]) -> tuple[str, ...]:
    """Which of the known clause identifiers appear in the text, as substring arithmetic.

    Matched against a closed vocabulary rather than by a pattern. A regular expression for "things
    that look like a clause identifier" is a guess about the corpus's naming, and it fails in both
    directions: it misses an identifier whose shape changes and it invents matches out of ordinary
    prose. The vocabulary is what the retrieval stage actually returned for this case, so every
    match is a real clause the case really saw.

    Sorted, so the report does not depend on the order the retriever happened to rank in.
    """
    return tuple(sorted({clause_id for clause_id in vocabulary if clause_id in narrative}))


def grounding_report(proposal: CorrectionProposal, vocabulary: Sequence[str]) -> GroundingReport:
    """Every clause the narrative names, set against every clause it cites."""
    cited = tuple(sorted({citation.clause_id for citation in proposal.citations}))
    # The cited identifiers are folded into the vocabulary so that a citation to a clause outside
    # the retrieved set is still checked. It cannot happen through `correction_request`, and a
    # vocabulary that silently excluded it would make the check depend on that fact staying true.
    mentioned = mentioned_clause_ids(proposal.narrative, tuple(vocabulary) + cited)
    return GroundingReport(
        mentioned=mentioned,
        cited=cited,
        ungrounded=tuple(name for name in mentioned if name not in set(cited)),
    )


def validated(proposal: CorrectionProposal, *, vocabulary: Sequence[str]) -> CorrectionProposal:
    """Enforce the grounding rule and the Article 50 disclosure on the way out.

    Returns the proposal unchanged or raises. It does not repair: a narrative with an ungrounded
    mention cannot be fixed by deleting the sentence that contains it, because whatever that
    sentence asserted was part of the argument the correction was making, and a document with a hole
    in its reasoning is worse than no document.

    This wrapper is applied to **every** composer, including the extractive one whose output is
    grounded by construction. Checking a property that cannot fail looks like ceremony and is the
    cheap half of a real guarantee: the abstractive arm shares this exit, a substituted composer in
    a test shares it, and the day the assembly changes the check is already in the path.
    """
    report = grounding_report(proposal, vocabulary)
    if not report.grounded:
        raise UngroundedCorrectionError(
            f"{proposal.claim_id}: the correction names {list(report.ungrounded)} and cites "
            f"{list(report.cited)}. A clause the case retrieved and did not cite is a clause the "
            f"gate never approved as evidence, and an adjudicator who looks it up finds an "
            f"authority this package never claimed."
        )
    if ARTICLE_50_DISCLOSURE not in proposal.ai_disclosure:
        raise DisclosureMissingError(
            f"{proposal.claim_id}: the correction left the composer without the disclosure its own "
            f"model declares by default. EU AI Act Article 50 asks that the reader be told the "
            f"text was assembled by an AI system, and the sentence is a property of the response "
            f"than of the screen it is shown on."
        )
    return proposal
