"""The types the whole system agrees on, and the invariants they refuse to be constructed without.

Everything here is frozen and forbids extra fields. `extra="forbid"` does real work in this domain:
a claim whose `serial_number` arrives as `serialNumber` would, under the permissive default, become
an ignored attribute and an eligibility check that silently checks nothing. The claim would be
approved, the resubmission would go out, and the manufacturer would reject it a second time — which
is the exact failure this project exists to prevent, arriving through a typo.

The invariants live in validators rather than in the code that builds these objects, because
there are several such places — the corpus generator, the API, the graph's intake node, the test
fixtures — and an invariant enforced in three of the four holds until someone adds a fifth.

Read `money.py` first. Every amount here is a `Money`, and the rules about currency and rounding
are that module's, not this one's.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from warranty_claim_recovery.money import Currency, Money

__all__ = [
    "AuditEvent",
    "CaseState",
    "Citation",
    "Claim",
    "ClaimWindow",
    "CorrectionProposal",
    "GateDecision",
    "GateSignals",
    "PolicyClause",
    "RecoveryComputation",
    "RecoveryOutcome",
    "RejectionCode",
    "Requirement",
    "RequirementStatus",
    "WarrantyProgram",
]


class _Frozen(BaseModel):
    """Frozen, and extras forbidden. See the module docstring for why the second one matters."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)


# ------------------------------------------------------------------------------------------------
# Enumerations. All closed, all `StrEnum` so an artifact serialises to something a person can read
# without a legend.
# ------------------------------------------------------------------------------------------------


class RejectionCode(StrEnum):
    """Why the manufacturer sent the claim back.

    Seven codes, and the set is closed because the requirement matrix in `requirements.py` is a
    total function over it. A code the matrix does not know must fail at the boundary rather than
    fall through to an empty requirement list — an empty list would mean "nothing is required", and
    a claim with nothing required is a claim the gate would happily resubmit.
    """

    MISSING_SERIAL = "MISSING_SERIAL"
    MISSING_INSTALL_PROOF = "MISSING_INSTALL_PROOF"
    WRONG_FAILURE_CODE = "WRONG_FAILURE_CODE"
    PART_NOT_COVERED = "PART_NOT_COVERED"
    OUTSIDE_WARRANTY_PERIOD = "OUTSIDE_WARRANTY_PERIOD"
    DUPLICATE_CLAIM = "DUPLICATE_CLAIM"
    LABOUR_RATE_EXCEEDED = "LABOUR_RATE_EXCEEDED"


class RequirementStatus(StrEnum):
    SATISFIED = "SATISFIED"
    MISSING = "MISSING"
    #: The evidence exists and contradicts itself, or two sources disagree. Distinct from MISSING
    #: because the remedies differ: MISSING is a request to a technician, CONFLICTING is a decision
    #: a person has to make.
    CONFLICTING = "CONFLICTING"


class RecoveryOutcome(StrEnum):
    """What the deterministic gate decided.

    Four, not two. Forcing an ambiguous claim to a binary outcome is the failure the blueprint names
    and the brief repeats: a claim that is partly covered and a claim nobody can adjudicate are
    different situations with different next actions, and collapsing either into "no" writes off
    money that was recoverable.
    """

    RECOVERABLE = "RECOVERABLE"
    PARTIALLY_RECOVERABLE = "PARTIALLY_RECOVERABLE"
    NOT_RECOVERABLE = "NOT_RECOVERABLE"
    REVIEW = "REVIEW"


class CaseState(StrEnum):
    """Where the case machine is. Mirrors the graph's nodes, one to one, deliberately.

    A state enum that drifts from the graph's node names makes the durability claim uncheckable:
    "resumes at the same node" needs one vocabulary, not a state name and a node name that a reader
    has to map between.
    """

    INTAKE = "INTAKE"
    REQUIREMENTS = "REQUIREMENTS"
    DEADLINE = "DEADLINE"
    ELIGIBILITY = "ELIGIBILITY"
    RETRIEVAL = "RETRIEVAL"
    COMPOSE = "COMPOSE"
    GATE = "GATE"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    SUBMITTED = "SUBMITTED"
    WRITTEN_OFF = "WRITTEN_OFF"


# ------------------------------------------------------------------------------------------------
# The policy corpus.
# ------------------------------------------------------------------------------------------------


class WarrantyProgram(_Frozen):
    """A manufacturer plus a policy version. The unit the hold-out splits on.

    Splitting on the claim would leak: every claim under one program is adjudicated against the same
    clause text, the same rejection-code table and the same deadline rule, so a claim-level split
    measures how well the system memorised a policy it was tuned on. ADR-001 §7.
    """

    program_id: str = Field(min_length=3)
    manufacturer: str = Field(min_length=2)
    policy_version: str = Field(min_length=1)
    currency: Currency
    #: Days after the rejection notice within which a correction must be filed. Deterministic
    #: arithmetic, never a judgement, and kill condition M is that nothing goes out after it.
    correction_window_days: int = Field(gt=0, le=365)
    #: Months of cover from the in-service date.
    warranty_months: int = Field(gt=0, le=240)
    labour_rate_cap_per_hour: Money
    #: Subtracted from the eligible amount before the cap. See `money.RecoveryComputation`.
    deductible: Money
    #: The ceiling on one claim, applied after the deductible.
    claim_cap: Money

    @model_validator(mode="after")
    def _amounts_share_the_program_currency(self) -> Self:
        for name in ("labour_rate_cap_per_hour", "deductible", "claim_cap"):
            amount: Money = getattr(self, name)
            if amount.currency is not self.currency:
                raise ValueError(
                    f"{name} is denominated in {amount.currency} and the program in "
                    f"{self.currency}; this system holds no rate that would reconcile them"
                )
        return self


class PolicyClause(_Frozen):
    """One retrievable chunk of warranty policy or service bulletin, with its coordinates.

    `text` is the verbatim chunk and `document_id` plus `start_offset` locate it in the document it
    came from. A citation is checked by slicing the document at those offsets and comparing, not by
    testing membership: a clause that appears twice in one policy is a coordinate that has drifted,
    and membership would call it faithful.
    """

    clause_id: str = Field(min_length=3)
    program_id: str = Field(min_length=3)
    document_id: str = Field(min_length=3)
    section: str = Field(min_length=1)
    text: str = Field(min_length=20)
    start_offset: int = Field(ge=0)
    end_offset: int = Field(gt=0)
    #: Which rejection codes this clause governs. Empty means the clause is context rather than
    #: authority, and a clause with no governed code may never be cited as the basis of a
    #: requirement being satisfied.
    governs: tuple[RejectionCode, ...] = ()

    @model_validator(mode="after")
    def _offsets_span_the_text(self) -> Self:
        if self.end_offset - self.start_offset != len(self.text):
            raise ValueError(
                f"{self.clause_id}: offsets span {self.end_offset - self.start_offset} characters "
                f"and the text is {len(self.text)}; a citation built from this would be unfaithful "
                f"by construction"
            )
        return self


class Citation(_Frozen):
    """A claim traced to a span of a document, by coordinates a reader can check.

    Kill condition I slices the named document version at `start_offset` and compares. The
    `policy_version` is part of the identity because a correct quote from a superseded policy is
    still the wrong authority, and the manufacturer will say so.
    """

    clause_id: str
    document_id: str
    policy_version: str
    section: str
    quote: str = Field(min_length=1)
    start_offset: int = Field(ge=0)
    end_offset: int = Field(gt=0)

    @model_validator(mode="after")
    def _offsets_span_the_quote(self) -> Self:
        if self.end_offset - self.start_offset != len(self.quote):
            raise ValueError(
                f"{self.clause_id}: the quote is {len(self.quote)} characters and the offsets span "
                f"{self.end_offset - self.start_offset}"
            )
        return self


# ------------------------------------------------------------------------------------------------
# The claim and its case.
# ------------------------------------------------------------------------------------------------


class ClaimWindow(_Frozen):
    """The manufacturer's correction deadline, and whether it has passed.

    Half-open: a correction filed **on** `closes_on` is in time and one filed the next day is not.
    Stated because "within 30 days" is ambiguous in exactly the way that costs a claim, and a
    half-open interval is the only reading that makes two adjacent windows partition the calendar.
    """

    rejected_on: date
    closes_on: date
    #: The date the system is reasoning as of. Explicit rather than `date.today()`, because a
    #: default that reads the wall clock makes every evaluation unreproducible and every test
    #: time-dependent.
    as_of: date

    @model_validator(mode="after")
    def _closes_after_rejection(self) -> Self:
        if self.closes_on <= self.rejected_on:
            raise ValueError("a correction window that closes before it opens is not a window")
        return self

    @property
    def is_open(self) -> bool:
        return self.as_of <= self.closes_on

    @property
    def days_remaining(self) -> int:
        return (self.closes_on - self.as_of).days


class Claim(_Frozen):
    """A rejected warranty claim, as it arrives.

    Everything here is fact from the source systems. Nothing here is a judgement, and nothing here
    is computed by this system — `RecoveryComputation` holds what this system concluded, so that a
    reader of an audit record can always tell input from output.
    """

    claim_id: str = Field(min_length=3)
    program_id: str = Field(min_length=3)
    part_number: str = Field(min_length=3)
    serial_number: str | None = None
    #: When the machine went into service. The warranty period runs from here.
    in_service_date: date
    failure_date: date
    repair_invoice_date: date
    rejection_code: RejectionCode
    rejected_on: date
    claimed_parts: Money
    claimed_labour_hours: Annotated[int, Field(ge=0, le=500)]
    claimed_labour_rate: Money
    #: Set when this claim's recovery identity has already produced an effect. The deterministic
    #: duplicate check reads this; it is not inferred from text.
    previously_recovered: Money | None = None

    @field_validator("claimed_labour_hours", mode="before")
    @classmethod
    def _hours_are_whole(cls, value: Any) -> Any:
        if isinstance(value, float):
            raise TypeError("labour hours arrive as whole units; a float here is a rounding bug")
        return value

    @model_validator(mode="after")
    def _dates_are_ordered(self) -> Self:
        if self.failure_date < self.in_service_date:
            raise ValueError(
                f"{self.claim_id}: failure on {self.failure_date} precedes the in-service date "
                f"{self.in_service_date}"
            )
        if self.repair_invoice_date < self.failure_date:
            raise ValueError(
                f"{self.claim_id}: the repair was invoiced on {self.repair_invoice_date}, before "
                f"the failure on {self.failure_date}"
            )
        if self.rejected_on < self.repair_invoice_date:
            raise ValueError(
                f"{self.claim_id}: rejected on {self.rejected_on}, before the claim could have "
                f"been made on {self.repair_invoice_date}"
            )
        return self

    @model_validator(mode="after")
    def _amounts_share_one_currency(self) -> Self:
        currency = self.claimed_parts.currency
        if self.claimed_labour_rate.currency is not currency:
            raise ValueError(
                f"{self.claim_id}: parts in {currency} and labour in "
                f"{self.claimed_labour_rate.currency}"
            )
        prior = self.previously_recovered
        if prior is not None and prior.currency is not currency:
            raise ValueError(f"{self.claim_id}: the prior recovery is in a different currency")
        return self

    @property
    def claimed_labour(self) -> Money:
        return self.claimed_labour_rate * self.claimed_labour_hours

    @property
    def claimed_total(self) -> Money:
        return self.claimed_parts + self.claimed_labour

    @property
    def recovery_identity(self) -> str:
        """What "the same recovery" means, for kill condition E.

        Claim plus part plus serial rather than claim alone: one claim can legitimately be
        resubmitted for a different part after a partial adjudication, and keying on the claim would
        refuse the second one as a duplicate. Serial is included because two identical parts on the
        same machine are two recoveries, and the manufacturer treats them as such.
        """
        return f"{self.claim_id}:{self.part_number}:{self.serial_number or '-'}"


class Requirement(_Frozen):
    """One thing the manufacturer's rejection demands, and whether the evidence supports it.

    `citation` is `None` unless `status` is `SATISFIED`, and `SATISFIED` without a citation is
    refused at construction — kill condition H made impossible rather than measured. The validator
    is the enforcement; the kill test then confirms that the enforcement was in the path.
    """

    requirement_id: str = Field(min_length=3)
    description: str = Field(min_length=10)
    status: RequirementStatus
    citation: Citation | None = None
    #: Why, in a sentence a technician can act on. Never model-generated for a MISSING requirement:
    #: the remedy is the requirement, and inventing a different one sends the wrong person looking
    #: for the wrong thing.
    detail: str = Field(min_length=3)

    @model_validator(mode="after")
    def _satisfied_requires_a_citation(self) -> Self:
        if self.status is RequirementStatus.SATISFIED and self.citation is None:
            raise ValueError(
                f"{self.requirement_id} is marked SATISFIED with no citation. A requirement "
                f"satisfied by nothing is how an unsupported resubmission leaves the system."
            )
        return self


class GateSignals(_Frozen):
    """The inputs to the gate, every one of them deterministic and none of them a model output.

    Recorded on the decision so the console can show *why* rather than asserting a verdict, and so a
    disagreement about an outcome is a disagreement about a number rather than about a vibe.
    """

    window_open: bool
    days_remaining: int
    within_warranty_period: bool
    part_covered: bool
    serial_in_range: bool
    requirements_total: int
    requirements_satisfied: int
    requirements_conflicting: int
    already_recovered: bool
    recoverable_is_positive: bool
    labour_rate_within_cap: bool

    @property
    def all_requirements_satisfied(self) -> bool:
        satisfied_all = self.requirements_satisfied == self.requirements_total
        return self.requirements_total > 0 and satisfied_all


class RecoveryComputation(_Frozen):
    """What this system concluded about the money, and every step it took to get there.

    Every intermediate is kept. A recovery package that shows only a total is a package the
    manufacturer's adjudicator has to reconstruct, and the one they cannot reconstruct is the one
    they refuse.

    **Rounding happens here and nowhere else** — `finalise` quantises once, at the end. See
    `money.py` failure 2.
    """

    currency: Currency
    claimed_total: Money
    #: Claimed labour above the program's rate cap. Excluded, and shown as excluded.
    labour_excess: Money
    #: Parts the policy does not cover.
    uncovered_parts: Money
    eligible_amount: Money
    deductible: Money
    #: What the cap removed, after the deductible. Zero when the cap did not bind.
    capped_amount: Money
    already_recovered: Money
    recoverable_amount: Money

    @model_validator(mode="after")
    def _one_currency_throughout(self) -> Self:
        for name in (
            "claimed_total",
            "labour_excess",
            "uncovered_parts",
            "eligible_amount",
            "deductible",
            "capped_amount",
            "already_recovered",
            "recoverable_amount",
        ):
            amount: Money = getattr(self, name)
            if amount.currency is not self.currency:
                raise ValueError(
                    f"{name} is in {amount.currency}, the computation in {self.currency}"
                )
        return self

    @model_validator(mode="after")
    def _recoverable_is_never_negative(self) -> Self:
        if self.recoverable_amount < Money.zero(self.currency):
            raise ValueError(
                "a negative recoverable amount is a claim against the distributor, not the "
                "manufacturer, and this system will not construct one"
            )
        return self

    @property
    def excluded_amount(self) -> Money:
        """Everything the claim asked for that the recovery does not include."""
        return self.claimed_total - self.recoverable_amount


class GateDecision(_Frozen):
    """The outcome, the reason a person can read, and the signals behind it.

    `RecoveryOutcome.RECOVERABLE` and `PARTIALLY_RECOVERABLE` are the two outcomes that let money
    leave the system, and `tests/test_gate.py` asserts over the AST that each is constructed in
    exactly one place in `gate.py`. Adding a second way to authorise a recovery would then be a
    visible diff rather than a line in a branch.
    """

    outcome: RecoveryOutcome
    reason: str = Field(min_length=10)
    signals: GateSignals
    computation: RecoveryComputation
    requirements: tuple[Requirement, ...]

    @model_validator(mode="after")
    def _authorised_outcomes_carry_evidence(self) -> Self:
        if self.outcome in {RecoveryOutcome.RECOVERABLE, RecoveryOutcome.PARTIALLY_RECOVERABLE}:
            if not self.requirements:
                raise ValueError(f"{self.outcome} with no requirements assessed")
            missing = [r for r in self.requirements if r.status is not RequirementStatus.SATISFIED]
            if missing:
                raise ValueError(
                    f"{self.outcome} with {len(missing)} unsatisfied requirement(s): "
                    f"{[r.requirement_id for r in missing]}"
                )
            if self.computation.recoverable_amount <= Money.zero(self.computation.currency):
                raise ValueError(f"{self.outcome} with a recoverable amount of zero or less")
        return self

    @property
    def authorises_recovery(self) -> bool:
        return self.outcome in {RecoveryOutcome.RECOVERABLE, RecoveryOutcome.PARTIALLY_RECOVERABLE}


class CorrectionProposal(_Frozen):
    """The text that would go back to the manufacturer, and what it is built from.

    The shipped composer is extractive: `narrative` is assembled from the cited spans, so the set of
    clause identifiers it mentions is a subset of the set in `citations` as string arithmetic rather
    than as a measured rate. The abstractive arm is a port; ADR-001 §3 says why it raises.
    """

    claim_id: str
    narrative: str = Field(min_length=20)
    citations: tuple[Citation, ...]
    composer: str = Field(min_length=3)
    #: EU AI Act Article 50. A property of the response rather than of the page it is drawn on.
    ai_disclosure: str = Field(
        default=(
            "This correction was assembled by an AI system from the cited policy passages. "
            "Verify against the referenced policy version before filing it."
        ),
        min_length=20,
    )

    @model_validator(mode="after")
    def _a_proposal_cites_something(self) -> Self:
        if not self.citations:
            raise ValueError("a correction with no citation is an assertion, not a proposal")
        return self


class AuditEvent(_Frozen):
    """One thing that happened, recorded independently of the thing that did it.

    An independent implementation of the audit contract, which is what `SKILL_MATRIX.md` asks of
    this project: the events are written by the node that acts, read by a verifier that imports
    none of the graph, and kill condition D is graded from them rather than from the graph's own
    belief about what it did.
    """

    event_id: str
    case_id: str
    #: Bumped whenever the case's decided content changes. An approval is bound to the version it
    #: approved, so approving a case and then changing it invalidates the approval — kill condition
    #: D checks exactly that, because "approved" and "approved *this*" are different claims.
    case_version: int = Field(ge=0)
    kind: str = Field(min_length=3)
    node: CaseState
    at: date
    actor: str = Field(min_length=1)
    detail: dict[str, Any] = Field(default_factory=dict)
