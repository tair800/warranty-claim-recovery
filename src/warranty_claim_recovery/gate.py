"""The deterministic gate: whether this claim may be resubmitted, decided before a word is written.

ADR-001 §4 claim 3 rests here. The gate reads eleven signals, **not one of which is a model
output**, and returns one of four outcomes. It runs *before* the correction is composed, so no
generated text can move a claim from a refusal to a resubmission — not because the composer is
instructed not to try, but because the composer is never consulted about the question.

Three properties are load-bearing, and each is checkable on its own.

**The loop only ever withholds; the fall-through is the only thing that authorises.** Every rule in
`_RULES` carries `NOT_RECOVERABLE` or `REVIEW`, and `tests/test_gate.py` asserts that over the
module's AST as well as over the table at run time. The shape is the argument: to create a new way
of authorising a recovery you would have to delete a rule, which a reviewer sees in a diff, rather
than add a branch, which a reviewer does not.

**`RECOVERABLE` and `PARTIALLY_RECOVERABLE` are each named in exactly one place in this file**, and
that place is after the loop. A second construction site is a second place for a guard to be
forgotten, and the only version of that check which survives a refactor is one that counts nodes in
the syntax tree rather than one that trusts a review.

**The claim window is checked first, before anything about evidence.** A window that has closed
makes the quality of the evidence irrelevant: filing a perfect correction the day after the deadline
consumes a person's afternoon and is refused on receipt. It is kill condition M, graded at zero over
the whole corpus, and putting the check anywhere but first would mean a case with missing paperwork
and a shut window reported as an evidence problem, which sends somebody to chase a document that
can no longer be used.

### Two signals that decide nothing here, and one that does

`part_covered` and `labour_rate_within_cap` are recorded on the decision and are not rules. They do
not need to be: both already move the money — an uncovered part becomes `uncovered_parts` and an
over-cap rate becomes `labour_excess` — so a claim they exhaust reaches the zero-recoverable rule on
the arithmetic, and each is additionally the subject of a rejection code with its own requirements.
They are withheld twice before this module sees them.

`within_warranty_period` is different and therefore **is** a rule. It moves no money under the
recovery arithmetic: a failure outside the warranty period still has covered parts and an in-cap
labour rate, so the recoverable amount comes out positive. Without a rule here, a claim for a
machine that went out of warranty last March, arriving with a complete and truthful evidence bundle,
would be authorised — which is precisely the false recovery kill condition G budgets at zero. A
signal that decides nothing is a signal that is not being used.

### Why `REVIEW` exists at all

Two outcomes would be cheaper and would be wrong. A requirement whose evidence contradicts itself is
not the same situation as a requirement with no evidence: the first needs a person to decide which
source is right, the second needs a technician to go and fetch something. Collapsing either into
"no" writes off money that was recoverable, which is the failure `domain.RecoveryOutcome` records in
its own docstring.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Final, NamedTuple

from warranty_claim_recovery.domain import (
    Claim,
    ClaimWindow,
    GateDecision,
    GateSignals,
    RecoveryComputation,
    RecoveryOutcome,
    Requirement,
    RequirementStatus,
    WarrantyProgram,
)
from warranty_claim_recovery.eligibility import Eligibility
from warranty_claim_recovery.money import Money

__all__ = [
    "decide",
    "signals",
    "withholds_recovery",
]


def withholds_recovery(outcome: RecoveryOutcome) -> bool:
    """Whether this outcome means nothing is filed with the manufacturer.

    A refusal and a review both withhold, and every count of "how many cases did the system decline
    to resubmit" must treat them together. Written once, here, so that the console, the evaluation
    and the queue do not become three slightly different opinions about what `REVIEW` means — and
    so that the test which asserts that no rule authorises money has one definition to test against.
    """
    return outcome in {RecoveryOutcome.NOT_RECOVERABLE, RecoveryOutcome.REVIEW}


def signals(
    window: ClaimWindow,
    eligibility: Eligibility,
    computation: RecoveryComputation,
    requirements: Sequence[Requirement],
) -> GateSignals:
    """The eleven inputs to the decision, without the decision.

    Exposed separately so the evaluation and the console can record what the gate saw without
    re-deriving it. A signal recomputed in two places is a signal that will eventually be computed
    two ways, and the disagreement surfaces as a decision nobody can explain.

    Counted over the requirements actually passed in, not over the matrix. The two should be the
    same tuple and `requirements_total == 0` is a rule rather than an assertion, because the gate
    cannot verify its caller and a claim assembled without its requirements must be visibly wrong
    rather than quietly compliant.
    """
    satisfied = sum(1 for item in requirements if item.status is RequirementStatus.SATISFIED)
    conflicting = sum(1 for item in requirements if item.status is RequirementStatus.CONFLICTING)
    return GateSignals(
        window_open=window.is_open,
        days_remaining=window.days_remaining,
        within_warranty_period=eligibility.within_warranty_period,
        part_covered=eligibility.part_covered,
        serial_in_range=eligibility.serial_in_range,
        requirements_total=len(requirements),
        requirements_satisfied=satisfied,
        requirements_conflicting=conflicting,
        already_recovered=eligibility.already_recovered,
        recoverable_is_positive=(computation.recoverable_amount > Money.zero(computation.currency)),
        labour_rate_within_cap=eligibility.labour_rate_within_cap,
    )


class _Context(NamedTuple):
    """Everything a rule is permitted to look at. Every field deterministic, every field frozen."""

    claim: Claim
    program: WarrantyProgram
    window: ClaimWindow
    computation: RecoveryComputation
    signals: GateSignals
    #: Identifiers only, in the order the requirement matrix declared them. The whole `Requirement`
    #: would let a reason string quote a `detail` written elsewhere, and a refusal that explains
    #: itself in somebody else's words is a refusal that cannot be traced to a rule.
    missing: tuple[str, ...]
    conflicting: tuple[str, ...]


class _Rule(NamedTuple):
    """A reason to withhold, the outcome it produces, and the sentence a person reads.

    `reason` is a callable rather than a format string because two of the seven rules have to list
    requirement identifiers, and a format language that can render a list is a format language that
    can render anything — at which point the reason is code with worse tooling.
    """

    fires: Callable[[_Context], bool]
    outcome: RecoveryOutcome
    reason: Callable[[_Context], str]


def _window_closed(context: _Context) -> str:
    return (
        f"the correction window for {context.claim.claim_id} closed on "
        f"{context.window.closes_on.isoformat()} and the case was assessed as of "
        f"{context.window.as_of.isoformat()}, {abs(context.signals.days_remaining)} day(s) later; "
        f"a correction filed now is refused on receipt"
    )


def _already_recovered(context: _Context) -> str:
    return (
        f"recovery identity {context.claim.recovery_identity} has already been settled for "
        f"{context.computation.already_recovered}; resubmitting it would be a duplicate claim "
        f"against {context.program.manufacturer}"
    )


def _outside_warranty_period(context: _Context) -> str:
    return (
        f"the failure on {context.claim.failure_date.isoformat()} fell outside the "
        f"{context.program.warranty_months}-month warranty period running from the in-service date "
        f"{context.claim.in_service_date.isoformat()}; no evidence makes an expired warranty cover "
        f"a repair"
    )


def _nothing_assessed(context: _Context) -> str:
    return (
        f"no requirement was assessed for rejection code {context.claim.rejection_code.value}. The "
        f"requirement matrix is a total function, so this case was assembled without one and a "
        f"person must find out why before anything is filed"
    )


def _conflicting_requirements(context: _Context) -> str:
    return (
        f"the evidence for {', '.join(context.conflicting)} contradicts itself or two sources "
        f"disagree; a person has to decide which is right, and this system will not choose for "
        f"them"
    )


def _missing_requirements(context: _Context) -> str:
    return (
        f"rejection code {context.claim.rejection_code.value} still has no evidence for "
        f"{', '.join(context.missing)}; resubmitting now repeats the rejection this case exists to "
        f"answer"
    )


def _nothing_left_to_recover(context: _Context) -> str:
    return (
        f"the recoverable amount is {context.computation.recoverable_amount} after a deductible of "
        f"{context.computation.deductible} and exclusions of "
        f"{context.computation.excluded_amount}; there is no money to file for"
    )


#: Evaluated in order, first match wins, and the order is the argument.
#:
#: The window is first because it makes every later question moot; the module docstring says why at
#: length. The duplicate check is second for the same reason from the other direction — a claim
#: already settled cannot be improved by evidence either, and reporting it as a paperwork problem
#: sends a technician to chase documents for money that has already been paid.
#:
#: `requirements_total == 0` is checked ahead of both requirement rules. With no requirements there
#: is nothing missing and nothing conflicting, so both of those rules are silent, and the case would
#: fall through to an authorisation earned by never having been asked a question.
#:
#: Conflict is checked before absence, which is the opposite of the intuitive order. A case with one
#: contradictory requirement and one missing requirement is a decision for a person, not an errand
#: for a technician: the contradiction will still be there when the missing document arrives, and
#: discovering it on the second pass costs another day against a window that is already running.
_RULES: Final[tuple[_Rule, ...]] = (
    _Rule(
        lambda context: not context.signals.window_open,
        RecoveryOutcome.NOT_RECOVERABLE,
        _window_closed,
    ),
    _Rule(
        lambda context: context.signals.already_recovered,
        RecoveryOutcome.NOT_RECOVERABLE,
        _already_recovered,
    ),
    _Rule(
        lambda context: not context.signals.within_warranty_period,
        RecoveryOutcome.NOT_RECOVERABLE,
        _outside_warranty_period,
    ),
    _Rule(
        lambda context: context.signals.requirements_total == 0,
        RecoveryOutcome.REVIEW,
        _nothing_assessed,
    ),
    _Rule(
        lambda context: bool(context.conflicting),
        RecoveryOutcome.REVIEW,
        _conflicting_requirements,
    ),
    _Rule(
        lambda context: bool(context.missing),
        RecoveryOutcome.NOT_RECOVERABLE,
        _missing_requirements,
    ),
    _Rule(
        lambda context: not context.signals.recoverable_is_positive,
        RecoveryOutcome.NOT_RECOVERABLE,
        _nothing_left_to_recover,
    ),
)


# Six positional parameters, one over the limit, and the suppression is argued rather than habitual.
# Every one of them is a distinct deterministic input the decision is graded against, and the
# alternative — bundling them into a "case" object — would give the gate a parameter whose contents
# no signature constrains, which is how a model output eventually arrives at a rule. The order is
# also the order of the graph's nodes, so a reader of `CLAUDE.md` §2.1 can check the call site
# against the pipeline without holding a mapping in their head.
def decide(  # noqa: PLR0917
    claim: Claim,
    program: WarrantyProgram,
    window: ClaimWindow,
    eligibility: Eligibility,
    computation: RecoveryComputation,
    requirements: tuple[Requirement, ...],
) -> GateDecision:
    """Whether this claim may go back to the manufacturer, and on what grounds.

    The loop withholds and the fall-through authorises; nothing else in this file constructs a
    decision. The distinction between the two authorising outcomes is a single subtraction —
    `excluded_amount` is the claimed total less the recoverable amount — and it is computed here
    rather than asked of the caller, because a caller that decides whether a recovery is full or
    partial is a caller that can decide it wrongly and be believed.

    A partial recovery is a success, not a degraded one. The claim goes out for the covered portion
    with the exclusions shown, which is the outcome the whole four-state design exists to make
    available; forcing it into `RECOVERABLE` would file a total the adjudicator refuses, and forcing
    it into `NOT_RECOVERABLE` writes off money that was owed.
    """
    computed = signals(window, eligibility, computation, requirements)
    context = _Context(
        claim=claim,
        program=program,
        window=window,
        computation=computation,
        signals=computed,
        missing=tuple(
            item.requirement_id for item in requirements if item.status is RequirementStatus.MISSING
        ),
        conflicting=tuple(
            item.requirement_id
            for item in requirements
            if item.status is RequirementStatus.CONFLICTING
        ),
    )

    for rule in _RULES:
        if rule.fires(context):
            return GateDecision(
                outcome=rule.outcome,
                reason=rule.reason(context),
                signals=computed,
                computation=computation,
                # Carried on every outcome, including the refusals. A refusal that discards the
                # requirements it refused over is a refusal nobody can appeal, and the console's
                # whole job is to show a handler which line to go and fix.
                requirements=requirements,
            )

    nothing_was_excluded = computation.excluded_amount == Money.zero(computation.currency)
    outcome = (
        RecoveryOutcome.RECOVERABLE
        if nothing_was_excluded
        else RecoveryOutcome.PARTIALLY_RECOVERABLE
    )
    return GateDecision(
        outcome=outcome,
        reason=(
            f"all {computed.requirements_total} requirement(s) for "
            f"{claim.rejection_code.value} are evidenced, the correction window for "
            f"{program.program_id} closes in {computed.days_remaining} day(s), and "
            f"{computation.recoverable_amount} of {computation.claimed_total} is recoverable"
        ),
        signals=computed,
        computation=computation,
        requirements=requirements,
    )
