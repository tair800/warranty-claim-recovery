"""The claims, the evidence they carry, and the ground truth — which is construction metadata.

**Nothing here is labelled.** The generator knows the answer to every claim because it built the
claim to have that answer: it chose the in-service date that put the failure one day outside cover,
it chose the serial that falls below the declared range, it withheld the repair invoice. No model
decided an outcome, no heuristic inferred one, and there is nothing in this corpus for a model to
have got right by accident. That is the only kind of ground truth kill condition F — exact `Decimal`
equality against the generator's answer — can be graded against.

### The ordered ground-truth rule

`GROUND_TRUTH_RULE` below is the rule, in order, first match wins, and `outcome_for` is its only
implementation. Two things about it are worth arguing rather than asserting.

**Why a missing invoice is `REVIEW` and not `NOT_RECOVERABLE`.** A claim whose evidence is
incomplete is not a claim that cannot be recovered; it is a claim nobody has finished. Writing it
off is the failure `CLAUDE.md` §1 names first — *a claim written off because nobody noticed* — and
the correct next action is a request to a technician, not a closure. `RecoveryOutcome` has four
members precisely so that this case does not have to be collapsed into one of two.

**Why a zero recovery is checked before the evidence is.** Chasing a technician for an installation
certificate that would unlock nothing wastes the one resource a warranty desk actually runs out of,
which is the correction window. So the money is checked first: if the deductible, the cap and a
prior recovery between them leave nothing, the claim is closed and no evidence is chased.

**Why `PARTIALLY_RECOVERABLE` means *something was excluded* rather than *the recovery is less than
the claim*.** The deductible applies to every claim under the programme; it is the price of the
cover rather than a refusal of part of it. If "less than claimed" decided the outcome then every
claim under a programme with a deductible would be partial and `RECOVERABLE` would be unreachable —
the outcome would be a property of the programme rather than of the claim. Rejected on those
grounds. A claim is partial when the labour-rate cap, the per-claim ceiling or a prior recovery
actually removed something.

### The evidence vocabulary, and what this module deliberately does not own

`EVIDENCE_KEYS` is a closed vocabulary of the documents a claim can carry, and each claim records a
status for every one of them. `missing_requirements` in the truth file is the sorted list of keys
the claim does not carry cleanly.

It is **evidence keys and not requirement identifiers**, and that is a deliberate boundary. The
requirement matrix — which evidence each rejection code *demands* — lives in `requirements.py` and
must live in exactly one place; restating it here would be a second implementation of a rule, and
the copy that drifts is always the one nobody thinks to edit. What this corpus owns is the opposite
question: what evidence a synthetic claim was built without. The two meet at
`RequirementSpec.evidence_key`, which is the join column it exists to be.

`EVIDENCE_RELEVANT_TO` exists so the generator can refuse an incoherent claim — a claim rejected for
a missing serial while carrying a perfect serial and lacking a labour-rate agreement teaches an
evaluation nothing. It records which evidence this corpus considers *withholdable* for a code. It is
not the requirement matrix, must not be used as one, and a cross-lane test comparing the two tables
would be worth writing: an evidence key the matrix demands and this corpus never withholds is a
requirement that is satisfied in every claim and therefore never exercised.

### Boundaries are set where the two readings agree

Every date in this corpus that decides an outcome sits where a half-open reading and a closed
reading give the same answer. The "warranty expired one day before the failure" claims put the end
of cover on the day *before* the failure, not on the failure date: on the failure date the two
readings disagree, and a corpus whose ground truth depends on an unstated interval convention is a
corpus that grades the convention rather than the system. Claims that are comfortably inside cover
are months inside it. The same applies to the correction window: a closed window is closed by days,
not by one.
"""

from __future__ import annotations

import calendar
from datetime import date, timedelta
from decimal import Decimal
from enum import StrEnum
from typing import Final, NamedTuple

from warranty_claim_recovery.corpus.clauses import BuiltDocument, governing_clause_id
from warranty_claim_recovery.corpus.programs import VARIANTS, ProgramSpec
from warranty_claim_recovery.corpus.rng import stream
from warranty_claim_recovery.domain import Claim, RecoveryOutcome, RejectionCode
from warranty_claim_recovery.money import Currency, Money

__all__ = [
    "ADVERSARIAL_KINDS",
    "EVIDENCE_KEYS",
    "EVIDENCE_RELEVANT_TO",
    "GROUND_TRUTH_RULE",
    "NOT_ADVERSARIAL",
    "RECIPE",
    "BuiltClaim",
    "ClaimRecipe",
    "EvidenceStatus",
    "GroundTruthMoney",
    "Shape",
    "add_months",
    "build_claims",
    "intended_outcome",
    "outcome_for",
    "recovery_of",
    "serial_is_in_range",
    "within_warranty_period",
]


class EvidenceStatus(StrEnum):
    """What the claim carries for one evidence key.

    `CONFLICTING` is distinct from `ABSENT` because the remedies differ and the outcomes should not
    be told apart by a reader guessing: absent evidence is a request to a technician, conflicting
    evidence is a decision a person has to make. `domain.RequirementStatus` draws the same line for
    the same reason.
    """

    PRESENT = "PRESENT"
    ABSENT = "ABSENT"
    CONFLICTING = "CONFLICTING"


#: The closed vocabulary of evidence a claim can carry. Closed rather than open because an evidence
#: key that only ever appears on one claim is a key nothing joins on, and a typo would produce one
#: silently.
EVIDENCE_KEYS: Final[tuple[str, ...]] = (
    "commissioning_date",
    "coverage_confirmation",
    "diagnostic_report",
    "failure_code",
    "installation_certificate",
    "labour_rate_agreement",
    "part_number_confirmation",
    "prior_claim_reference",
    "repair_invoice",
    "serial_plate_photo",
    "supplier_declaration",
    "warranty_start_evidence",
)

#: Which evidence this corpus is willing to withhold or corrupt for each rejection code. See the
#: module docstring: this is **not** the requirement matrix and must not be used as one.
EVIDENCE_RELEVANT_TO: Final[dict[RejectionCode, tuple[str, ...]]] = {
    RejectionCode.MISSING_SERIAL: ("serial_plate_photo", "part_number_confirmation"),
    RejectionCode.MISSING_INSTALL_PROOF: (
        "installation_certificate",
        "commissioning_date",
        "repair_invoice",
    ),
    RejectionCode.WRONG_FAILURE_CODE: ("diagnostic_report", "failure_code"),
    RejectionCode.PART_NOT_COVERED: (
        "part_number_confirmation",
        "coverage_confirmation",
        "supplier_declaration",
    ),
    RejectionCode.OUTSIDE_WARRANTY_PERIOD: ("warranty_start_evidence", "commissioning_date"),
    RejectionCode.DUPLICATE_CLAIM: ("prior_claim_reference", "repair_invoice"),
    RejectionCode.LABOUR_RATE_EXCEEDED: ("labour_rate_agreement", "repair_invoice"),
}

#: The rule, in order, first match wins. `outcome_for` is its only implementation and this tuple is
#: what the artifact publishes, so the claim and the implementation are the same strings.
GROUND_TRUTH_RULE: Final[tuple[str, ...]] = (
    "1. the claim is not denominated in the programme's currency -> REVIEW",
    "2. the correction window had closed at the as-of date -> NOT_RECOVERABLE",
    "3. the failure fell outside the warranty period -> NOT_RECOVERABLE",
    "4. the part is not covered, or its serial is outside the declared range -> NOT_RECOVERABLE",
    "5. the recoverable amount is zero -> NOT_RECOVERABLE",
    "6. evidence is absent or conflicting -> REVIEW",
    "7. the labour cap, the claim ceiling or a prior recovery removed something "
    "-> PARTIALLY_RECOVERABLE",
    "8. otherwise -> RECOVERABLE",
)


class Shape(StrEnum):
    """How a claim was constructed. One shape, one intended outcome, asserted by the generator."""

    CLEAN = "CLEAN"
    WINDOW_CLOSED = "WINDOW_CLOSED"
    EXPIRED_BY_ONE_DAY = "EXPIRED_BY_ONE_DAY"
    SERIAL_OUT_OF_RANGE = "SERIAL_OUT_OF_RANGE"
    PART_EXCLUDED = "PART_EXCLUDED"
    WRONG_VARIANT = "WRONG_VARIANT"
    SUPPLIER_CHANGED = "SUPPLIER_CHANGED"
    CAP_BINDS = "CAP_BINDS"
    LABOUR_RATE_OVER_CAP = "LABOUR_RATE_OVER_CAP"
    PARTIAL_COVERAGE = "PARTIAL_COVERAGE"
    ALREADY_RECOVERED_PART = "ALREADY_RECOVERED_PART"
    ALREADY_RECOVERED_FULL = "ALREADY_RECOVERED_FULL"
    MISSING_EVIDENCE = "MISSING_EVIDENCE"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    CURRENCY_MISMATCH = "CURRENCY_MISMATCH"


#: What each shape is built to produce. The generator computes the outcome from the rule and
#: refuses to write a corpus where the computed outcome disagrees with this table. That check is
#: the whole value of the table: it makes a construction that quietly stopped producing the case it
#: was written for into a build failure rather than into a silently easier evaluation.
_INTENDED: Final[dict[Shape, RecoveryOutcome]] = {
    Shape.CLEAN: RecoveryOutcome.RECOVERABLE,
    Shape.WINDOW_CLOSED: RecoveryOutcome.NOT_RECOVERABLE,
    Shape.EXPIRED_BY_ONE_DAY: RecoveryOutcome.NOT_RECOVERABLE,
    Shape.SERIAL_OUT_OF_RANGE: RecoveryOutcome.NOT_RECOVERABLE,
    Shape.PART_EXCLUDED: RecoveryOutcome.NOT_RECOVERABLE,
    Shape.WRONG_VARIANT: RecoveryOutcome.NOT_RECOVERABLE,
    Shape.SUPPLIER_CHANGED: RecoveryOutcome.NOT_RECOVERABLE,
    Shape.CAP_BINDS: RecoveryOutcome.PARTIALLY_RECOVERABLE,
    Shape.LABOUR_RATE_OVER_CAP: RecoveryOutcome.PARTIALLY_RECOVERABLE,
    Shape.PARTIAL_COVERAGE: RecoveryOutcome.PARTIALLY_RECOVERABLE,
    Shape.ALREADY_RECOVERED_PART: RecoveryOutcome.PARTIALLY_RECOVERABLE,
    Shape.ALREADY_RECOVERED_FULL: RecoveryOutcome.NOT_RECOVERABLE,
    Shape.MISSING_EVIDENCE: RecoveryOutcome.REVIEW,
    Shape.CONFLICTING_EVIDENCE: RecoveryOutcome.REVIEW,
    Shape.CURRENCY_MISMATCH: RecoveryOutcome.REVIEW,
}

#: Every adversarial construction the brief names, and the label carried on the claim record. The
#: ordinary claims carry `NOT_ADVERSARIAL`, which is a value rather than an absent field: a null
#: here would make "this claim is ordinary" and "nobody recorded what this claim is" the same
#: thing.
NOT_ADVERSARIAL: Final = "none"

ADVERSARIAL_KINDS: Final[tuple[str, ...]] = (
    "warranty_expired_one_day_before_failure",
    "correct_part_wrong_serial_range",
    "matching_name_conflicting_part_number",
    "already_recovered_once",
    "cap_lower_than_repair_amount",
    "partial_coverage",
    "exclusion_clause",
    "superseded_bulletin",
    "supplier_changed_after_manufacture_date",
    "missing_invoice",
    "duplicate_invoice",
    "ambiguous_manufacturer",
    "similar_part_family_wrong_exact_variant",
    "claim_amount_correct_but_currency_wrong",
)


class ClaimRecipe(NamedTuple):
    """One slot in every programme's claim set.

    The recipe is a fixed table rather than a seeded draw, and that is the point. Every programme
    gets the same forty constructions, so the corpus-wide outcome mix, the rejection-code mix and
    the adversarial coverage are arithmetic over the programme count instead of something the seed
    might or might not have produced. Adding a programme cannot quietly drop the only claim that
    exercised a construction; removing one cannot silently take a rejection code below its contract
    floor.
    """

    slot: int
    code: RejectionCode
    shape: Shape
    adversarial_kind: str
    evidence_key: str | None
    family_position: int
    variant_index: int


#: Forty claims per programme. Fourteen are the named adversarial constructions, one of each;
#: twenty-six are ordinary claims that spread the seven rejection codes and the four outcomes.
RECIPE: Final[tuple[ClaimRecipe, ...]] = (
    # --- the fourteen adversarial constructions ----------------------------------------------
    ClaimRecipe(
        1,
        RejectionCode.OUTSIDE_WARRANTY_PERIOD,
        Shape.EXPIRED_BY_ONE_DAY,
        "warranty_expired_one_day_before_failure",
        None,
        1,
        0,
    ),
    ClaimRecipe(
        2,
        RejectionCode.MISSING_SERIAL,
        Shape.SERIAL_OUT_OF_RANGE,
        "correct_part_wrong_serial_range",
        None,
        0,
        0,
    ),
    ClaimRecipe(
        3,
        RejectionCode.PART_NOT_COVERED,
        Shape.CONFLICTING_EVIDENCE,
        "matching_name_conflicting_part_number",
        "part_number_confirmation",
        0,
        1,
    ),
    ClaimRecipe(
        4,
        RejectionCode.DUPLICATE_CLAIM,
        Shape.ALREADY_RECOVERED_PART,
        "already_recovered_once",
        None,
        1,
        1,
    ),
    ClaimRecipe(
        5,
        RejectionCode.LABOUR_RATE_EXCEEDED,
        Shape.CAP_BINDS,
        "cap_lower_than_repair_amount",
        None,
        0,
        0,
    ),
    ClaimRecipe(
        6,
        RejectionCode.PART_NOT_COVERED,
        Shape.PARTIAL_COVERAGE,
        "partial_coverage",
        None,
        1,
        2,
    ),
    ClaimRecipe(
        7,
        RejectionCode.PART_NOT_COVERED,
        Shape.PART_EXCLUDED,
        "exclusion_clause",
        None,
        2,
        0,
    ),
    ClaimRecipe(
        8,
        RejectionCode.WRONG_FAILURE_CODE,
        Shape.CONFLICTING_EVIDENCE,
        "superseded_bulletin",
        "diagnostic_report",
        1,
        3,
    ),
    ClaimRecipe(
        9,
        RejectionCode.PART_NOT_COVERED,
        Shape.SUPPLIER_CHANGED,
        "supplier_changed_after_manufacture_date",
        None,
        3,
        1,
    ),
    ClaimRecipe(
        10,
        RejectionCode.MISSING_INSTALL_PROOF,
        Shape.MISSING_EVIDENCE,
        "missing_invoice",
        "repair_invoice",
        1,
        0,
    ),
    ClaimRecipe(
        11,
        RejectionCode.DUPLICATE_CLAIM,
        Shape.CONFLICTING_EVIDENCE,
        "duplicate_invoice",
        "repair_invoice",
        3,
        0,
    ),
    ClaimRecipe(
        12,
        RejectionCode.PART_NOT_COVERED,
        Shape.CONFLICTING_EVIDENCE,
        "ambiguous_manufacturer",
        "coverage_confirmation",
        3,
        2,
    ),
    ClaimRecipe(
        13,
        RejectionCode.PART_NOT_COVERED,
        Shape.WRONG_VARIANT,
        "similar_part_family_wrong_exact_variant",
        None,
        0,
        2,
    ),
    ClaimRecipe(
        14,
        RejectionCode.LABOUR_RATE_EXCEEDED,
        Shape.CURRENCY_MISMATCH,
        "claim_amount_correct_but_currency_wrong",
        None,
        0,
        0,
    ),
    # --- twelve ordinary claims the manufacturer rejected and the evidence now answers ---------
    ClaimRecipe(15, RejectionCode.MISSING_SERIAL, Shape.CLEAN, NOT_ADVERSARIAL, None, 1, 0),
    ClaimRecipe(16, RejectionCode.MISSING_SERIAL, Shape.CLEAN, NOT_ADVERSARIAL, None, 0, 1),
    ClaimRecipe(17, RejectionCode.MISSING_SERIAL, Shape.CLEAN, NOT_ADVERSARIAL, None, 3, 0),
    ClaimRecipe(18, RejectionCode.MISSING_INSTALL_PROOF, Shape.CLEAN, NOT_ADVERSARIAL, None, 1, 1),
    ClaimRecipe(19, RejectionCode.MISSING_INSTALL_PROOF, Shape.CLEAN, NOT_ADVERSARIAL, None, 0, 0),
    ClaimRecipe(20, RejectionCode.MISSING_INSTALL_PROOF, Shape.CLEAN, NOT_ADVERSARIAL, None, 3, 2),
    ClaimRecipe(21, RejectionCode.WRONG_FAILURE_CODE, Shape.CLEAN, NOT_ADVERSARIAL, None, 1, 2),
    ClaimRecipe(22, RejectionCode.WRONG_FAILURE_CODE, Shape.CLEAN, NOT_ADVERSARIAL, None, 0, 1),
    ClaimRecipe(23, RejectionCode.WRONG_FAILURE_CODE, Shape.CLEAN, NOT_ADVERSARIAL, None, 3, 3),
    ClaimRecipe(
        24, RejectionCode.OUTSIDE_WARRANTY_PERIOD, Shape.CLEAN, NOT_ADVERSARIAL, None, 1, 3
    ),
    ClaimRecipe(
        25, RejectionCode.OUTSIDE_WARRANTY_PERIOD, Shape.CLEAN, NOT_ADVERSARIAL, None, 0, 0
    ),
    ClaimRecipe(
        26, RejectionCode.OUTSIDE_WARRANTY_PERIOD, Shape.CLEAN, NOT_ADVERSARIAL, None, 3, 0
    ),
    # --- three whose correction window had already shut: kill condition M's denominator --------
    ClaimRecipe(27, RejectionCode.MISSING_SERIAL, Shape.WINDOW_CLOSED, NOT_ADVERSARIAL, None, 1, 0),
    ClaimRecipe(
        28, RejectionCode.WRONG_FAILURE_CODE, Shape.WINDOW_CLOSED, NOT_ADVERSARIAL, None, 0, 1
    ),
    ClaimRecipe(
        29, RejectionCode.MISSING_INSTALL_PROOF, Shape.WINDOW_CLOSED, NOT_ADVERSARIAL, None, 3, 2
    ),
    # --- four where the rate exceeds the cap, one of them on the amended family ----------------
    ClaimRecipe(
        30,
        RejectionCode.LABOUR_RATE_EXCEEDED,
        Shape.LABOUR_RATE_OVER_CAP,
        NOT_ADVERSARIAL,
        None,
        0,
        0,
    ),
    ClaimRecipe(
        31,
        RejectionCode.LABOUR_RATE_EXCEEDED,
        Shape.LABOUR_RATE_OVER_CAP,
        NOT_ADVERSARIAL,
        None,
        0,
        1,
    ),
    ClaimRecipe(
        32,
        RejectionCode.LABOUR_RATE_EXCEEDED,
        Shape.LABOUR_RATE_OVER_CAP,
        NOT_ADVERSARIAL,
        None,
        1,
        0,
    ),
    ClaimRecipe(
        33,
        RejectionCode.LABOUR_RATE_EXCEEDED,
        Shape.LABOUR_RATE_OVER_CAP,
        NOT_ADVERSARIAL,
        None,
        3,
        3,
    ),
    # --- three where the per-claim ceiling bites ------------------------------------------------
    ClaimRecipe(
        34, RejectionCode.OUTSIDE_WARRANTY_PERIOD, Shape.CAP_BINDS, NOT_ADVERSARIAL, None, 1, 1
    ),
    ClaimRecipe(
        35, RejectionCode.MISSING_INSTALL_PROOF, Shape.CAP_BINDS, NOT_ADVERSARIAL, None, 0, 1
    ),
    ClaimRecipe(36, RejectionCode.WRONG_FAILURE_CODE, Shape.CAP_BINDS, NOT_ADVERSARIAL, None, 3, 0),
    # --- two already recovered in full: a positive claim with nothing left to recover -----------
    ClaimRecipe(
        37, RejectionCode.DUPLICATE_CLAIM, Shape.ALREADY_RECOVERED_FULL, NOT_ADVERSARIAL, None, 1, 2
    ),
    ClaimRecipe(
        38, RejectionCode.DUPLICATE_CLAIM, Shape.ALREADY_RECOVERED_FULL, NOT_ADVERSARIAL, None, 0, 0
    ),
    # --- two whose evidence is simply not there -------------------------------------------------
    ClaimRecipe(
        39,
        RejectionCode.MISSING_SERIAL,
        Shape.MISSING_EVIDENCE,
        NOT_ADVERSARIAL,
        "serial_plate_photo",
        1,
        3,
    ),
    ClaimRecipe(
        40,
        RejectionCode.MISSING_INSTALL_PROOF,
        Shape.MISSING_EVIDENCE,
        NOT_ADVERSARIAL,
        "installation_certificate",
        1,
        1,
    ),
)

#: The fraction of the per-claim ceiling each shape aims the eligible amount at, above the
#: deductible. Below one, the ceiling does not bite; above it, it does. Written as a table so that a
#: reader can see at a glance which constructions are meant to hit the cap.
_ELIGIBLE_TARGET: Final[dict[Shape, Decimal]] = {
    Shape.CAP_BINDS: Decimal("1.6"),
    Shape.PARTIAL_COVERAGE: Decimal("1.4"),
}
_DEFAULT_TARGET: Final = Decimal("0.5")

#: Shapes whose claimed labour rate is above the programme cap.
_RATE_ABOVE_CAP: Final[frozenset[Shape]] = frozenset(
    {Shape.LABOUR_RATE_OVER_CAP, Shape.PARTIAL_COVERAGE}
)

#: Where a mismatched claim's currency comes from. A cycle rather than a draw, so the wrong currency
#: is a fixed function of the right one and no programme's adversarial claim is accidentally
#: denominated in its own currency.
_NEXT_CURRENCY: Final[dict[Currency, Currency]] = {
    Currency.GBP: Currency.EUR,
    Currency.EUR: Currency.USD,
    Currency.USD: Currency.GBP,
}

#: The window failure and in-service dates are drawn from. Fixed years and days 2..28 only:
#: month-end clamping would make `add_months` non-invertible, and the "expired one day before the
#: failure" construction needs it to be exactly invertible or the boundary it sets is not the
#: boundary it claims.
_FAILURE_YEARS: Final[tuple[int, ...]] = (2024, 2025)
_FIRST_DAY: Final = 2
_LAST_DAY: Final = 28

#: The smallest parts figure the generator will emit. It is never reached under the ranges this
#: corpus draws from — the assertion in `_amounts` says so — and exists to make the floor explicit
#: rather than implicit in four arithmetic ranges a reader would have to compose in their head.
_MINIMUM_PARTS_MINOR: Final = 2_500


class GroundTruthMoney(NamedTuple):
    """Every step of the money, kept rather than collapsed into a total.

    The intermediates are what makes a disagreement about an outcome a disagreement about a number.
    A truth file that published only the recoverable amount would leave anyone investigating a
    mismatch to reconstruct four subtractions from the claim, and the one they cannot reconstruct is
    the one that was wrong.
    """

    claimed_total: Money
    labour_excess: Money
    uncovered_parts: Money
    eligible_amount: Money
    deductible: Money
    capped_amount: Money
    already_recovered: Money
    recoverable_amount: Money

    @property
    def anything_excluded(self) -> bool:
        zero = Money.zero(self.recoverable_amount.currency)
        return (
            self.labour_excess > zero
            or self.uncovered_parts > zero
            or self.capped_amount > zero
            or self.already_recovered > zero
        )


class BuiltClaim(NamedTuple):
    """A claim, the facts that decided it, and the answer the generator built it to have."""

    claim: Claim
    program_id: str
    slot: int
    shape: Shape
    adversarial_kind: str
    as_of: date
    closes_on: date
    part_family_id: str
    family_position: int
    evidence: tuple[tuple[str, str], ...]
    missing_requirements: tuple[str, ...]
    currency_matches_program: bool
    window_open: bool
    within_warranty: bool
    part_covered: bool
    serial_in_range: bool
    money: GroundTruthMoney | None
    outcome: RecoveryOutcome
    recoverable_amount: Money
    governing_clause_id: str


# ----------------------------------------------------------------------------------- date helpers


def add_months(start: date, months: int) -> date:
    """Calendar month arithmetic, clamping to the end of the target month.

    The clamp is here for correctness rather than for use: every date this corpus feeds it has a day
    between 2 and 28, so no clamp ever fires and `add_months(add_months(d, -n), n) == d` holds
    exactly. That invertibility is what the one-day-outside-cover construction rests on, and the
    generator asserts it rather than trusting the ranges to stay as they are.
    """
    total = start.month - 1 + months
    year = start.year + total // 12
    month = total % 12 + 1
    day = min(start.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def within_warranty_period(in_service: date, failure: date, months: int) -> bool:
    """Half-open: cover runs from the in-service date up to, and excluding, the month anniversary.

    Half-open because two adjacent cover periods must partition the calendar, which they only do if
    the day one ends is the day the next begins. Every claim in this corpus is either months inside
    the period or a day outside the exclusive bound, so no ground truth here depends on this choice
    — stated so that a reader can check that claim rather than take it.
    """
    return failure < add_months(in_service, months)


def serial_is_in_range(serial: str | None, first: int | None, last: int | None) -> bool:
    """Whether the claimed serial falls in the programme's declared range for the part.

    A part with no declared range covers every serial, including an unknown one. A part *with* a
    declared range and a claim with no serial cannot be adjudicated, and this returns `False` — but
    the generator never emits that combination, because the honest answer there is "nobody knows"
    and a boolean cannot say it.
    """
    if first is None or last is None:
        return True
    if serial is None:
        return False
    return first <= int(serial) <= last


# ----------------------------------------------------------------------------------- the money


def recovery_of(
    claim: Claim,
    *,
    labour_rate_cap: Money,
    deductible: Money,
    claim_cap: Money,
    part_covered: bool,
    serial_in_range: bool,
) -> GroundTruthMoney:
    """The recovery arithmetic, in the order ADR-001 and `money.py` fix it.

    The order is the argument and it is not interchangeable. Exclusions come off first, because an
    amount the policy never covered was never part of the claim. The deductible comes next, because
    it applies to what is eligible and not to what was asked for. The ceiling comes last, because a
    ceiling applied before the deductible would defeat the deductible entirely — `money.py` names
    that as failure 3. Every intermediate keeps full precision and `quantize` is called once, here,
    at the end, on every field.
    """
    currency = claim.claimed_parts.currency
    zero = Money.zero(currency)

    over_cap = (claim.claimed_labour_rate - labour_rate_cap).floored_at_zero()
    labour_excess = over_cap * claim.claimed_labour_hours
    uncovered = zero if part_covered and serial_in_range else claim.claimed_parts

    eligible = claim.claimed_total - labour_excess - uncovered
    after_deductible = (eligible - deductible).floored_at_zero()
    capped = (after_deductible - claim_cap).floored_at_zero()
    already = claim.previously_recovered or zero
    recoverable = (after_deductible.clamped_to(claim_cap) - already).floored_at_zero()

    return GroundTruthMoney(
        claimed_total=claim.claimed_total.quantize(),
        labour_excess=labour_excess.quantize(),
        uncovered_parts=uncovered.quantize(),
        eligible_amount=eligible.quantize(),
        deductible=deductible.quantize(),
        capped_amount=capped.quantize(),
        already_recovered=already.quantize(),
        recoverable_amount=recoverable.quantize(),
    )


def outcome_for(  # noqa: PLR0911
    *,
    currency_matches_program: bool,
    window_open: bool,
    within_warranty: bool,
    part_covered: bool,
    serial_in_range: bool,
    money: GroundTruthMoney | None,
    evidence_complete: bool,
) -> RecoveryOutcome:
    """`GROUND_TRUTH_RULE`, and its only implementation. Ordered, first match wins.

    PLR0911 is suppressed rather than satisfied. The rule has eight ordered clauses and each one
    returns; folding two of them together to please a branch counter would put two unrelated
    conditions behind one `return` and make the next person guess which of them fired. A table of
    predicate-and-outcome pairs was considered and rejected: it passes the counter and costs the
    reader the order, which is the only thing about this function that is load-bearing.
    """
    if not currency_matches_program or money is None:
        return RecoveryOutcome.REVIEW
    if not window_open:
        return RecoveryOutcome.NOT_RECOVERABLE
    if not within_warranty:
        return RecoveryOutcome.NOT_RECOVERABLE
    if not part_covered or not serial_in_range:
        return RecoveryOutcome.NOT_RECOVERABLE
    if money.recoverable_amount == Money.zero(money.recoverable_amount.currency):
        return RecoveryOutcome.NOT_RECOVERABLE
    if not evidence_complete:
        return RecoveryOutcome.REVIEW
    if money.anything_excluded:
        return RecoveryOutcome.PARTIALLY_RECOVERABLE
    return RecoveryOutcome.RECOVERABLE


# ----------------------------------------------------------------------------------- construction


def _minor(amount: Money) -> int:
    """Whole minor units of an exact amount.

    Every figure this corpus draws is exact at two decimal places, so the multiplication is
    exact and the `int` truncates nothing. A figure that was not would silently lose its
    third place here rather than failing, which is why `quantize` is applied first.
    """
    return int(amount.quantize().amount * 100)


def _from_minor(minor: int, currency: Currency) -> Money:
    return Money(Decimal(minor).scaleb(-2), currency)


def _dates(spec: ProgramSpec, recipe: ClaimRecipe, claim_id: str) -> tuple[date, date, date, date]:
    """In-service, failure, repair-invoice and rejection dates, in that order."""
    program = spec.program
    draw = stream("claim-dates", program.program_id, claim_id)
    failure = date(
        draw.choice(_FAILURE_YEARS),
        draw.randrange(1, 13),
        draw.randrange(_FIRST_DAY, _LAST_DAY + 1),
    )

    if recipe.shape is Shape.EXPIRED_BY_ONE_DAY:
        # Cover ends the day before the failure. Both the half-open and the closed reading of the
        # warranty period call this failure outside cover, which is why the boundary is set a day
        # early rather than on the failure date itself.
        bound = failure - timedelta(days=1)
        in_service = add_months(bound, -program.warranty_months)
    else:
        # Comfortably inside: at least two whole months of cover left at the failure date.
        elapsed = draw.randrange(1, program.warranty_months - 1)
        in_service = add_months(failure, -elapsed)

    repair = failure + timedelta(days=draw.randrange(2, 12))
    rejected = repair + timedelta(days=draw.randrange(10, 45))
    return in_service, failure, repair, rejected


def _as_of(spec: ProgramSpec, recipe: ClaimRecipe, claim_id: str, rejected_on: date) -> date:
    """The date the system reasons as of. Explicit and stored; never `date.today()`.

    A corpus whose window arithmetic reads the wall clock produces a different answer every day it
    is evaluated, and kill condition M — nothing resubmitted after the window closed — would then
    pass or fail depending on when the build ran.
    """
    window = spec.program.correction_window_days
    draw = stream("claim-as-of", spec.program.program_id, claim_id)
    if recipe.shape is Shape.WINDOW_CLOSED:
        return rejected_on + timedelta(days=window + draw.randrange(3, 20))
    return rejected_on + timedelta(days=draw.randrange(1, max(2, window - 5)))


def _amounts(spec: ProgramSpec, recipe: ClaimRecipe, claim_id: str) -> tuple[Money, int, Money]:
    """Parts, labour hours and labour rate, in the programme's currency.

    Every figure is drawn in whole minor units and converted once, so no decimal string in this
    corpus was ever a float. The parts figure is solved backwards from the eligible amount the shape
    is aiming at, which is what lets `_ELIGIBLE_TARGET` decide whether the per-claim ceiling bites
    instead of leaving it to whatever the draw happened to produce.
    """
    program = spec.program
    currency = program.currency
    draw = stream("claim-amounts", program.program_id, claim_id)

    hours = draw.randrange(2, 7)
    cap_minor = _minor(program.labour_rate_cap_per_hour)
    if recipe.shape in _RATE_ABOVE_CAP:
        rate_minor = cap_minor + draw.randrange(300, 1_550, 50)
    else:
        rate_minor = cap_minor - draw.randrange(0, 850, 50)
    rate = _from_minor(rate_minor, currency)

    fraction = _ELIGIBLE_TARGET.get(recipe.shape, _DEFAULT_TARGET)
    target = program.deductible + program.claim_cap * fraction
    parts_minor = _minor(target) - cap_minor * hours
    if parts_minor < _MINIMUM_PARTS_MINOR:  # pragma: no cover - guarded by the ranges above
        raise ValueError(
            f"{claim_id}: the labour bill at the capped rate exceeds the eligible amount this "
            f"shape "
            f"targets, so the parts figure would have to be negative. The programme ranges in "
            f"programs.py are meant to make that impossible; one of them has moved."
        )
    return _from_minor(parts_minor, currency), hours, rate


def _serial(spec: ProgramSpec, recipe: ClaimRecipe, claim_id: str, part: str) -> str | None:
    row = spec.coverage_for(part)
    if recipe.evidence_key == "serial_plate_photo":
        # The one construction where the serial is genuinely unknown. It is routed at a part with no
        # declared range on purpose; see `serial_is_in_range`.
        return None
    draw = stream("claim-serial", spec.program.program_id, claim_id)
    if row.serial_first is None or row.serial_last is None:
        return str(4_000_000 + draw.randrange(0, 900_000))
    if recipe.shape is Shape.SERIAL_OUT_OF_RANGE:
        return str(row.serial_last + draw.randrange(1, 500))
    return str(row.serial_first + draw.randrange(0, 50_000))


def _evidence(recipe: ClaimRecipe) -> tuple[tuple[str, str], ...]:
    """The claim's evidence bundle over the whole closed vocabulary, in key order.

    A tuple of pairs rather than a dictionary, because the serialised order must be fixed and the
    iteration order of a dictionary built from a draw is a decision nobody made.
    """
    status = dict.fromkeys(EVIDENCE_KEYS, EvidenceStatus.PRESENT)
    if recipe.evidence_key is not None:
        status[recipe.evidence_key] = (
            EvidenceStatus.ABSENT
            if recipe.shape is Shape.MISSING_EVIDENCE
            else EvidenceStatus.CONFLICTING
        )
    return tuple((key, status[key].value) for key in EVIDENCE_KEYS)


def _prior_recovery(shape: Shape, after_deductible_capped: Money) -> Money | None:
    """What has already been recovered against this identity, if anything.

    Computed in whole minor units so the value entering the claim is exact at two places. It is an
    *input* to the claim rather than an intermediate of the computation, which is why it is rounded
    here and not at the end: `recovery_of` quantises what it computes, and what it is given must
    already be a real amount somebody could have been paid.
    """
    if shape is Shape.ALREADY_RECOVERED_FULL:
        return after_deductible_capped
    if shape is Shape.ALREADY_RECOVERED_PART:
        minor = _minor(after_deductible_capped) * 2 // 5
        return _from_minor(minor, after_deductible_capped.currency)
    return None


def _coverage_facts(spec: ProgramSpec, part: str, serial: str | None) -> tuple[bool, bool]:
    row = spec.coverage_for(part)
    return row.covered, serial_is_in_range(serial, row.serial_first, row.serial_last)


def build_claims(spec: ProgramSpec, documents: tuple[BuiltDocument, ...]) -> tuple[BuiltClaim, ...]:
    """Every claim of one programme, in recipe order, with the answer it was built to have."""
    return tuple(_build_one(spec, documents, recipe) for recipe in RECIPE)


def _build_one(
    spec: ProgramSpec, documents: tuple[BuiltDocument, ...], recipe: ClaimRecipe
) -> BuiltClaim:
    program = spec.program
    program_id = program.program_id
    claim_id = f"{program_id}-c{recipe.slot:02d}"

    family = spec.families[recipe.family_position]
    part = f"{family.family_id}-{VARIANTS[recipe.variant_index]}"

    in_service, failure, repair, rejected = _dates(spec, recipe, claim_id)
    as_of = _as_of(spec, recipe, claim_id, rejected)
    closes_on = rejected + timedelta(days=program.correction_window_days)

    parts, hours, rate = _amounts(spec, recipe, claim_id)
    serial = _serial(spec, recipe, claim_id, part)
    part_covered, serial_in_range = _coverage_facts(spec, part, serial)

    mismatched = recipe.shape is Shape.CURRENCY_MISMATCH
    currency = _NEXT_CURRENCY[program.currency] if mismatched else program.currency
    if mismatched:
        # The amounts are right and the currency is wrong. That is the whole defect: a claim whose
        # figures reconcile against the invoice and whose denomination reconciles against nothing.
        parts = Money(parts.amount, currency)
        rate = Money(rate.amount, currency)

    prior = _prior_recovery(
        recipe.shape,
        _provisional_after_deductible(
            parts=parts,
            hours=hours,
            rate=rate,
            program_deductible=program.deductible,
            program_cap=program.claim_cap,
            labour_rate_cap=program.labour_rate_cap_per_hour,
            part_covered=part_covered,
            serial_in_range=serial_in_range,
            mismatched=mismatched,
        ),
    )

    claim = Claim(
        claim_id=claim_id,
        program_id=program_id,
        part_number=part,
        serial_number=serial,
        in_service_date=in_service,
        failure_date=failure,
        repair_invoice_date=repair,
        rejection_code=recipe.code,
        rejected_on=rejected,
        claimed_parts=parts,
        claimed_labour_hours=hours,
        claimed_labour_rate=rate,
        previously_recovered=prior,
    )

    money = (
        None
        if mismatched
        else recovery_of(
            claim,
            labour_rate_cap=program.labour_rate_cap_per_hour,
            deductible=program.deductible,
            claim_cap=program.claim_cap,
            part_covered=part_covered,
            serial_in_range=serial_in_range,
        )
    )
    evidence = _evidence(recipe)
    missing = tuple(key for key, status in evidence if status != EvidenceStatus.PRESENT.value)
    window_open = as_of <= closes_on
    within = within_warranty_period(in_service, failure, program.warranty_months)

    outcome = outcome_for(
        currency_matches_program=not mismatched,
        window_open=window_open,
        within_warranty=within,
        part_covered=part_covered,
        serial_in_range=serial_in_range,
        money=money,
        evidence_complete=not missing,
    )
    recoverable = money.recoverable_amount if money is not None else Money.zero(currency)

    return BuiltClaim(
        claim=claim,
        program_id=program_id,
        slot=recipe.slot,
        shape=recipe.shape,
        adversarial_kind=recipe.adversarial_kind,
        as_of=as_of,
        closes_on=closes_on,
        part_family_id=family.family_id,
        family_position=recipe.family_position,
        evidence=evidence,
        missing_requirements=missing,
        currency_matches_program=not mismatched,
        window_open=window_open,
        within_warranty=within,
        part_covered=part_covered,
        serial_in_range=serial_in_range,
        money=money,
        outcome=outcome,
        recoverable_amount=recoverable,
        governing_clause_id=governing_clause_id(
            documents, code=recipe.code, family_position=recipe.family_position
        ),
    )


def _provisional_after_deductible(
    *,
    parts: Money,
    hours: int,
    rate: Money,
    program_deductible: Money,
    program_cap: Money,
    labour_rate_cap: Money,
    part_covered: bool,
    serial_in_range: bool,
    mismatched: bool,
) -> Money:
    """What would be settled before any prior recovery is deducted.

    The prior recovery has to be a real fraction of the settlement it is a prior recovery *of*, and
    the settlement depends on the claim's own figures, so this repeats the first three steps of
    `recovery_of` before the claim object exists. It deliberately stops at the ceiling and does not
    subtract anything: continuing would be the fourth step, and the fourth step is the one this
    value is an input to.
    """
    currency = parts.currency
    if mismatched:
        # A mismatched claim never reaches the money path at all, and computing a figure here would
        # mean comparing two currencies this system holds no rate for.
        return Money.zero(currency)
    over_cap = (rate - labour_rate_cap).floored_at_zero()
    labour_excess = over_cap * hours
    uncovered = Money.zero(currency) if part_covered and serial_in_range else parts
    eligible = parts + rate * hours - labour_excess - uncovered
    after = (eligible - program_deductible).floored_at_zero()
    return after.clamped_to(program_cap).quantize()


def intended_outcome(shape: Shape) -> RecoveryOutcome:
    """What the shape was written to produce. `generate.py` refuses a corpus that disagrees."""
    return _INTENDED[shape]
