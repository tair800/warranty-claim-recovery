"""The recovery arithmetic, tested at the five places a warranty team gets a different number.

Nobody disputes that 400 plus 200 is 600. The disputes are about sequence: whether the cap came
before or after the deductible, whether the labour excess was taken off the rate or the hours,
whether the prior recovery was subtracted from the capped figure or the gross, and whether the
rounding happened once or four times. Every test below pins one of those, and several of them
assert the number the *wrong* order would have produced, so that a future refactor which quietly
swaps two lines fails with a message that says which two.

Kill condition F is exact `Decimal` equality against the corpus generator's ground truth, with no
tolerance. That is the standard these tests hold the arithmetic to: an amount that is right to the
nearest penny by the wrong route is a failure here, not a rounding difference.
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

import pytest

from warranty_claim_recovery.domain import (
    Claim,
    RecoveryComputation,
    RejectionCode,
    WarrantyProgram,
)
from warranty_claim_recovery.eligibility import Eligibility, PartCoverage, assess
from warranty_claim_recovery.money import Currency, CurrencyMismatchError, Money
from warranty_claim_recovery.recovery import compute, finalise

GBP = Currency.GBP
EUR = Currency.EUR

PART = "AB-1234-C"

#: Every amount in a finalised computation is quantised to the currency's minor unit, which for
#: all three currencies in this corpus is two places. Asserted as an exponent rather than by
#: comparing strings, because `Decimal("200.0")` and `Decimal("200.00")` compare equal and only one
#: of them is a rounded amount.
MINOR_UNIT_EXPONENT = -2


def make_program(
    *,
    currency: Currency = GBP,
    deductible: str = "50.00",
    claim_cap: str = "5000.00",
    labour_rate_cap: str = "55.00",
) -> WarrantyProgram:
    return WarrantyProgram(
        program_id="PRG-ACME-V2",
        manufacturer="Acme Drivetrain (synthetic)",
        policy_version="v2",
        currency=currency,
        correction_window_days=30,
        warranty_months=24,
        labour_rate_cap_per_hour=Money(labour_rate_cap, currency),
        deductible=Money(deductible, currency),
        claim_cap=Money(claim_cap, currency),
    )


def make_claim(
    *,
    currency: Currency = GBP,
    parts: str = "400.00",
    labour_hours: int = 4,
    labour_rate: str = "50.00",
    previously_recovered: str | None = None,
) -> Claim:
    failure = date(2025, 6, 10)
    invoiced = failure + timedelta(days=2)
    return Claim(
        claim_id="CLM-0001",
        program_id="PRG-ACME-V2",
        part_number=PART,
        serial_number="SN-004821",
        in_service_date=date(2024, 1, 15),
        failure_date=failure,
        repair_invoice_date=invoiced,
        rejection_code=RejectionCode.MISSING_SERIAL,
        rejected_on=invoiced + timedelta(days=14),
        claimed_parts=Money(Decimal(parts), currency),
        claimed_labour_hours=labour_hours,
        claimed_labour_rate=Money(Decimal(labour_rate), currency),
        previously_recovered=(
            None if previously_recovered is None else Money(Decimal(previously_recovered), currency)
        ),
    )


def covered(*, is_covered: bool = True) -> PartCoverage:
    return PartCoverage(part_number=PART, covered=is_covered, serial_first=None, serial_last=None)


def recover(
    claim: Claim, program: WarrantyProgram, *, is_covered: bool = True
) -> RecoveryComputation:
    """The whole deterministic money path, from a claim to a finalised computation."""
    return compute(claim, program, assess(claim, program, covered(is_covered=is_covered)))


# ------------------------------------------------------------------------------ the happy path


def test_a_fully_covered_claim_recovers_everything_but_the_deductible() -> None:
    computation = recover(make_claim(), make_program())

    assert computation.claimed_total == Money("600.00", GBP)
    assert computation.eligible_amount == Money("600.00", GBP)
    assert computation.deductible == Money("50.00", GBP)
    assert computation.capped_amount == Money.zero(GBP)
    assert computation.recoverable_amount == Money("550.00", GBP)


def test_a_programme_with_no_deductible_recovers_the_whole_claim() -> None:
    """The only shape that produces a full rather than a partial recovery at the gate.

    Worth its own test because it is the one case where `excluded_amount` is zero, and the gate
    distinguishes `RECOVERABLE` from `PARTIALLY_RECOVERABLE` on exactly that subtraction.
    """
    computation = recover(make_claim(), make_program(deductible="0.00"))

    assert computation.recoverable_amount == computation.claimed_total
    assert computation.excluded_amount == Money.zero(GBP)


def test_the_excluded_amount_accounts_for_every_penny_not_recovered() -> None:
    """An adjudicator who cannot see the subtraction refuses the package rather than rebuilding it.

    Claimed 680.00: 60.00 of labour above the rate cap and a 50.00 deductible, so 570.00 is
    recoverable and 110.00 is excluded. The two have to add back to the claimed total or the
    package does not balance.
    """
    computation = recover(make_claim(labour_rate="70.00"), make_program())

    assert computation.claimed_total == Money("680.00", GBP)
    assert computation.labour_excess == Money("60.00", GBP)
    assert computation.recoverable_amount == Money("570.00", GBP)
    assert computation.excluded_amount == Money("110.00", GBP)
    assert computation.recoverable_amount + computation.excluded_amount == computation.claimed_total


# ------------------------------------------------------- the cap, and the order it is applied in


def test_a_cap_lower_than_the_repair_produces_a_partial_recovery() -> None:
    """The adversarial case the brief names. The cap binds and what it removed is recorded.

    Eligible 600.00, less a 50.00 deductible is 550.00, capped at 500.00. The cap removed 50.00,
    and `capped_amount` is that 50.00 rather than the cap itself — a recovery package has to show
    what was taken out, and the cap is already on the programme.
    """
    computation = recover(make_claim(), make_program(claim_cap="500.00"))

    assert computation.capped_amount == Money("50.00", GBP)
    assert computation.recoverable_amount == Money("500.00", GBP)


def test_the_cap_is_applied_after_the_deductible_and_not_before() -> None:
    """The one ordering error that silently pays the manufacturer's money back to it.

    Eligible 600.00, deductible 50.00, cap 500.00. In this order the recovery is 500.00. Applying
    the cap to the gross first gives 600 clamped to 500, then less the deductible, which is 450.00
    — and the 50.00 difference is a deductible charged twice.
    """
    computation = recover(make_claim(), make_program(claim_cap="500.00"))

    assert computation.recoverable_amount == Money("500.00", GBP)
    assert computation.recoverable_amount != Money("450.00", GBP)


def test_a_cap_that_does_not_bind_removes_nothing() -> None:
    computation = recover(make_claim(), make_program(claim_cap="5000.00"))

    assert computation.capped_amount == Money.zero(GBP)


def test_a_claim_exactly_at_the_cap_is_not_reduced() -> None:
    """The boundary. Off by one here and every claim written to the cap loses a penny."""
    computation = recover(make_claim(), make_program(deductible="0.00", claim_cap="600.00"))

    assert computation.capped_amount == Money.zero(GBP)
    assert computation.recoverable_amount == Money("600.00", GBP)


# ----------------------------------------------------------------- the deductible, and the floor


def test_a_deductible_larger_than_the_eligible_amount_recovers_zero_and_never_less() -> None:
    """The adversarial case the brief names, and the fourth failure `money.py` is shaped by.

    A negative recoverable amount is a claim by the manufacturer against the distributor, which is
    not a thing this system is entitled to construct. `RecoveryComputation` refuses one at its
    validator; the floor in `compute` is what stops one being built.
    """
    computation = recover(make_claim(), make_program(deductible="1000.00"))

    assert computation.recoverable_amount == Money.zero(GBP)
    assert computation.recoverable_amount.amount == Decimal("0.00")
    assert computation.capped_amount == Money.zero(GBP)


def test_a_deductible_exactly_equal_to_the_eligible_amount_recovers_zero() -> None:
    computation = recover(make_claim(), make_program(deductible="600.00"))

    assert computation.recoverable_amount == Money.zero(GBP)


# ------------------------------------------------------------------------------ prior recovery


def test_a_claim_already_recovered_once_recovers_only_the_remainder() -> None:
    """The adversarial case the brief names. 550.00 was recoverable, 300.00 has been paid."""
    computation = recover(make_claim(previously_recovered="300.00"), make_program())

    assert computation.already_recovered == Money("300.00", GBP)
    assert computation.recoverable_amount == Money("250.00", GBP)


def test_a_prior_recovery_larger_than_the_remainder_leaves_nothing_and_not_a_debt() -> None:
    computation = recover(make_claim(previously_recovered="900.00"), make_program())

    assert computation.recoverable_amount == Money.zero(GBP)


def test_the_prior_recovery_is_subtracted_after_the_cap_and_not_before() -> None:
    """Eligible 600.00, deductible 50.00, cap 500.00, already paid 200.00.

    In this order: 550.00 capped to 500.00, less 200.00 paid, is 300.00. Subtracting the prior
    recovery first would give 350.00 capped to 350.00 — money claimed twice against a cap the
    manufacturer applies to the claim as a whole.
    """
    computation = recover(
        make_claim(previously_recovered="200.00"), make_program(claim_cap="500.00")
    )

    assert computation.recoverable_amount == Money("300.00", GBP)
    assert computation.recoverable_amount != Money("350.00", GBP)


def test_a_recorded_nil_prior_recovery_changes_nothing() -> None:
    computation = recover(make_claim(previously_recovered="0.00"), make_program())

    assert computation.already_recovered == Money.zero(GBP)
    assert computation.recoverable_amount == Money("550.00", GBP)


# ----------------------------------------------------------------------------- the exclusions


def test_an_uncovered_part_is_excluded_and_the_labour_still_recovers() -> None:
    """Parts 400.00 uncovered, labour 200.00 covered, deductible 50.00, so 150.00 recovers.

    The exclusion is shown rather than the claim refused: writing the whole claim off because one
    component is outside the schedule is money the manufacturer would have paid for the labour.
    """
    computation = recover(make_claim(), make_program(), is_covered=False)

    assert computation.uncovered_parts == Money("400.00", GBP)
    assert computation.eligible_amount == Money("200.00", GBP)
    assert computation.recoverable_amount == Money("150.00", GBP)


def test_the_labour_excess_is_taken_off_the_rate_over_every_hour_charged() -> None:
    """Rate 70.00 against a cap of 55.00 over four hours is 60.00, not one refused hour."""
    computation = recover(make_claim(labour_rate="70.00", labour_hours=4), make_program())

    assert computation.labour_excess == Money("60.00", GBP)
    assert computation.eligible_amount == Money("620.00", GBP)


def test_both_exclusions_apply_together_without_double_counting() -> None:
    """Parts 400.00 uncovered and 60.00 of labour excess on a claimed total of 680.00.

    The two are disjoint components of the claim — one is all of the parts, the other is part of
    the labour — so the eligible amount is 220.00 and cannot go negative however they combine.
    """
    computation = recover(make_claim(labour_rate="70.00"), make_program(), is_covered=False)

    assert computation.uncovered_parts == Money("400.00", GBP)
    assert computation.labour_excess == Money("60.00", GBP)
    assert computation.eligible_amount == Money("220.00", GBP)
    assert computation.recoverable_amount == Money("170.00", GBP)


# --------------------------------------------------------------------------------- rounding


def test_rounding_happens_once_at_the_end_and_not_on_each_component() -> None:
    """The drift kill condition F fails on, built from amounts that each round up on their own.

    Parts of 100.005 and labour of 3 hours at 33.333 is a claimed total of 200.004, which rounds to
    200.00. Rounding the two components first gives 100.01 and 100.00, whose sum is 200.01 — right
    to the nearest penny, and a penny away from the generator's ground truth.
    """
    computation = recover(
        make_claim(parts="100.005", labour_hours=3, labour_rate="33.333"),
        make_program(deductible="0.00"),
    )

    assert computation.claimed_total == Money("200.00", GBP)
    assert computation.recoverable_amount == Money("200.00", GBP)
    assert computation.recoverable_amount != Money("200.01", GBP)


def test_every_field_of_a_finalised_computation_is_quantised() -> None:
    """Not just the total. A package that shows a rounded total beside full-precision components
    is a package whose arithmetic does not add up on the adjudicator's screen."""
    computation = recover(
        make_claim(parts="100.005", labour_hours=3, labour_rate="33.333"),
        make_program(deductible="0.005"),
    )

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
        amount: Money = getattr(computation, name)
        assert amount.amount.as_tuple().exponent == MINOR_UNIT_EXPONENT, name


def test_finalise_rounds_what_it_is_given_and_nothing_else_does() -> None:
    """`finalise` called directly, because it is the single rounding point the whole design rests
    on and a test that only ever reaches it through `compute` would not notice it being bypassed."""
    unrounded = Money("1.005", GBP)

    computation = finalise(
        currency=GBP,
        claimed_total=unrounded,
        labour_excess=Money.zero(GBP),
        uncovered_parts=Money.zero(GBP),
        eligible_amount=unrounded,
        deductible=Money.zero(GBP),
        capped_amount=Money.zero(GBP),
        already_recovered=Money.zero(GBP),
        recoverable_amount=unrounded,
    )

    assert computation.claimed_total == Money("1.01", GBP)
    assert computation.recoverable_amount == Money("1.01", GBP)


# ---------------------------------------------------------------------------------- currency


def test_a_claim_denominated_against_a_foreign_programme_raises() -> None:
    """The adversarial case the brief names, checked on the money path as well as the eligibility
    one: `compute` takes a plain `NamedTuple` any caller can build, so the guarantee that the
    amounts share a currency is not carried by the type and has to be re-checked here."""
    hand_built = Eligibility(
        within_warranty_period=True,
        part_covered=True,
        serial_in_range=True,
        labour_rate_within_cap=True,
        already_recovered=False,
        uncovered_parts=Money.zero(EUR),
        labour_excess=Money.zero(EUR),
    )

    with pytest.raises(CurrencyMismatchError, match="CLM-0001"):
        compute(make_claim(currency=EUR), make_program(currency=GBP), hand_built)


def test_a_computation_in_another_currency_stays_in_that_currency() -> None:
    computation = recover(make_claim(currency=EUR), make_program(currency=EUR))

    assert computation.currency is EUR
    assert computation.recoverable_amount == Money("550.00", EUR)


# ---------------------------------------------------------------- independence from `assess`


def test_compute_takes_a_hand_built_eligibility_without_any_other_module() -> None:
    """The deterministic core has to be exercisable one piece at a time.

    If `compute` could only be reached through `assess`, a coverage bug and an arithmetic bug would
    fail the same test, and the corpus lane's ground-truth generator would have to import the
    eligibility rules to produce an expected amount — at which point kill condition F would be
    comparing the system against itself.
    """
    hand_built = Eligibility(
        within_warranty_period=True,
        part_covered=False,
        serial_in_range=True,
        labour_rate_within_cap=False,
        already_recovered=False,
        uncovered_parts=Money("400.00", GBP),
        labour_excess=Money("60.00", GBP),
    )

    computation = compute(make_claim(labour_rate="70.00"), make_program(), hand_built)

    assert computation.eligible_amount == Money("220.00", GBP)
    assert computation.recoverable_amount == Money("170.00", GBP)
