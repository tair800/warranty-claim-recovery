"""Eligibility, tested against claims whose paperwork is perfect and whose money is not.

Every fixture in this file has a complete evidence bundle by construction — there is no requirement
machinery here at all. That separation is the point of the module: a claim can be immaculate on
paper and still be worth nothing because the machine went out of warranty, or the unit falls outside
the covered build range, or somebody recovered against it last quarter. If these checks needed the
requirement matrix or a store to run, an eligibility bug and a paperwork bug would fail the same
test and nobody could tell them apart.

Every monetary literal here is a `Decimal` or a string. A float in a fixture is a float that gets
copied into production code the next time somebody needs a similar case.
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

import pytest

from warranty_claim_recovery.domain import Claim, RejectionCode, WarrantyProgram
from warranty_claim_recovery.eligibility import (
    PartCoverage,
    assess,
    refuse_cross_currency_claim,
    serial_ordinal,
)
from warranty_claim_recovery.money import Currency, CurrencyMismatchError, Money

GBP = Currency.GBP
EUR = Currency.EUR

PART = "AB-1234-C"


def make_program(
    *,
    currency: Currency = GBP,
    warranty_months: int = 24,
    labour_rate_cap: str = "55.00",
) -> WarrantyProgram:
    return WarrantyProgram(
        program_id="PRG-ACME-V2",
        manufacturer="Acme Drivetrain (synthetic)",
        policy_version="v2",
        currency=currency,
        correction_window_days=30,
        warranty_months=warranty_months,
        labour_rate_cap_per_hour=Money(labour_rate_cap, currency),
        deductible=Money("50.00", currency),
        claim_cap=Money("5000.00", currency),
    )


def make_claim(
    *,
    currency: Currency = GBP,
    in_service_date: date = date(2024, 1, 15),
    failure_date: date = date(2025, 6, 10),
    serial_number: str | None = "SN-004821",
    parts: str = "400.00",
    labour_hours: int = 4,
    labour_rate: str = "50.00",
    previously_recovered: str | None = None,
) -> Claim:
    invoiced = failure_date + timedelta(days=2)
    return Claim(
        claim_id="CLM-0001",
        program_id="PRG-ACME-V2",
        part_number=PART,
        serial_number=serial_number,
        in_service_date=in_service_date,
        failure_date=failure_date,
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


def covered(
    *,
    serial_first: int | None = None,
    serial_last: int | None = None,
    is_covered: bool = True,
) -> PartCoverage:
    return PartCoverage(
        part_number=PART,
        covered=is_covered,
        serial_first=serial_first,
        serial_last=serial_last,
    )


# ------------------------------------------------------------------------- the warranty period


def test_a_failure_one_day_after_the_warranty_expired_is_outside_the_period() -> None:
    """The adversarial case the brief names, on the side that costs the manufacturer nothing.

    Twenty-four months from 15 January 2024 expires on 15 January 2026. A failure on the 16th is
    outside, and every piece of evidence in the world does not change that — which is why the gate
    carries this as a rule rather than leaving it to the requirement matrix.
    """
    eligibility = assess(
        make_claim(in_service_date=date(2024, 1, 15), failure_date=date(2026, 1, 16)),
        make_program(warranty_months=24),
        covered(),
    )

    assert eligibility.within_warranty_period is False


def test_a_failure_on_the_last_covered_day_is_inside_the_period() -> None:
    """The mirror. An off-by-one here writes off a claim the manufacturer would have paid."""
    eligibility = assess(
        make_claim(in_service_date=date(2024, 1, 15), failure_date=date(2026, 1, 15)),
        make_program(warranty_months=24),
        covered(),
    )

    assert eligibility.within_warranty_period is True


def test_the_period_check_does_not_move_the_money() -> None:
    """Recorded deliberately, because it is the reason `gate.py` carries a warranty-period rule.

    An expired warranty leaves the parts covered and the labour rate inside the cap, so the
    recovery arithmetic still produces a positive amount. Nothing in this module excludes it. If
    the gate did not check the period, that positive amount would be authorised — kill condition
    G's false recovery, arriving through a claim whose paperwork was in perfect order.
    """
    eligibility = assess(
        make_claim(in_service_date=date(2024, 1, 15), failure_date=date(2026, 1, 16)),
        make_program(warranty_months=24),
        covered(),
    )

    assert eligibility.within_warranty_period is False
    assert eligibility.uncovered_parts == Money.zero(GBP)
    assert eligibility.labour_excess == Money.zero(GBP)


# --------------------------------------------------------------- the part and its build range


def test_the_right_part_on_a_unit_outside_the_covered_build_range_is_not_covered() -> None:
    """The adversarial case the brief names: correct part number, wrong serial.

    Coverage is published per part *and* per build range. Treating the two as independent — part
    covered, serial merely noted — authorises a claim for the right component on the wrong machine,
    and the manufacturer refuses it on the serial without ever looking at the part.
    """
    eligibility = assess(
        make_claim(serial_number="SN-004821"),
        make_program(),
        covered(serial_first=1000, serial_last=4000),
    )

    assert eligibility.serial_in_range is False
    assert eligibility.part_covered is False
    assert eligibility.uncovered_parts == Money("400.00", GBP)


def test_a_serial_inside_the_covered_range_leaves_the_part_covered() -> None:
    eligibility = assess(
        make_claim(serial_number="SN-004821"),
        make_program(),
        covered(serial_first=1000, serial_last=9000),
    )

    assert eligibility.serial_in_range is True
    assert eligibility.part_covered is True
    assert eligibility.uncovered_parts == Money.zero(GBP)


@pytest.mark.parametrize("serial", ["SN-001000", "SN-009000"])
def test_both_ends_of_a_build_range_are_inside_it(serial: str) -> None:
    """The range published in a schedule is inclusive at both ends. One unit either way is a
    machine whose claim is accepted or refused on an off-by-one."""
    eligibility = assess(
        make_claim(serial_number=serial),
        make_program(),
        covered(serial_first=1000, serial_last=9000),
    )

    assert eligibility.serial_in_range is True


def test_a_schedule_entry_with_no_build_range_covers_every_unit() -> None:
    """A consumable covered across every build, which is the common case, not the exception.

    Checking the serial first would report a missing serial as out of range for a part whose
    coverage never depended on one, and the technician would be sent to find a number that changes
    nothing.
    """
    eligibility = assess(
        make_claim(serial_number=None),
        make_program(),
        covered(),
    )

    assert eligibility.serial_in_range is True
    assert eligibility.part_covered is True


def test_an_open_ended_range_has_no_upper_bound() -> None:
    """ "From build 4000" covers builds that did not exist when the schedule was written.

    Substituting a large finite number for the missing bound would put an expiry on a range the
    manufacturer did not put one on, and the first unit past it would be refused for no reason.
    """
    eligibility = assess(
        make_claim(serial_number="SN-999999"),
        make_program(),
        covered(serial_first=4000, serial_last=None),
    )

    assert eligibility.serial_in_range is True


def test_a_missing_serial_against_a_restricted_range_is_not_in_range() -> None:
    """Not shown to be inside a range is not inside it. This is what `MISSING_SERIAL` is for."""
    eligibility = assess(
        make_claim(serial_number=None),
        make_program(),
        covered(serial_first=1000, serial_last=9000),
    )

    assert eligibility.serial_in_range is False
    assert eligibility.part_covered is False


def test_a_part_the_schedule_excludes_is_uncovered_whatever_its_serial() -> None:
    eligibility = assess(
        make_claim(serial_number="SN-004821"),
        make_program(),
        covered(serial_first=1000, serial_last=9000, is_covered=False),
    )

    assert eligibility.serial_in_range is True
    assert eligibility.part_covered is False
    assert eligibility.uncovered_parts == Money("400.00", GBP)


def test_a_coverage_record_for_a_different_part_is_refused() -> None:
    """A wiring bug, and the honest answer would be indistinguishable from a real exclusion.

    Answering "not covered" here would write off a covered claim with a finding that looks exactly
    like a genuine one, and no test downstream could tell the two apart.
    """
    mismatched = PartCoverage(
        part_number="ZZ-9999-X", covered=True, serial_first=None, serial_last=None
    )

    with pytest.raises(ValueError, match="coverage record for"):
        assess(make_claim(), make_program(), mismatched)


# ------------------------------------------------------------------------------ serial parsing


@pytest.mark.parametrize(
    ("serial", "expected"),
    [
        ("SN-004821", 4821),
        ("004821", 4821),
        ("SN-9", 9),
        ("SN-0", 0),
        (None, None),
        ("SN-", None),
        ("NO-DIGITS", None),
        ("4821-REV-B", None),
    ],
)
def test_the_serial_ordinal_is_the_trailing_digit_run(serial: str | None, expected: int) -> None:
    """The prefix identifies the plant, not the build sequence.

    Comparing serials lexically would place `SN-9` above `SN-10`, so the numeric tail is what a
    build range is checked against. A serial whose tail is not numeric returns `None` rather than
    raising: it is a data-quality problem the requirement matrix already has a remedy for, and a
    case that crashes is a case nobody chases before the window closes.
    """
    assert serial_ordinal(serial) == expected


def test_a_serial_with_no_numeric_tail_is_not_placed_in_a_range() -> None:
    eligibility = assess(
        make_claim(serial_number="REV-B"),
        make_program(),
        covered(serial_first=1000, serial_last=9000),
    )

    assert eligibility.serial_in_range is False


# --------------------------------------------------------------------------- the labour rate cap


def test_a_labour_rate_above_the_cap_produces_an_excess_over_every_hour_charged() -> None:
    """The adversarial case the brief names. The excess is excluded and shown as excluded.

    A rate of 70.00 against a cap of 55.00 over four hours is 60.00 of excess. Treating the cap as
    an hours cap instead — refusing the fourth hour — would exclude money the policy does cover and
    would give a number the manufacturer's own schedule contradicts.
    """
    eligibility = assess(
        make_claim(labour_rate="70.00", labour_hours=4),
        make_program(labour_rate_cap="55.00"),
        covered(),
    )

    assert eligibility.labour_rate_within_cap is False
    assert eligibility.labour_excess == Money("60.00", GBP)


def test_a_rate_exactly_at_the_cap_is_within_it() -> None:
    """The cap is a ceiling, not a strict bound. An off-by-one here excludes a compliant invoice."""
    eligibility = assess(
        make_claim(labour_rate="55.00", labour_hours=4),
        make_program(labour_rate_cap="55.00"),
        covered(),
    )

    assert eligibility.labour_rate_within_cap is True
    assert eligibility.labour_excess == Money.zero(GBP)


def test_a_rate_below_the_cap_produces_no_credit() -> None:
    """A rate under the cap is not a discount the distributor may claim back.

    Without the floor, an under-cap rate would produce a negative excess, which would be *added*
    to the eligible amount three lines later and would inflate the recovery.
    """
    eligibility = assess(
        make_claim(labour_rate="40.00", labour_hours=4),
        make_program(labour_rate_cap="55.00"),
        covered(),
    )

    assert eligibility.labour_excess == Money.zero(GBP)


def test_an_over_cap_rate_on_zero_hours_costs_nothing() -> None:
    """A parts-only claim carries a rate that was never charged. Excluding on it alone would
    subtract money from a claim with no labour in it."""
    eligibility = assess(
        make_claim(labour_rate="70.00", labour_hours=0),
        make_program(labour_rate_cap="55.00"),
        covered(),
    )

    assert eligibility.labour_rate_within_cap is False
    assert eligibility.labour_excess == Money.zero(GBP)


# ------------------------------------------------------------------------------ prior recovery


def test_a_claim_already_recovered_once_is_flagged() -> None:
    """The adversarial case the brief names, and the input to the gate's duplicate rule.

    Read from the claim's recorded prior recovery, never inferred from the text of a rejection
    letter: a duplicate detected by reading prose is a duplicate that depends on wording.
    """
    eligibility = assess(make_claim(previously_recovered="300.00"), make_program(), covered())

    assert eligibility.already_recovered is True


def test_a_recorded_nil_recovery_is_a_ledger_entry_and_not_a_recovery() -> None:
    """A settled nil adjudication against a machine must not refuse every later claim on it."""
    eligibility = assess(make_claim(previously_recovered="0.00"), make_program(), covered())

    assert eligibility.already_recovered is False


def test_no_prior_recovery_at_all_is_not_a_recovery() -> None:
    eligibility = assess(make_claim(previously_recovered=None), make_program(), covered())

    assert eligibility.already_recovered is False


# ---------------------------------------------------------------------------------- currency


def test_a_claim_in_another_currency_raises_rather_than_being_converted() -> None:
    """The adversarial case the brief names: the amount is right and the currency is not.

    This system holds no exchange rate it is entitled to apply. A converted recovery amount would
    be wrong in a way kill condition F — exact `Decimal` equality against ground truth — could
    report and never attribute, and the error has to name the claim rather than a decimal.
    """
    with pytest.raises(CurrencyMismatchError, match="CLM-0001"):
        assess(make_claim(currency=EUR), make_program(currency=GBP), covered())


def test_the_currency_guard_returns_the_shared_currency_when_they_agree() -> None:
    """Returned rather than merely checked, so that no caller re-derives "the currency of this
    case" from a second field and the two readings drift apart."""
    assert refuse_cross_currency_claim(make_claim(), make_program()) is GBP


def test_a_claim_and_programme_in_the_same_foreign_currency_are_fine() -> None:
    """The guard is about disagreement, not about GBP being privileged."""
    eligibility = assess(make_claim(currency=EUR), make_program(currency=EUR), covered())

    assert eligibility.uncovered_parts == Money.zero(EUR)
