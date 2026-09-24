"""The two clocks, tested at the boundaries, because the boundary is where the money goes.

Nobody argues about a claim rejected three months ago. The disputes are all one day wide: a failure
on the anniversary of the in-service date, a correction filed on the closing date of the window, a
twenty-four month warranty on a machine commissioned on the 31st of a month that the target month
does not have. Every test here is one of those.

`as_of` is passed explicitly in every case, and one test asserts that it cannot be omitted. A
deadline module that reads the wall clock produces an evaluation nobody can reproduce and a test
suite that goes red on a date nobody chose.
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

import pytest

from warranty_claim_recovery.deadline import add_months, claim_window, warranty_expires_on
from warranty_claim_recovery.domain import Claim, RejectionCode, WarrantyProgram
from warranty_claim_recovery.money import Currency, Money

GBP = Currency.GBP


def make_program(*, correction_window_days: int = 30, warranty_months: int = 24) -> WarrantyProgram:
    return WarrantyProgram(
        program_id="PRG-ACME-V2",
        manufacturer="Acme Drivetrain (synthetic)",
        policy_version="v2",
        currency=GBP,
        correction_window_days=correction_window_days,
        warranty_months=warranty_months,
        labour_rate_cap_per_hour=Money("55.00", GBP),
        deductible=Money("50.00", GBP),
        claim_cap=Money("5000.00", GBP),
    )


def make_claim(
    *,
    in_service_date: date = date(2024, 1, 15),
    failure_date: date = date(2025, 6, 10),
    rejected_on: date | None = None,
) -> Claim:
    invoiced = failure_date + timedelta(days=2)
    return Claim(
        claim_id="CLM-0001",
        program_id="PRG-ACME-V2",
        part_number="AB-1234-C",
        serial_number="SN-004821",
        in_service_date=in_service_date,
        failure_date=failure_date,
        repair_invoice_date=invoiced,
        rejection_code=RejectionCode.MISSING_SERIAL,
        rejected_on=rejected_on if rejected_on is not None else invoiced + timedelta(days=14),
        claimed_parts=Money(Decimal("400.00"), GBP),
        claimed_labour_hours=4,
        claimed_labour_rate=Money(Decimal("50.00"), GBP),
    )


# ------------------------------------------------------------------------------ month arithmetic


def test_adding_months_clamps_to_the_end_of_a_shorter_month() -> None:
    """31 January plus one month is 28 February, not 3 March.

    Rolling the overflow forward would extend cover by two or three days at exactly the boundary
    where claims are contested, which is money paid out that the policy never promised.
    """
    assert add_months(date(2025, 1, 31), 1) == date(2025, 2, 28)
    assert add_months(date(2025, 3, 31), 1) == date(2025, 4, 30)


def test_a_leap_day_in_service_date_lands_on_the_28th_in_a_common_year() -> None:
    """The case that breaks a naive `date(year + n, month, day)` with a ValueError."""
    assert add_months(date(2024, 2, 29), 12) == date(2025, 2, 28)
    assert add_months(date(2024, 2, 29), 48) == date(2028, 2, 29)


def test_adding_whole_years_of_months_keeps_the_day() -> None:
    assert add_months(date(2024, 1, 15), 24) == date(2026, 1, 15)
    assert add_months(date(2024, 12, 1), 1) == date(2025, 1, 1)
    assert add_months(date(2024, 1, 15), 0) == date(2024, 1, 15)


def test_a_negative_month_count_is_refused() -> None:
    """Warranty periods run forwards. A negative count here is a sign flip somewhere upstream, and
    silently producing a date in the past would make a live machine look out of warranty."""
    with pytest.raises(ValueError, match="run forwards"):
        add_months(date(2024, 1, 15), -1)


# ------------------------------------------------------------------------- the warranty period


def test_a_failure_on_the_expiry_date_is_still_inside_the_warranty_period() -> None:
    """The inclusive boundary, stated in the module docstring and worth a penny-exact test.

    A twenty-four month warranty on a machine commissioned on 15 January 2024 covers a failure on
    15 January 2026. The alternative reading stops cover on the 14th, which is twenty-three months
    and thirty days, and no manufacturer's adjudicator would accept that arithmetic.
    """
    program = make_program(warranty_months=24)
    claim = make_claim(in_service_date=date(2024, 1, 15), failure_date=date(2026, 1, 15))

    expires = warranty_expires_on(claim, program)

    assert expires == date(2026, 1, 15)
    assert claim.failure_date <= expires


def test_a_failure_the_day_after_expiry_is_outside_the_warranty_period() -> None:
    """The mirror of the test above. One day, and the claim is worth nothing."""
    program = make_program(warranty_months=24)
    claim = make_claim(in_service_date=date(2024, 1, 15), failure_date=date(2026, 1, 16))

    assert claim.failure_date > warranty_expires_on(claim, program)


def test_the_period_is_measured_from_the_in_service_date_and_not_the_invoice() -> None:
    """The months a machine spends in a yard before commissioning belong to the distributor.

    A machine despatched in January and commissioned in June has five months of shelf time, and
    measuring the warranty from the earlier date would write off five months of cover the policy
    granted. `REQ-IN-SERVICE-DATE` exists so a claimant can prove which date applies.
    """
    program = make_program(warranty_months=12)
    claim = make_claim(in_service_date=date(2024, 6, 1), failure_date=date(2025, 5, 20))

    assert warranty_expires_on(claim, program) == date(2025, 6, 1)
    assert claim.failure_date <= warranty_expires_on(claim, program)


# -------------------------------------------------------------------------- the claim window


def test_the_window_closes_the_declared_number_of_days_after_the_rejection() -> None:
    program = make_program(correction_window_days=30)
    claim = make_claim(rejected_on=date(2026, 3, 1))

    window = claim_window(claim, program, as_of=date(2026, 3, 1))

    assert window.rejected_on == date(2026, 3, 1)
    assert window.closes_on == date(2026, 3, 31)
    assert window.days_remaining == 30


def test_a_correction_on_the_closing_date_is_in_time() -> None:
    """The adversarial case the brief names, from the side that costs money if it is wrong.

    A window this system believes has shut while the manufacturer would still have accepted the
    correction is a claim written off for nothing. `ClaimWindow` fixes the convention: the closing
    date is inside the window.
    """
    program = make_program(correction_window_days=30)
    claim = make_claim(rejected_on=date(2026, 3, 1))

    window = claim_window(claim, program, as_of=date(2026, 3, 31))

    assert window.is_open is True
    assert window.days_remaining == 0


def test_a_correction_the_day_after_the_closing_date_is_not_in_time() -> None:
    """Kill condition M, at the only boundary where it can be got wrong.

    The other side of the same day. A correction filed here consumes a handler's afternoon, is
    refused on receipt, and counts against a criterion budgeted at zero over the whole corpus.
    """
    program = make_program(correction_window_days=30)
    claim = make_claim(rejected_on=date(2026, 3, 1))

    window = claim_window(claim, program, as_of=date(2026, 4, 1))

    assert window.is_open is False
    assert window.days_remaining == -1


def test_the_window_runs_from_the_rejection_and_not_from_the_day_it_was_read() -> None:
    """A rejection that sat in an inbox for a fortnight has a fortnight less window, not the same.

    Counting from the day this system first saw the rejection would hand the distributor days it
    does not have, and everything downstream — retrieval, composition, approval — would run to
    completion before the manufacturer refused it.
    """
    program = make_program(correction_window_days=30)
    claim = make_claim(rejected_on=date(2026, 3, 1))

    window = claim_window(claim, program, as_of=date(2026, 3, 15))

    assert window.closes_on == date(2026, 3, 31)
    assert window.days_remaining == 16


def test_a_one_day_window_is_still_a_window() -> None:
    """`ClaimWindow` refuses a window that closes before it opens; one day is the smallest legal
    one, and the shortest window in a corpus is the one an off-by-one is found on."""
    program = make_program(correction_window_days=1)
    claim = make_claim(rejected_on=date(2026, 3, 1))

    window = claim_window(claim, program, as_of=date(2026, 3, 2))

    assert window.closes_on == date(2026, 3, 2)
    assert window.is_open is True


def test_the_as_of_date_cannot_be_omitted() -> None:
    """Determinism, asserted as behaviour rather than by grepping for `date.today`.

    If `as_of` had a wall-clock default, this call would succeed. It does not, so every evaluation
    of this system states the date it reasoned as of, and a run in six months' time reproduces the
    same answer from the same corpus.
    """
    program = make_program()
    claim = make_claim()

    with pytest.raises(TypeError):
        claim_window(claim, program)  # type: ignore[call-arg]


def test_the_same_inputs_give_the_same_window_whenever_it_is_asked() -> None:
    program = make_program()
    claim = make_claim(rejected_on=date(2026, 3, 1))

    first = claim_window(claim, program, as_of=date(2026, 3, 10))
    second = claim_window(claim, program, as_of=date(2026, 3, 10))

    assert first == second
