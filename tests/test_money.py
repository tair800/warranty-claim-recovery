"""Money, exercised against the four failures its module docstring names.

These tests are not about arithmetic. Addition works. They are about the four ways a warranty
recovery gets a number wrong while every individual operation looks correct: a currency silently
converted, a rounding step applied twice, a cap applied before the deductible, and a negative
recovery escaping into a package. Three of those are tested here, at the level of the type; the
fourth belongs to `recovery.py` and is tested in `test_recovery.py`, where the order of operations
lives.

Every literal in this file is a `Decimal`, a `str` or an `int`. There is no float anywhere,
including in the values a test is asserting a failure about, because a float written into a test
fixture is a float that will eventually be copied out of it into production code.
"""

from __future__ import annotations

import json
import operator
from collections.abc import Callable
from decimal import Decimal

import pytest

from warranty_claim_recovery.money import (
    CURRENCY_EXPONENT,
    ROUNDING,
    Currency,
    CurrencyMismatchError,
    Money,
    parse_money,
)

GBP = Currency.GBP
EUR = Currency.EUR


# ------------------------------------------------------------------------------- construction


def test_an_amount_can_be_built_from_a_decimal_a_string_or_an_integer() -> None:
    """The three exact spellings, all of which arrive in practice.

    A string is what a JSON corpus carries, an integer is what a whole-pound cap looks like in a
    fixture, and a `Decimal` is what the rest of the system passes around. All three have to land
    on the same value, because a cap written as `500` and a claim written as `"500.00"` are the
    same money and a system that treats them as different amounts fails at the boundary.
    """
    assert Money(Decimal("500.00"), GBP) == Money("500", GBP)
    assert Money(500, GBP) == Money(Decimal("500"), GBP)
    assert Money("500.00", GBP).amount == Decimal("500.00")


def test_a_value_that_is_not_an_exact_decimal_is_refused_where_it_enters() -> None:
    """A malformed amount has to fail here, naming itself, and not three modules later."""
    with pytest.raises(ValueError, match="not an exact decimal amount"):
        Money("four hundred", GBP)


@pytest.mark.parametrize("raw", ["NaN", "Infinity", "-Infinity"])
def test_a_non_finite_amount_is_never_money(raw: str) -> None:
    """`Decimal` accepts NaN and infinity. A recovery package cannot.

    NaN is the dangerous one: it compares false against everything, so a NaN recoverable amount
    would pass `recoverable_amount < zero` — the validator in `RecoveryComputation` that exists to
    stop a negative recovery — and be written into a package as a total nobody can add up.
    """
    with pytest.raises(ValueError, match="not finite"):
        Money(raw, GBP)


def test_parse_money_refuses_a_float_rather_than_converting_it() -> None:
    """The one runtime boundary that stops binary floating point entering an exact system.

    `Money.__init__` is typed to exclude `float` and mypy enforces that at every call site inside
    this package. `parse_money` is what stands at the edges where types are not checked — a JSON
    payload, a corpus file, a form field — and it refuses rather than converts, because a float
    that reached it was computed by arithmetic this system cannot audit and `Decimal(str(0.1))`
    would launder that into a value which looks exact and is already wrong.
    """
    with pytest.raises(TypeError, match="is a float"):
        parse_money(0.1, GBP)


def test_parse_money_accepts_the_exact_spellings_and_keeps_their_precision() -> None:
    parsed = parse_money("100.005", GBP)

    assert parsed.amount == Decimal("100.005")
    assert parsed.currency is GBP


def test_zero_is_built_per_currency() -> None:
    assert Money.zero(GBP).amount == Decimal(0)
    assert Money.zero("EUR").currency is EUR


# --------------------------------------------------------------- failure 1: the wrong currency


def test_adding_two_currencies_raises_rather_than_converting() -> None:
    """The German subsidiary's invoice against the British cap.

    There is no exchange rate this system holds and none it is entitled to invent. A converted
    total would be wrong in a way kill condition F — exact `Decimal` equality against the
    generator's ground truth — could report but never attribute.
    """
    with pytest.raises(CurrencyMismatchError, match="no exchange rate"):
        Money("100", GBP) + Money("100", EUR)


def test_subtracting_two_currencies_raises() -> None:
    with pytest.raises(CurrencyMismatchError):
        Money("100", GBP) - Money("100", EUR)


@pytest.mark.parametrize(
    "compare",
    [operator.lt, operator.le, operator.gt, operator.ge],
    ids=["lt", "le", "gt", "ge"],
)
def test_every_ordering_comparison_across_currencies_raises(
    compare: Callable[[Money, Money], bool],
) -> None:
    """Ordering is where the silent failure would be, not addition.

    `eligibility.assess` asks whether a claimed labour rate exceeds a cap. If that comparison
    compared the numbers and ignored the currencies, a rate of 60 EUR against a cap of 55 GBP
    would produce an excess denominated in nothing, and the resulting recovery would be wrong by
    whatever the two currencies happen to differ by that week.
    """
    with pytest.raises(CurrencyMismatchError):
        compare(Money("100", GBP), Money("100", EUR))


def test_clamping_against_a_foreign_ceiling_raises() -> None:
    """A cap in the wrong currency is the failure that would silently halve a recovery."""
    with pytest.raises(CurrencyMismatchError):
        Money("100", GBP).clamped_to(Money("50", EUR))


def test_equality_across_currencies_is_false_rather_than_an_error() -> None:
    """Deliberately not symmetrical with the orderings, and the asymmetry is the design.

    Equality is a total relation in Python: `x == y` appears inside `assert`, inside `in`, and
    inside every dictionary lookup, and a version that raised would make an ordinary containment
    check explode. The orderings raise because there is no defensible answer; equality answers
    "no", which is true — 100 GBP is not 100 EUR.
    """
    assert Money("100", GBP) != Money("100", EUR)
    assert Money("100", GBP) != "100 GBP"


# ----------------------------------------------------------- failure 2: rounding applied twice


def test_rounding_is_half_up_because_that_is_what_the_invoice_says() -> None:
    """Banker's rounding is the better statistical choice and the wrong one here.

    A recovery total that disagrees with the arithmetic a human does on the invoice is a total the
    human does not trust, and the package is then argued rather than paid. 0.005 goes up.
    """
    assert Money("100.005", GBP).quantize().amount == Decimal("100.01")
    assert Money("100.015", GBP).quantize().amount == Decimal("100.02")
    assert ROUNDING == "ROUND_HALF_UP"


def test_quantising_is_idempotent_so_a_second_pass_changes_nothing() -> None:
    once = Money("100.005", GBP).quantize()

    assert once.quantize() == once


def test_rounding_the_parts_before_summing_them_gives_a_different_total() -> None:
    """The drift this module's single rounding point exists to prevent, shown as a number.

    Two components that each round up put a penny into a total that the full-precision sum does not
    contain. One penny is enough: kill condition F is exact `Decimal` equality, so a total that is
    right to the nearest penny by the wrong route is a failure.
    """
    parts = Money("100.005", GBP)
    labour = Money("99.999", GBP)

    rounded_once_at_the_end = (parts + labour).quantize()
    rounded_on_the_way_in = parts.quantize() + labour.quantize()

    assert rounded_once_at_the_end.amount == Decimal("200.00")
    assert rounded_on_the_way_in.amount == Decimal("200.01")
    assert rounded_once_at_the_end != rounded_on_the_way_in


def test_intermediate_precision_is_kept_until_it_is_asked_for() -> None:
    """Arithmetic does not quantise. Only `quantize` does, and `recovery.finalise` calls it."""
    total = Money("33.333", GBP) * 3

    assert total.amount == Decimal("99.999")
    assert total.quantize().amount == Decimal("100.00")


def test_every_currency_declares_its_minor_unit() -> None:
    """A hard-coded two-decimal quantum is a defect that stays silent until the day it meets JPY."""
    for currency in Currency:
        assert currency in CURRENCY_EXPONENT


# ------------------------------------------------------- failures 3 and 4: the cap and the floor


def test_clamping_returns_the_smaller_of_the_two() -> None:
    assert Money("100", GBP).clamped_to(Money("60", GBP)) == Money("60", GBP)
    assert Money("40", GBP).clamped_to(Money("60", GBP)) == Money("40", GBP)


def test_clamping_at_the_boundary_keeps_the_amount() -> None:
    """A claim exactly at the cap is fully recoverable. An off-by-one here costs the last penny."""
    assert Money("60.00", GBP).clamped_to(Money("60.00", GBP)) == Money("60.00", GBP)


def test_flooring_turns_a_negative_into_zero_and_leaves_a_positive_alone() -> None:
    """A deductible larger than the eligible amount produces no recovery, never a debt.

    A negative recoverable amount is a claim by the manufacturer against the distributor, which is
    not a thing this system is entitled to construct. `RecoveryComputation` refuses one at its
    validator; this is the operation that stops one being built in the first place.
    """
    assert Money("-1.00", GBP).floored_at_zero() == Money.zero(GBP)
    assert Money("0", GBP).floored_at_zero() == Money.zero(GBP)
    assert Money("1.00", GBP).floored_at_zero() == Money("1.00", GBP)


def test_multiplication_takes_whole_hours_and_exact_factors() -> None:
    """Labour is a rate times hours, and hours are whole units in this domain."""
    assert Money("45.50", GBP) * 4 == Money("182.00", GBP)
    assert Money("45.50", GBP) * Decimal("0.5") == Money("22.750", GBP)
    assert Money("45.50", GBP) * 0 == Money.zero(GBP)


# ------------------------------------------------------------------------ the wire, and the type


def test_the_wire_form_is_strings_and_never_a_float() -> None:
    """The one boundary where this system's numbers are read by something else.

    `json.dumps` on a float reintroduces binary floating point at exactly the point where the
    numbers leave: 100.01 serialises as 100.01000000000000512, and the manufacturer's portal sees
    a total this system never computed.
    """
    payload = Money("100.005", GBP).as_json()

    assert payload == {"amount": "100.01", "currency": "GBP"}
    assert isinstance(payload["amount"], str)
    assert "100.01" in json.dumps(payload)


def test_the_readable_form_is_rounded_and_carries_its_currency() -> None:
    assert str(Money("100.005", GBP)) == "100.01 GBP"
    assert repr(Money("100.005", GBP)) == "Money(100.005, 'GBP')"


def test_an_amount_cannot_be_changed_after_it_is_built() -> None:
    """Immutability is what lets an amount be passed into a computation without being copied.

    `RecoveryComputation` holds eight amounts and is handed to the console, the audit log and the
    evaluation. If any holder could mutate one, the record of what the system decided and the thing
    it decided would be the same object with two histories.
    """
    amount = Money("100.00", GBP)

    with pytest.raises(AttributeError, match="immutable"):
        amount.something = Decimal(1)  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        amount.amount = Decimal(1)  # type: ignore[misc]


def test_an_amount_is_hashable_so_it_can_key_a_dictionary() -> None:
    """Used as a dictionary key in the evaluation, which is why `Money` is not a Pydantic model."""
    ledger = {Money("100.00", GBP): "settled"}

    assert ledger[Money("100.00", GBP)] == "settled"
    assert hash(Money("100.00", GBP)) != hash(Money("100.00", EUR))


def test_two_amounts_with_the_same_value_in_different_currencies_do_not_collide() -> None:
    ledger = {Money("100.00", GBP): "gbp", Money("100.00", EUR): "eur"}

    assert len(ledger) == 2
