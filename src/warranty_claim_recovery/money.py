"""Money, and the four ways a warranty recovery gets it wrong.

Every monetary value in this system is a `Decimal` with an explicit currency. That is not a style
preference and it is not defensive programming; it is the difference between a system that can be
audited and one that cannot. `0.1 + 0.2 != 0.3` in binary floating point, and a recovery package
whose total is out by a hundredth is a package the manufacturer rejects a second time — which is the
failure this project exists to prevent, arriving by a different door.

The four failures this module is shaped by, each of which a warranty-recovery team meets:

1. **The wrong currency.** A German subsidiary's invoice in EUR compared against a cap denominated
   in GBP. `Money.__add__` and every comparison **raise** on a currency mismatch rather than
   converting, because there is no exchange rate this system is entitled to invent and a silent
   conversion is a wrong number with a plausible shape.
2. **Rounding applied twice.** Round at the deductible, round at the cap, round at the total, and
   the answer drifts. Rounding happens at exactly one point — `recovery.finalise` — and every
   intermediate value keeps full precision.
3. **A cap that is not a cap.** The recoverable amount is clamped after the deductible, never
   before, because a cap applied to the gross defeats the deductible entirely.
4. **Negative recovery.** A deductible larger than the eligible amount must produce zero, not a
   negative claim against the manufacturer. `Money` refuses to construct a negative recoverable
   amount at the boundary rather than letting one flow into a package.

`ROUNDING` is `ROUND_HALF_UP`, the rule invoices are written with. Banker's rounding is the better
statistical choice and the wrong one here: a recovery total that disagrees with the arithmetic a
human does on the invoice is a total the human does not trust, and the whole package is then argued
rather than paid.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from enum import StrEnum
from typing import Any, Final, Self

__all__ = [
    "CURRENCY_EXPONENT",
    "ROUNDING",
    "Currency",
    "CurrencyMismatchError",
    "Money",
    "parse_money",
]

#: The rule invoices are written with. See the module docstring for why not banker's rounding.
ROUNDING: Final = ROUND_HALF_UP


class Currency(StrEnum):
    """The currencies this corpus uses.

    A closed enum rather than a free string, because `extra="forbid"` cannot protect a field whose
    type accepts anything. A claim denominated in a currency the system has never seen should fail
    at the boundary, where the error names the currency, and not three modules later where it names
    a decimal.
    """

    GBP = "GBP"
    EUR = "EUR"
    USD = "USD"


#: Minor units per currency. All three here are two, and the constant exists anyway: a system that
#: hard-codes `.00` acquires a defect the day it meets JPY, and the defect is silent.
CURRENCY_EXPONENT: Final[dict[Currency, int]] = {
    Currency.GBP: 2,
    Currency.EUR: 2,
    Currency.USD: 2,
}


class CurrencyMismatchError(ValueError):
    """Two amounts in different currencies met in an arithmetic or comparison context.

    A distinct type because the caller has something specific and true to say: not "invalid amount"
    but "these two numbers are not comparable, and this system holds no rate that would make them
    so". Kill condition F is exact equality against ground truth, and a converted amount would be
    wrong in a way no test could attribute.
    """


def _quantum(currency: Currency) -> Decimal:
    return Decimal(1).scaleb(-CURRENCY_EXPONENT[currency])


class Money:
    """An exact amount in one currency.

    Immutable, hashable, and comparable only against its own currency. Deliberately **not** a
    Pydantic model: it is used in hot deterministic paths and in dictionary keys, and a validation
    pass on every intermediate would cost more than it protects. Validation happens where a value
    enters the system, in `parse_money`.

    Full precision is kept through every intermediate. `quantize` is called once, by
    `recovery.finalise`, and calling it here would reintroduce failure 2 above.
    """

    __slots__ = ("_amount", "_currency")

    # Annotated so `--strict` knows the slot types. Annotations create no class attribute, so
    # `__slots__` still does its job; without them every private read is an attr-defined error.
    _amount: Decimal
    _currency: Currency

    def __init__(self, amount: Decimal | int | str, currency: Currency | str) -> None:
        try:
            value = amount if isinstance(amount, Decimal) else Decimal(str(amount))
        except InvalidOperation as error:
            raise ValueError(f"{amount!r} is not an exact decimal amount") from error
        if not value.is_finite():
            raise ValueError(f"{amount!r} is not finite; money cannot be NaN or infinite")
        object.__setattr__(self, "_amount", value)
        object.__setattr__(self, "_currency", Currency(currency))

    # -- construction -------------------------------------------------------------------------

    @classmethod
    def zero(cls, currency: Currency | str) -> Self:
        return cls(Decimal(0), currency)

    # -- reading ------------------------------------------------------------------------------

    @property
    def amount(self) -> Decimal:
        return self._amount

    @property
    def currency(self) -> Currency:
        return self._currency

    def quantize(self) -> Money:
        """Round to the currency's minor unit. Called once, at the end, and nowhere else."""
        rounded = self._amount.quantize(_quantum(self.currency), rounding=ROUNDING)
        return Money(rounded, self.currency)

    # -- arithmetic ---------------------------------------------------------------------------

    def _same_currency(self, other: Money) -> None:
        if self.currency is not other.currency:
            raise CurrencyMismatchError(
                f"{self.currency} and {other.currency} are not comparable, and this system holds "
                f"no exchange rate it is entitled to apply"
            )

    def __add__(self, other: Money) -> Money:
        self._same_currency(other)
        return Money(self._amount + other._amount, self.currency)

    def __sub__(self, other: Money) -> Money:
        self._same_currency(other)
        return Money(self._amount - other._amount, self.currency)

    def __mul__(self, factor: Decimal | int) -> Money:
        multiplier = factor if isinstance(factor, Decimal) else Decimal(factor)
        return Money(self._amount * multiplier, self.currency)

    def clamped_to(self, ceiling: Money) -> Money:
        """The smaller of the two. Applied **after** the deductible; see failure 3 above."""
        self._same_currency(ceiling)
        return self if self._amount <= ceiling._amount else ceiling

    def floored_at_zero(self) -> Money:
        """Never a negative claim against a manufacturer. See failure 4 above."""
        return self if self._amount >= 0 else Money.zero(self.currency)

    # -- comparison ---------------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Money):
            return NotImplemented
        # Currencies differing makes two amounts unequal rather than incomparable: equality is a
        # total relation and raising here would make `x == y` unusable in a test assertion.
        return self.currency is other.currency and self._amount == other._amount

    def __lt__(self, other: Money) -> bool:
        self._same_currency(other)
        return bool(self._amount < other._amount)

    def __le__(self, other: Money) -> bool:
        self._same_currency(other)
        return bool(self._amount <= other._amount)

    def __gt__(self, other: Money) -> bool:
        self._same_currency(other)
        return bool(self._amount > other._amount)

    def __ge__(self, other: Money) -> bool:
        self._same_currency(other)
        return bool(self._amount >= other._amount)

    def __hash__(self) -> int:
        return hash((self._amount, self._currency))

    # -- representation -----------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"Money({self._amount}, {self.currency.value!r})"

    def __str__(self) -> str:
        return f"{self.quantize().amount} {self.currency.value}"

    def as_json(self) -> dict[str, str]:
        """The wire form: a string, never a float.

        `json.dumps` on a float would reintroduce binary floating point at the one boundary where
        this system's numbers are read by something else, which is the boundary that matters.
        """
        return {"amount": str(self.quantize().amount), "currency": self.currency.value}

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Money is immutable")


def parse_money(raw: Any, currency: Currency | str) -> Money:
    """Validate an amount at the boundary, where the error can still name the field.

    Rejects `float` outright rather than converting it. A float that reached here was computed
    somewhere by binary floating point, and accepting it would launder that arithmetic into an exact
    type — the value would look exact and would already be wrong.
    """
    if isinstance(raw, float):
        raise TypeError(
            f"{raw!r} is a float. Money is exact; a float arrived from arithmetic this system "
            f"cannot audit, and converting it here would make a wrong number look right."
        )
    return Money(raw, currency)
