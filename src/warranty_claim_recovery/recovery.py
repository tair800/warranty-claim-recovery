"""The recoverable amount, in the order the policy applies its terms and with one rounding step.

Five subtractions, and the order of them is the whole module. Every one of the five is a place a
warranty team gets a different number from the manufacturer, and in every case the argument is not
about the arithmetic but about the sequence it was done in.

```
claimed total        parts plus labour, as invoiced
  less labour excess       the hours charged, at the amount by which the rate beat the cap
  less uncovered parts     parts the schedule does not cover on this build
= eligible amount    what the policy ever covered
  less deductible          the programme's excess, applied to the eligible amount
= after deductible   floored at zero, never negative
  capped at claim cap      the ceiling on one claim, applied after the deductible
  less already recovered   what this recovery identity has already been paid
= recoverable amount floored at zero
```

**The cap is applied after the deductible, and this is the one that costs money.** Applying it to
the gross defeats the deductible entirely: a £5,000 claim against a £4,000 cap and a £250 deductible
recovers £3,750 in this order and £4,000 in the other, and the £250 difference is the manufacturer's
money being claimed. `money.py` records it as the third of the four failures the type exists for.

**`capped_amount` is what the cap removed, not the cap.** It is `after_deductible - claim_cap`,
floored at zero, and it is zero whenever the cap did not bind. The rejected alternative was to store
the cap itself, which is already on the programme and tells a reader nothing about this claim; a
recovery package has to show what was taken out, because an adjudicator who cannot see the
subtraction cannot check the total and refuses the package rather than reconstructing it.

**Rounding happens once, in `finalise`, on every field at the same moment.** Round at the deductible
and again at the cap and again at the total, and the answer drifts by a penny or two — which is
enough for kill condition F, exact `Decimal` equality against the generator's ground truth, to fail
on a number that is right to the nearest penny and wrong to the system. Every intermediate above
keeps full precision; `finalise` is the only place `Money.quantize` is called on a recovery figure.

**Nothing here is floored except where a floor means something.** `eligible_amount` is deliberately
not floored at zero, even though a negative one would be absurd, because the two subtractions that
produce it are bounded components of the claimed total and so it cannot go negative unless something
upstream is broken. A floor there would turn a broken exclusion into a plausible zero and hide it;
the floors that do exist — after the deductible, and on the recoverable amount — are policy, because
a deductible larger than the claim produces no recovery and never a debt owed the other way.
"""

from __future__ import annotations

from warranty_claim_recovery.domain import Claim, RecoveryComputation, WarrantyProgram
from warranty_claim_recovery.eligibility import Eligibility, refuse_cross_currency_claim
from warranty_claim_recovery.money import Currency, Money

__all__ = [
    "compute",
    "finalise",
]


def finalise(
    *,
    currency: Currency,
    claimed_total: Money,
    labour_excess: Money,
    uncovered_parts: Money,
    eligible_amount: Money,
    deductible: Money,
    capped_amount: Money,
    already_recovered: Money,
    recoverable_amount: Money,
) -> RecoveryComputation:
    """Round every figure to the minor unit, once, and build the record.

    The eight `quantize` calls are written out one per line rather than looped over a mapping, and
    the repetition is the point: a reader can see at a glance that every field of the record was
    rounded at the same moment and by the same rule, which is the property `money.py` asks for and
    which a comprehension would leave them taking on trust. A field added to `RecoveryComputation`
    and not to this function fails to construct, because the model forbids extras and requires all
    of them — so the compiler-shaped failure is the one that happens, rather than a silent
    full-precision value appearing in a package next to seven rounded ones.

    Keyword-only throughout. Eight amounts of the same type in a row is the argument order nobody
    gets right twice, and swapping `deductible` with `capped_amount` positionally would produce a
    record that validates cleanly and is wrong.
    """
    return RecoveryComputation(
        currency=currency,
        claimed_total=claimed_total.quantize(),
        labour_excess=labour_excess.quantize(),
        uncovered_parts=uncovered_parts.quantize(),
        eligible_amount=eligible_amount.quantize(),
        deductible=deductible.quantize(),
        capped_amount=capped_amount.quantize(),
        already_recovered=already_recovered.quantize(),
        recoverable_amount=recoverable_amount.quantize(),
    )


def compute(
    claim: Claim,
    program: WarrantyProgram,
    eligibility: Eligibility,
) -> RecoveryComputation:
    """What this claim is worth, with every intermediate step kept.

    Re-checks the currency although `eligibility.assess` already did. The duplication is deliberate:
    `compute` takes an `Eligibility`, which is a plain `NamedTuple` that any caller can build by
    hand — the tests here do exactly that — so the guarantee that the amounts share a currency is
    not carried by the type. Without the re-check, a hand-built `Eligibility` would reach the first
    subtraction and raise from inside `Money`, naming two currencies and no claim.
    """
    currency = refuse_cross_currency_claim(claim, program)
    zero = Money.zero(currency)

    claimed_total = claim.claimed_total
    eligible_amount = claimed_total - eligibility.labour_excess - eligibility.uncovered_parts

    after_deductible = (eligible_amount - program.deductible).floored_at_zero()

    # What the cap removed. Computed before the clamp rather than derived from it afterwards,
    # because `after_deductible - min(after_deductible, cap)` is the same number by a longer route
    # and the longer route is the one where a future edit puts the clamp before the deductible.
    capped_amount = (after_deductible - program.claim_cap).floored_at_zero()

    # Read from the claim rather than reconstructed from `eligibility.already_recovered`, which is
    # a boolean and cannot carry an amount. The two agree by construction: the flag is true exactly
    # when this value is positive.
    prior = claim.previously_recovered
    already_recovered = prior if prior is not None else zero

    recoverable_amount = (
        after_deductible.clamped_to(program.claim_cap) - already_recovered
    ).floored_at_zero()

    return finalise(
        currency=currency,
        claimed_total=claimed_total,
        labour_excess=eligibility.labour_excess,
        uncovered_parts=eligibility.uncovered_parts,
        eligible_amount=eligible_amount,
        deductible=program.deductible,
        capped_amount=capped_amount,
        already_recovered=already_recovered,
        recoverable_amount=recoverable_amount,
    )
