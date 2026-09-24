"""The five questions that decide what part of a claim the policy ever covered.

Paperwork and eligibility are different things, and conflating them is the mistake this module is
shaped around. A claim can arrive with a complete evidence bundle — serial stamped, commissioning
certificate attached, failure code corrected — and still be worth nothing, because the machine went
out of warranty in March, or the part is not in the covered schedule, or the unit's serial falls
outside the build range the programme covers, or somebody already recovered against it. The
requirement matrix answers "is the paperwork in order". This module answers "was any of it ever
covered", and the gate needs both before it authorises anything.

`assess` returns five booleans and two amounts, and the two amounts are the reason this is not
simply a predicate. A claim that is 90% covered is the ordinary case, not the exception: the parts
are covered, the labour rate is eight pounds over the cap, and the recoverable amount is the whole
claim minus the excess. Collapsing that to `eligible: False` writes off money the manufacturer
would have paid, and collapsing it to `eligible: True` files a claim that gets rejected a second
time for the same reason. Both amounts are therefore carried forward and **shown as excluded** in
the recovery package rather than quietly dropped, because an adjudicator who cannot see what was
taken out cannot check the total and refuses the package instead.

### Cross-currency claims raise; they are never converted

A claim denominated in EUR assessed against a programme whose cap and deductible are in GBP has no
answer this system is entitled to give. `refuse_cross_currency_claim` raises at the top of `assess`
rather than letting the first subtraction raise three frames deeper, so the error names the claim
and the two currencies instead of naming a decimal. `money.py` records the underlying rule: there is
no exchange rate this system holds, and a converted amount would be wrong in a way kill condition F
— exact `Decimal` equality against ground truth — could not attribute to anything.

### Why coverage is passed in rather than looked up

`PartCoverage` is an argument, not a database query made inside this function. The whole
deterministic core is testable with hand-built fixtures for that reason: if these checks needed a
store to run, an eligibility bug and a connection bug would fail the same test and nobody could
tell them apart. The corpus lane owns where coverage records come from; this module owns what they
mean.
"""

from __future__ import annotations

from typing import NamedTuple

from warranty_claim_recovery.deadline import warranty_expires_on
from warranty_claim_recovery.domain import Claim, WarrantyProgram
from warranty_claim_recovery.money import Currency, CurrencyMismatchError, Money

__all__ = [
    "Eligibility",
    "PartCoverage",
    "assess",
    "refuse_cross_currency_claim",
    "serial_ordinal",
]


class PartCoverage(NamedTuple):
    """What the covered-parts schedule says about one part number, including its build range.

    `serial_first` and `serial_last` are both `None` when the schedule places no build-range
    restriction on the part. That is a real and common case — a consumable covered across every
    build — and it is deliberately not spelled as a range from zero to a large number, because a
    sentinel range is indistinguishable from a real one that happens to be wide, and the day
    somebody has to explain why the coverage record says 0 to 999999999 is the day the sentinel
    starts being treated as data.
    """

    part_number: str
    covered: bool
    serial_first: int | None
    serial_last: int | None

    @property
    def restricts_serials(self) -> bool:
        return self.serial_first is not None or self.serial_last is not None


class Eligibility(NamedTuple):
    """What the policy covered, before any question of deductibles, caps or prior recoveries.

    Five findings and two amounts. The amounts are full-precision `Money` and are **not** rounded
    here: rounding happens once, in `recovery.finalise`, and a value quantised on the way in would
    be quantised again on the way out — the second of the four failures `money.py` is shaped by.
    """

    within_warranty_period: bool
    part_covered: bool
    serial_in_range: bool
    labour_rate_within_cap: bool
    already_recovered: bool
    #: The claimed parts cost, in full, when the policy does not cover this part on this unit. Not
    #: a proportion: a schedule either lists the part for the build or it does not, and inventing a
    #: partial coverage percentage would be this system deciding policy.
    uncovered_parts: Money
    #: Claimed labour above the programme's hourly cap, over the hours claimed. The hours
    #: themselves are not challenged here; the cap is a rate cap and treating it as an hours cap
    #: would exclude money the policy does cover.
    labour_excess: Money


def refuse_cross_currency_claim(claim: Claim, program: WarrantyProgram) -> Currency:
    """The one currency the claim and the programme share, or an error that names both.

    Returns the currency rather than `None` so that callers have something to build zero amounts
    from and cannot reach for `claim.claimed_parts.currency` again out of habit — two readings of
    "the currency of this case" is one more than there should be.
    """
    claimed = claim.claimed_parts.currency
    if claimed is not program.currency:
        raise CurrencyMismatchError(
            f"{claim.claim_id} is claimed in {claimed.value} against programme "
            f"{program.program_id}, which is denominated in {program.currency.value}. This system "
            f"holds no exchange rate it is entitled to apply, and a converted recovery amount "
            f"would be wrong in a way no audit could attribute."
        )
    return program.currency


def serial_ordinal(serial_number: str | None) -> int | None:
    """The numeric part of a serial, for comparison against a build range.

    Serials in this corpus are a prefix and a number — `SN-004821` — and the range published in a
    covered-parts schedule is numeric. The trailing digit run is taken and the prefix ignored,
    because the prefix identifies the plant rather than the build sequence and comparing it
    lexically would place `SN-9` above `SN-10`.

    Returns `None` for a serial that carries no trailing digits, and deliberately does **not**
    raise. A serial that cannot be placed in a range is a serial that has not been shown to be
    inside one, which is exactly the finding `REQ-SERIAL-IN-COVERED-RANGE` exists to record and
    which sends the case to a technician. Raising would instead take the whole case down over a
    data-quality problem the requirement machinery already has a remedy for, and a case that
    crashes is a case nobody chases before the window closes.
    """
    if serial_number is None:
        return None
    digits = ""
    for character in reversed(serial_number):
        if not character.isdigit():
            break
        digits = character + digits
    return int(digits) if digits else None


def _serial_is_in_range(claim: Claim, coverage: PartCoverage) -> bool:
    """Whether this unit's serial falls inside the build range the schedule publishes.

    An unrestricted schedule entry covers every unit, so the answer is `True` before a serial has
    even been read. The order matters: checking the serial first would report a missing serial as
    out of range for a part whose coverage never depended on one, and the technician would be sent
    to find a number that changes nothing.
    """
    if not coverage.restricts_serials:
        return True
    ordinal = serial_ordinal(claim.serial_number)
    if ordinal is None:
        return False
    # An open-ended bound is an open end, not a missing one. A schedule that says "from build 4000"
    # covers everything after it, including builds that did not exist when the schedule was
    # written, and substituting a large finite number here would put an expiry on a range the
    # manufacturer did not put one on.
    if coverage.serial_first is not None and ordinal < coverage.serial_first:
        return False
    return not (coverage.serial_last is not None and ordinal > coverage.serial_last)


def assess(claim: Claim, program: WarrantyProgram, coverage: PartCoverage) -> Eligibility:
    """What the policy covered on this claim, with nothing about deductibles or caps in it.

    Refuses a coverage record for a different part number. That mismatch is a wiring bug in the
    caller, and the reason it raises rather than answering is that the honest answer would be "not
    covered" — a plausible-looking finding that would write off a covered claim, and one that no
    test downstream could distinguish from a real exclusion.
    """
    currency = refuse_cross_currency_claim(claim, program)

    if coverage.part_number != claim.part_number:
        raise ValueError(
            f"{claim.claim_id} claims part {claim.part_number} and was assessed against a coverage "
            f"record for {coverage.part_number}. Answering that question would produce a coverage "
            f"finding about the wrong part, which is indistinguishable downstream from a real one."
        )

    within_warranty_period = claim.failure_date <= warranty_expires_on(claim, program)
    serial_in_range = _serial_is_in_range(claim, coverage)
    # Coverage is per part *and* per build range, so a correct part number on a unit outside the
    # covered builds is not covered. Treating those two as independent — part covered, serial
    # merely noted — is how a claim for the right component on the wrong machine gets authorised,
    # and the manufacturer refuses it on the serial without ever looking at the part.
    part_covered = coverage.covered and serial_in_range

    labour_rate_within_cap = claim.claimed_labour_rate <= program.labour_rate_cap_per_hour
    over_cap_per_hour = (
        claim.claimed_labour_rate - program.labour_rate_cap_per_hour
    ).floored_at_zero()

    prior = claim.previously_recovered
    # A recorded recovery of zero is a ledger entry, not a recovery. Reading it as one would refuse
    # every claim against a machine that has a settled nil adjudication against it.
    already_recovered = prior is not None and prior > Money.zero(currency)

    return Eligibility(
        within_warranty_period=within_warranty_period,
        part_covered=part_covered,
        serial_in_range=serial_in_range,
        labour_rate_within_cap=labour_rate_within_cap,
        already_recovered=already_recovered,
        uncovered_parts=claim.claimed_parts if not part_covered else Money.zero(currency),
        labour_excess=over_cap_per_hour * claim.claimed_labour_hours,
    )
