"""Two dates that decide whether a claim is still worth anything, computed and never judged.

A warranty recovery has two clocks running against it and they measure different things.

The **warranty period** runs from the date the machine went into service and says whether the
failure was ever covered. The **correction window** runs from the date the manufacturer rejected the
claim and says whether there is still time to answer the rejection. A claim can be inside the first
and outside the second — that is the ordinary way a recoverable claim becomes a write-off, and it
happens while somebody waits for a technician to reply.

Both failures this module prevents are expensive and neither is dramatic. A claim written off
because nobody noticed the window closing is money the distributor was owed and did not ask for. A
correction filed after the window shut is worse: it consumes a person's afternoon, it is refused on
receipt, and it is kill condition M, graded at zero over the whole corpus.

**Everything here takes `as_of` as an argument.** There is no `date.today()` anywhere in this
module and there is no default for it. A deadline function that reads the wall clock produces a
different answer on a different day, which means an evaluation that cannot be reproduced, a test
that passes until it does not, and an audit record whose arithmetic nobody can re-run. The cost is
one extra argument at every call site, which is the right price.

### Two conventions, both stated because both are ambiguous in exactly the way that costs a claim

**A boundary date is inside its interval.** A correction filed *on* `closes_on` is in time and one
filed the next day is not; a failure *on* the warranty expiry date is covered and one the next day
is not. `ClaimWindow` in `domain.py` already fixes this for the correction window and this module
matches it for the warranty period, because two adjacent intervals with different conventions is
how a system comes to accept a claim the manufacturer refuses. The rejected alternative — an
exclusive upper bound — was rejected because it makes a twenty-four month warranty taken out on 15
January 2024 stop covering on 14 January 2026, which reads as twenty-three months and thirty days
to every human being who checks it, and a deadline a human cannot reproduce is a deadline that gets
argued rather than accepted.

**Adding months clamps to the end of the shorter month.** Thirty-one months from 31 March is 31
October; twelve months from 29 February 2024 is 28 February 2025, not 1 March. The rejected
alternative — rolling the overflow into the next month — was rejected because it silently extends
cover by a day or three at exactly the boundary where claims are contested, and a warranty that
lasts longer than the policy says is money paid out that the policy never promised. Clamping is
also what a policy administrator does with a paper calendar, which is the arithmetic the
manufacturer will check against.
"""

from __future__ import annotations

from calendar import monthrange
from datetime import date, timedelta

from warranty_claim_recovery.domain import Claim, ClaimWindow, WarrantyProgram

__all__ = [
    "add_months",
    "claim_window",
    "warranty_expires_on",
]


def add_months(start: date, months: int) -> date:
    """`start` advanced by whole months, clamped to the end of the target month.

    Written out rather than taken from `dateutil.relativedelta`, which does the same thing. The
    dependency is not the objection; the objection is that the clamping convention is a policy
    decision this project has to be able to defend, and a decision delegated to a library is one
    nobody on the team can quote the rule for when a manufacturer disputes a single day.
    """
    if months < 0:
        raise ValueError("warranty periods run forwards; a negative month count is a caller bug")
    zero_based = start.month - 1 + months
    year = start.year + zero_based // 12
    month = zero_based % 12 + 1
    # `monthrange` returns (weekday of the first, length of the month). The clamp is the second.
    last_day_of_target_month = monthrange(year, month)[1]
    return date(year, month, min(start.day, last_day_of_target_month))


def warranty_expires_on(claim: Claim, program: WarrantyProgram) -> date:
    """The last date on which a failure is still inside the warranty period.

    Measured from the in-service date and not from the invoice date or the despatch date. Those
    three are the same date for a machine that is commissioned the week it arrives and are months
    apart for one that sits in a distributor's yard over a winter, and the months in between are
    the ones that get argued about. The in-service date is the one the policy names, so it is the
    one this system measures from, and `REQ-IN-SERVICE-DATE` exists to make a claimant produce it.

    The returned date is **inclusive**: a failure on it is covered. See the module docstring.
    """
    return add_months(claim.in_service_date, program.warranty_months)


def claim_window(claim: Claim, program: WarrantyProgram, as_of: date) -> ClaimWindow:
    """The correction window opened by this rejection, seen from `as_of`.

    Counted in days from the rejection date, because that is what a manufacturer's terms say — "30
    days from the date of this notice" — and converting it into months here would introduce the
    clamping question above into a place where the policy does not ask it.

    Counted from `rejected_on` rather than from the date this system first saw the rejection. The
    two differ by however long the rejection sat in an inbox, and using the later of them would
    hand the distributor days it does not have. A window this system believes is open and the
    manufacturer believes is shut is the one arrangement that guarantees wasted work, because
    everything downstream — the retrieval, the composition, the approval — runs to completion
    before the refusal comes back.

    `as_of` is passed through onto the window rather than being consumed here, so that every
    downstream reader of a `ClaimWindow` can see the date the answer was computed for. A window
    that records only "open" is a window whose answer cannot be checked six months later, and
    kill condition M is graded from records written at the time.
    """
    closes_on = claim.rejected_on + timedelta(days=program.correction_window_days)
    return ClaimWindow(rejected_on=claim.rejected_on, closes_on=closes_on, as_of=as_of)
