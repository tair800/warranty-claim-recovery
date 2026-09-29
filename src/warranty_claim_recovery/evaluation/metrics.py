"""Every number that decides whether this project ships, computed from nothing but its arguments.

This module imports `domain`, `money` and the standard library. It reads no file, opens no
connection and knows nothing about a corpus. That constraint is the whole design and it is worth
stating why, because it costs a layer of plumbing everywhere else.

**A metric that can only be exercised by running the system has never been seen at its boundaries,
and the boundaries are where metrics fail.** A precision with no predictions of its class, a recall
over an empty support, a mean reciprocal rank over a query set where nothing was found, a median of
an empty list: each of those is a division by zero waiting behind a number that will otherwise look
plausible. Project 7 published a kill-condition pass whose numerator was empty by construction and
nobody could tell from the artifact. `tests/test_metrics.py` hands every function here inputs a
reader wrote out by hand, including the empty ones, and asserts what comes back.

### Three decisions that could have gone the other way

**Every rate carries its own numerator and denominator into the artifact.** `Ratio` is not a float.
ADR-001's vacuity guard fails the build when a graded criterion's denominator is zero, and a rate
alone cannot be checked for that: `0/0` and `47/47` both serialise as a number, and only one of them
means anything. Publishing the fraction makes a criterion graded over nothing visible in the file
rather than discoverable by rerunning the evaluation.

**A false recovery is counted on its own and never folded into an F1.** An F1 over four outcomes
averages the expensive failure with the cheap one. A false recovery is an attempt to take money from
a manufacturer on a basis that does not exist; a false denial writes off money the distributor was
owed; a premature write-off sends a recoverable claim to the bin instead of to a person. They have
different costs, different remedies and different audiences, and a single number that moves when any
of them moves tells a reader which way the wind blew rather than what happened. The confusion matrix
is published in full for the same reason: it is the only form in which a reader can compute the
figure this module did not think to publish.

**An undefined rate is reported as undefined rather than as zero.** `Ratio(0, 0).value` is `0.0`
because a float has to be something, and `Ratio.defined` is `False` so that a caller who cares — the
release gate, the macro average — can refuse to treat it as a measurement. Averaging an undefined
per-class F1 into a macro score as though it were a zero is how a system that never predicts a class
comes to look merely mediocre at it instead of blind to it.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any, Final, NamedTuple

from warranty_claim_recovery.domain import RecoveryOutcome
from warranty_claim_recovery.money import Money

__all__ = [
    "AUTHORISING_OUTCOMES",
    "OUTCOME_ORDER",
    "RATE_PLACES",
    "CandidateProfile",
    "ConfusionMatrix",
    "OutcomeScores",
    "Pair",
    "Rank",
    "Ratio",
    "amount_agreement",
    "candidate_profile",
    "false_denials",
    "false_recoveries",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "outcome_agreement",
    "partial_recovery_accuracy",
    "premature_write_offs",
    "recall_at_k",
    "review_rate",
]

#: The order every confusion matrix and every per-outcome table is written in. Fixed here rather
#: than taken from the enum's iteration order at each call site, because an artifact whose rows
#: reorder between runs produces a diff that says nothing and is therefore a diff nobody reads.
OUTCOME_ORDER: Final[tuple[RecoveryOutcome, ...]] = (
    RecoveryOutcome.RECOVERABLE,
    RecoveryOutcome.PARTIALLY_RECOVERABLE,
    RecoveryOutcome.NOT_RECOVERABLE,
    RecoveryOutcome.REVIEW,
)

#: The two outcomes that let a correction leave the system. Written once here and imported by every
#: counter below, so that "the system authorised money" has a single definition. `gate.py` names the
#: same pair in `GateDecision.authorises_recovery`; this module cannot import that property because
#: it counts pairs of outcomes rather than decisions, and two definitions of the same set is one
#: more than there should be — `tests/test_metrics.py` asserts the two agree.
AUTHORISING_OUTCOMES: Final[frozenset[RecoveryOutcome]] = frozenset(
    {RecoveryOutcome.RECOVERABLE, RecoveryOutcome.PARTIALLY_RECOVERABLE}
)

#: Rates are rounded before they are written. Six places is far beyond anything this corpus can
#: resolve — one case in seven hundred moves the fourth place — and it exists so that two runs over
#: the same corpus produce the same bytes rather than differing in a trailing digit of a repeating
#: binary fraction. Rounding can only make two distinct values compare equal, never make an equal
#: pair compare distinct, so it can tighten a strict comparison in a kill criterion and never loosen
#: one.
RATE_PLACES: Final = 6


class Ratio(NamedTuple):
    """A measurement and the population it was measured over, kept together.

    The pair travels together everywhere because they are only meaningful together. ADR-001's
    vacuity guard fails a build whose graded criterion has a zero denominator, and that guard can
    only be applied to something that still carries its denominator by the time it reaches the
    artifact. A function returning a bare float would discard the one field the guard reads.
    """

    numerator: int
    denominator: int

    @property
    def defined(self) -> bool:
        """Whether anything was measured. A rate over nothing is not a rate of zero."""
        return self.denominator > 0

    @property
    def value(self) -> float:
        """The rate, or `0.0` when nothing was measured.

        A float has to be something and `None` would push the question into every caller. The
        honest reading of the zero is `defined`, which is published beside it in every artifact this
        module feeds, so a reader never sees the number without the fact that it measured nothing.
        """
        if not self.defined:
            return 0.0
        return round(self.numerator / self.denominator, RATE_PLACES)

    def as_json(self) -> dict[str, Any]:
        return {
            "numerator": self.numerator,
            "denominator": self.denominator,
            "rate": self.value,
            "defined": self.defined,
        }


Pair = tuple[RecoveryOutcome, RecoveryOutcome]
"""One graded case: the ground truth first, the system's answer second.

The order is truth-then-prediction throughout this module and it is stated because the opposite
convention is equally common and the two are indistinguishable in a tuple. Getting it backwards
swaps precision with recall, which is the one error in this family that produces plausible numbers
in both directions.
"""


class OutcomeScores(NamedTuple):
    """Precision, recall and F1 for one outcome, each still carrying its population.

    F1 is stored as a float rather than a `Ratio` because it is not a count over a population; it is
    a function of two rates. `defined` is therefore explicit: an outcome the system never predicted
    and that never occurred has no F1 at all, and reporting `0.0` for it would say the system is bad
    at something that never came up.
    """

    outcome: RecoveryOutcome
    support: int
    predicted: int
    correct: int
    precision: Ratio
    recall: Ratio
    f1: float
    f1_defined: bool

    def as_json(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome.value,
            "support": self.support,
            "predicted": self.predicted,
            "correct": self.correct,
            "precision": self.precision.as_json(),
            "recall": self.recall.as_json(),
            "f1": self.f1,
            "f1_defined": self.f1_defined,
        }


class ConfusionMatrix:
    """Every (truth, prediction) pair counted, with all sixteen cells present.

    The cells are materialised even when they are zero. A matrix serialised as only its non-zero
    entries makes "the system never once produced this outcome" and "this run did not look at that
    outcome" the same absence, and those are the two readings a person investigating a regression
    most needs to tell apart.

    Not a Pydantic model: it is built inside the evaluation from data the evaluation just computed,
    so there is no boundary here for validation to sit on, and a frozen model would buy nothing that
    the absence of a mutator does not already buy.
    """

    __slots__ = ("_cells", "_total")

    def __init__(self, pairs: Iterable[Pair]) -> None:
        cells: dict[Pair, int] = {
            (truth, predicted): 0 for truth in OUTCOME_ORDER for predicted in OUTCOME_ORDER
        }
        total = 0
        for pair in pairs:
            cells[pair] += 1
            total += 1
        self._cells = cells
        self._total = total

    @property
    def total(self) -> int:
        return self._total

    def count(self, truth: RecoveryOutcome, predicted: RecoveryOutcome) -> int:
        return self._cells[(truth, predicted)]

    def support(self, outcome: RecoveryOutcome) -> int:
        """How often the ground truth said this. The row total."""
        return sum(self._cells[(outcome, predicted)] for predicted in OUTCOME_ORDER)

    def predicted(self, outcome: RecoveryOutcome) -> int:
        """How often the system said this. The column total."""
        return sum(self._cells[(truth, outcome)] for truth in OUTCOME_ORDER)

    def scores(self, outcome: RecoveryOutcome) -> OutcomeScores:
        """Precision, recall and F1 for one outcome, with the two populations they divide by.

        F1 is defined only when both precision and recall are, which is the honest condition: the
        harmonic mean of a number and a non-measurement is not a number. The usual shortcut —
        treating an undefined precision as zero so that F1 always exists — reports a system that has
        never predicted an outcome as being poor at it rather than as having no opinion about it,
        and those need different fixes.
        """
        correct = self.count(outcome, outcome)
        precision = Ratio(correct, self.predicted(outcome))
        recall = Ratio(correct, self.support(outcome))
        defined = precision.defined and recall.defined
        denominator = precision.value + recall.value
        f1 = 0.0
        if defined and denominator > 0:
            f1 = round(2 * precision.value * recall.value / denominator, RATE_PLACES)
        return OutcomeScores(
            outcome=outcome,
            support=self.support(outcome),
            predicted=self.predicted(outcome),
            correct=correct,
            precision=precision,
            recall=recall,
            f1=f1,
            f1_defined=defined,
        )

    def as_json(self) -> dict[str, Any]:
        """The matrix as nested objects keyed by outcome name, in `OUTCOME_ORDER`.

        Keyed by name rather than positionally. A four-by-four array of integers is shorter and
        needs a legend to read, and the legend is the thing that goes stale: a reader six months
        later cannot tell from the numbers alone whether the rows or the columns are the truth.
        """
        return {
            "rows_are_ground_truth": True,
            "columns_are_system_outcomes": True,
            "counts": {
                truth.value: {
                    predicted.value: self.count(truth, predicted) for predicted in OUTCOME_ORDER
                }
                for truth in OUTCOME_ORDER
            },
        }


def outcome_agreement(pairs: Sequence[Pair]) -> Ratio:
    """How often the system's outcome equalled the ground truth's.

    Published beside the confusion matrix and never instead of it. Accuracy over four outcomes with
    an uneven mix is the number that moves least when the expensive failure changes, which is
    exactly the wrong property for the headline figure of a system whose expensive failure is worth
    money.
    """
    return Ratio(sum(1 for truth, predicted in pairs if truth is predicted), len(pairs))


def false_recoveries(pairs: Sequence[Pair]) -> Ratio:
    """Cases the system authorised whose ground truth is `NOT_RECOVERABLE`.

    The expensive failure, counted alone. Filing a correction for a claim that was never recoverable
    asks a manufacturer for money on a basis that does not exist: it is refused, it costs the
    distributor's credibility on every other claim in the same batch, and the person who filed it
    spent an afternoon on it. Kill condition G budgets zero of these on the hold-out.

    The denominator is the number of cases whose truth is `NOT_RECOVERABLE`, not the corpus. A rate
    over the whole corpus would fall as the corpus grew, which would let a system that authorised
    exactly as many bad claims as before report an improvement.
    """
    population = [pair for pair in pairs if pair[0] is RecoveryOutcome.NOT_RECOVERABLE]
    return Ratio(
        sum(1 for _, predicted in population if predicted in AUTHORISING_OUTCOMES),
        len(population),
    )


def false_denials(pairs: Sequence[Pair]) -> Ratio:
    """Cases the system refused outright whose ground truth authorises a recovery.

    The other expensive failure, and the one a system tuned only against kill condition G will
    acquire: refusing everything scores zero false recoveries. `CLAUDE.md` §1 names it first among
    the failures this project exists to prevent — *a claim written off because nobody noticed* — and
    it is money the distributor was owed and did not ask for.

    `REVIEW` is deliberately not counted here. Sending a case to a person is a deferral, not a
    denial: the money is still recoverable the next morning. `premature_write_offs` counts the
    opposite mistake, and keeping the two apart is what lets a reader see whether a system is
    cautious or simply wrong.
    """
    population = [pair for pair in pairs if pair[0] in AUTHORISING_OUTCOMES]
    return Ratio(
        sum(1 for _, predicted in population if predicted is RecoveryOutcome.NOT_RECOVERABLE),
        len(population),
    )


def premature_write_offs(pairs: Sequence[Pair]) -> Ratio:
    """Cases the system closed as `NOT_RECOVERABLE` whose ground truth is `REVIEW`.

    A case whose evidence is incomplete is not a case that cannot be recovered; it is a case nobody
    has finished. Writing it off closes a claim that a person, given the missing document, would
    have recovered — and unlike a false denial it also removes the case from the queue, so nobody
    ever finds out. Counted separately from `false_denials` because the remedies differ: a false
    denial is an arithmetic or a rule error, a premature write-off is a routing error.
    """
    population = [pair for pair in pairs if pair[0] is RecoveryOutcome.REVIEW]
    return Ratio(
        sum(1 for _, predicted in population if predicted is RecoveryOutcome.NOT_RECOVERABLE),
        len(population),
    )


def partial_recovery_accuracy(pairs: Sequence[Pair]) -> Ratio:
    """How often a partially recoverable claim was reported as partially recoverable.

    Its own figure because the four-state design exists for this case. A partial recovery collapsed
    into `RECOVERABLE` files a total the adjudicator refuses; collapsed into `NOT_RECOVERABLE` it
    writes off money that was owed. Both collapses are invisible in an accuracy figure that treats
    the four outcomes as interchangeable labels.
    """
    population = [pair for pair in pairs if pair[0] is RecoveryOutcome.PARTIALLY_RECOVERABLE]
    return Ratio(
        sum(1 for _, predicted in population if predicted is RecoveryOutcome.PARTIALLY_RECOVERABLE),
        len(population),
    )


def review_rate(pairs: Sequence[Pair]) -> Ratio:
    """The share of cases the system sent to a person.

    Not a quality measure in either direction, and published because it is the cost of the
    system: every point of it is somebody's afternoon. A system that escalates everything commits
    no false recoveries and saves nobody any work, and this is the number that makes that visible
    rather than leaving it hidden behind a clean confusion matrix.
    """
    return Ratio(
        sum(1 for _, predicted in pairs if predicted is RecoveryOutcome.REVIEW), len(pairs)
    )


def amount_agreement(pairs: Sequence[tuple[Money, Money]]) -> Ratio:
    """How many computed recoverable amounts equal the ground truth exactly.

    Exact `Decimal` equality including the currency, with no tolerance, which is kill condition F.
    `Money.__eq__` compares the currency and the exact amount, so an amount that is right to the
    penny in the wrong currency is a mismatch here — as it must be, because there is no exchange
    rate this system holds and a converted figure would be wrong in a way no audit could attribute.

    A tolerance was rejected rather than omitted. Every arithmetic step in this system is exact and
    rounded once, so a discrepancy of a penny is not a rounding artefact to be absorbed; it is a
    step applied in the wrong order, and the tolerance would hide exactly the defect the criterion
    exists to catch.
    """
    return Ratio(sum(1 for expected, actual in pairs if expected == actual), len(pairs))


# ------------------------------------------------------------------------------------ retrieval

Rank = int | None
"""Where the governing clause was found, one-based, or `None` when it was not in the results.

One-based because that is how a person reads a result list, and `None` rather than a sentinel such
as zero or `k + 1` because every metric below has to branch on "was it found at all" and a sentinel
that participates in arithmetic is a sentinel that eventually gets averaged.
"""


def recall_at_k(ranks: Sequence[Rank], k: int) -> Ratio:
    """How often the governing clause appeared in the first `k` results.

    There is exactly one relevant clause per query — the corpus names it — so recall at k, hit rate
    at k and precision at k over the relevant set are the same number here, and this one is called
    recall because that is the name ADR-001 §5 fixed for kill condition J.

    A rank beyond `k` counts as a miss even when the caller retrieved more than `k`. The criterion
    is about what a composer would actually be handed, and a clause ranked eleventh is not evidence
    that was available to anybody.
    """
    if k < 1:
        raise ValueError(f"k={k} asks how often the clause was in the first {k} results")
    return Ratio(sum(1 for rank in ranks if rank is not None and rank <= k), len(ranks))


def mean_reciprocal_rank(ranks: Sequence[Rank]) -> float:
    """The mean of one over the rank, counting a miss as zero.

    Published because recall at k cannot tell a system that puts the governing clause first from one
    that puts it fifth, and on a corpus where the candidate pool is small those are very different
    systems: the composer cites the top-ranked authority, so rank one is the answer and rank five is
    a clause that was merely available. A miss contributes zero rather than being dropped, so the
    denominator stays the whole query set and a system cannot improve its mean by failing more
    often.
    """
    if not ranks:
        return 0.0
    total = sum(1.0 / rank for rank in ranks if rank is not None)
    return round(total / len(ranks), RATE_PLACES)


def ndcg_at_k(ranks: Sequence[Rank], k: int) -> float:
    """Normalised discounted cumulative gain, for the single-relevant-document case.

    With one relevant clause per query the ideal gain is `1 / log2(2) == 1`, so the normalisation is
    a division by one and the figure is the mean discounted gain. It is published anyway, under its
    usual name, because a reader comparing this project against a retrieval benchmark expects the
    name and would otherwise have to work out that the simplification is safe. The discount is
    `1 / log2(rank + 1)`, which is the standard one; nothing here is a variant.
    """
    if k < 1:
        raise ValueError(f"k={k} asks for a gain over the first {k} results")
    if not ranks:
        return 0.0
    gain = sum(1.0 / math.log2(rank + 1) for rank in ranks if rank is not None and rank <= k)
    return round(gain / len(ranks), RATE_PLACES)


class CandidateProfile(NamedTuple):
    """How many clauses survived the metadata filter per query, and whether ranking can matter.

    This exists because of a specific failure elsewhere in this portfolio: project 7's retrieval
    criterion was saturated by construction — its filter left so few candidates that every one of
    them fitted inside `k` — and the recall it published was a property of the filter rather than of
    the ranker. Nobody noticed until after the scoring was done, because nothing in the artifact
    said how many candidates there had been.

    So the profile is published beside every recall figure, and `ranking_can_change_recall` states
    the conclusion in the artifact rather than leaving a reader to compute it. When the median
    candidate count is at or below `k`, every candidate is returned whatever the ranking, recall is
    pinned at one, and the criterion is measuring the `WHERE` clause.
    """

    queries: int
    median: float
    p95: int
    minimum: int
    maximum: int
    at_or_below_k: Ratio
    k: int

    @property
    def ranking_can_change_recall(self) -> bool:
        return self.median > self.k

    @property
    def verdict(self) -> str:
        if self.ranking_can_change_recall:
            return (
                f"the median query has {self.median} candidates against k={self.k}, so the ranker "
                f"chooses which {self.k} of them are returned and recall at k is a measurement of "
                f"the ranking"
            )
        return (
            f"SATURATED: the median query has only {self.median} candidates against k={self.k}, so "
            f"every candidate is returned whatever the ranking. Recall at k here measures the "
            f"metadata filter and not the ranker, and no ranking change can move it."
        )

    def as_json(self) -> dict[str, Any]:
        return {
            "queries": self.queries,
            "k": self.k,
            "median_candidates": self.median,
            "p95_candidates": self.p95,
            "min_candidates": self.minimum,
            "max_candidates": self.maximum,
            "queries_with_at_most_k_candidates": self.at_or_below_k.as_json(),
            "ranking_can_change_recall": self.ranking_can_change_recall,
            "verdict": self.verdict,
        }


def candidate_profile(counts: Sequence[int], k: int) -> CandidateProfile:
    """Summarise the candidate pool sizes. Deterministic, with no interpolation anywhere.

    The 95th percentile is the nearest-rank value — the smallest observed count at or above 95% of
    the sorted population — rather than an interpolated one. An interpolated percentile of a set of
    integers is a number that was never observed, and a reader who goes looking for the query that
    produced it will not find one. The median is the stdlib's, which averages the middle pair on an
    even population; it is a summary rather than an observation and is labelled as a float so that
    nobody reads `15.0` as a query that existed.
    """
    if k < 1:
        raise ValueError(f"k={k} is not a result-list length")
    if not counts:
        return CandidateProfile(
            queries=0, median=0.0, p95=0, minimum=0, maximum=0, at_or_below_k=Ratio(0, 0), k=k
        )
    ordered = sorted(counts)
    middle = len(ordered) // 2
    median = (
        float(ordered[middle]) if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    )
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return CandidateProfile(
        queries=len(ordered),
        median=median,
        p95=ordered[index],
        minimum=ordered[0],
        maximum=ordered[-1],
        at_or_below_k=Ratio(sum(1 for count in ordered if count <= k), len(ordered)),
        k=k,
    )
