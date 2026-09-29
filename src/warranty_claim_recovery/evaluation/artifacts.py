"""The three payloads `tests/test_kill_criteria.py` grades, assembled from counted facts.

This module takes data and returns dictionaries. It opens no connection, reads no file and embeds
nothing, so `tests/test_evaluation.py` can hand it cases a reader wrote out by hand and assert on
the exact keys that come back. That matters more here than anywhere else in the package: the kill
test reads these files by name, key by key, and it imports nothing from this repository. A key
renamed here and not there does not fail loudly — the grader raises a `KeyError` in a CI step
nobody was reading, or worse, `metric()` on the console prints *not measured* beside a criterion
that was in fact measured and passed. The key contract is therefore a thing to be tested, not a
thing to be careful about.

### Why the shapes are what they are

**Every criterion carries its own denominator into the file.** ADR-001's vacuity guard fails a
build whose graded criterion was graded over an empty population, and that guard can only read a
denominator that survived into the artifact. `cases_scored`, `holdout_not_recoverable_cases`,
`cases_with_a_closed_window`, `requirements_marked_satisfied`, `citations_checked` and
`holdout_queries` are all denominators before they are anything else, and each is published beside
the number it divides. Project 7 published a passing criterion whose numerator was empty by
construction and nothing in the file said so.

**False recoveries are a top-level integer and appear in no average.** `metrics.false_recoveries`
argues the case at length; the consequence for this module is that `holdout_false_recoveries` sits
at the top of `recovery.json`, above the confusion matrix and above every rate, and the per-outcome
F1 table is published *underneath* it rather than in place of it. There is no number of correct
recoveries that pays for one claim filed against a manufacturer with nothing behind it, so there is
no average in which that count belongs.

**Disagreements are published as pairs with the system's own words.** A confusion matrix says that
a hundred and fifty cases came out one way rather than another; it does not say why, and the reader
who wants to know has to rerun the evaluation. So `outcome_divergences` groups every disagreement
by its `(ground truth, system)` pair, counts it on each side of the split, names the first few
claims, and quotes the reason **the gate itself wrote** for one of them. Nothing in that block is
this module's opinion: the reason string is `GateDecision.reason`, produced by the rule that fired.

**Retrieval and composition errors are separated, and the rule is stated in the file.** The skill
matrix asks this project for retrieval-versus-generation attribution, and an attribution that a
reader cannot check is a label. So `groundedness.json` publishes the rule it applied — the
governing clause was not in the k results, or it was and the answer was still wrong — along with
the population each side was counted over. A case can be a retrieval error and a composition error
in the informal sense; this rule makes it exactly one of them, because two overlapping counts that
sum to more than the errors is a table nobody can read.

### Rejected

**A single `summary.json`.** One file is easier to write and it makes the kill test's failure
message useless: `assert evidence["amount_mismatches"] <= 0` failing tells a reader which of three
concerns broke when the file is named after the concern, and tells them nothing when it is not. The
three names are also the names ADR-001 §5 uses, so a criterion and the file that grades it can be
matched without a mapping.

**Rounding rates at the point of writing.** They are rounded in `metrics`, once, at the point of
computation, so that the number in the artifact and the number a caller compared against are the
same number. Rounding here as well would give the console and the release gate two slightly
different figures to quote, and the disagreement would appear exactly when somebody was reading
both.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final, NamedTuple

from warranty_claim_recovery.corpus.holdout import DEVELOPMENT, HOLDOUT
from warranty_claim_recovery.domain import Citation, RecoveryOutcome
from warranty_claim_recovery.evaluation.metrics import (
    OUTCOME_ORDER,
    ConfusionMatrix,
    Pair,
    Rank,
    amount_agreement,
    candidate_profile,
    false_denials,
    false_recoveries,
    mean_reciprocal_rank,
    ndcg_at_k,
    outcome_agreement,
    partial_recovery_accuracy,
    premature_write_offs,
    recall_at_k,
    review_rate,
)
from warranty_claim_recovery.evaluation.pipeline import CaseAssessment
from warranty_claim_recovery.money import Money
from warranty_claim_recovery.retrieval.citations import citation_failure, verify_citation

__all__ = [
    "ATTRIBUTION_RULE",
    "MAX_EXAMPLES",
    "SYNTHETIC_NOTICE",
    "GradedCase",
    "GradedRequirement",
    "Measurement",
    "graded",
    "groundedness_artifact",
    "recovery_artifact",
    "retrieval_artifact",
    "summary_lines",
]

#: Repeated in the body of every artifact this module writes. `CLAUDE.md` §3.8 requires the corpus
#: to be described as synthetic in the body of every generated file, not only in a README: a JSON
#: file travels, and the sentence that says what it is has to travel with it.
SYNTHETIC_NOTICE: Final = (
    "Every figure here was measured over a synthetic corpus generated from a committed seed. It "
    "describes no real warranty programme and no manufacturer's policy, and the rates mean nothing "
    "outside this corpus."
)

#: How many claim identifiers to name per divergence class or per failure class. Enough that a
#: reader can go and look at one, few enough that the file stays readable. The list is a sample and
#: is labelled as one; the count beside it is the whole population.
MAX_EXAMPLES: Final = 5

#: The rule `groundedness.json` applies, published in the file. Stated as one string here so the
#: artifact's description of the rule and the code that applies it cannot drift apart.
ATTRIBUTION_RULE: Final = (
    "a case whose outcome disagrees with the ground truth is a RETRIEVAL error when the governing "
    "clause was not among the k retrieved, and a COMPOSITION error when it was retrieved and the "
    "answer was still wrong. The two are exclusive and sum to the number of wrong cases."
)


class GradedRequirement(NamedTuple):
    """One requirement as the grader sees it: whether it was satisfied, and what backs it.

    `citation` is carried rather than a boolean saying whether one exists, because kill condition I
    has to slice a document at the offsets this citation names. A grader that recorded only
    *whether* a requirement was cited could answer H and would have nothing to check for I, and the
    two criteria exist precisely because a citation that is present and a citation that is true are
    different claims.
    """

    requirement_id: str
    satisfied: bool
    citation: Citation | None


class GradedCase(NamedTuple):
    """One assessed case, reduced to the facts the three artifacts count.

    A deliberate narrowing of `CaseAssessment`. The artifacts take this rather than the assessment
    so that a test can build the inputs by hand — a `CaseAssessment` carries a `Claim`, a
    `WarrantyProgram`, a `RecoveryComputation` and a `GateDecision`, none of which any counter here
    reads, and a test forced to construct all four to check a key name would be a test nobody
    writes. `graded` is the only place the narrowing happens, so there is one definition of what
    the grader looks at.

    `reason` is the gate's own sentence, carried so that `outcome_divergences` can quote the system
    rather than paraphrase it. It is `None` on the two paths that never reach the gate.
    """

    claim_id: str
    program_id: str
    split: str
    rejection_code: str
    truth_outcome: RecoveryOutcome
    system_outcome: RecoveryOutcome
    truth_amount: Money
    system_amount: Money
    window_closed: bool
    authorises_recovery: bool
    governing_clause_id: str
    governing_clause_rank: Rank
    authority_clause_id: str | None
    requirements: tuple[GradedRequirement, ...]
    reason: str | None
    escalation: str | None

    @property
    def outcome_pair(self) -> Pair:
        """Ground truth first, system second. `metrics.Pair` fixes the order and says why."""
        return (self.truth_outcome, self.system_outcome)

    @property
    def amount_pair(self) -> tuple[Money, Money]:
        return (self.truth_amount, self.system_amount)

    @property
    def correct(self) -> bool:
        return self.truth_outcome is self.system_outcome

    @property
    def governing_clause_retrieved(self) -> bool:
        return self.governing_clause_rank is not None


def graded(assessment: CaseAssessment) -> GradedCase:
    """Reduce one assessment to the facts the artifacts count, and nothing else.

    The `requirements_satisfied` count is not taken from `GateSignals`, although the gate computed
    one. `GateSignals` is absent on the two paths that never reach the gate — a claim in the wrong
    currency, and a case whose governing authority was not retrieved — and a grader that read the
    count from the decision would silently stop counting the requirements of exactly the cases most
    likely to be interesting. Counting the requirement list directly means the denominator of kill
    condition H is the same on every path.
    """
    return GradedCase(
        claim_id=assessment.case.claim.claim_id,
        program_id=assessment.case.claim.program_id,
        split=assessment.case.split,
        rejection_code=assessment.case.claim.rejection_code.value,
        truth_outcome=assessment.case.truth_outcome,
        system_outcome=assessment.outcome,
        truth_amount=assessment.case.truth_amount,
        system_amount=assessment.recoverable_amount,
        window_closed=assessment.window_closed,
        authorises_recovery=assessment.authorises_recovery,
        governing_clause_id=assessment.case.governing_clause_id,
        governing_clause_rank=assessment.gold_rank,
        authority_clause_id=assessment.authority_clause_id,
        requirements=tuple(
            GradedRequirement(
                requirement_id=item.requirement_id,
                satisfied=item.status.value == "SATISFIED",
                citation=item.citation,
            )
            for item in assessment.requirements
        ),
        reason=None if assessment.decision is None else assessment.decision.reason,
        escalation=assessment.escalation,
    )


def _of_split(cases: Sequence[GradedCase], split: str) -> tuple[GradedCase, ...]:
    return tuple(case for case in cases if case.split == split)


def _examples(cases: Sequence[GradedCase]) -> list[str]:
    """The first few claim identifiers, in corpus order.

    Corpus order rather than sorted, so the sample is the first few a reader would meet stepping
    through the evaluation rather than the first few alphabetically — which on this corpus means
    the same manufacturer every time, and a sample drawn from one manufacturer says nothing about
    whether a failure class is general.
    """
    return [case.claim_id for case in cases[:MAX_EXAMPLES]]


# ------------------------------------------------------------------------------------- recovery


def _outcome_block(cases: Sequence[GradedCase]) -> dict[str, Any]:
    """Everything counted about one population of cases: the matrix, the errors, the money.

    Written once and applied to the hold-out, to the development split and to the whole corpus, so
    that the three blocks cannot drift into three slightly different definitions of the same word.
    ADR-001 §7 forbids tuning against the hold-out, and the development numbers are published
    beside it so that a reader can see whether the two populations behave alike — a system that
    scores far better on the split it was built against is a system that learned the split.
    """
    pairs = [case.outcome_pair for case in cases]
    matrix = ConfusionMatrix(pairs)
    closed = [case for case in cases if case.window_closed]
    return {
        "cases": len(cases),
        # First, before any rate. The expensive failure, counted alone.
        "false_recoveries": false_recoveries(pairs).numerator,
        "false_recovery_rate": false_recoveries(pairs).as_json(),
        "false_denials": false_denials(pairs).numerator,
        "false_denial_rate": false_denials(pairs).as_json(),
        "premature_write_offs": premature_write_offs(pairs).numerator,
        "premature_write_off_rate": premature_write_offs(pairs).as_json(),
        "partial_recovery_accuracy": partial_recovery_accuracy(pairs).as_json(),
        "review_rate": review_rate(pairs).as_json(),
        "outcome_agreement": outcome_agreement(pairs).as_json(),
        "confusion_matrix": matrix.as_json(),
        "per_outcome": [matrix.scores(outcome).as_json() for outcome in OUTCOME_ORDER],
        "amount_agreement": amount_agreement([case.amount_pair for case in cases]).as_json(),
        "amount_mismatches": sum(1 for case in cases if case.truth_amount != case.system_amount),
        "cases_with_a_closed_window": len(closed),
        "resubmissions_after_window_closed": sum(1 for case in closed if case.authorises_recovery),
        "not_recoverable_cases": sum(
            1 for case in cases if case.truth_outcome is RecoveryOutcome.NOT_RECOVERABLE
        ),
    }


def _divergences(cases: Sequence[GradedCase]) -> list[dict[str, Any]]:
    """Every disagreement grouped by its `(truth, system)` pair, with the gate's own reason.

    Published because a confusion matrix answers *how many* and never *which*. A systematic
    divergence — one construction, one rule, the same hundred and fifty cases — and a scatter of
    unrelated mistakes produce identical matrices, and they are not the same finding: the first is
    one decision to examine and the second is a system that does not work. Grouping by pair and
    naming the claims makes the difference visible in the file rather than discoverable by rerunning
    the evaluation with a debugger.

    The quoted reason is `GateDecision.reason`, written by the rule that fired. Nothing here
    paraphrases it, because a paraphrase of a refusal is a second opinion about why the refusal
    happened, and the day the two differ the reader believes the wrong one.
    """
    grouped: dict[Pair, list[GradedCase]] = {}
    for case in cases:
        if not case.correct:
            grouped.setdefault(case.outcome_pair, []).append(case)

    order = {outcome: position for position, outcome in enumerate(OUTCOME_ORDER)}
    rows: list[dict[str, Any]] = []
    for pair in sorted(grouped, key=lambda item: (order[item[0]], order[item[1]])):
        group = grouped[pair]
        reasons = [case.reason for case in group if case.reason]
        escalations = [case.escalation for case in group if case.escalation]
        rows.append(
            {
                "ground_truth": pair[0].value,
                "system": pair[1].value,
                "cases": len(group),
                "holdout": sum(1 for case in group if case.split == HOLDOUT),
                "development": sum(1 for case in group if case.split == DEVELOPMENT),
                "rejection_codes": sorted({case.rejection_code for case in group}),
                "example_claim_ids": _examples(group),
                "example_reason_from_the_gate": reasons[0] if reasons else None,
                "example_escalation": escalations[0] if escalations else None,
            }
        )
    return rows


def recovery_artifact(cases: Sequence[GradedCase]) -> dict[str, Any]:
    """`artifacts/recovery.json` — kill conditions F, G and M, and the confusion behind them.

    The three criteria are graded over three different populations and the file says so beside each
    one. F is the whole corpus, because an arithmetic error is an arithmetic error wherever it
    happens. G is the hold-out, because ADR-001 §7 fixes that as the population no decision was
    tuned against. M is the whole corpus, because the claim window is arithmetic and a case with a
    shut window is refusable on every split.

    `amount_exact_match_rate` is published as a bare float as well as a fraction, because the kill
    test asserts equality with `1.0` and a nested object would make that assertion read through two
    keys. The fraction is the one a reader should quote; the float is there so the grader stays a
    one-line comparison a person can check.
    """
    holdout = _of_split(cases, HOLDOUT)
    development = _of_split(cases, DEVELOPMENT)
    whole = _outcome_block(cases)
    held = _outcome_block(holdout)
    mismatched = [case for case in cases if case.truth_amount != case.system_amount]

    return {
        "is_synthetic_corpus": True,
        "notice": SYNTHETIC_NOTICE,
        "what_this_is": (
            "kill conditions F, G and M: the recoverable amount against the generator's ground "
            "truth at exact Decimal equality, the number of hold-out cases authorised whose truth "
            "is NOT_RECOVERABLE, and the number of cases authorised after their correction window "
            "had closed"
        ),
        "graded_by": "tests/test_kill_criteria.py",
        # --- kill condition G. First in the file, and its own integer, deliberately.
        "holdout_false_recoveries": held["false_recoveries"],
        "holdout_not_recoverable_cases": held["not_recoverable_cases"],
        "false_recoveries_whole_corpus": whole["false_recoveries"],
        "why_false_recoveries_stand_alone": (
            "a false recovery is a claim filed against a manufacturer on a basis that does not "
            "exist. There is no number of correct recoveries that pays for one, so it is never "
            "folded into an F1 or an accuracy. ADR-001 §5 G budgets zero of them on the hold-out."
        ),
        # --- kill condition F.
        "cases_scored": len(cases),
        "amount_mismatches": whole["amount_mismatches"],
        "amount_exact_match_rate": whole["amount_agreement"]["rate"],
        "amount_agreement": whole["amount_agreement"],
        "amount_comparison": (
            "exact Decimal equality including the currency, parsed from the string form in "
            "truth.json. No tolerance: every step in this system is exact and rounded once, so a "
            "discrepancy of a penny is a step applied in the wrong order rather than a rounding "
            "artefact, and a tolerance would hide exactly the defect the criterion exists to catch"
        ),
        "amount_mismatch_examples": _examples(mismatched),
        # --- kill condition M.
        "cases_with_a_closed_window": whole["cases_with_a_closed_window"],
        "resubmissions_after_window_closed": whole["resubmissions_after_window_closed"],
        # --- the console's headline figures, hold-out.
        "holdout_false_denials": held["false_denials"],
        "holdout_review_rate": held["review_rate"]["rate"],
        # --- everything else, both splits separately and then together.
        "splits": {
            HOLDOUT: held,
            DEVELOPMENT: _outcome_block(development),
        },
        "whole_corpus": whole,
        "outcome_divergences": _divergences(cases),
        "outcome_divergence_note": (
            "grouped by (ground truth, system) pair with the gate's own reason quoted. A "
            "systematic divergence and a scatter of unrelated mistakes produce the same confusion "
            "matrix and are not the same finding; this block is what tells them apart."
        ),
        "split_note": (
            "the hold-out is fixed by ADR-001 §7's rule and was frozen before anything was scored "
            "against it. The development figures are published beside it so a reader can see "
            "whether the two populations behave alike: a system that scores far better on the "
            "split it was built against is a system that learned the split."
        ),
    }


# ------------------------------------------------------------------------------------ retrieval


class Measurement(NamedTuple):
    """One retrieval system's answers, as the rank of the governing clause per query.

    A rank per query rather than a score, because every figure in `retrieval.json` — recall at k,
    the reciprocal rank, the discounted gain — is a function of where the governing clause came,
    and a system that published a rate directly would be a system whose rate nobody could
    recompute. `metrics.Rank` is `None` for a miss rather than a sentinel, for the reason recorded
    there.
    """

    name: str
    description: str
    holdout_ranks: tuple[Rank, ...]
    development_ranks: tuple[Rank, ...]


def _scores(ranks: Sequence[Rank], k: int) -> dict[str, Any]:
    """Recall at k, MRR and nDCG at k over one query set, each with its population.

    All three are published together and none of them alone. Recall at k cannot distinguish a
    system that puts the governing clause first from one that puts it fifth, and on this corpus
    that is the difference between a composer citing the authority and a composer citing a
    distractor that happened to be in the list. MRR cannot tell a reader whether the clause was
    inside the window a composer actually sees. nDCG is published under its usual name so a reader
    comparing against a retrieval benchmark does not have to work out that the single-relevant-
    document simplification is safe.
    """
    found = recall_at_k(ranks, k)
    positions = [rank for rank in ranks if rank is not None and rank <= k]
    return {
        "queries": len(ranks),
        "recall_at_k": found.value,
        "hits": found.numerator,
        "misses": len(ranks) - found.numerator,
        "recall": found.as_json(),
        "mean_reciprocal_rank": mean_reciprocal_rank(ranks),
        "ndcg_at_k": ndcg_at_k(ranks, k),
        "rank_histogram": {
            str(position): positions.count(position) for position in range(1, k + 1)
        },
    }


def _measured(measurement: Measurement, k: int, *, ranks: Sequence[Rank]) -> dict[str, Any]:
    return {"name": measurement.name, "description": measurement.description, **_scores(ranks, k)}


def retrieval_artifact(
    *,
    k: int,
    system: Measurement,
    baselines: Sequence[Measurement],
    diagnostics: Sequence[Measurement] = (),
    holdout_candidates: Sequence[int],
    development_candidates: Sequence[int],
) -> dict[str, Any]:
    """`artifacts/retrieval.json` — kill condition J, its four baselines and the candidate profile.

    `baselines` holds exactly the four ADR-001 §6 predeclared and nothing else, because the kill
    test iterates that mapping and requires the system to beat every entry. A diagnostic added to
    it would become a criterion, and a criterion invented after the fact is the thing the whole
    predeclaration exists to prevent. `diagnostics` is a separate mapping for the same reason: the
    code-lookup ceiling is worth publishing and is not one of the four.

    **The candidate profile is not decoration.** Project 7's retrieval criterion was saturated by
    construction — its metadata filter left so few candidates that all of them fitted inside k, so
    recall was pinned at one whatever the ranker did — and nobody noticed until after the scoring,
    because nothing in the artifact said how many candidates there had been. `metrics.
    CandidateProfile.verdict` states the conclusion in the file, in words, and says SATURATED when
    the median query has k candidates or fewer. A reader should not have to derive that from a
    number they would have to know to look for.
    """
    held = candidate_profile(holdout_candidates, k)
    scored = {
        measurement.name: _measured(measurement, k, ranks=measurement.holdout_ranks)
        for measurement in baselines
    }
    system_recall = float(_scores(system.holdout_ranks, k)["recall_at_k"])
    floors = {name: float(row["recall_at_k"]) for name, row in scored.items()}
    best = max(floors.values(), default=0.0)

    return {
        "is_synthetic_corpus": True,
        "notice": SYNTHETIC_NOTICE,
        "what_this_is": (
            "kill condition J: hold-out recall at k for the governing clause, against the four "
            "baselines ADR-001 §6 predeclared, each of which removes or replaces a component of "
            "the retrieval stage itself"
        ),
        "graded_by": "tests/test_kill_criteria.py",
        "k": k,
        "holdout_queries": len(system.holdout_ranks),
        "development_queries": len(system.development_ranks),
        "system": _measured(system, k, ranks=system.holdout_ranks),
        "baselines": scored,
        "beats_every_baseline": all(system_recall > floor for floor in floors.values()),
        "best_baseline_recall_at_k": best,
        "margin_over_best_baseline": round(system_recall - best, 6),
        "baseline_note": (
            "each baseline differs in the retrieval stage. ADR-001 §5 records why: project 7's "
            "equivalent criterion required beating a baseline that removed only a downstream "
            "component and therefore ran the identical retriever, which was an impossible target "
            "and a defect in the criterion rather than in the system."
        ),
        "development": {
            "system": _measured(system, k, ranks=system.development_ranks),
            "baselines": {
                measurement.name: _measured(measurement, k, ranks=measurement.development_ranks)
                for measurement in baselines
            },
            "candidate_profile": candidate_profile(development_candidates, k).as_json(),
        },
        "development_note": (
            "published beside the hold-out and graded by nothing. Nothing was tuned against the "
            "hold-out after it was scored; the development figures are here so a reader can see "
            "whether the two populations behave alike."
        ),
        "diagnostics": {
            measurement.name: _measured(measurement, k, ranks=measurement.holdout_ranks)
            for measurement in diagnostics
        },
        "diagnostic_note": (
            "not baselines and not graded. ADR-001 §6 fixed four baselines before any score "
            "existed, and anything measured afterwards is a diagnostic however interesting it is."
        ),
        "candidate_profile": held.as_json(),
        "candidate_profile_verdict": held.verdict,
        "ranking_can_change_recall": held.ranking_can_change_recall,
        "why_the_candidate_profile_is_here": (
            "project 7's retrieval criterion was saturated by construction: its metadata filter "
            "left so few candidates that every one of them fitted inside k, recall could not move "
            "whatever the ranker did, and the figure published was a property of the WHERE clause. "
            "Nothing in that artifact said how many candidates there had been, so nobody noticed "
            "until after the scoring. If the median here is at or below k, this criterion is "
            "measuring the filter and not the ranker, and the verdict above says so in words."
        ),
    }


# --------------------------------------------------------------------------------- groundedness


def _citation_rows(cases: Sequence[GradedCase]) -> list[tuple[GradedCase, GradedRequirement]]:
    return [
        (case, requirement)
        for case in cases
        for requirement in case.requirements
        if requirement.satisfied and requirement.citation is not None
    ]


def groundedness_artifact(
    cases: Sequence[GradedCase], documents: Mapping[str, str]
) -> dict[str, Any]:
    """`artifacts/groundedness.json` — kill conditions H and I, and the error attribution.

    **Every citation is checked by slicing the document at the offsets it names.** The check is
    `retrieval.citations.verify_citation`, the same function the loader used when it refused a
    clause that was not where it said it was — not a second implementation of the same idea, which
    is how a verifier comes to agree with a bug. `retrieval/citations.py` records at length why the
    membership test `quote in document_text` is refused: these policies are written from templates
    and the same sentence appears in three sections of one document, so membership succeeds no
    matter which of the three the offsets address.

    **A citation naming a document that is not in the corpus is unfaithful, not unchecked.** The
    alternative — skipping it and leaving the denominator smaller — is the vacuity failure ADR-001
    was written against: a citation into a document nobody can produce is exactly the citation an
    adjudicator cannot follow, and dropping it from the count would make an unverifiable citation
    cheaper than a wrong one.

    **The attribution is exclusive.** `ATTRIBUTION_RULE` is published in the file. A case that is
    wrong is a retrieval error when the governing clause was not among the k retrieved, and a
    composition error otherwise; the two counts sum to the number of wrong cases, so a reader can
    check the arithmetic in the file rather than trusting it.
    """
    requirements = [requirement for case in cases for requirement in case.requirements]
    satisfied = [requirement for requirement in requirements if requirement.satisfied]
    uncited = [requirement for requirement in satisfied if requirement.citation is None]

    rows = _citation_rows(cases)
    failures: list[dict[str, Any]] = []
    for case, requirement in rows:
        citation = requirement.citation
        if citation is None:  # pragma: no cover - excluded by `_citation_rows`
            continue
        text = documents.get(citation.document_id)
        if text is None:
            failures.append(
                {
                    "claim_id": case.claim_id,
                    "requirement_id": requirement.requirement_id,
                    "clause_id": citation.clause_id,
                    "document_id": citation.document_id,
                    "problem": (
                        f"{citation.document_id} is not in the corpus, so the span this citation "
                        f"names cannot be produced by anybody"
                    ),
                }
            )
            continue
        if not verify_citation(citation, text):
            failures.append(
                {
                    "claim_id": case.claim_id,
                    "requirement_id": requirement.requirement_id,
                    "clause_id": citation.clause_id,
                    "document_id": citation.document_id,
                    "problem": citation_failure(citation, text),
                }
            )

    wrong = [case for case in cases if not case.correct]
    retrieval_errors = [case for case in wrong if not case.governing_clause_retrieved]
    composition_errors = [case for case in wrong if case.governing_clause_retrieved]
    misrouted = [
        case
        for case in composition_errors
        if case.authority_clause_id is not None
        and case.authority_clause_id != case.governing_clause_id
    ]

    return {
        "is_synthetic_corpus": True,
        "notice": SYNTHETIC_NOTICE,
        "what_this_is": (
            "kill conditions H and I: every requirement this system reported satisfied carries a "
            "citation, and every cited span is present verbatim at the offsets it names in the "
            "document version it names"
        ),
        "graded_by": "tests/test_kill_criteria.py",
        "cases_scored": len(cases),
        # --- kill condition H.
        "requirements_assessed": len(requirements),
        "requirements_marked_satisfied": len(satisfied),
        "requirements_satisfied_without_citation": len(uncited),
        "how_h_is_enforced": (
            "domain.Requirement refuses to construct a SATISFIED requirement with no citation, so "
            "this count is zero by construction rather than by luck. It is measured anyway: the "
            "criterion asks what the running system produced, and a validator that was removed or "
            "bypassed would show up here rather than in a review of a diff."
        ),
        # --- kill condition I.
        "citations_checked": len(rows),
        "unfaithful_citations": len(failures),
        "unfaithful_citation_detail": failures[:MAX_EXAMPLES],
        "how_i_is_checked": (
            "document_text[start:end] == quote, against the document read from the corpus, using "
            "retrieval.citations.verify_citation — the same function the loader used to refuse a "
            "clause that was not where it said it was. A membership test would pass for a citation "
            "whose offsets had drifted onto a different section carrying the same template "
            "sentence, which is the failure this check exists to catch."
        ),
        "documents_available": len(documents),
        # --- the skill matrix's retrieval-versus-generation attribution.
        "error_attribution": {
            "rule": ATTRIBUTION_RULE,
            "cases_wrong": len(wrong),
            "retrieval_errors": len(retrieval_errors),
            "composition_errors": len(composition_errors),
            "sums_to_cases_wrong": len(retrieval_errors) + len(composition_errors) == len(wrong),
            "retrieval_error_examples": _examples(retrieval_errors),
            "composition_error_examples": _examples(composition_errors),
            "composition_errors_citing_another_authority": len(misrouted),
            "composition_errors_citing_another_authority_note": (
                "a subset of the composition errors: the governing clause was retrieved and a "
                "different clause governing the same rejection code was cited instead. Reported "
                "separately because the remedy is a ranking change, whereas a composition error "
                "that cited the right clause and still came out wrong is a rule to look at."
            ),
        },
        "error_attribution_by_split": {
            split: {
                "cases_wrong": sum(1 for case in _of_split(cases, split) if not case.correct),
                "retrieval_errors": sum(
                    1
                    for case in _of_split(cases, split)
                    if not case.correct and not case.governing_clause_retrieved
                ),
                "composition_errors": sum(
                    1
                    for case in _of_split(cases, split)
                    if not case.correct and case.governing_clause_retrieved
                ),
            }
            for split in (HOLDOUT, DEVELOPMENT)
        },
    }


def summary_lines(recovery: Mapping[str, Any], retrieval: Mapping[str, Any]) -> tuple[str, ...]:
    """The four numbers a reader should see on the terminal, in the order they matter.

    Printed by `scripts/build_artifacts.py` so that the person who just ran the evaluation sees the
    false-recovery count before anything else, rather than a line saying three files were written.
    An evaluation whose headline is *done* is an evaluation whose result nobody reads until CI goes
    red.
    """
    system = retrieval["system"]
    return (
        f"false recoveries (hold-out): {recovery['holdout_false_recoveries']} "
        f"over {recovery['holdout_not_recoverable_cases']} NOT_RECOVERABLE case(s)",
        f"amount mismatches: {recovery['amount_mismatches']} over "
        f"{recovery['cases_scored']} case(s), exact match rate "
        f"{recovery['amount_exact_match_rate']}",
        f"resubmissions after the window closed: "
        f"{recovery['resubmissions_after_window_closed']} over "
        f"{recovery['cases_with_a_closed_window']} closed window(s)",
        f"hold-out recall@{retrieval['k']}: {system['recall_at_k']} against a best baseline of "
        f"{retrieval['best_baseline_recall_at_k']}",
    )
