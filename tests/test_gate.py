"""The gate, exercised without a database, a retriever, an embedding model or a corpus.

Every fixture here is built by hand, and that is the property the gate was designed for: if these
tests needed a store to run, a decision bug and a connection bug would fail the same test and nobody
could tell which had broken.

The first three tests are structural and are asserted over this module's syntax tree rather than by
reading it. ADR-001 §3 and `CLAUDE.md` §3.3 fix one property — "`RECOVERABLE` is reachable from
exactly one place" — and the only version of that check which survives a refactor is one that counts
nodes. A review does not count nodes, and a grep counts one spelling.

The behavioural tests that follow are each an adversarial case rather than a path through the code:
a window that shut yesterday, a claim already settled, a warranty that ran out in March, evidence
that contradicts itself. The ordering tests are the ones that matter most, because every rule reads
as obviously right on its own and the expensive failures all come from two being true at once.
"""

from __future__ import annotations

import ast
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from warranty_claim_recovery import gate
from warranty_claim_recovery.deadline import claim_window
from warranty_claim_recovery.domain import (
    Citation,
    Claim,
    GateDecision,
    RecoveryOutcome,
    RejectionCode,
    Requirement,
    RequirementStatus,
    WarrantyProgram,
)
from warranty_claim_recovery.eligibility import PartCoverage, assess
from warranty_claim_recovery.money import Currency, Money
from warranty_claim_recovery.recovery import compute
from warranty_claim_recovery.requirements import (
    MAX_REQUIREMENTS_PER_CODE,
    MIN_REQUIREMENTS_PER_CODE,
    REQUIREMENT_MATRIX,
    UnknownRejectionCodeError,
    assert_matrix_is_total,
    required_for,
)

GATE_SOURCE = Path(gate.__file__)

GBP = Currency.GBP
PART = "AB-1234-C"

#: The claim's dates are fixed so that every window figure in this file can be read off by hand.
#: Failure on 10 June, invoiced two days later, rejected a fortnight after that, and a thirty-day
#: correction window therefore closes on 26 July 2025.
FAILURE_DATE = date(2025, 6, 10)
REJECTED_ON = FAILURE_DATE + timedelta(days=16)
WINDOW_CLOSES_ON = REJECTED_ON + timedelta(days=30)
INSIDE_THE_WINDOW = date(2025, 7, 1)

QUOTE = "Claims must cite the serial number stamped on the failed component."


def make_program(
    *,
    deductible: str = "0.00",
    claim_cap: str = "5000.00",
    warranty_months: int = 24,
) -> WarrantyProgram:
    return WarrantyProgram(
        program_id="PRG-ACME-V2",
        manufacturer="Acme Drivetrain (synthetic)",
        policy_version="v2",
        currency=GBP,
        correction_window_days=30,
        warranty_months=warranty_months,
        labour_rate_cap_per_hour=Money("55.00", GBP),
        deductible=Money(deductible, GBP),
        claim_cap=Money(claim_cap, GBP),
    )


def make_claim(*, previously_recovered: str | None = None) -> Claim:
    return Claim(
        claim_id="CLM-0001",
        program_id="PRG-ACME-V2",
        part_number=PART,
        serial_number="SN-004821",
        in_service_date=date(2024, 1, 15),
        failure_date=FAILURE_DATE,
        repair_invoice_date=FAILURE_DATE + timedelta(days=2),
        rejection_code=RejectionCode.MISSING_SERIAL,
        rejected_on=REJECTED_ON,
        claimed_parts=Money(Decimal("400.00"), GBP),
        claimed_labour_hours=4,
        claimed_labour_rate=Money(Decimal("50.00"), GBP),
        previously_recovered=(
            None if previously_recovered is None else Money(Decimal(previously_recovered), GBP)
        ),
    )


def requirement(
    requirement_id: str = "REQ-SERIAL-STAMPED",
    status: RequirementStatus = RequirementStatus.SATISFIED,
) -> Requirement:
    """One requirement, cited when satisfied because `domain.Requirement` refuses it otherwise."""
    cited = (
        Citation(
            clause_id="CL-0007",
            document_id="DOC-ACME-V2",
            policy_version="v2",
            section="4.1",
            quote=QUOTE,
            start_offset=120,
            end_offset=120 + len(QUOTE),
        )
        if status is RequirementStatus.SATISFIED
        else None
    )
    return Requirement(
        requirement_id=requirement_id,
        description="The serial number stamped on the failed part, transcribed from the part.",
        status=status,
        citation=cited,
        detail="Read from the technician's photograph of the component.",
    )


SATISFIED_PAIR = (requirement("REQ-SERIAL-STAMPED"), requirement("REQ-SERIAL-IN-COVERED-RANGE"))


def decide_for(
    *,
    as_of: date = INSIDE_THE_WINDOW,
    requirements: tuple[Requirement, ...] = SATISFIED_PAIR,
    deductible: str = "0.00",
    claim_cap: str = "5000.00",
    warranty_months: int = 24,
    previously_recovered: str | None = None,
    part_is_covered: bool = True,
) -> GateDecision:
    """Assemble one case through the real deterministic pipeline and gate it.

    Deliberately runs `assess` and `compute` rather than hand-building their outputs: the gate's
    job is to decide over what those two actually produce, and a test that fed it invented signals
    would keep passing after the arithmetic beneath it changed shape.
    """
    claim = make_claim(previously_recovered=previously_recovered)
    program = make_program(
        deductible=deductible, claim_cap=claim_cap, warranty_months=warranty_months
    )
    coverage = PartCoverage(
        part_number=PART, covered=part_is_covered, serial_first=None, serial_last=None
    )
    eligibility = assess(claim, program, coverage)
    return gate.decide(
        claim,
        program,
        claim_window(claim, program, as_of=as_of),
        eligibility,
        compute(claim, program, eligibility),
        requirements,
    )


# ------------------------------------------------------------------ the structural guarantees


def test_each_authorising_outcome_is_constructed_in_exactly_one_place() -> None:
    """ADR-001 §4 and `CLAUDE.md` §3.3: `RECOVERABLE` is reachable from exactly one place.

    Counted over the syntax tree rather than trusted to review. A second construction site is a
    second place a guard can be forgotten, and the whole shape of `decide` — a loop that only
    withholds, and one fall-through that authorises — is only a guarantee if the count is one.
    Adding a second way to authorise money is then a visible diff rather than a line in a branch.
    """
    tree = ast.parse(GATE_SOURCE.read_text(encoding="utf-8"))

    for outcome in (RecoveryOutcome.RECOVERABLE, RecoveryOutcome.PARTIALLY_RECOVERABLE):
        sites = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and node.attr == outcome.name
            and isinstance(node.value, ast.Name)
            and node.value.id == "RecoveryOutcome"
        ]
        assert len(sites) == 1, (
            f"{outcome.name} is named at {len(sites)} places in {GATE_SOURCE.name}, at lines "
            f"{[node.lineno for node in sites]}; ADR-001 fixes it at one"
        )


def test_the_authorising_outcomes_are_named_only_after_the_withholding_loop() -> None:
    """One construction site is not enough if that site sits inside the rules loop.

    A rule that authorised would still be a single site and would still be wrong: the loop exists
    to withhold, and a rule that returns a recovery is a rule that can be reordered ahead of the
    window check. The site has to be in the fall-through, which is everything after the loop ends.
    """
    tree = ast.parse(GATE_SOURCE.read_text(encoding="utf-8"))
    decide = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "decide"
    )
    rules_loop = next(node for node in ast.walk(decide) if isinstance(node, ast.For))

    for outcome in (RecoveryOutcome.RECOVERABLE, RecoveryOutcome.PARTIALLY_RECOVERABLE):
        site = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and node.attr == outcome.name
            and isinstance(node.value, ast.Name)
            and node.value.id == "RecoveryOutcome"
        )
        assert site.lineno > (rules_loop.end_lineno or rules_loop.lineno), (
            f"{outcome.name} is named at line {site.lineno}, inside or before the rules loop that "
            f"ends at line {rules_loop.end_lineno}; only the fall-through may authorise"
        )


def test_no_authorising_outcome_is_spelled_as_a_bare_string() -> None:
    """The attribute form is not the only way to name an outcome.

    `RecoveryOutcome` is a `StrEnum` and `GateDecision` is a Pydantic model, so
    `GateDecision(outcome="RECOVERABLE", ...)` builds exactly the same authorised decision and is
    invisible to the node count above. An AST guard that looks for one spelling guards one
    spelling, so every string literal equal to an authorising outcome's value is counted too.
    """
    tree = ast.parse(GATE_SOURCE.read_text(encoding="utf-8"))
    values = {RecoveryOutcome.RECOVERABLE.value, RecoveryOutcome.PARTIALLY_RECOVERABLE.value}

    literals = [
        node for node in ast.walk(tree) if isinstance(node, ast.Constant) and node.value in values
    ]

    assert not literals, (
        f"{GATE_SOURCE.name} names an authorising outcome as a bare string at lines "
        f"{[node.lineno for node in literals]}; a StrEnum makes that the same construction"
    )


def test_a_decision_is_constructed_in_exactly_two_places() -> None:
    """One withholding construction inside the loop, one authorising construction after it.

    Counted independently of the outcome names because a third `GateDecision(...)` anywhere in the
    module would be a third exit from the gate, and an exit whose outcome came from a variable is
    invisible to a check that only looks at enum attributes.
    """
    tree = ast.parse(GATE_SOURCE.read_text(encoding="utf-8"))

    sites = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "GateDecision"
    ]

    assert len(sites) == 2, (
        f"{GATE_SOURCE.name} constructs a decision at {len(sites)} places, at lines "
        f"{[node.lineno for node in sites]}; there are two exits from the gate and no more"
    )


def test_every_rule_in_the_table_withholds() -> None:
    """The run-time half of the structural claim, and the one that survives a rename.

    The AST checks assert that authorising happens in one place. This asserts the other direction
    over the object the running system actually iterates: no entry in the rules table carries an
    outcome that lets money leave, so reordering the table can change which refusal a case gets and
    can never turn a refusal into a resubmission.
    """
    assert gate._RULES, "an empty rules table would authorise everything by falling through"
    for rule in gate._RULES:
        assert gate.withholds_recovery(rule.outcome), (
            f"a rule in the table produces {rule.outcome}, which authorises a recovery from inside "
            f"the loop that exists to withhold"
        )


def test_the_gate_imports_nothing_that_could_generate_text() -> None:
    """No signal the gate reads may be influenced by a model.

    ADR-001 §2 lists what the deterministic core owns and the gate decision is on the list. An
    import of the composer or the retriever here would be the first step towards a signal a model
    can move, and it would arrive as a convenience rather than as a decision.
    """
    tree = ast.parse(GATE_SOURCE.read_text(encoding="utf-8"))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    forbidden = ("compose", "llm", "retrieval", "corpus", "store")
    assert not [
        name
        for name in imported
        if name.startswith(tuple(f"warranty_claim_recovery.{part}" for part in forbidden))
    ], f"the gate imports {sorted(imported)}"


# ------------------------------------------------------------------- the claim window, first


def test_a_closed_window_refuses_a_claim_whose_evidence_is_perfect() -> None:
    """Kill condition M, and the reason the window is the first rule rather than the last.

    Every requirement is satisfied, the part is covered, the warranty is live and the money is
    positive. None of that matters: a correction filed after the window shut consumes a handler's
    afternoon and is refused on receipt.
    """
    decision = decide_for(as_of=WINDOW_CLOSES_ON + timedelta(days=1))

    assert decision.outcome is RecoveryOutcome.NOT_RECOVERABLE
    assert decision.authorises_recovery is False
    assert "correction window" in decision.reason
    assert decision.signals.window_open is False


def test_a_claim_assessed_on_the_closing_date_is_still_decided_on_its_merits() -> None:
    """The other side of the same day. One day early and the recovery is authorised."""
    decision = decide_for(as_of=WINDOW_CLOSES_ON)

    assert decision.outcome is RecoveryOutcome.RECOVERABLE
    assert decision.signals.days_remaining == 0


def test_the_window_is_checked_before_the_evidence_and_before_the_money() -> None:
    """The ordering test, which is the one that matters.

    This case is wrong in four ways at once: the window shut a week ago, the claim was already
    recovered, a requirement is missing and the deductible leaves nothing. A gate that checked the
    evidence first would report it as a paperwork problem and send somebody to chase a document
    that can no longer be used.
    """
    decision = decide_for(
        as_of=WINDOW_CLOSES_ON + timedelta(days=7),
        requirements=(requirement(status=RequirementStatus.MISSING),),
        previously_recovered="300.00",
        deductible="5000.00",
    )

    assert decision.outcome is RecoveryOutcome.NOT_RECOVERABLE
    assert "correction window" in decision.reason


# ------------------------------------------------------------------------ duplicates, second


def test_a_claim_already_recovered_once_is_refused_as_a_duplicate() -> None:
    """The adversarial case the brief names. Evidence cannot improve a claim already settled."""
    decision = decide_for(previously_recovered="300.00")

    assert decision.outcome is RecoveryOutcome.NOT_RECOVERABLE
    assert "duplicate" in decision.reason
    assert decision.signals.already_recovered is True


def test_the_duplicate_check_runs_before_the_evidence_check() -> None:
    """A settled claim with missing paperwork is settled, not incomplete.

    Reporting it as a paperwork problem sends a technician to chase documents for money that has
    already been paid, and the case comes back a week later to be refused for the real reason.
    """
    decision = decide_for(
        previously_recovered="300.00",
        requirements=(requirement(status=RequirementStatus.MISSING),),
    )

    assert "duplicate" in decision.reason


# --------------------------------------------------------------------- the warranty period


def test_a_failure_outside_the_warranty_period_is_refused_however_good_the_evidence() -> None:
    """Kill condition G's false recovery, caught by the one signal that moves no money.

    An expired warranty leaves the parts covered and the labour rate inside the cap, so the
    recovery arithmetic produces a positive amount and every requirement can be satisfied. Without
    this rule the claim would be authorised on paperwork alone.
    """
    decision = decide_for(warranty_months=6)

    assert decision.outcome is RecoveryOutcome.NOT_RECOVERABLE
    assert "warranty period" in decision.reason
    assert decision.signals.within_warranty_period is False
    assert decision.signals.recoverable_is_positive is True


# --------------------------------------------------------------------------- the evidence


def test_a_case_with_no_requirements_assessed_goes_to_review() -> None:
    """Should be unreachable, and is a rule anyway.

    `requirements.REQUIREMENT_MATRIX` is a total function, so a case reaching the gate with an
    empty requirement tuple was assembled wrongly. Falling through would authorise a recovery
    earned by never having been asked a question, which is the failure `RejectionCode`'s own
    docstring names.
    """
    decision = decide_for(requirements=())

    assert decision.outcome is RecoveryOutcome.REVIEW
    assert decision.signals.requirements_total == 0


def test_contradictory_evidence_goes_to_review_rather_than_being_refused() -> None:
    """A contradiction is a decision for a person, not an errand for a technician.

    Refusing it writes off money that may well be recoverable once somebody says which source is
    right, and `RecoveryOutcome` exists in four states precisely so this case has somewhere to go.
    """
    decision = decide_for(
        requirements=(
            requirement("REQ-SERIAL-STAMPED"),
            requirement("REQ-SERIAL-IN-COVERED-RANGE", RequirementStatus.CONFLICTING),
        )
    )

    assert decision.outcome is RecoveryOutcome.REVIEW
    assert "REQ-SERIAL-IN-COVERED-RANGE" in decision.reason
    assert decision.signals.requirements_conflicting == 1


def test_a_contradiction_is_raised_before_a_missing_document_is_chased() -> None:
    """The counter-intuitive ordering, and the reason for it.

    The contradiction will still be there when the missing document arrives, and discovering it on
    the second pass costs another day against a window that is already running.
    """
    decision = decide_for(
        requirements=(
            requirement("REQ-SERIAL-STAMPED", RequirementStatus.MISSING),
            requirement("REQ-SERIAL-IN-COVERED-RANGE", RequirementStatus.CONFLICTING),
        )
    )

    assert decision.outcome is RecoveryOutcome.REVIEW


def test_a_missing_requirement_is_refused_and_the_refusal_names_it() -> None:
    """A refusal that does not say what is missing is a refusal nobody can act on.

    The console's whole job is to show a handler which line to go and fix, so the identifiers are
    in the reason and the requirements themselves are carried on the decision.
    """
    decision = decide_for(
        requirements=(
            requirement("REQ-SERIAL-STAMPED"),
            requirement("REQ-SERIAL-IN-COVERED-RANGE", RequirementStatus.MISSING),
        )
    )

    assert decision.outcome is RecoveryOutcome.NOT_RECOVERABLE
    assert "REQ-SERIAL-IN-COVERED-RANGE" in decision.reason
    assert "REQ-SERIAL-STAMPED" not in decision.reason
    assert len(decision.requirements) == 2


def test_a_refusal_still_carries_the_requirements_it_refused_over() -> None:
    decision = decide_for(requirements=(requirement(status=RequirementStatus.MISSING),))

    assert decision.requirements
    assert decision.signals.requirements_satisfied == 0


# ------------------------------------------------------------------------------- the money


def test_a_claim_with_nothing_left_to_recover_is_refused() -> None:
    """A deductible larger than the claim leaves nothing to file for.

    Filing a zero-value correction is not free: it occupies an adjudicator, it produces a second
    rejection, and `GateDecision` refuses to construct an authorised outcome over a zero amount —
    so without this rule the gate would raise rather than decide.
    """
    decision = decide_for(deductible="5000.00")

    assert decision.outcome is RecoveryOutcome.NOT_RECOVERABLE
    assert decision.signals.recoverable_is_positive is False
    assert decision.computation.recoverable_amount == Money.zero(GBP)


def test_a_claim_with_nothing_excluded_is_fully_recoverable() -> None:
    """The fall-through's first branch. Nothing was taken out, so the whole claim goes back."""
    decision = decide_for(deductible="0.00")

    assert decision.outcome is RecoveryOutcome.RECOVERABLE
    assert decision.authorises_recovery is True
    assert decision.computation.excluded_amount == Money.zero(GBP)
    assert decision.computation.recoverable_amount == Money("600.00", GBP)


def test_a_deductible_makes_the_recovery_partial_rather_than_full() -> None:
    """A partial recovery is a success, not a degraded one.

    The claim goes out for the covered portion with the exclusion shown. Forcing it into
    `RECOVERABLE` would file a total the adjudicator refuses; forcing it into `NOT_RECOVERABLE`
    writes off 550.00 that was owed.
    """
    decision = decide_for(deductible="50.00")

    assert decision.outcome is RecoveryOutcome.PARTIALLY_RECOVERABLE
    assert decision.authorises_recovery is True
    assert decision.computation.recoverable_amount == Money("550.00", GBP)
    assert decision.computation.excluded_amount == Money("50.00", GBP)


def test_a_binding_cap_makes_the_recovery_partial() -> None:
    """The adversarial case the brief names, seen from the gate rather than the arithmetic."""
    decision = decide_for(deductible="0.00", claim_cap="500.00")

    assert decision.outcome is RecoveryOutcome.PARTIALLY_RECOVERABLE
    assert decision.computation.capped_amount == Money("100.00", GBP)


def test_an_uncovered_part_still_recovers_the_labour_as_a_partial() -> None:
    decision = decide_for(part_is_covered=False)

    assert decision.outcome is RecoveryOutcome.PARTIALLY_RECOVERABLE
    assert decision.signals.part_covered is False
    assert decision.computation.recoverable_amount == Money("200.00", GBP)


# ------------------------------------------------------------------------------ the record


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, RecoveryOutcome.RECOVERABLE),
        ({"deductible": "50.00"}, RecoveryOutcome.PARTIALLY_RECOVERABLE),
        ({"deductible": "5000.00"}, RecoveryOutcome.NOT_RECOVERABLE),
        ({"requirements": ()}, RecoveryOutcome.REVIEW),
    ],
    ids=["recoverable", "partial", "refused", "review"],
)
def test_every_outcome_carries_the_signals_and_the_computation_it_was_decided_from(
    kwargs: dict[str, object],
    expected: RecoveryOutcome,
) -> None:
    """A verdict without its inputs is an assertion, and an assertion cannot be disagreed with.

    The console shows the signals beside the outcome so that a disagreement about a decision is a
    disagreement about a number rather than about a judgement, and the evaluation reads the same
    record rather than recomputing one.
    """
    decision = decide_for(**kwargs)  # type: ignore[arg-type]

    assert decision.outcome is expected
    assert decision.signals.requirements_total == len(decision.requirements)
    assert decision.computation.currency is GBP
    assert len(decision.reason) >= 10


def test_the_signals_helper_agrees_with_the_signals_on_the_decision() -> None:
    """One definition of what the gate saw, not two.

    `signals` is exposed so the evaluation and the console can record the inputs without
    re-deriving them. A signal computed in two places is a signal that will eventually be computed
    two ways, and the disagreement surfaces as a decision nobody can explain.
    """
    claim = make_claim()
    program = make_program()
    coverage = PartCoverage(part_number=PART, covered=True, serial_first=None, serial_last=None)
    eligibility = assess(claim, program, coverage)
    computation = compute(claim, program, eligibility)
    window = claim_window(claim, program, as_of=INSIDE_THE_WINDOW)

    decision = gate.decide(claim, program, window, eligibility, computation, SATISFIED_PAIR)

    assert gate.signals(window, eligibility, computation, SATISFIED_PAIR) == decision.signals


def test_withholding_and_authorising_partition_the_four_outcomes() -> None:
    """Written once, here, so the console, the queue and the evaluation cannot disagree."""
    withheld = [outcome for outcome in RecoveryOutcome if gate.withholds_recovery(outcome)]

    assert set(withheld) == {RecoveryOutcome.NOT_RECOVERABLE, RecoveryOutcome.REVIEW}
    assert len(withheld) + 2 == len(RecoveryOutcome)


# ------------------------------------------------------------------- the requirement matrix
#
# These live here rather than in a file of their own because the matrix has no consumer other than
# the gate: its totality is what stops `requirements_total == 0` from being reachable in practice,
# and the rule above is what catches it if the table is ever wrong. Testing the two apart would
# leave the connection between them written down in prose and checked by nobody.


def test_the_matrix_answers_for_every_rejection_code() -> None:
    """Totality, checked over the enum rather than over the table's own keys.

    A code with no entry falls through to an empty requirement list, which does not mean "unknown"
    — it means "nothing is required", and the gate reads a fully satisfied list as permission.
    """
    for code in RejectionCode:
        specs = required_for(code)
        assert MIN_REQUIREMENTS_PER_CODE <= len(specs) <= MAX_REQUIREMENTS_PER_CODE


def test_every_declared_requirement_can_be_built_into_a_domain_requirement() -> None:
    """The matrix's descriptions have to survive `domain.Requirement`'s own validators.

    A description of "serial" passes review and fails at run time on the first case that carries
    that code, which is the worst possible moment for a table of constants to be found wrong.
    """
    for code in RejectionCode:
        for spec in required_for(code):
            built = Requirement(
                requirement_id=spec.requirement_id,
                description=spec.description,
                status=RequirementStatus.MISSING,
                citation=None,
                detail=f"Evidence key {spec.evidence_key} is not present on the case.",
            )
            assert built.requirement_id == spec.requirement_id


def test_requirement_identifiers_are_unique_across_the_whole_matrix() -> None:
    """A satisfied requirement has to be traceable to the rejection that raised it.

    Kill condition H is graded from audit records, and a `REQ-` identifier emitted by two codes
    makes a record that cannot be attributed to either.
    """
    seen = [spec.requirement_id for code in RejectionCode for spec in required_for(code)]

    assert len(seen) == len(set(seen))


def test_a_matrix_missing_a_code_is_refused() -> None:
    """The guard, shown rejecting something, rather than only ever run against the correct table.

    `CLAUDE.md` §3.6: a guard must fail from behaviour. The broken table is handed to the guard and
    the refusal is observed, which is a different claim from "the shipped table is fine".
    """
    incomplete = {
        code: specs
        for code, specs in REQUIREMENT_MATRIX.items()
        if code is not RejectionCode.DUPLICATE_CLAIM
    }

    with pytest.raises(UnknownRejectionCodeError, match="DUPLICATE_CLAIM"):
        assert_matrix_is_total(incomplete)


def test_a_code_declaring_no_requirements_is_refused() -> None:
    """The same failure wearing a different hat: the key exists and the list behind it is empty."""
    hollow = dict(REQUIREMENT_MATRIX)
    hollow[RejectionCode.DUPLICATE_CLAIM] = ()

    with pytest.raises(ValueError, match="DUPLICATE_CLAIM declares 0"):
        assert_matrix_is_total(hollow)


def test_a_requirement_identifier_declared_by_two_codes_is_refused() -> None:
    duplicated = dict(REQUIREMENT_MATRIX)
    duplicated[RejectionCode.DUPLICATE_CLAIM] = REQUIREMENT_MATRIX[RejectionCode.MISSING_SERIAL]

    with pytest.raises(ValueError, match="REQ-SERIAL-STAMPED is declared by both"):
        assert_matrix_is_total(duplicated)


def test_an_unknown_code_raises_rather_than_returning_an_empty_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lookup path, tested by removing an entry from the live table at run time.

    Patching the mapping the running function actually reads is the behavioural version of this
    check. Asserting that the source contains a `raise` would pass against a `raise` that is
    unreachable.
    """
    monkeypatch.delitem(REQUIREMENT_MATRIX, RejectionCode.DUPLICATE_CLAIM)

    with pytest.raises(UnknownRejectionCodeError, match="will not treat an unknown rejection"):
        required_for(RejectionCode.DUPLICATE_CLAIM)
