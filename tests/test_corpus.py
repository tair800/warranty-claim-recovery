"""What the corpus has to be true about itself before any number measured over it means anything.

These are not a restatement of the generator's own checks. The generator refuses to *write* a
corpus that fails; this file reads what was written and re-derives the answers independently, which
is a different question. In particular `test_the_money_recomputes_from_the_written_records`
implements the recovery arithmetic a second time, from the JSON alone, and compares. A single
implementation checked against itself proves that the code is consistent; two implementations
agreeing proves that the arithmetic is right, and kill condition F is exact `Decimal` equality
against these amounts.

The corpus is rebuilt in memory rather than read from disk wherever a check is about the generator,
and read from disk wherever the check is about the file a later stage will load. Both matter: a
generator that is correct and a writer that drops a field are the same failure to everything
downstream.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from warranty_claim_recovery.corpus.claims import (
    ADVERSARIAL_KINDS,
    EVIDENCE_KEYS,
    GROUND_TRUTH_RULE,
    NOT_ADVERSARIAL,
    RECIPE,
    EvidenceStatus,
    add_months,
    serial_is_in_range,
)
from warranty_claim_recovery.corpus.clauses import SYNTHETIC_NOTICE
from warranty_claim_recovery.corpus.generate import FLOORS, GeneratedCorpus, build, generate_corpus
from warranty_claim_recovery.corpus.holdout import is_holdout, split_of
from warranty_claim_recovery.domain import Claim, RecoveryOutcome, RejectionCode, WarrantyProgram
from warranty_claim_recovery.money import Currency, Money

JsonDict = dict[str, Any]


def _serialise(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False)


@pytest.fixture(scope="module")
def corpus() -> GeneratedCorpus:
    return build()


@pytest.fixture(scope="module")
def written(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The corpus as files, which is the form every later stage actually loads."""
    out = tmp_path_factory.mktemp("corpus")
    generate_corpus(out, artifact_path=out / "corpus.json")
    return out


def _load(written: Path, name: str, key: str) -> list[JsonDict]:
    payload = json.loads((written / name).read_text(encoding="utf-8"))
    records: list[JsonDict] = payload[key]
    return records


def _money(payload: JsonDict) -> Money:
    return Money(payload["amount"], payload["currency"])


def _programs_by_id(written: Path) -> dict[str, JsonDict]:
    records = _load(written, "programs.json", "programs")
    return {r["program"]["program_id"]: r["program"] for r in records}


# --------------------------------------------------------------------------------- determinism


def test_two_builds_are_byte_identical(corpus: GeneratedCorpus) -> None:
    """The whole point of a committed seed. A clock, a hostname, a path or a set iteration deciding
    an order would each show up here and nowhere else."""
    again = build()
    for (name, left), (_, right) in zip(corpus.files, again.files, strict=True):
        assert _serialise(left) == _serialise(right), f"{name} differs between two builds"
    assert _serialise(corpus.artifact) == _serialise(again.artifact)


def _keys_of(node: object) -> set[str]:
    if isinstance(node, dict):
        found: set[str] = set()
        for key, value in node.items():
            found.add(key)
            found |= _keys_of(value)
        return found
    if isinstance(node, list):
        return set().union(*(_keys_of(value) for value in node)) if node else set()
    return set()


def test_nothing_written_carries_a_clock_or_a_path(corpus: GeneratedCorpus) -> None:
    banned = {"generated_at", "created_at", "timestamp", "hostname", "cwd", "user", "machine"}
    for name, payload in (*corpus.files, ("artifacts/corpus.json", corpus.artifact)):
        offending = sorted(_keys_of(payload) & banned)
        assert not offending, f"{name} carries {offending}"
    assert corpus.artifact["contains_timestamp"] is False


# --------------------------------------------------------------------------------- the contract


def test_every_declared_floor_is_cleared(corpus: GeneratedCorpus) -> None:
    observed = corpus.artifact["observed"]
    for name, floor in FLOORS:
        assert observed[name] >= floor, f"{name} is {observed[name]}, the floor is {floor}"


def test_every_rejection_code_is_represented(corpus: GeneratedCorpus) -> None:
    by_code = corpus.artifact["claims_by_rejection_code"]
    for code in RejectionCode:
        seen = by_code.get(code.value, 0)
        assert seen >= 20, f"{code} appears {seen} times"


def test_the_outcome_mix_clears_its_floors(corpus: GeneratedCorpus) -> None:
    by_outcome = corpus.artifact["claims_by_outcome"]
    assert by_outcome[RecoveryOutcome.NOT_RECOVERABLE.value] >= 60
    assert by_outcome[RecoveryOutcome.PARTIALLY_RECOVERABLE.value] >= 60
    assert by_outcome[RecoveryOutcome.REVIEW.value] >= 40
    assert by_outcome[RecoveryOutcome.RECOVERABLE.value] >= 1


def test_every_adversarial_construction_the_brief_names_is_present(
    corpus: GeneratedCorpus,
) -> None:
    by_kind = corpus.artifact["claims_by_adversarial_kind"]
    assert sorted(by_kind) == sorted(ADVERSARIAL_KINDS)
    programs = corpus.artifact["counts"]["programs"]
    for kind in ADVERSARIAL_KINDS:
        assert by_kind[kind] == programs, (
            f"{kind} appears {by_kind[kind]} times and the recipe puts one in every programme"
        )


def test_the_recipe_covers_every_slot_exactly_once() -> None:
    slots = [recipe.slot for recipe in RECIPE]
    assert slots == sorted(slots)
    assert len(set(slots)) == len(slots)
    kinds = {r.adversarial_kind for r in RECIPE} - {NOT_ADVERSARIAL}
    assert kinds == set(ADVERSARIAL_KINDS)


# --------------------------------------------------------------------------------- no floats


CORPUS_FILES = (
    "programs.json",
    "clauses.json",
    "claims.json",
    "coverage.json",
    "truth.json",
    "documents.json",
    "corpus.json",
)


def _floats_in(node: object, where: str) -> list[str]:
    if isinstance(node, bool):
        return []
    if isinstance(node, float):
        return [f"{where} = {node!r}"]
    if isinstance(node, dict):
        return [f for key, value in node.items() for f in _floats_in(value, f"{where}.{key}")]
    if isinstance(node, list):
        return [f for i, value in enumerate(node) for f in _floats_in(value, f"{where}[{i}]")]
    return []


def test_no_float_appears_anywhere_in_any_written_payload(written: Path) -> None:
    """Read back from disk, because the question is about the bytes a later stage parses.

    `json.load` turns `1.5` into a float and `"1.50"` into a string, so this is a real check on the
    file rather than on the object the generator held.
    """
    for name in CORPUS_FILES:
        payload = json.loads((written / name).read_text(encoding="utf-8"))
        offenders = _floats_in(payload, name)
        assert not offenders, offenders[:5]


# --------------------------------------------------------------------------------- citations


def test_every_clause_slices_out_of_its_document_at_the_offsets_it_records(written: Path) -> None:
    documents = {d["document_id"]: d for d in _load(written, "documents.json", "documents")}
    clauses = _load(written, "clauses.json", "clauses")
    assert clauses
    for record in clauses:
        clause = record["clause"]
        document = documents[clause["document_id"]]
        sliced = document["text"][clause["start_offset"] : clause["end_offset"]]
        assert sliced == clause["text"], clause["clause_id"]


def test_every_clause_body_is_long_enough_to_state_a_rule(written: Path) -> None:
    for record in _load(written, "clauses.json", "clauses"):
        assert len(record["clause"]["text"]) >= 200, record["clause"]["clause_id"]


def test_the_withdrawn_bulletin_governs_nothing(written: Path) -> None:
    """A clause with no governed code is context, never authority. `domain.PolicyClause` says so and
    the withdrawn bulletin is the reason the distinction exists."""
    withdrawn = [
        record
        for record in _load(written, "clauses.json", "clauses")
        if not record["document_is_current"]
    ]
    assert withdrawn, "no withdrawn bulletin survived into the corpus"
    for record in withdrawn:
        assert record["clause"]["governs"] == []


def test_every_governing_clause_governs_its_claims_code_and_shares_its_programme(
    written: Path,
) -> None:
    clauses = {r["clause"]["clause_id"]: r for r in _load(written, "clauses.json", "clauses")}
    claims = {r["claim"]["claim_id"]: r for r in _load(written, "claims.json", "claims")}
    truth = json.loads((written / "truth.json").read_text(encoding="utf-8"))["truth"]
    assert len(truth) == len(claims)
    for claim_id, entry in truth.items():
        clause = clauses[entry["governing_clause_id"]]
        claim = claims[claim_id]["claim"]
        assert clause["clause"]["program_id"] == claim["program_id"]
        assert claim["rejection_code"] in clause["clause"]["governs"]
        assert clause["document_is_current"] is True


def test_the_bulletin_amendment_makes_the_governing_clause_depend_on_more_than_the_code(
    written: Path,
) -> None:
    """If a rejection code had one governing clause per programme, the `exact_code_lookup` baseline
    in ADR-001 §6 would score a perfect recall and kill condition J could never be met by any
    retriever. This asserts that the corpus does not have that shape."""
    claims = {r["claim"]["claim_id"]: r for r in _load(written, "claims.json", "claims")}
    truth = json.loads((written / "truth.json").read_text(encoding="utf-8"))["truth"]
    per_program_code: dict[tuple[str, str], set[str]] = {}
    for claim_id, entry in truth.items():
        claim = claims[claim_id]["claim"]
        key = (claim["program_id"], claim["rejection_code"])
        per_program_code.setdefault(key, set()).add(entry["governing_clause_id"])
    ambiguous = [key for key, ids in per_program_code.items() if len(ids) > 1]
    assert ambiguous, "every code resolves to one clause per programme; the task is a table lookup"


# --------------------------------------------------------------------------------- the money


def _recompute(claim: Claim, program: WarrantyProgram, covered: bool, in_range: bool) -> Money:
    """The recovery arithmetic, implemented a second time from ADR-001's formula.

    Deliberately written out step by step rather than by calling `claims.recovery_of`. The value of
    a second implementation is that it can disagree; importing the first one back would make this
    test a tautology.
    """
    zero = Money.zero(claim.claimed_parts.currency)
    over = claim.claimed_labour_rate - program.labour_rate_cap_per_hour
    labour_excess = (over * claim.claimed_labour_hours) if over > zero else zero
    uncovered = zero if covered and in_range else claim.claimed_parts
    eligible = claim.claimed_total - labour_excess - uncovered
    after = max(eligible - program.deductible, zero)
    settled = after if after < program.claim_cap else program.claim_cap
    already = claim.previously_recovered or zero
    recoverable = settled - already
    return (recoverable if recoverable > zero else zero).quantize()


def test_the_money_recomputes_from_the_written_records(written: Path) -> None:
    programs = _programs_by_id(written)
    coverage = {
        (r["program_id"], r["coverage"]["part_number"]): r
        for r in _load(written, "coverage.json", "coverage")
    }
    truth = json.loads((written / "truth.json").read_text(encoding="utf-8"))["truth"]

    checked = 0
    for record in _load(written, "claims.json", "claims"):
        raw = record["claim"]
        claim = _claim_from(raw)
        program = _program_from(programs[raw["program_id"]])
        entry = truth[claim.claim_id]

        if not record["facts"]["currency_matches_program"]:
            # The one case the money path cannot run at all: two currencies and no rate this system
            # is entitled to apply. The truth says nothing is recoverable, in the claim's currency.
            assert entry["recoverable_amount"] == "0.00"
            assert entry["currency"] == claim.claimed_parts.currency.value
            assert entry["outcome"] == RecoveryOutcome.REVIEW.value
            continue

        row = coverage[(raw["program_id"], raw["part_number"])]["coverage"]
        in_range = serial_is_in_range(raw["serial_number"], row["serial_first"], row["serial_last"])
        assert row["covered"] == record["facts"]["part_covered"]
        assert in_range == record["facts"]["serial_in_range"]

        expected = _recompute(claim, program, row["covered"], in_range)
        assert str(expected.amount) == entry["recoverable_amount"], claim.claim_id
        assert expected.currency.value == entry["currency"]
        checked += 1

    assert checked > 0, "nothing was recomputed, so this test graded an empty population"


def test_every_truth_amount_is_an_exact_two_place_string(written: Path) -> None:
    truth = json.loads((written / "truth.json").read_text(encoding="utf-8"))["truth"]
    for claim_id, entry in truth.items():
        raw = entry["recoverable_amount"]
        assert isinstance(raw, str), claim_id
        assert "e" not in raw.lower(), claim_id
        value = Decimal(raw)
        assert value == value.quantize(Decimal("0.01")), claim_id
        assert value >= 0, claim_id
        Currency(entry["currency"])


def test_no_recovery_exceeds_the_claim_it_came_from(written: Path) -> None:
    truth = json.loads((written / "truth.json").read_text(encoding="utf-8"))["truth"]
    for record in _load(written, "claims.json", "claims"):
        if not record["facts"]["currency_matches_program"]:
            continue
        claim = _claim_from(record["claim"])
        recovered = Money(truth[claim.claim_id]["recoverable_amount"], claim.claimed_parts.currency)
        assert recovered <= claim.claimed_total, claim.claim_id


# --------------------------------------------------------------------------------- the records


def _claim_from(raw: JsonDict) -> Claim:
    prior = raw["previously_recovered"]
    return Claim(
        claim_id=raw["claim_id"],
        program_id=raw["program_id"],
        part_number=raw["part_number"],
        serial_number=raw["serial_number"],
        in_service_date=date.fromisoformat(raw["in_service_date"]),
        failure_date=date.fromisoformat(raw["failure_date"]),
        repair_invoice_date=date.fromisoformat(raw["repair_invoice_date"]),
        rejection_code=RejectionCode(raw["rejection_code"]),
        rejected_on=date.fromisoformat(raw["rejected_on"]),
        claimed_parts=_money(raw["claimed_parts"]),
        claimed_labour_hours=raw["claimed_labour_hours"],
        claimed_labour_rate=_money(raw["claimed_labour_rate"]),
        previously_recovered=None if prior is None else _money(prior),
    )


def _program_from(raw: JsonDict) -> WarrantyProgram:
    return WarrantyProgram(
        program_id=raw["program_id"],
        manufacturer=raw["manufacturer"],
        policy_version=raw["policy_version"],
        currency=Currency(raw["currency"]),
        correction_window_days=raw["correction_window_days"],
        warranty_months=raw["warranty_months"],
        labour_rate_cap_per_hour=_money(raw["labour_rate_cap_per_hour"]),
        deductible=_money(raw["deductible"]),
        claim_cap=_money(raw["claim_cap"]),
    )


def test_every_written_claim_rebuilds_into_the_frozen_model(written: Path) -> None:
    """The `claim` sub-object holds exactly the model's fields and nothing else.

    `extra="forbid"` means a record carrying one stray key cannot be loaded at all, so keeping the
    corpus metadata outside that object is a contract rather than a convention. A loader that has to
    strip keys before constructing a model is a loader that will one day strip the wrong one.
    """
    records = _load(written, "claims.json", "claims")
    assert len(records) >= 480
    for record in records:
        assert set(record["claim"]) == set(Claim.model_fields), record["claim"]["claim_id"]
        assert _claim_from(record["claim"]).claim_id == record["claim"]["claim_id"]


def test_every_written_program_rebuilds_into_the_frozen_model(written: Path) -> None:
    records = _load(written, "programs.json", "programs")
    assert len(records) >= 12
    currencies = {r["program"]["currency"] for r in records}
    assert currencies == {c.value for c in Currency}
    for record in records:
        assert set(record["program"]) == set(WarrantyProgram.model_fields)
        rebuilt = _program_from(record["program"])
        assert rebuilt.program_id == record["program"]["program_id"]
        assert record["split"] == split_of(rebuilt.program_id)


def test_every_claim_has_a_coverage_row_in_its_own_programme(written: Path) -> None:
    coverage = {
        (r["program_id"], r["coverage"]["part_number"])
        for r in _load(written, "coverage.json", "coverage")
    }
    for record in _load(written, "claims.json", "claims"):
        key = (record["claim"]["program_id"], record["claim"]["part_number"])
        assert key in coverage, key


def test_every_claim_carries_the_whole_evidence_vocabulary(written: Path) -> None:
    allowed = {status.value for status in EvidenceStatus}
    for record in _load(written, "claims.json", "claims"):
        evidence = record["evidence"]
        assert tuple(evidence) == EVIDENCE_KEYS
        assert set(evidence.values()) <= allowed


def test_missing_requirements_name_exactly_the_evidence_that_is_not_present(written: Path) -> None:
    truth = json.loads((written / "truth.json").read_text(encoding="utf-8"))["truth"]
    incomplete = 0
    for record in _load(written, "claims.json", "claims"):
        expected = sorted(
            key
            for key, status in record["evidence"].items()
            if status != EvidenceStatus.PRESENT.value
        )
        assert truth[record["claim"]["claim_id"]]["missing_requirements"] == expected
        incomplete += bool(expected)
    assert incomplete > 0, "no claim in the corpus is missing evidence, so this graded nothing"


# --------------------------------------------------------------------------------- the boundaries


def test_the_expired_claims_sit_where_both_readings_of_the_period_agree(written: Path) -> None:
    """The one-day construction puts the end of cover on the day *before* the failure.

    A half-open and a closed reading of the warranty period then give the same answer. A
    ground truth that depended on an unstated interval convention would be grading the
    convention rather than the system.
    """
    programs = _programs_by_id(written)
    expired = [
        r
        for r in _load(written, "claims.json", "claims")
        if r["adversarial_kind"] == "warranty_expired_one_day_before_failure"
    ]
    assert expired
    for record in expired:
        raw = record["claim"]
        months = programs[raw["program_id"]]["warranty_months"]
        in_service = date.fromisoformat(raw["in_service_date"])
        failure = date.fromisoformat(raw["failure_date"])
        bound = add_months(in_service, months)
        assert bound < failure, raw["claim_id"]
        assert (failure - bound).days == 1, raw["claim_id"]
        assert record["facts"]["within_warranty_period"] is False


def test_the_closed_windows_are_closed_by_days_and_not_by_one(written: Path) -> None:
    closed = [r for r in _load(written, "claims.json", "claims") if not r["facts"]["window_open"]]
    assert len(closed) >= 1, "kill condition M would be graded over an empty population"
    for record in closed:
        as_of = date.fromisoformat(record["as_of"])
        closes_on = date.fromisoformat(record["closes_on"])
        assert (as_of - closes_on).days >= 2, record["claim"]["claim_id"]


def test_no_claim_leaves_its_serial_unknown_under_a_declared_range(written: Path) -> None:
    coverage = {
        (r["program_id"], r["coverage"]["part_number"]): r["coverage"]
        for r in _load(written, "coverage.json", "coverage")
    }
    unknown = 0
    for record in _load(written, "claims.json", "claims"):
        raw = record["claim"]
        if raw["serial_number"] is not None:
            continue
        unknown += 1
        row = coverage[(raw["program_id"], raw["part_number"])]
        assert row["serial_first"] is None and row["serial_last"] is None, raw["claim_id"]
    assert unknown > 0, "no claim has an unknown serial, so this graded nothing"


# --------------------------------------------------------------------------------- the hold-out


def test_the_holdout_side_carries_the_populations_the_kill_test_grades(written: Path) -> None:
    """Kill condition G is graded over hold-out claims whose ground truth is NOT_RECOVERABLE, and
    the vacuity guard fails a criterion whose denominator is zero."""
    truth = json.loads((written / "truth.json").read_text(encoding="utf-8"))["truth"]
    held = [
        r for r in _load(written, "claims.json", "claims") if is_holdout(r["claim"]["program_id"])
    ]
    assert held
    outcomes = Counter(truth[r["claim"]["claim_id"]]["outcome"] for r in held)
    assert outcomes[RecoveryOutcome.NOT_RECOVERABLE.value] >= 1
    assert outcomes[RecoveryOutcome.RECOVERABLE.value] >= 1


def test_both_splits_are_populated_and_disjoint(written: Path) -> None:
    records = _load(written, "claims.json", "claims")
    held = {r["claim"]["claim_id"] for r in records if r["split"] == "holdout"}
    development = {r["claim"]["claim_id"] for r in records if r["split"] == "development"}
    assert held and development
    assert not (held & development)
    assert len(held) + len(development) == len(records)


# --------------------------------------------------------------------------------- the notice


def test_every_generated_file_says_in_its_own_body_that_it_is_synthetic(written: Path) -> None:
    for name in CORPUS_FILES:
        payload = json.loads((written / name).read_text(encoding="utf-8"))
        assert payload["is_synthetic"] is True, name
        assert "synthetic" in payload["notice"].lower(), name
        assert "not a manufacturer publication" in payload["notice"], name


def test_every_document_says_so_in_the_text_a_citation_would_show(written: Path) -> None:
    documents = _load(written, "documents.json", "documents")
    assert documents
    for document in documents:
        assert document["text"].startswith(SYNTHETIC_NOTICE), document["document_id"]


def test_the_artifact_publishes_the_rule_it_implements(corpus: GeneratedCorpus) -> None:
    assert corpus.artifact["ground_truth_rule"] == list(GROUND_TRUTH_RULE)
    assert "construction metadata" in corpus.artifact["ground_truth"]
    assert corpus.artifact["split_unit"] == "warranty_program"
