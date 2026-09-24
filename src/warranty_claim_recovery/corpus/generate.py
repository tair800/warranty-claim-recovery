"""Assemble the corpus, check it, and only then write it — refusing to write one that fails.

The checks in here are not tests. Tests run when somebody runs them; these run every time the
corpus is built, and a failure stops the build rather than producing files a later stage has to
discover are wrong. Five of them earn their place on their own:

- **Offsets.** `document_text[start:end] == clause.text`, for every clause in every document. Kill
  condition I is graded on exactly this arithmetic, and the generator is the only thing in the
  system that can guarantee it rather than measure it.
- **The intended outcome.** Every claim is built by a named construction with a declared answer, and
  the generator recomputes the answer from the ordered ground-truth rule and refuses a corpus where
  the two disagree. Without it, a construction that quietly stopped producing the case it was
  written for would silently make the evaluation easier and nothing would say so.
- **The partition.** A hold-out claim whose governing clause belongs to a development programme is
  the leak the hold-out exists to prevent, and it is checked against the *references* rather than
  against the rule — the rule cannot disagree with itself.
- **No floats.** Every payload is walked before it is written and a `float` anywhere in it is a
  build failure. `CLAUDE.md` §2.2 says a float in a monetary path is a defect rather than a style
  preference; this is where that becomes enforceable instead of aspirational, because JSON is the
  one boundary where an exact amount could quietly become a binary approximation.
- **The contract floors.** ADR-001's corpus table, restated so the generator fails with a message
  naming what is short, rather than leaving the kill test to fail later on an assertion about a
  number in a file.

Nothing written here contains a timestamp, a hostname, a path or a wall-clock reading. Two runs must
be byte-identical, and a `generated_at` field would make that impossible for the most boring reason
imaginable. The writer pins `newline="\\n"` for the same class of reason: the platform default on
Windows rewrites every newline, and a corpus whose bytes depend on the operating system cannot be
compared across two machines.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any, Final, NamedTuple

from warranty_claim_recovery.corpus.claims import (
    ADVERSARIAL_KINDS,
    EVIDENCE_KEYS,
    EVIDENCE_RELEVANT_TO,
    GROUND_TRUTH_RULE,
    NOT_ADVERSARIAL,
    RECIPE,
    BuiltClaim,
    add_months,
    build_claims,
    intended_outcome,
)
from warranty_claim_recovery.corpus.clauses import (
    SYNTHETIC_NOTICE,
    BuiltDocument,
    build_documents,
)
from warranty_claim_recovery.corpus.holdout import (
    HOLDOUT_RULE,
    SPLIT_UNIT,
    membership_of,
    partition_problems,
    split_of,
)
from warranty_claim_recovery.corpus.programs import ProgramSpec, build_programs
from warranty_claim_recovery.corpus.rng import CORPUS_SEED, GENERATOR_VERSION
from warranty_claim_recovery.domain import RecoveryOutcome, RejectionCode
from warranty_claim_recovery.money import Currency, Money

__all__ = [
    "FLOORS",
    "CorpusContractError",
    "GeneratedCorpus",
    "build",
    "generate_corpus",
]

JsonDict = dict[str, Any]

# ADR-001's corpus table and the brief's size contract, restated as module constants so a floor that
# moves is a diff a reviewer sees rather than a number buried in a comparison.
MIN_PROGRAMS: Final = 12
MIN_CLAUSES_PER_PROGRAM: Final = 10
MIN_CLAUSE_CHARACTERS: Final = 200
MIN_CLAIMS_PER_PROGRAM: Final = 40
MIN_CLAIMS_TOTAL: Final = 480
MIN_PER_REJECTION_CODE: Final = 20
MIN_NOT_RECOVERABLE: Final = 60
MIN_PARTIALLY_RECOVERABLE: Final = 60
MIN_REVIEW: Final = 40
MIN_CURRENCIES: Final = 3
#: The hold-out has floors of its own, and they are corpus properties rather than score properties.
#: A hold-out of one programme makes every rate measured over it a rate decided by forty claims
#: under one policy, and a hold-out missing a currency cannot detect a currency-specific defect at
#: all. Declared here, before any score exists, and checked by the generator.
MIN_HOLDOUT_PROGRAMS: Final = 3
MIN_HOLDOUT_CURRENCIES: Final = 3

#: Every floor, as the artifact publishes it. Imported by `scripts/generate_corpus.py` so the number
#: printed on the console and the number enforced in the build are the same constant.
FLOORS: Final[tuple[tuple[str, int], ...]] = (
    ("programs", MIN_PROGRAMS),
    ("clauses_per_program", MIN_CLAUSES_PER_PROGRAM),
    ("clause_characters", MIN_CLAUSE_CHARACTERS),
    ("claims_per_program", MIN_CLAIMS_PER_PROGRAM),
    ("claims_total", MIN_CLAIMS_TOTAL),
    ("claims_per_rejection_code", MIN_PER_REJECTION_CODE),
    ("not_recoverable_claims", MIN_NOT_RECOVERABLE),
    ("partially_recoverable_claims", MIN_PARTIALLY_RECOVERABLE),
    ("review_claims", MIN_REVIEW),
    ("currencies", MIN_CURRENCIES),
    ("holdout_programs", MIN_HOLDOUT_PROGRAMS),
    ("holdout_currencies", MIN_HOLDOUT_CURRENCIES),
)

_TRUTH_VOCABULARY: Final = (
    "missing_requirements is stated in evidence-key vocabulary — the join column "
    "RequirementSpec.evidence_key exists to be. The corpus owns what evidence a claim was built "
    "without; requirements.py owns what each rejection code demands, and restating that table here "
    "would be a second implementation of a rule that must live in one place."
)

_TRUTH_AMOUNT_NOTE: Final = (
    "recoverable_amount is the output of the deterministic money formula and is stated for every "
    "claim, including claims the gate refuses. The amount and the outcome answer different "
    "questions: a claim for a part the policy does not cover can still have positive labour "
    "arithmetic behind it, and refusing it is the gate's job rather than the formula's."
)


class CorpusContractError(RuntimeError):
    """The generated corpus does not meet its contract. Raised instead of writing the files."""


class BuiltProgram(NamedTuple):
    spec: ProgramSpec
    documents: tuple[BuiltDocument, ...]
    claims: tuple[BuiltClaim, ...]


class GeneratedCorpus(NamedTuple):
    programs: JsonDict
    clauses: JsonDict
    claims: JsonDict
    coverage: JsonDict
    documents: JsonDict
    truth: JsonDict
    artifact: JsonDict

    @property
    def files(self) -> tuple[tuple[str, JsonDict], ...]:
        return (
            ("programs.json", self.programs),
            ("clauses.json", self.clauses),
            ("claims.json", self.claims),
            ("coverage.json", self.coverage),
            ("documents.json", self.documents),
            ("truth.json", self.truth),
        )


# ------------------------------------------------------------------------------------- checks


def _check_offsets(built: Sequence[BuiltProgram]) -> int:
    """Every clause's offsets must index its own document's text exactly.

    Compared by slicing rather than by membership. A clause whose body also occurs elsewhere in the
    same document — and the withdrawn bulletin is written to make that nearly true — would pass a
    membership test with coordinates that point at the wrong copy, and a citation built from those
    coordinates would quote the withdrawn authority while naming the current one.
    """
    checked = 0
    for program in built:
        for document in program.documents:
            for clause in document.clauses:
                located = document.text[clause.start_offset : clause.end_offset]
                if located != clause.text:
                    raise CorpusContractError(
                        f"{clause.clause_id} claims offsets "
                        f"[{clause.start_offset}:{clause.end_offset}] of {document.document_id}, "
                        f"which hold {located[:60]!r}, not {clause.text[:60]!r}. A citation built "
                        f"from this corpus would be unfaithful by construction."
                    )
                checked += 1
    return checked


def _check_clause_bodies(built: Sequence[BuiltProgram]) -> None:
    for program in built:
        clauses = [c for document in program.documents for c in document.clauses]
        if len(clauses) < MIN_CLAUSES_PER_PROGRAM:
            raise CorpusContractError(
                f"{program.spec.program.program_id} has {len(clauses)} clauses and the contract "
                f"requires at least {MIN_CLAUSES_PER_PROGRAM}"
            )
        for clause in clauses:
            if len(clause.text) < MIN_CLAUSE_CHARACTERS:
                raise CorpusContractError(
                    f"{clause.clause_id} is {len(clause.text)} characters and the contract "
                    f"requires "
                    f"at least {MIN_CLAUSE_CHARACTERS}; a clause too short to state a rule is a "
                    f"clause a retrieval score would be measured over for nothing"
                )
        for document in program.documents:
            if SYNTHETIC_NOTICE not in document.text:
                raise CorpusContractError(
                    f"{document.document_id} does not say in its own body that it is synthetic. "
                    f"The body is what a citation shows a reader; the envelope is what a reader "
                    f"never sees."
                )


def _check_governing_clauses(built: Sequence[BuiltProgram]) -> None:
    for program in built:
        by_id = {
            clause.clause_id: clause
            for document in program.documents
            for clause in document.clauses
        }
        for claim in program.claims:
            clause = by_id.get(claim.governing_clause_id)
            if clause is None:
                raise CorpusContractError(
                    f"{claim.claim.claim_id} names governing clause {claim.governing_clause_id}, "
                    f"which does not exist in its own programme"
                )
            if clause.program_id != claim.program_id:
                raise CorpusContractError(
                    f"{claim.claim.claim_id} is under {claim.program_id} and its governing clause "
                    f"belongs to {clause.program_id}"
                )
            if claim.claim.rejection_code not in clause.governs:
                raise CorpusContractError(
                    f"{claim.claim.claim_id} was rejected under {claim.claim.rejection_code} and "
                    f"its governing clause {clause.clause_id} governs {list(clause.governs)}. A "
                    f"clause that governs nothing is context and may never be the authority behind "
                    f"a satisfied requirement."
                )


def _check_intended_outcomes(built: Sequence[BuiltProgram]) -> None:
    for program in built:
        for claim in program.claims:
            expected = intended_outcome(claim.shape)
            if claim.outcome is not expected:
                raise CorpusContractError(
                    f"{claim.claim.claim_id} was built by the {claim.shape} construction, which "
                    f"exists to produce {expected}, and the ground-truth rule made it "
                    f"{claim.outcome}. Either the construction stopped doing what it was written "
                    f"to do, or the rule changed; both are build failures rather than a quietly "
                    f"different corpus."
                )


def _check_evidence_coherence() -> None:
    """A withheld document must be one the rejection code could plausibly have turned on.

    A claim rejected for a missing serial that carries a perfect serial and is missing its labour
    rate agreement is incoherent: the evidence gap has nothing to do with the rejection, and an
    evaluation over such a claim measures nothing anyone would act on.
    """
    for recipe in RECIPE:
        if recipe.evidence_key is None:
            continue
        if recipe.evidence_key not in EVIDENCE_KEYS:
            raise CorpusContractError(
                f"recipe slot {recipe.slot} names evidence key {recipe.evidence_key!r}, which is "
                f"not in the closed vocabulary"
            )
        if recipe.evidence_key not in EVIDENCE_RELEVANT_TO[recipe.code]:
            raise CorpusContractError(
                f"recipe slot {recipe.slot} withholds {recipe.evidence_key!r} from a claim "
                f"rejected "
                f"under {recipe.code}, and this corpus does not consider that evidence relevant to "
                f"that code"
            )


def _check_month_arithmetic(built: Sequence[BuiltProgram]) -> None:
    """`add_months` must be exactly invertible for every date this corpus fed it.

    The one-day-outside-cover construction inverts it. If a draw ever produced a day of the month
    that clamps — the twenty-ninth of a January, say — the inverse would land a day early and the
    claim would be two days outside cover while claiming to be one. Checked rather than trusted,
    because the check costs nothing and the ranges it depends on live in another module.
    """
    for program in built:
        months = program.spec.program.warranty_months
        for claim in program.claims:
            in_service = claim.claim.in_service_date
            if add_months(add_months(in_service, months), -months) != in_service:
                raise CorpusContractError(
                    f"{claim.claim.claim_id}: add_months is not invertible at {in_service} over "
                    f"{months} months, so the warranty boundary this claim sets is not the "
                    f"boundary "
                    f"it records"
                )


def _check_coverage(built: Sequence[BuiltProgram]) -> None:
    for program in built:
        known = {row.part_number for row in program.spec.coverage}
        for claim in program.claims:
            if claim.claim.part_number not in known:
                raise CorpusContractError(
                    f"{claim.claim.claim_id} claims part {claim.claim.part_number}, which has no "
                    f"row in {claim.program_id}'s covered-parts schedule. An eligibility check "
                    f"with "
                    f"nothing to check passes silently."
                )
            if claim.claim.serial_number is None and not claim.serial_in_range:
                raise CorpusContractError(
                    f"{claim.claim.claim_id} has no serial under a part that declares a serial "
                    f"range. The honest answer there is that nobody knows, and a boolean cannot "
                    f"say "
                    f"it, so this corpus does not emit the combination."
                )


def _no_floats(payload: Any, path: str = "$") -> None:
    """Refuse a payload containing a `float` anywhere, at any depth.

    JSON is the one boundary where an exact amount could quietly become a binary approximation, and
    `0.1 + 0.2 != 0.3` is not a rounding inconvenience in a recovery total — it is a package the
    manufacturer rejects a second time. `bool` is a subclass of `int` and is fine; `float` never is.
    """
    if isinstance(payload, float):
        raise CorpusContractError(
            f"{path} is the float {payload!r}. Money in this system is Decimal and every numeric "
            f"amount is written as a string; a float here would launder binary floating-point "
            f"arithmetic into a file that looks exact."
        )
    if isinstance(payload, dict):
        for key, value in payload.items():
            _no_floats(value, f"{path}.{key}")
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            _no_floats(value, f"{path}[{index}]")


def _check_contract(artifact: JsonDict) -> None:
    observed = artifact["observed"]
    for name, floor in FLOORS:
        if int(observed[name]) < floor:
            raise CorpusContractError(
                f"the corpus contract requires at least {floor} {name} and this corpus has "
                f"{observed[name]}"
            )
    by_kind = artifact["claims_by_adversarial_kind"]
    missing_kinds = [kind for kind in ADVERSARIAL_KINDS if by_kind.get(kind, 0) < 1]
    if missing_kinds:
        raise CorpusContractError(
            f"these adversarial constructions produced no claim: {missing_kinds}"
        )
    if artifact["leaked_programs"] or artifact["leaked_cases"]:
        raise CorpusContractError(
            f"the split does not partition cleanly: {artifact['partition_problems'][:5]}"
        )


# ------------------------------------------------------------------------------------- records


def _money_json(amount: Money) -> JsonDict:
    return amount.as_json()


def _program_record(program: BuiltProgram) -> JsonDict:
    spec = program.spec
    model = spec.program
    return {
        "program": {
            "program_id": model.program_id,
            "manufacturer": model.manufacturer,
            "policy_version": model.policy_version,
            "currency": model.currency.value,
            "correction_window_days": model.correction_window_days,
            "warranty_months": model.warranty_months,
            "labour_rate_cap_per_hour": _money_json(model.labour_rate_cap_per_hour),
            "deductible": _money_json(model.deductible),
            "claim_cap": _money_json(model.claim_cap),
        },
        "manufacturer_slug": spec.manufacturer.slug,
        "split": split_of(model.program_id),
        "part_families": [
            {"family_id": family.family_id, "label": family.label, "position": family.position}
            for family in spec.families
        ],
        "document_ids": [document.document_id for document in program.documents],
    }


def _clause_records(program: BuiltProgram) -> list[JsonDict]:
    records: list[JsonDict] = []
    for document in program.documents:
        for clause in document.clauses:
            records.append(
                {
                    "clause": {
                        "clause_id": clause.clause_id,
                        "program_id": clause.program_id,
                        "document_id": clause.document_id,
                        "section": clause.section,
                        "text": clause.text,
                        "start_offset": clause.start_offset,
                        "end_offset": clause.end_offset,
                        "governs": [code.value for code in clause.governs],
                    },
                    "policy_version": document.policy_version,
                    "document_kind": document.kind,
                    "document_is_current": document.is_current,
                    "split": split_of(clause.program_id),
                    "characters": len(clause.text),
                }
            )
    return records


def _document_records(program: BuiltProgram) -> list[JsonDict]:
    return [
        {
            "document_id": document.document_id,
            "program_id": document.program_id,
            "policy_version": document.policy_version,
            "kind": document.kind,
            "title": document.title,
            "revision": document.revision,
            "is_current": document.is_current,
            "superseded_by": document.superseded_by,
            "split": split_of(document.program_id),
            "clause_ids": [clause.clause_id for clause in document.clauses],
            # The full text ships with the documents so a citation's offsets can be checked against
            # the source without rebuilding the corpus. Kill condition I is graded on this string.
            "text": document.text,
        }
        for document in program.documents
    ]


def _coverage_records(program: BuiltProgram) -> list[JsonDict]:
    return [
        {
            # Exactly the fields `eligibility.PartCoverage` takes, so a loader can build the tuple
            # by field name. Everything else is corpus metadata and is kept outside that object
            # rather than inside it, because every model in this system forbids an extra field.
            "coverage": {
                "part_number": row.part_number,
                "covered": row.covered,
                "serial_first": row.serial_first,
                "serial_last": row.serial_last,
            },
            "program_id": row.program_id,
            "family_id": row.family_id,
            "variant": row.variant,
            "note": row.note,
            "split": split_of(row.program_id),
        }
        for row in program.spec.coverage
    ]


def _claim_records(program: BuiltProgram) -> list[JsonDict]:
    records: list[JsonDict] = []
    for built in program.claims:
        claim = built.claim
        record: JsonDict = {
            "claim": {
                "claim_id": claim.claim_id,
                "program_id": claim.program_id,
                "part_number": claim.part_number,
                "serial_number": claim.serial_number,
                "in_service_date": claim.in_service_date.isoformat(),
                "failure_date": claim.failure_date.isoformat(),
                "repair_invoice_date": claim.repair_invoice_date.isoformat(),
                "rejection_code": claim.rejection_code.value,
                "rejected_on": claim.rejected_on.isoformat(),
                "claimed_parts": _money_json(claim.claimed_parts),
                "claimed_labour_hours": claim.claimed_labour_hours,
                "claimed_labour_rate": _money_json(claim.claimed_labour_rate),
                "previously_recovered": (
                    None
                    if claim.previously_recovered is None
                    else _money_json(claim.previously_recovered)
                ),
            },
            "program_id": built.program_id,
            "slot": built.slot,
            "split": split_of(built.program_id),
            "as_of": built.as_of.isoformat(),
            "closes_on": built.closes_on.isoformat(),
            "construction": built.shape.value,
            "adversarial_kind": built.adversarial_kind,
            "part_family_id": built.part_family_id,
            "evidence": dict(built.evidence),
            "facts": {
                "currency_matches_program": built.currency_matches_program,
                "window_open": built.window_open,
                "within_warranty_period": built.within_warranty,
                "part_covered": built.part_covered,
                "serial_in_range": built.serial_in_range,
            },
            "recovery_identity": claim.recovery_identity,
        }
        if built.money is not None:
            record["money"] = {
                "claimed_total": _money_json(built.money.claimed_total),
                "labour_excess": _money_json(built.money.labour_excess),
                "uncovered_parts": _money_json(built.money.uncovered_parts),
                "eligible_amount": _money_json(built.money.eligible_amount),
                "deductible": _money_json(built.money.deductible),
                "capped_amount": _money_json(built.money.capped_amount),
                "already_recovered": _money_json(built.money.already_recovered),
                "recoverable_amount": _money_json(built.money.recoverable_amount),
            }
        records.append(record)
    return records


def _truth_entry(built: BuiltClaim) -> JsonDict:
    return {
        "outcome": built.outcome.value,
        # A string, never a float. `json.dumps` on a Decimal would raise and on a float would write
        # a binary approximation; a string is the only form that survives the round trip exactly.
        "recoverable_amount": str(built.recoverable_amount.quantize().amount),
        "currency": built.recoverable_amount.currency.value,
        "governing_clause_id": built.governing_clause_id,
        "missing_requirements": list(built.missing_requirements),
    }


# ------------------------------------------------------------------------------------- assembly


def _envelope(kind: str, records: Sequence[JsonDict]) -> JsonDict:
    """Every file states, in its own body, that it is synthetic. ADR-001 requires it by name."""
    return {
        "is_synthetic": True,
        "notice": SYNTHETIC_NOTICE,
        "generator_version": GENERATOR_VERSION,
        "seed": CORPUS_SEED,
        "count": len(records),
        kind: list(records),
    }


def _tally(values: Sequence[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _artifact(
    built: Sequence[BuiltProgram],
    *,
    offsets_checked: int,
    leaked_programs: Sequence[str],
    leaked_cases: Sequence[str],
    problems: Sequence[str],
) -> JsonDict:
    claims = [claim for program in built for claim in program.claims]
    clauses = [c for program in built for document in program.documents for c in document.clauses]
    coverage = [row for program in built for row in program.spec.coverage]
    programs = [program.spec.program for program in built]

    by_outcome = _tally([claim.outcome.value for claim in claims])
    by_code = _tally([claim.claim.rejection_code.value for claim in claims])
    holdout_programs = sorted(p.program_id for p in programs if split_of(p.program_id) == "holdout")
    holdout_currencies = sorted(
        {p.currency.value for p in programs if p.program_id in set(holdout_programs)}
    )
    development_programs = sorted(
        p.program_id for p in programs if split_of(p.program_id) != "holdout"
    )

    observed = {
        "programs": len(programs),
        "clauses_per_program": min(
            len([c for document in program.documents for c in document.clauses])
            for program in built
        ),
        "clause_characters": min(len(clause.text) for clause in clauses),
        "claims_per_program": min(len(program.claims) for program in built),
        "claims_total": len(claims),
        "claims_per_rejection_code": min(by_code.get(code.value, 0) for code in RejectionCode),
        "not_recoverable_claims": by_outcome.get(RecoveryOutcome.NOT_RECOVERABLE.value, 0),
        "partially_recoverable_claims": by_outcome.get(
            RecoveryOutcome.PARTIALLY_RECOVERABLE.value, 0
        ),
        "review_claims": by_outcome.get(RecoveryOutcome.REVIEW.value, 0),
        "currencies": len({p.currency for p in programs}),
        "holdout_programs": len(holdout_programs),
        "holdout_currencies": len(holdout_currencies),
    }

    return {
        "is_synthetic": True,
        "notice": SYNTHETIC_NOTICE,
        "generator_version": GENERATOR_VERSION,
        "seed": CORPUS_SEED,
        "ground_truth": (
            "construction metadata — the generator knows every answer because it built the claim "
            "that way. Nothing in this corpus was labelled by a model."
        ),
        "ground_truth_rule": list(GROUND_TRUTH_RULE),
        "split_rule": HOLDOUT_RULE,
        "split_unit": SPLIT_UNIT,
        "counts": {
            "programs": len(programs),
            "documents": sum(len(program.documents) for program in built),
            "clauses": len(clauses),
            "claims": len(claims),
            "coverage_rows": len(coverage),
            "manufacturers": len({p.manufacturer for p in programs}),
        },
        "observed": observed,
        "floors": dict(FLOORS),
        "claims_by_outcome": by_outcome,
        "claims_by_rejection_code": by_code,
        "claims_by_construction": _tally([claim.shape.value for claim in claims]),
        "claims_by_adversarial_kind": _tally(
            [
                claim.adversarial_kind
                for claim in claims
                if claim.adversarial_kind != NOT_ADVERSARIAL
            ]
        ),
        "claims_by_split": _tally([split_of(claim.program_id) for claim in claims]),
        "claims_with_a_closed_window": sum(1 for claim in claims if not claim.window_open),
        "claims_with_currency_mismatch": sum(
            1 for claim in claims if not claim.currency_matches_program
        ),
        "currencies": sorted({p.currency.value for p in programs}),
        "holdout_programs": holdout_programs,
        "holdout_currencies": holdout_currencies,
        "development_programs": development_programs,
        "clause_offsets_verified": offsets_checked,
        "leaked_programs": len(leaked_programs),
        "leaked_cases": len(leaked_cases),
        "partition_problems": list(problems),
        "evidence_keys": list(EVIDENCE_KEYS),
        "adversarial_kinds": list(ADVERSARIAL_KINDS),
        # Deliberately absent: a generation timestamp, a hostname and a path. Two runs must be
        # byte-identical, and a clock in the output is the cheapest possible way to lose that.
        "contains_timestamp": False,
    }


def build() -> GeneratedCorpus:
    """The corpus in memory, fully checked. Raises `CorpusContractError` rather than returning."""
    built: list[BuiltProgram] = []
    for spec in build_programs():
        documents = build_documents(spec)
        claims = build_claims(spec, documents)
        built.append(BuiltProgram(spec=spec, documents=documents, claims=claims))

    _check_evidence_coherence()
    offsets_checked = _check_offsets(built)
    _check_clause_bodies(built)
    _check_governing_clauses(built)
    _check_intended_outcomes(built)
    _check_month_arithmetic(built)
    _check_coverage(built)

    claim_programs = {
        claim.claim.claim_id: claim.program_id for program in built for claim in program.claims
    }
    clause_programs = {
        clause.clause_id: clause.program_id
        for program in built
        for document in program.documents
        for clause in document.clauses
    }
    governing = {
        claim.claim.claim_id: claim.governing_clause_id
        for program in built
        for claim in program.claims
    }
    membership = membership_of(
        tuple(program.spec.program.program_id for program in built), claim_programs
    )
    report = partition_problems(
        membership,
        claim_programs=claim_programs,
        clause_programs=clause_programs,
        governing_clause=governing,
    )

    artifact = _artifact(
        built,
        offsets_checked=offsets_checked,
        leaked_programs=report.leaked_programs,
        leaked_cases=report.leaked_cases,
        problems=report.problems,
    )
    _check_contract(artifact)

    generated = GeneratedCorpus(
        programs=_envelope("programs", [_program_record(program) for program in built]),
        clauses=_envelope(
            "clauses", [record for program in built for record in _clause_records(program)]
        ),
        claims=_envelope(
            "claims", [record for program in built for record in _claim_records(program)]
        ),
        coverage=_envelope(
            "coverage", [record for program in built for record in _coverage_records(program)]
        ),
        documents=_envelope(
            "documents", [record for program in built for record in _document_records(program)]
        ),
        truth=_truth_file(built),
        artifact=artifact,
    )

    for _, payload in generated.files:
        _no_floats(payload)
    _no_floats(generated.artifact)
    _check_truth_amounts(generated.truth)
    return generated


def _truth_file(built: Sequence[BuiltProgram]) -> JsonDict:
    entries = {
        claim.claim.claim_id: _truth_entry(claim) for program in built for claim in program.claims
    }
    return {
        "is_synthetic": True,
        "notice": SYNTHETIC_NOTICE,
        "generator_version": GENERATOR_VERSION,
        "seed": CORPUS_SEED,
        "ground_truth": (
            "construction metadata — the generator knows every answer because it built the claim "
            "that way. Nothing in this corpus was labelled by a model."
        ),
        "ground_truth_rule": list(GROUND_TRUTH_RULE),
        "missing_requirements_vocabulary": _TRUTH_VOCABULARY,
        "recoverable_amount_note": _TRUTH_AMOUNT_NOTE,
        "count": len(entries),
        "truth": entries,
    }


def _check_truth_amounts(truth: JsonDict) -> None:
    """Every recovered amount must be an exact decimal string with the currency's own scale.

    Exponent notation would round-trip through `Decimal` correctly and would still be wrong here:
    the file is read by a console and by a reviewer as well as by a parser, and `1.2E+3` in a
    settlement column is a number somebody will mistype.
    """
    for claim_id, entry in truth["truth"].items():
        raw = entry["recoverable_amount"]
        if not isinstance(raw, str):
            raise CorpusContractError(f"{claim_id}: recoverable_amount is {type(raw).__name__}")
        value = Decimal(raw)
        if value != value.quantize(Decimal("0.01")):
            raise CorpusContractError(f"{claim_id}: {raw} is not stated to the currency's scale")
        if "E" in raw or "e" in raw:
            raise CorpusContractError(f"{claim_id}: {raw} is in exponent notation")
        Currency(entry["currency"])


# ------------------------------------------------------------------------------------- writing


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False)
    path.write_text(body + "\n", encoding="utf-8", newline="\n")


def generate_corpus(out_dir: Path, *, artifact_path: Path | None = None) -> GeneratedCorpus:
    """Build, check and write the corpus. Returns what was written."""
    generated = build()
    for name, payload in generated.files:
        _write_json(out_dir / name, payload)
    if artifact_path is not None:
        _write_json(artifact_path, generated.artifact)
    return generated
