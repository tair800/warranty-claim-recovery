"""The deterministic path, run over every claim in the committed corpus, once per claim.

```
claim -> requirements   the rejection code's matrix, with evidence resolved from the case
      -> deadline       the correction window, as of the date the corpus recorded
      -> eligibility    warranty period, part and serial, prior recovery
      -> recovery       the money, Decimal throughout, rounded once
      -> retrieval      the governing clause, from pgvector, filtered before it is ranked
      -> citation       one per satisfied requirement, at the policy version searched
      -> gate           the outcome, computed from signals no model produced
```

The order is `CLAUDE.md` §2.1's, node for node, and it is run here rather than approximated, because
an evaluation that scores a reimplementation of the pipeline scores the reimplementation. What this
module adds to the graph is a loop and a comparison against `data/generated/truth.json`; everything
between the arrows is the shipped module.

### The join this module owns, and why it exists at all

The requirement matrix in `requirements.py` says what each **rejection code demands**, in the
manufacturer's vocabulary: a serial transcribed from the part, an entry in the covered-parts
schedule, a labour rate agreement. The corpus in `corpus/claims.py` says what **artefacts a claim
carries**, in a distributor's document-store vocabulary: `serial_plate_photo`,
`coverage_confirmation`, `labour_rate_agreement`. The two vocabularies overlap in two strings out of
fourteen, and that is not an accident to be fixed by renaming one of them.

They are different tables answering different questions and each has exactly one home. Renaming the
matrix's keys to match the corpus would make the requirement matrix a description of this synthetic
corpus, so that a real document store with different filenames would silently satisfy nothing.
Renaming the corpus's keys to match the matrix would make the generator's evidence bundle a
restatement of the matrix, and `corpus/claims.py` records at length why it refuses to hold a second
copy of a rule that must live in one place.

So the join is stated here, once, in `EVIDENCE_JOIN`, because this module is the only place the two
tables meet. In a deployment the same join is the intake adapter between the document store and the
requirement matrix, and it would be written against that store's real key names.

**The join is checked for the property that matters, and the check runs at import.** Over-mapping is
harmless: a requirement that consults an artefact the code did not really turn on can only be
unsatisfied when that artefact is genuinely absent, and an absent artefact is an incomplete case
however it is routed. Under-mapping is the dangerous direction, and it is silent: a claim built
without its repair invoice, whose requirements consult nothing that reads the repair invoice, comes
out with every requirement satisfied and is authorised for a resubmission on evidence nobody has.
`assert_join_is_total` therefore checks that for every rejection code, every artefact this corpus is
willing to withhold for that code is reachable from at least one of that code's requirements. It is
the cross-lane check `corpus/claims.py` asks for by name.

### Three requirements are answered by the policy's own schedules, not by a document

`REQ-PART-IN-SCHEDULE`, `REQ-SERIAL-IN-COVERED-RANGE` and `REQ-SERIAL-STAMPED` name things the
system already knows. Whether the claimed part number appears in the covered-parts schedule is a
fact about the schedule; whether the stamped serial falls inside the declared build range is
arithmetic over that schedule; whether a serial was recorded at all is a property of the claim.

Resolving those three from the evidence bundle alone would mark them satisfied because a document
called `coverage_confirmation` is on file — while the schedule says the part is not covered. A
confirmation that confirms something untrue is worse than no confirmation, because it ends up
quoted in the resubmission. So those three requirements are satisfied only when the document **and**
the deterministic finding agree, and `FACT_BACKED` names them. Without this, four of the corpus's
constructions — an excluded part, a superseded variant, a part whose supplier changed, and a serial
below the covered build range — arrive at the gate with a complete evidence bundle, positive labour
arithmetic and nothing to refuse them, and the system authorises a recovery for a part the policy
does not cover. That is a false recovery, and kill condition G budgets zero.

### What this module deliberately does not do

It does not decide any outcome. Every outcome here comes from `gate.decide`, with one exception that
withholds rather than authorises: a case whose governing authority could not be retrieved is
escalated to `REVIEW` without reaching the gate, because a satisfied requirement cannot be
constructed without a citation and this system will not report a requirement as satisfied by
nothing. `RECOVERABLE` and `PARTIALLY_RECOVERABLE` remain reachable from exactly one place in this
repository, which is `gate.py`'s fall-through.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, Final, NamedTuple

from sqlalchemy import text
from sqlalchemy.orm import Session

from warranty_claim_recovery import deadline, gate, recovery
from warranty_claim_recovery.corpus.claims import EVIDENCE_KEYS, EVIDENCE_RELEVANT_TO
from warranty_claim_recovery.corpus.holdout import (
    DEVELOPMENT,
    HOLDOUT,
    Membership,
    load_frozen,
    split_of,
)
from warranty_claim_recovery.domain import (
    Citation,
    Claim,
    ClaimWindow,
    GateDecision,
    PolicyClause,
    RecoveryComputation,
    RecoveryOutcome,
    RejectionCode,
    Requirement,
    RequirementStatus,
    WarrantyProgram,
)
from warranty_claim_recovery.eligibility import Eligibility, PartCoverage, assess
from warranty_claim_recovery.money import Currency, CurrencyMismatchError, Money
from warranty_claim_recovery.requirements import REQUIREMENT_MATRIX, RequirementSpec, required_for
from warranty_claim_recovery.retrieval.citations import citation_for
from warranty_claim_recovery.retrieval.pipeline import DEFAULT_K, Retriever, ScoredClause

__all__ = [
    "EVIDENCE_JOIN",
    "FACT_BACKED",
    "PRESENT",
    "CaseAssessment",
    "CorpusCase",
    "DeterminedFacts",
    "JoinError",
    "MemoisingEncoder",
    "assert_join_is_total",
    "assess_case",
    "candidate_counts",
    "case_question",
    "load_cases",
    "pgvector_search",
    "requirements_for_case",
]

#: The evidence status a corpus record carries when the artefact is on file and consistent. The
#: other two values — `ABSENT` and `CONFLICTING` — are read as their `RequirementStatus`
#: counterparts. Compared as a string rather than imported as an enum because the corpus files are
#: read as JSON and the string is what is actually in them; importing the enum and comparing against
#: its `.value` would be the same comparison with an import that suggests otherwise.
PRESENT: Final = "PRESENT"


class JoinError(ValueError):
    """The join between the requirement matrix and the corpus's evidence vocabulary is not total.

    A distinct type because the remedy is specific and is never "add a fallback": a requirement that
    reads no artefact the corpus can withhold makes a claim built without its evidence look
    complete, and the fix is an entry in `EVIDENCE_JOIN`, not a default.
    """


#: Which artefacts in the corpus's document-store vocabulary evidence each requirement of the
#: matrix. The keys are `RequirementSpec.evidence_key`; the values are members of
#: `corpus.claims.EVIDENCE_KEYS`.
#:
#: A requirement is satisfied only when **every** artefact listed against it is on file and
#: consistent, so listing more artefacts against a requirement makes it harder to satisfy, never
#: easier. That asymmetry is why the pairs below are read generously: a repair invoice bears on
#: whether an installation can be matched to a claim (policy clause 3 says so in as many words), on
#: whether this recovery is distinct from one already settled, and on the hours charged, so it
#: appears against three requirements. The alternative — one artefact per requirement, chosen for
#: tidiness — leaves the artefacts this corpus withholds unreachable from the requirements they
#: bear on, which is the silent failure `assert_join_is_total` exists to refuse.
EVIDENCE_JOIN: Final[dict[str, tuple[str, ...]]] = {
    # MISSING_SERIAL
    "part_serial_number": ("serial_plate_photo",),
    "serial_coverage_range": ("coverage_confirmation", "part_number_confirmation"),
    # MISSING_INSTALL_PROOF. The repair invoice is here because policy clause 3 rejects a claim
    # "where the repair invoice cannot be matched to an installation on file"; the invoice is part
    # of proving the installation, not merely a billing document.
    "installation_certificate": ("installation_certificate", "repair_invoice"),
    "installer_identity": ("commissioning_date", "supplier_declaration"),
    # WRONG_FAILURE_CODE
    "failure_code": ("failure_code",),
    "technician_report": ("diagnostic_report",),
    # PART_NOT_COVERED
    "covered_parts_schedule": ("coverage_confirmation", "part_number_confirmation"),
    "failure_causation": ("diagnostic_report", "supplier_declaration"),
    # OUTSIDE_WARRANTY_PERIOD
    "in_service_record": (
        "commissioning_date",
        "warranty_start_evidence",
        "installation_certificate",
    ),
    "warranty_period_clause": ("warranty_start_evidence",),
    # DUPLICATE_CLAIM
    "prior_recovery_ledger": ("prior_claim_reference",),
    "recovery_identity": ("repair_invoice", "prior_claim_reference"),
    # LABOUR_RATE_EXCEEDED
    "labour_rate_schedule": ("labour_rate_agreement",),
    "repair_time_allowance": ("repair_invoice", "labour_rate_agreement"),
}


class DeterminedFacts(NamedTuple):
    """The three findings the deterministic core has already made about coverage.

    Carried as a named tuple rather than passed as three booleans so that a caller cannot transpose
    two of them. All three are `True` in the ordinary case, and a transposition would therefore be
    invisible on most of the corpus and wrong on exactly the claims the hold-out is scored over.
    """

    serial_recorded: bool
    serial_in_covered_range: bool
    part_in_schedule: bool


#: The requirements that a document alone cannot satisfy. See the module docstring: each of these
#: names something the policy's own schedules answer, and a document asserting the opposite of the
#: schedule is not evidence — it is a contradiction the resubmission would carry into the
#: manufacturer's hands.
FACT_BACKED: Final[dict[str, Callable[[DeterminedFacts], bool]]] = {
    "REQ-SERIAL-STAMPED": lambda facts: facts.serial_recorded,
    "REQ-SERIAL-IN-COVERED-RANGE": lambda facts: facts.serial_in_covered_range,
    "REQ-PART-IN-SCHEDULE": lambda facts: facts.part_in_schedule,
}


def assert_join_is_total(
    join: Mapping[str, tuple[str, ...]] | None = None,
    *,
    fact_backed: Mapping[str, Callable[[DeterminedFacts], bool]] | None = None,
) -> None:
    """Refuse a join that would let a claim with withheld evidence look complete.

    Four checks, each corresponding to a way this table goes wrong rather than to a way it could in
    principle be malformed.

    First, **every requirement is joined**. A requirement whose evidence key has no entry would
    raise a `KeyError` at the first claim carrying its rejection code, three modules from here.

    Second, **every artefact named is one the corpus can carry**. A typo in an artefact name reads
    as an artefact that is never present, so every requirement mentioning it would be permanently
    unsatisfied and the affected rejection code would never recover anything.

    Third, and this is the one with teeth: **for every rejection code, every artefact this corpus is
    willing to withhold for that code is reachable from one of that code's own requirements.** A gap
    here is invisible in every other test — the claim validates, the requirements construct, the
    gate runs — and it authorises a resubmission whose missing document nothing looked for.

    Fourth, **every fact-backed requirement identifier exists in the matrix**. A renamed requirement
    would silently stop being fact-backed, and the four constructions that depend on it would
    quietly become false recoveries.

    Both tables are parameters, defaulting to the shipped ones, so `tests/test_evaluation.py` can
    hand this a table with a hole and observe the refusal. A guard that has only ever been run
    against the correct input has never been shown to reject anything.
    """
    subject = EVIDENCE_JOIN if join is None else join
    facts = FACT_BACKED if fact_backed is None else fact_backed

    for code in RejectionCode:
        for spec in REQUIREMENT_MATRIX[code]:
            if spec.evidence_key not in subject:
                raise JoinError(
                    f"{spec.requirement_id} needs artefact {spec.evidence_key!r} and the join has "
                    f"no entry for it. The requirement would raise at the first claim rejected "
                    f"under {code.value}."
                )

    for evidence_key, artefacts in sorted(subject.items()):
        if not artefacts:
            raise JoinError(
                f"{evidence_key!r} is joined to no artefact at all, so the requirement it backs is "
                f"satisfied by anything, including by a claim that carries nothing"
            )
        for artefact in artefacts:
            if artefact not in EVIDENCE_KEYS:
                raise JoinError(
                    f"{evidence_key!r} names artefact {artefact!r}, which is not in the corpus's "
                    f"closed vocabulary {sorted(EVIDENCE_KEYS)}. A name no record carries reads as "
                    f"an artefact that is never present."
                )

    for code in RejectionCode:
        reachable = {
            artefact
            for spec in REQUIREMENT_MATRIX[code]
            for artefact in subject.get(spec.evidence_key, ())
        }
        for artefact in EVIDENCE_RELEVANT_TO[code]:
            if artefact not in reachable:
                raise JoinError(
                    f"this corpus withholds {artefact!r} from claims rejected under {code.value}, "
                    f"and no requirement of that code consults it. A claim built without that "
                    f"artefact would reach the gate with every requirement satisfied and be "
                    f"authorised on evidence nobody looked for."
                )

    known = {spec.requirement_id for specs in REQUIREMENT_MATRIX.values() for spec in specs}
    for requirement_id in sorted(facts):
        if requirement_id not in known:
            raise JoinError(
                f"{requirement_id!r} is declared as answered by a deterministic finding and is not "
                f"in the requirement matrix. It would never be applied, and the coverage "
                f"constructions it exists for would become false recoveries."
            )


# Run at import, for the reason `requirements.assert_matrix_is_total` runs at import: a join that is
# wrong is wrong for every claim that follows, and the cheapest place to find out is before a corpus
# has been read, while the traceback still points at this file.
assert_join_is_total()


# ------------------------------------------------------------------------------------ the corpus


class CorpusCase(NamedTuple):
    """One claim with everything needed to run the deterministic path and grade the answer.

    The generator's answer is carried alongside the inputs rather than looked up later, so that a
    case and the truth it is graded against cannot come from two different reads of two different
    files. `truth_amount` is parsed from the string form in `truth.json` with `Decimal`; kill
    condition F is exact equality, and a float parsed from `"1176.00"` would already be a different
    number from the one the generator computed.
    """

    claim: Claim
    program: WarrantyProgram
    coverage: PartCoverage
    as_of: date
    evidence: Mapping[str, str]
    split: str
    truth_outcome: RecoveryOutcome
    truth_amount: Money
    governing_clause_id: str


def _read(path: Path, key: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing. The corpus is rebuilt from a committed seed rather than kept in "
            f"git; run `make corpus` before scoring anything against it."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload[key]
    if not isinstance(entries, list):
        raise ValueError(f"{path}: {key!r} is not a list of records")
    return [entry for entry in entries if isinstance(entry, dict)]


def _money(raw: Mapping[str, Any]) -> Money:
    """An amount from the corpus, as a `Decimal` parsed from its string form.

    `Decimal(str)` and never `Decimal(float)`. The corpus writes every amount as a string precisely
    so that this boundary can be exact, and taking the shorter route through `float` here would
    reintroduce binary floating point at the one place kill condition F is graded.
    """
    return Money(Decimal(str(raw["amount"])), Currency(str(raw["currency"])))


def _program(record: Mapping[str, Any]) -> WarrantyProgram:
    payload = record["program"]
    return WarrantyProgram(
        program_id=str(payload["program_id"]),
        manufacturer=str(payload["manufacturer"]),
        policy_version=str(payload["policy_version"]),
        currency=Currency(str(payload["currency"])),
        correction_window_days=int(payload["correction_window_days"]),
        warranty_months=int(payload["warranty_months"]),
        labour_rate_cap_per_hour=_money(payload["labour_rate_cap_per_hour"]),
        deductible=_money(payload["deductible"]),
        claim_cap=_money(payload["claim_cap"]),
    )


def _claim(payload: Mapping[str, Any]) -> Claim:
    prior = payload.get("previously_recovered")
    return Claim(
        claim_id=str(payload["claim_id"]),
        program_id=str(payload["program_id"]),
        part_number=str(payload["part_number"]),
        serial_number=None if payload["serial_number"] is None else str(payload["serial_number"]),
        in_service_date=date.fromisoformat(str(payload["in_service_date"])),
        failure_date=date.fromisoformat(str(payload["failure_date"])),
        repair_invoice_date=date.fromisoformat(str(payload["repair_invoice_date"])),
        rejection_code=RejectionCode(str(payload["rejection_code"])),
        rejected_on=date.fromisoformat(str(payload["rejected_on"])),
        claimed_parts=_money(payload["claimed_parts"]),
        claimed_labour_hours=int(payload["claimed_labour_hours"]),
        claimed_labour_rate=_money(payload["claimed_labour_rate"]),
        previously_recovered=None if prior is None else _money(prior),
    )


def _split_from(membership: Membership, claim_id: str, program_id: str) -> str:
    """Which side of the line this case is on, taken from the committed freeze.

    Read from `artifacts/holdout.json` rather than recomputed from the rule. The freeze is the
    artefact whose position in git history proves the split was fixed before anything was scored
    over it; recomputing the rule here would produce the same answer today and would silently follow
    the corpus if a programme identifier ever changed, which is the whole failure the freeze
    prevents. The rule is then used as a cross-check, so a freeze that has drifted from it fails the
    evaluation rather than quietly regrading it.
    """
    if claim_id in membership.holdout_cases:
        declared = HOLDOUT
    elif claim_id in membership.development_cases:
        declared = DEVELOPMENT
    else:
        raise ValueError(
            f"{claim_id} is enumerated in neither side of the frozen hold-out. The corpus and "
            f"artifacts/holdout.json describe different claim sets, and every figure measured over "
            f"either of them would be measured over a population nobody declared."
        )
    if declared != split_of(program_id):
        raise ValueError(
            f"{claim_id} is frozen as {declared} and the hold-out rule makes its programme "
            f"{program_id} {split_of(program_id)}. Re-freezing to resolve this invalidates every "
            f"score taken against the old membership; ADR-001 §7 requires a new recorded decision."
        )
    return declared


def load_cases(corpus_dir: Path, artifacts_dir: Path) -> tuple[CorpusCase, ...]:
    """Every claim in the corpus, in the order the generator wrote them, with its ground truth.

    The generator's order is kept rather than sorted. It is already deterministic — the recipe is a
    fixed table and the programmes are built in a fixed order — and re-sorting here would make the
    evaluation's iteration order a second thing that has to agree with the corpus's, for no benefit.
    """
    membership = load_frozen(artifacts_dir)
    if membership is None:
        raise FileNotFoundError(
            f"{artifacts_dir / 'holdout.json'} is missing. The hold-out is frozen before anything "
            f"is scored against it, and an evaluation that decided the split for itself would be "
            f"an evaluation that could decide it again after seeing a result."
        )

    programs = {
        str(record["program"]["program_id"]): _program(record)
        for record in _read(corpus_dir / "programs.json", "programs")
    }
    coverage: dict[tuple[str, str], PartCoverage] = {}
    for record in _read(corpus_dir / "coverage.json", "coverage"):
        row = record["coverage"]
        coverage[(str(record["program_id"]), str(row["part_number"]))] = PartCoverage(
            part_number=str(row["part_number"]),
            covered=bool(row["covered"]),
            serial_first=None if row["serial_first"] is None else int(row["serial_first"]),
            serial_last=None if row["serial_last"] is None else int(row["serial_last"]),
        )
    truth_payload = json.loads((corpus_dir / "truth.json").read_text(encoding="utf-8"))["truth"]

    cases: list[CorpusCase] = []
    for record in _read(corpus_dir / "claims.json", "claims"):
        claim = _claim(record["claim"])
        answer = truth_payload[claim.claim_id]
        cases.append(
            CorpusCase(
                claim=claim,
                program=programs[claim.program_id],
                coverage=coverage[(claim.program_id, claim.part_number)],
                as_of=date.fromisoformat(str(record["as_of"])),
                evidence={str(key): str(value) for key, value in record["evidence"].items()},
                split=_split_from(membership, claim.claim_id, claim.program_id),
                truth_outcome=RecoveryOutcome(str(answer["outcome"])),
                truth_amount=Money(
                    Decimal(str(answer["recoverable_amount"])), Currency(str(answer["currency"]))
                ),
                governing_clause_id=str(answer["governing_clause_id"]),
            )
        )
    return tuple(cases)


# ------------------------------------------------------------------------------------ the query


def case_question(claim: Claim, program: WarrantyProgram) -> str:
    """The question this case asks the policy, in one fixed form.

    Every retrieval figure in `artifacts/retrieval.json` is a figure about this text, and the four
    baselines are given the same text, so the comparison is between rankers rather than between
    phrasings. **The graph's retrieval node must call this function**; a node that composes its own
    question has a recall nobody has measured.

    Three decisions, each of which could have gone the other way.

    **The part number is in the query because the governing clause depends on it.** The current
    service bulletin amends two of the seven rejection codes for one named part family, so for those
    claims the governing authority is the bulletin clause and not the policy clause it replaces. A
    query without the part could not distinguish the two even in principle, and the recall it
    measured would be a measurement of a question that cannot be answered.

    **What the rejection demands is taken from `REQUIREMENT_MATRIX` rather than written out per
    code.** Seven hand-written sentences would be seven tuning knobs, and a knob turned after the
    hold-out has been scored is precisely what ADR-001 §7 forbids. Deriving the text from the table
    the deterministic core already uses means the query changes only when the requirements change,
    and that change is visible in a diff a reviewer reads.

    **The manufacturer's name and the claim identifier are left out.** The metadata filter has
    already fixed the manufacturer, so its name appears in every surviving candidate and can only
    add noise; the claim identifier and the serial appear in no policy text at all. A term that
    every candidate shares and a term no candidate holds are the two kinds of term that cost
    ranking quality without buying anything.

    The form was fixed against the development split and scored once against the hold-out. It has
    not been adjusted since a hold-out number existed.
    """
    demands = " ".join(spec.description for spec in required_for(claim.rejection_code))
    return (
        f"The corrected claim for part {claim.part_number} under policy version "
        f"{program.policy_version} must show: {demands}"
    )


# ------------------------------------------------------------------------------ the requirements


def _status_of(
    spec: RequirementSpec, evidence: Mapping[str, str], facts: DeterminedFacts
) -> RequirementStatus:
    """One requirement's status, from the artefacts it rests on and the findings it rests on.

    Conflict is checked before absence, which mirrors `gate.py`'s rule order and is deliberate: a
    requirement with one contradictory artefact and one missing artefact is a decision for a person
    rather than an errand for a technician, because the contradiction will still be there when the
    missing document arrives.

    A deterministic finding that says no makes the requirement `MISSING` rather than `CONFLICTING`,
    even when the document is on file. The two situations are genuinely different and the label
    follows the remedy: the covered-parts schedule does not list this part, so there is nothing for
    a person to adjudicate between — what is missing is cover, and the case is refused rather than
    argued.
    """
    statuses = {evidence[artefact] for artefact in EVIDENCE_JOIN[spec.evidence_key]}
    if RequirementStatus.CONFLICTING.value in statuses:
        return RequirementStatus.CONFLICTING
    if statuses != {PRESENT}:
        return RequirementStatus.MISSING
    determines = FACT_BACKED.get(spec.requirement_id)
    if determines is not None and not determines(facts):
        return RequirementStatus.MISSING
    return RequirementStatus.SATISFIED


def _detail(spec: RequirementSpec, status: RequirementStatus, evidence: Mapping[str, str]) -> str:
    """The sentence a technician acts on, naming the artefacts that were actually consulted.

    Never generated, and never a restatement of the requirement's own description. A detail that
    paraphrases the requirement tells the person reading it nothing they did not have; naming the
    artefact and its state tells them which document to fetch and from where.
    """
    artefacts = EVIDENCE_JOIN[spec.evidence_key]
    states = ", ".join(f"{artefact}={evidence[artefact]}" for artefact in artefacts)
    if status is RequirementStatus.SATISFIED:
        return f"evidenced by {states}"
    if status is RequirementStatus.CONFLICTING:
        return f"the evidence contradicts itself or two sources disagree: {states}"
    return f"not evidenced: {states}"


def requirements_for_case(
    case: CorpusCase, eligibility: Eligibility | None, citation: Citation | None
) -> tuple[Requirement, ...]:
    """The rejection code's requirements, with the case's evidence and findings weighed.

    `eligibility` is `None` for a claim this system refused to assess at all — a claim denominated
    in a currency the programme does not use. Its coverage findings are then unknown rather than
    false, and the fact-backed requirements are treated as unmet: a claim whose currency cannot be
    reconciled is going to a person regardless, and asserting a coverage finding that was never
    computed would put a fabricated fact into an audit record.

    `citation` is the authority the retrieval stage found. Without one, nothing can be reported as
    satisfied — `domain.Requirement` refuses to construct such a thing, which is kill condition H
    made impossible rather than measured — so every requirement whose evidence is complete is
    reported as `MISSING` instead, and the case is escalated by `assess_case` rather than refused.
    """
    facts = DeterminedFacts(
        serial_recorded=case.claim.serial_number is not None,
        serial_in_covered_range=eligibility is not None and eligibility.serial_in_range,
        part_in_schedule=eligibility is not None and eligibility.part_covered,
    )
    built: list[Requirement] = []
    for spec in required_for(case.claim.rejection_code):
        status = _status_of(spec, case.evidence, facts)
        if status is RequirementStatus.SATISFIED and citation is None:
            status = RequirementStatus.MISSING
            detail = (
                f"the evidence is on file and no clause of policy version "
                f"{case.program.policy_version} governing "
                f"{case.claim.rejection_code.value} was retrieved, so nothing can be cited for it"
            )
        else:
            detail = _detail(spec, status, case.evidence)
        built.append(
            Requirement(
                requirement_id=spec.requirement_id,
                description=spec.description,
                status=status,
                citation=citation if status is RequirementStatus.SATISFIED else None,
                detail=detail,
            )
        )
    return tuple(built)


# ------------------------------------------------------------------------------- the retrieval


Search = Callable[[CorpusCase, int], tuple[ScoredClause, ...]]
"""How a case's clauses are fetched. Injected so the assessment can be tested without a database.

The shipped implementation is `pgvector_search`, which is the system's own `Retriever` and nothing
else. A test supplies a function returning hand-built clauses, which is what lets the requirement
and escalation logic be exercised without a 285MB ONNX session and a live server.
"""


class MemoisingEncoder:
    """The shipped encoder with a dictionary in front of it, for the evaluation only.

    The evaluation embeds each case's question twice — once for the system's filtered search and
    once for the `dense_no_metadata` baseline — and the two must embed the *same* vector or the
    comparison is between a ranker and a paraphrase. Memoising guarantees they do, and it halves the
    slowest step of the run as a side effect rather than as its purpose.

    Deliberately not shipped inside `FastEmbedEncoder`. A cache in the deployed encoder would grow
    without bound in a long-running worker and would make a query's latency depend on what other
    cases happened to run first, which is exactly the property a latency figure must not have.
    """

    __slots__ = ("_cache", "_inner")

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self._cache: dict[str, list[float]] = {}

    @property
    def dimensions(self) -> int:
        dimensions: int = self._inner.dimensions
        return dimensions

    def encode_passages(self, texts: Sequence[str]) -> list[list[float]]:
        encoded: list[list[float]] = self._inner.encode_passages(texts)
        return encoded

    # The parameter is named `text` and not `text_`, although `text` is also the name of the
    # SQLAlchemy helper imported at the top of this module. The shadowing is local to this method,
    # which touches no SQL, and the name has to match: `retrieval.embeddings.TextEncoder` is a
    # protocol whose `encode_query` parameter is called `text`, and a structural type with a
    # differently named positional parameter does not satisfy it. A class that is a `TextEncoder`
    # at run time and not one to the type checker would force a cast at every call site.
    def encode_query(self, text: str) -> list[float]:
        cached = self._cache.get(text)
        if cached is None:
            cached = list(self._inner.encode_query(text))
            self._cache[text] = cached
        return list(cached)


def pgvector_search(session: Session, retriever: Retriever) -> Search:
    """The system's own retrieval stage, bound to a session, as a callable.

    A closure rather than a class so that nothing about the evaluation can reach into the retriever
    and change what it does. The only arguments it supplies are the case's own facts and `k`, which
    is what the graph's retrieval node supplies too.
    """

    def search(case: CorpusCase, k: int) -> tuple[ScoredClause, ...]:
        return retriever.retrieve(
            session,
            query=case_question(case.claim, case.program),
            program_id=case.claim.program_id,
            policy_version=case.program.policy_version,
            rejection_code=case.claim.rejection_code,
            k=k,
        ).clauses

    return search


def candidate_counts(session: Session) -> dict[tuple[str, str], int]:
    """How many clauses survive the metadata filter, per programme and policy version.

    Asked of the server in one grouped statement rather than counted from the corpus files. The
    profile in `artifacts/retrieval.json` is a statement about what the `WHERE` clause actually
    leaves for the ranker to order, and a count taken from the JSON on disk would describe the
    corpus that was *meant* to be indexed — which is the same number right up until the day a load
    was interrupted, and then it is the one number that would have explained the recall.
    """
    rows = session.execute(
        text(
            "SELECT program_id, policy_version, count(*) AS clauses FROM clause "
            "GROUP BY program_id, policy_version"
        )
    ).all()
    return {(str(row[0]), str(row[1])): int(row[2]) for row in rows}


def _authority(
    clauses: Sequence[ScoredClause], code: RejectionCode
) -> tuple[PolicyClause, int] | None:
    """The highest-ranked retrieved clause that takes authority over this rejection code.

    Not simply the top-ranked clause. `domain.PolicyClause` records that a clause governing nothing
    is context rather than authority and may never be cited as the basis of a satisfied requirement,
    and the corpus contains a withdrawn bulletin written to read almost exactly like the clause
    that replaced it. Citing the top-ranked clause would eventually cite that one, and a correct
    quotation from a withdrawn authority is still the wrong authority — the manufacturer rejects the
    resubmission a second time and the correction window has meanwhile run.

    Returns the clause and its rank, because the rank is what `artifacts/retrieval.json` reports and
    a rank recomputed by whichever grader iterates the tuple is a rank the next grader can recompute
    differently.
    """
    for scored in clauses:
        if code in scored.clause.governs:
            return scored.clause, scored.rank
    return None


# --------------------------------------------------------------------------------- the assessment


class CaseAssessment(NamedTuple):
    """What the system concluded about one case, and everything needed to grade it.

    `outcome` is `gate.decide`'s, except on the escalation path named in `escalation`, which
    withholds rather than authorises. `decision` is `None` on that path and on the currency path,
    because no gate decision was made and recording a fabricated one would put an outcome into an
    audit record that no rule produced.
    """

    case: CorpusCase
    outcome: RecoveryOutcome
    recoverable_amount: Money
    window: ClaimWindow | None
    computation: RecoveryComputation | None
    decision: GateDecision | None
    requirements: tuple[Requirement, ...]
    citations: tuple[Citation, ...]
    retrieved_clause_ids: tuple[str, ...]
    gold_rank: int | None
    authority_clause_id: str | None
    escalation: str | None

    @property
    def authorises_recovery(self) -> bool:
        return not gate.withholds_recovery(self.outcome)

    @property
    def window_closed(self) -> bool:
        return self.window is not None and not self.window.is_open


def assess_case(case: CorpusCase, search: Search, *, k: int = DEFAULT_K) -> CaseAssessment:
    """Run the deterministic path over one case and record what every stage concluded.

    Retrieval runs for every case, including the ones the money path refuses. The graph runs it
    after eligibility, so a refused case never reaches it in production and nothing is lost by that;
    here the governing clause is the thing kill condition J is graded over, and skipping the refused
    cases would quietly measure retrieval on the easy half of the corpus.

    A claim denominated in a currency the programme does not use does not reach the gate. There is
    no exchange rate this system holds, so there is no eligibility finding, no computation and no
    outcome to compute — the case goes to a person, and the recoverable amount is recorded as zero
    in the currency the claim was actually presented in rather than in the programme's, because
    restating it in the programme's currency would be the conversion this system refuses to perform.
    """
    clauses = search(case, k)
    retrieved = tuple(scored.clause.clause_id for scored in clauses)
    gold_rank = next(
        (scored.rank for scored in clauses if scored.clause.clause_id == case.governing_clause_id),
        None,
    )
    found = _authority(clauses, case.claim.rejection_code)
    citation = None if found is None else citation_for(found[0], case.program.policy_version)

    try:
        eligibility = assess(case.claim, case.program, case.coverage)
    except CurrencyMismatchError as error:
        requirements = requirements_for_case(case, None, citation)
        return CaseAssessment(
            case=case,
            outcome=RecoveryOutcome.REVIEW,
            recoverable_amount=Money.zero(case.claim.claimed_parts.currency),
            window=None,
            computation=None,
            decision=None,
            requirements=requirements,
            citations=_citations_of(requirements),
            retrieved_clause_ids=retrieved,
            gold_rank=gold_rank,
            authority_clause_id=None if found is None else found[0].clause_id,
            escalation=str(error),
        )

    window = deadline.claim_window(case.claim, case.program, case.as_of)
    computation = recovery.compute(case.claim, case.program, eligibility)
    requirements = requirements_for_case(case, eligibility, citation)

    if found is None:
        # Withholding, not deciding. No requirement can be reported satisfied without a citation,
        # so the gate would be handed a requirement list that understates the evidence on file and
        # would refuse the case outright. Refusing is worse than escalating: the evidence is there
        # and the correction is filable the moment a person supplies the authority, whereas a
        # refusal closes the claim and nobody looks at it again.
        return CaseAssessment(
            case=case,
            outcome=RecoveryOutcome.REVIEW,
            recoverable_amount=computation.recoverable_amount,
            window=window,
            computation=computation,
            decision=None,
            requirements=requirements,
            citations=(),
            retrieved_clause_ids=retrieved,
            gold_rank=gold_rank,
            authority_clause_id=None,
            escalation=(
                f"no clause of {case.program.program_id} at policy version "
                f"{case.program.policy_version} governing "
                f"{case.claim.rejection_code.value} was among the {len(retrieved)} retrieved; "
                f"nothing may be filed on an authority this system could not locate"
            ),
        )

    decision = gate.decide(case.claim, case.program, window, eligibility, computation, requirements)
    return CaseAssessment(
        case=case,
        outcome=decision.outcome,
        recoverable_amount=computation.recoverable_amount,
        window=window,
        computation=computation,
        decision=decision,
        requirements=requirements,
        citations=_citations_of(requirements),
        retrieved_clause_ids=retrieved,
        gold_rank=gold_rank,
        authority_clause_id=found[0].clause_id,
        escalation=None,
    )


def _citations_of(requirements: Sequence[Requirement]) -> tuple[Citation, ...]:
    """One citation per satisfied requirement, in requirement order.

    Kept per requirement rather than deduplicated to one per case. Kill conditions H and I are
    graded over requirements: H counts requirements reported satisfied with nothing behind them and
    I checks each cited span against the document it names. Collapsing two satisfied requirements
    that share an authority into one citation would halve I's denominator and would make H's
    question — *is every satisfied requirement cited* — unanswerable from the artifact.
    """
    return tuple(item.citation for item in requirements if item.citation is not None)
