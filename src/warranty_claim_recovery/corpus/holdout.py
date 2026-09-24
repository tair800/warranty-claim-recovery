"""The hold-out rule, the partition check that can actually fail, and the freeze that has teeth.

ADR-001 §7 fixes the rule before anything was built:

    a program is held out iff blake2b(program_id, digest_size=8) % 100 < 30 (big-endian)

Four properties matter, and each of them is here because of a way the thing has gone wrong before.

**It splits by warranty programme, never by claim.** Forty claims under one programme are
adjudicated against one clause set, one rejection-code table, one deadline rule and one
covered-parts schedule. A claim-level split leaves thirty of them in development and ten in the
hold-out, and the ten then measure how well the system memorised a policy it was already tuned on.
That number is always good and always meaningless.

**It consults no seed and no score.** There is nothing to re-draw. A split taking a seed could be
re-rolled until it flattered a result, and nobody reading the number afterwards could tell that it
had been.

**The byte order is stated.** ADR-001 does not name one, so big-endian is chosen here, written into
`HOLDOUT_RULE`, shipped in the artifact, and never changed. A split that silently reverses is a new
experiment wearing the old one's numbers.

**The partition check is a real check.** `partition_problems` compares facts that were derived
separately — the declared membership against the rule, each claim's split against its programme's,
each claim's governing clause against its own split — rather than deriving one from the other and
finding them equal. Project 7's split-leak guard read a key its corpus did not use, so it iterated
an empty sequence and reported clean because it had looked at nothing; a guard that examines
nothing is indistinguishable from a guard that found nothing wrong, which is the worst property a
guard can have. `tests/test_holdout.py` plants five separate leaks and requires each to be caught,
because the only evidence that a zero means anything is a demonstration that the same code can
produce a non-zero.

The freeze is separate from the rule and is the part with consequences. `freeze` materialises the
membership into `artifacts/holdout.json` — every programme and every claim identifier, with a
digest — and refuses to overwrite an existing freeze whose content would change. Once that file is
committed, the hold-out is a fact in git history with a timestamp, and any later score can be
checked against it by a reader who does not trust the author.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, NamedTuple

__all__ = [
    "DEVELOPMENT",
    "FREEZE_FILENAME",
    "HOLDOUT",
    "HOLDOUT_RULE",
    "HOLDOUT_SHARE",
    "SPLIT_UNIT",
    "CorpusView",
    "HoldoutDriftError",
    "HoldoutLeakError",
    "Membership",
    "PartitionReport",
    "digest_of",
    "freeze",
    "is_holdout",
    "load_corpus",
    "load_frozen",
    "membership_of",
    "partition_problems",
    "split_of",
]

#: Out of a hundred. Fixed in ADR-001 before any corpus existed and never tuned: moving it after a
#: score would be re-drawing the hold-out with extra steps.
HOLDOUT_SHARE: Final = 30
_BUCKETS: Final = 100

#: The rule string, shipped verbatim in `artifacts/holdout.json`. The artifact and the
#: implementation are the same string because a rule described in one place and implemented in
#: another is a rule that will one day be described wrongly.
HOLDOUT_RULE: Final = (
    "a program is held out iff blake2b(program_id, digest_size=8) % 100 < 30 (big-endian)"
)

#: What the split partitions. Read by `tests/test_kill_criteria.py`, which asserts it is the
#: programme and not the claim.
SPLIT_UNIT: Final = "warranty_program"

HOLDOUT: Final = "holdout"
DEVELOPMENT: Final = "development"

#: The filename is part of the contract: the kill test and the evaluation both look for it here, and
#: a freeze written somewhere else is a freeze nobody checks.
FREEZE_FILENAME: Final = "holdout.json"

_NOTICE: Final = (
    "This file is synthetic evaluation data derived from a corpus generated from a committed seed. "
    "It is not a manufacturer publication and describes no real warranty programme."
)


class HoldoutDriftError(RuntimeError):
    """A freeze exists and the corpus would now produce a different one.

    This is the error that protects the whole evaluation. It fires when the corpus changed in a way
    that moves a programme or a claim across the line — which may be entirely innocent, and is never
    something to resolve by deleting the file. Every score taken against the old membership was
    measured over a different set and has to be re-run.
    """


class HoldoutLeakError(RuntimeError):
    """The split does not partition cleanly, so freezing it would freeze the leak."""


def is_holdout(program_id: str) -> bool:
    """Whether a warranty programme is in the hold-out. The rule, and nothing else."""
    digest = hashlib.blake2b(program_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % _BUCKETS < HOLDOUT_SHARE


def split_of(program_id: str) -> str:
    return HOLDOUT if is_holdout(program_id) else DEVELOPMENT


class Membership(NamedTuple):
    """What was held out, enumerated rather than described.

    A rule plus a corpus implies a membership, but only while both are unchanged. Enumerating the
    identifiers means a reader can check a later score against the actual set without regenerating
    anything, and without trusting the generator that produced the score.
    """

    holdout_programs: tuple[str, ...]
    development_programs: tuple[str, ...]
    holdout_cases: tuple[str, ...]
    development_cases: tuple[str, ...]
    digest: str


class PartitionReport(NamedTuple):
    """Every way the split could leak, looked at rather than assumed."""

    problems: tuple[str, ...]
    leaked_programs: tuple[str, ...]
    leaked_cases: tuple[str, ...]

    @property
    def clean(self) -> bool:
        return not self.problems


def digest_of(
    holdout_programs: Sequence[str],
    development_programs: Sequence[str],
    holdout_cases: Sequence[str],
    development_cases: Sequence[str],
) -> str:
    """A digest over both sides of the split, not only the hold-out side.

    Digesting the hold-out alone would be blind to a claim quietly leaving the development set: the
    hold-out would be unchanged, the digest would match, and the denominator of every development
    figure would have moved. Sorted before hashing, because the digest must not depend on the order
    the generator happened to emit the records in.
    """
    payload = json.dumps(
        {
            "holdout_programs": sorted(holdout_programs),
            "development_programs": sorted(development_programs),
            "holdout_cases": sorted(holdout_cases),
            "development_cases": sorted(development_cases),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def membership_of(program_ids: Sequence[str], claim_programs: Mapping[str, str]) -> Membership:
    """Apply the rule and enumerate both sides."""
    holdout_programs = sorted(p for p in program_ids if is_holdout(p))
    development_programs = sorted(p for p in program_ids if not is_holdout(p))
    held = set(holdout_programs)
    holdout_cases = sorted(c for c, p in claim_programs.items() if p in held)
    development_cases = sorted(c for c, p in claim_programs.items() if p not in held)
    return Membership(
        holdout_programs=tuple(holdout_programs),
        development_programs=tuple(development_programs),
        holdout_cases=tuple(holdout_cases),
        development_cases=tuple(development_cases),
        digest=digest_of(holdout_programs, development_programs, holdout_cases, development_cases),
    )


def partition_problems(
    membership: Membership,
    *,
    claim_programs: Mapping[str, str],
    clause_programs: Mapping[str, str],
    governing_clause: Mapping[str, str],
) -> PartitionReport:
    """Check that the two splits really are a partition, over programmes **and** over claims.

    Six independent failures are looked for, and none of them is derived from the thing it is
    checking:

    1. a programme listed on both sides;
    2. a programme whose listed side disagrees with the rule;
    3. a claim listed on both sides;
    4. a claim whose listed side disagrees with the side of the programme it belongs to;
    5. a claim whose governing clause belongs to a programme on the other side — the leak that
       matters most, because it is the one that puts hold-out evidence in front of a development
       case without either record looking wrong on its own;
    6. a claim, or a programme, that is enumerated nowhere.

    `membership` is taken as data rather than recomputed from the rule, which is what makes a
    planted leak possible to plant. A function that recomputed both sides from `is_holdout` and
    then compared them would be comparing a value with itself.
    """
    problems: list[str] = []
    leaked_programs: set[str] = set()
    leaked_cases: set[str] = set()

    held_programs = set(membership.holdout_programs)
    dev_programs = set(membership.development_programs)

    for program_id in sorted(held_programs & dev_programs):
        problems.append(f"programme {program_id} is listed in both splits")
        leaked_programs.add(program_id)

    for program_id in sorted(held_programs | dev_programs):
        declared = HOLDOUT if program_id in held_programs else DEVELOPMENT
        if program_id in held_programs and program_id in dev_programs:
            continue
        if declared != split_of(program_id):
            problems.append(
                f"programme {program_id} is listed as {declared} and the rule makes it "
                f"{split_of(program_id)}"
            )
            leaked_programs.add(program_id)

    held_cases = set(membership.holdout_cases)
    dev_cases = set(membership.development_cases)

    for claim_id in sorted(held_cases & dev_cases):
        problems.append(f"claim {claim_id} is listed in both splits")
        leaked_cases.add(claim_id)

    for claim_id, program_id in sorted(claim_programs.items()):
        if claim_id not in held_cases and claim_id not in dev_cases:
            problems.append(f"claim {claim_id} is enumerated in neither split")
            leaked_cases.add(claim_id)
            continue
        declared = HOLDOUT if claim_id in held_cases else DEVELOPMENT
        if program_id not in held_programs and program_id not in dev_programs:
            problems.append(
                f"claim {claim_id} belongs to programme {program_id}, which is enumerated in "
                f"neither split"
            )
            leaked_cases.add(claim_id)
            leaked_programs.add(program_id)
            continue
        owning = HOLDOUT if program_id in held_programs else DEVELOPMENT
        if declared != owning:
            problems.append(
                f"claim {claim_id} is listed as {declared} and its programme {program_id} is "
                f"{owning}"
            )
            leaked_cases.add(claim_id)
            leaked_programs.add(program_id)

        clause_id = governing_clause.get(claim_id)
        if clause_id is None:
            problems.append(f"claim {claim_id} names no governing clause")
            leaked_cases.add(claim_id)
            continue
        clause_program = clause_programs.get(clause_id)
        if clause_program is None:
            problems.append(f"claim {claim_id} cites unknown clause {clause_id}")
            leaked_cases.add(claim_id)
            continue
        clause_side = HOLDOUT if clause_program in held_programs else DEVELOPMENT
        if clause_side != declared:
            problems.append(
                f"claim {claim_id} is {declared} and its governing clause {clause_id} belongs to "
                f"{clause_program}, which is {clause_side}"
            )
            leaked_cases.add(claim_id)
            leaked_programs.add(clause_program)

    return PartitionReport(
        problems=tuple(problems),
        leaked_programs=tuple(sorted(leaked_programs)),
        leaked_cases=tuple(sorted(leaked_cases)),
    )


class CorpusView(NamedTuple):
    """The three facts the freeze needs, read off the generated files rather than rebuilt.

    Read rather than regenerated on purpose. Freezing against a fresh in-memory build would freeze
    the membership of a corpus that may not be the one on disk, and the one on disk is the one every
    later stage loads.
    """

    program_ids: tuple[str, ...]
    claim_programs: dict[str, str]
    clause_programs: dict[str, str]
    governing_clause: dict[str, str]


def _read(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing. The corpus is rebuilt from a committed seed rather than kept in "
            f"git; run `python scripts/generate_corpus.py` first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_corpus(corpus_dir: Path) -> CorpusView:
    programs = _read(corpus_dir / "programs.json")["programs"]
    claims = _read(corpus_dir / "claims.json")["claims"]
    clauses = _read(corpus_dir / "clauses.json")["clauses"]
    truth = _read(corpus_dir / "truth.json")["truth"]

    return CorpusView(
        program_ids=tuple(str(record["program"]["program_id"]) for record in programs),
        claim_programs={
            str(record["claim"]["claim_id"]): str(record["claim"]["program_id"])
            for record in claims
        },
        clause_programs={
            str(record["clause"]["clause_id"]): str(record["clause"]["program_id"])
            for record in clauses
        },
        governing_clause={
            str(claim_id): str(entry["governing_clause_id"]) for claim_id, entry in truth.items()
        },
    )


def _artifact(membership: Membership, report: PartitionReport, *, matches: bool) -> dict[str, Any]:
    return {
        "is_synthetic": True,
        "notice": _NOTICE,
        "what_this_file_is": (
            "the materialised membership of the hold-out, written and committed before any score "
            "over it existed. Its position in git history is the evidence that the split was not "
            "chosen after seeing a result."
        ),
        "rule": HOLDOUT_RULE,
        "share_percent": HOLDOUT_SHARE,
        "split_unit": SPLIT_UNIT,
        "programs_total": len(membership.holdout_programs) + len(membership.development_programs),
        "cases_total": len(membership.holdout_cases) + len(membership.development_cases),
        "holdout_programs": list(membership.holdout_programs),
        "development_programs": list(membership.development_programs),
        "holdout_cases": list(membership.holdout_cases),
        "development_cases": list(membership.development_cases),
        "counts": {
            "holdout_programs": len(membership.holdout_programs),
            "development_programs": len(membership.development_programs),
            "holdout_cases": len(membership.holdout_cases),
            "development_cases": len(membership.development_cases),
        },
        "leaked_programs": len(report.leaked_programs),
        "leaked_cases": len(report.leaked_cases),
        "leaked_program_ids": list(report.leaked_programs),
        "leaked_case_ids": list(report.leaked_cases),
        "partition_checks": [
            "a programme listed in both splits",
            "a programme whose listed split disagrees with the rule",
            "a claim listed in both splits",
            "a claim whose split disagrees with its programme's",
            "a claim whose governing clause belongs to a programme in the other split",
            "a claim or programme enumerated in neither split",
        ],
        "digest": membership.digest,
        "digest_matches_corpus": matches,
        "digest_covers": (
            "both splits, programmes and claims. A digest over the hold-out alone would be blind "
            "to a claim leaving the development set."
        ),
    }


def load_frozen(artifacts_dir: Path) -> Membership | None:
    path = artifacts_dir / FREEZE_FILENAME
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return Membership(
        holdout_programs=tuple(payload["holdout_programs"]),
        development_programs=tuple(payload["development_programs"]),
        holdout_cases=tuple(payload["holdout_cases"]),
        development_cases=tuple(payload["development_cases"]),
        digest=payload["digest"],
    )


def freeze(corpus_dir: Path, artifacts_dir: Path, *, allow_refreeze: bool = False) -> None:
    """Materialise the membership into `artifacts/holdout.json`, or refuse to.

    On a first run this writes the file. On every later run it recomputes from the corpus on disk
    and compares, and raises `HoldoutDriftError` if the membership moved — because a hold-out that
    silently follows the corpus is not frozen, and the number measured over it means whatever the
    last edit decided.

    `digest_matches_corpus` in the written artifact is the result of reading the corpus directory a
    second time, independently, and recomputing. On a first freeze it can only be true, and the
    field is honest about what it is: the mechanism with teeth is `HoldoutDriftError`, not the
    boolean.

    `allow_refreeze` exists for a corpus that legitimately grew. It is not a way to make an
    inconvenient drift go away: re-freezing invalidates every score taken against the old
    membership, and ADR-001 §7 requires those to be re-run under a new recorded decision rather
    than carried over.
    """
    view = load_corpus(corpus_dir)
    membership = membership_of(view.program_ids, view.claim_programs)
    report = partition_problems(
        membership,
        claim_programs=view.claim_programs,
        clause_programs=view.clause_programs,
        governing_clause=view.governing_clause,
    )
    if not report.clean:
        raise HoldoutLeakError(
            "the split does not partition cleanly, so freezing it would freeze a leak:\n  "
            + "\n  ".join(report.problems[:10])
        )

    existing = load_frozen(artifacts_dir)
    if existing is not None and existing.digest != membership.digest and not allow_refreeze:
        raise HoldoutDriftError(
            f"artifacts/{FREEZE_FILENAME} already holds a different membership.\n"
            f"  frozen:   {len(existing.holdout_programs)} hold-out programmes, "
            f"{len(existing.holdout_cases)} cases, digest {existing.digest[:16]}\n"
            f"  computed: {len(membership.holdout_programs)} hold-out programmes, "
            f"{len(membership.holdout_cases)} cases, digest {membership.digest[:16]}\n"
            "The corpus moved a programme or a claim across the line. That may be entirely "
            "innocent, and it still invalidates every score taken against the old membership. "
            "Re-run them, or pass allow_refreeze and record the decision in DECISIONS.md."
        )

    # The second, independent read. If the files on disk changed between the two reads, or if the
    # membership depends on anything other than the corpus, the digests disagree and the artifact
    # says so rather than asserting a match nobody checked.
    recomputed = membership_of(*_reread(corpus_dir))
    matches = recomputed.digest == membership.digest

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / FREEZE_FILENAME).write_text(
        json.dumps(_artifact(membership, report, matches=matches), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _reread(corpus_dir: Path) -> tuple[tuple[str, ...], dict[str, str]]:
    view = load_corpus(corpus_dir)
    return view.program_ids, view.claim_programs
