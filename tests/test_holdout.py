"""The hold-out rule, and — more importantly — proof that its leak check can fail.

A zero in an artifact means nothing on its own. Project 7's split-leak guard read a key its corpus
did not use, iterated an empty sequence, and reported clean because it had looked at nothing; the
kill test graded that zero and the pass was worthless. So the substantial half of this file plants a
leak of each kind the check claims to catch and requires each one to be caught. `leaked_programs: 0`
in `artifacts/holdout.json` is only evidence because the same function, on data one field different,
returns a non-zero.

The five plants are chosen to be independent failures rather than five spellings of one:

1. a programme listed on both sides;
2. a programme whose listed side contradicts the rule;
3. a claim listed on the side its programme is not on;
4. a claim listed on both sides;
5. a claim whose governing clause belongs to a programme on the other side — the one that matters
   most, because neither record looks wrong on its own and hold-out evidence reaches a development
   case without anything in either file being individually false.

`freeze` is then exercised against real files: it writes, it refuses to overwrite a changed
membership, it accepts a deliberate re-draw, and it refuses outright to freeze a corpus that leaks.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from warranty_claim_recovery.corpus.holdout import (
    DEVELOPMENT,
    FREEZE_FILENAME,
    HOLDOUT,
    HOLDOUT_RULE,
    HOLDOUT_SHARE,
    SPLIT_UNIT,
    HoldoutDriftError,
    HoldoutLeakError,
    Membership,
    digest_of,
    freeze,
    is_holdout,
    load_corpus,
    load_frozen,
    membership_of,
    partition_problems,
    split_of,
)

JsonDict = dict[str, Any]

#: Identifiers the rule puts on each side. Written out rather than searched for at run time, so that
#: a change to the rule fails this file loudly instead of quietly re-selecting fixtures to suit it.
HELD = ("prog-00", "prog-08", "prog-16")
DEV = ("prog-01", "prog-02", "prog-03")


# ------------------------------------------------------------------------------------- the rule


def test_the_rule_string_states_what_the_code_does() -> None:
    assert "blake2b" in HOLDOUT_RULE
    assert "program_id" in HOLDOUT_RULE
    assert str(HOLDOUT_SHARE) in HOLDOUT_RULE
    assert "big-endian" in HOLDOUT_RULE
    assert SPLIT_UNIT == "warranty_program"


def test_the_rule_is_the_hash_it_claims_to_be() -> None:
    """Recomputed by hand, including the byte order.

    ADR-001 does not name a byte order, so the choice is the implementation's, and a silent
    reversal would be a new experiment wearing the old one's numbers.
    """
    for program_id in (*HELD, *DEV, "kestrel-hydraulics-2019.1", ""):
        digest = hashlib.blake2b(program_id.encode("utf-8"), digest_size=8).digest()
        expected = int.from_bytes(digest, "big") % 100 < HOLDOUT_SHARE
        assert is_holdout(program_id) is expected
        assert split_of(program_id) == (HOLDOUT if expected else DEVELOPMENT)


def test_the_rule_consults_nothing_but_the_identifier() -> None:
    """Called twice, called in a different order, same answer. A rule that took a seed could be
    re-rolled until it flattered a result and nobody reading the number could tell."""
    first = [is_holdout(p) for p in (*HELD, *DEV)]
    second = [is_holdout(p) for p in reversed((*HELD, *DEV))]
    assert first == list(reversed(second))


def test_the_written_out_fixtures_fall_where_this_file_says_they_do() -> None:
    assert all(is_holdout(p) for p in HELD)
    assert not any(is_holdout(p) for p in DEV)


# ------------------------------------------------------------------------------- the membership


def _claim_programs() -> dict[str, str]:
    return {f"{program}-c{index}": program for program in (*HELD, *DEV) for index in (1, 2)}


def _clause_programs() -> dict[str, str]:
    return {f"{program}-clause": program for program in (*HELD, *DEV)}


def _governing() -> dict[str, str]:
    return {claim: f"{program}-clause" for claim, program in _claim_programs().items()}


def _membership() -> Membership:
    return membership_of((*HELD, *DEV), _claim_programs())


def test_the_membership_partitions_both_programmes_and_claims() -> None:
    membership = _membership()
    assert set(membership.holdout_programs) == set(HELD)
    assert set(membership.development_programs) == set(DEV)
    assert not set(membership.holdout_cases) & set(membership.development_cases)
    assert len(membership.holdout_cases) + len(membership.development_cases) == len(
        _claim_programs()
    )


def test_the_digest_does_not_depend_on_the_order_the_generator_emitted_records_in() -> None:
    forward = digest_of(HELD, DEV, ("a", "b"), ("c",))
    backward = digest_of(tuple(reversed(HELD)), tuple(reversed(DEV)), ("b", "a"), ("c",))
    assert forward == backward


def test_the_digest_covers_the_development_side_too() -> None:
    """A digest over the hold-out alone would be blind to a claim leaving the development set: the
    hold-out would be unchanged, the digest would match, and every development denominator would
    have
    moved."""
    assert digest_of(HELD, DEV, ("a",), ("c",)) != digest_of(HELD, DEV, ("a",), ())


def test_the_real_corpus_membership_partitions_cleanly() -> None:
    report = partition_problems(
        _membership(),
        claim_programs=_claim_programs(),
        clause_programs=_clause_programs(),
        governing_clause=_governing(),
    )
    assert report.clean, report.problems
    assert report.leaked_programs == ()
    assert report.leaked_cases == ()


# ------------------------------------------------------------------------------- planted leaks


def test_a_programme_listed_on_both_sides_is_caught() -> None:
    honest = _membership()
    planted = honest._replace(development_programs=(*honest.development_programs, HELD[0]))
    report = partition_problems(
        planted,
        claim_programs=_claim_programs(),
        clause_programs=_clause_programs(),
        governing_clause=_governing(),
    )
    assert not report.clean
    assert HELD[0] in report.leaked_programs
    assert any("both splits" in line for line in report.problems)


def test_a_programme_whose_listed_side_contradicts_the_rule_is_caught() -> None:
    honest = _membership()
    planted = honest._replace(
        holdout_programs=tuple(p for p in honest.holdout_programs if p != HELD[0]),
        development_programs=(*honest.development_programs, HELD[0]),
    )
    report = partition_problems(
        planted,
        claim_programs=_claim_programs(),
        clause_programs=_clause_programs(),
        governing_clause=_governing(),
    )
    assert not report.clean
    assert HELD[0] in report.leaked_programs
    assert any("the rule makes it" in line for line in report.problems)


def test_a_claim_on_the_wrong_side_of_its_own_programme_is_caught() -> None:
    honest = _membership()
    moved = honest.holdout_cases[0]
    planted = honest._replace(
        holdout_cases=tuple(c for c in honest.holdout_cases if c != moved),
        development_cases=(*honest.development_cases, moved),
    )
    report = partition_problems(
        planted,
        claim_programs=_claim_programs(),
        clause_programs=_clause_programs(),
        governing_clause=_governing(),
    )
    assert not report.clean
    assert moved in report.leaked_cases


def test_a_claim_listed_on_both_sides_is_caught() -> None:
    honest = _membership()
    duplicated = honest.holdout_cases[0]
    planted = honest._replace(development_cases=(*honest.development_cases, duplicated))
    report = partition_problems(
        planted,
        claim_programs=_claim_programs(),
        clause_programs=_clause_programs(),
        governing_clause=_governing(),
    )
    assert not report.clean
    assert duplicated in report.leaked_cases


def test_a_claim_citing_a_clause_from_the_other_split_is_caught() -> None:
    """The leak that matters.

    Neither record is wrong on its own: the claim is in the right split and the clause is in
    the right programme. What is wrong is the edge between them.
    """
    honest = _membership()
    victim = honest.development_cases[0]
    governing = _governing()
    governing[victim] = f"{HELD[0]}-clause"
    report = partition_problems(
        honest,
        claim_programs=_claim_programs(),
        clause_programs=_clause_programs(),
        governing_clause=governing,
    )
    assert not report.clean
    assert victim in report.leaked_cases
    assert HELD[0] in report.leaked_programs
    assert any("governing clause" in line for line in report.problems)


def test_a_claim_enumerated_nowhere_is_caught() -> None:
    honest = _membership()
    claims = _claim_programs()
    claims["orphan-claim"] = DEV[0]
    report = partition_problems(
        honest,
        claim_programs=claims,
        clause_programs=_clause_programs(),
        governing_clause=_governing(),
    )
    assert not report.clean
    assert "orphan-claim" in report.leaked_cases


def test_a_claim_citing_an_unknown_clause_is_caught() -> None:
    governing = _governing()
    victim = next(iter(governing))
    governing[victim] = "no-such-clause"
    report = partition_problems(
        _membership(),
        claim_programs=_claim_programs(),
        clause_programs=_clause_programs(),
        governing_clause=governing,
    )
    assert not report.clean
    assert victim in report.leaked_cases


# ------------------------------------------------------------------------------- the freeze


def _write_corpus(directory: Path, *, governing: dict[str, str] | None = None) -> None:
    """A minimal corpus on disk, in the shape `load_corpus` reads.

    Deliberately not the real corpus. The freeze's behaviour under drift and under a leak has to be
    exercised with a corpus that can be changed one field at a time, and regenerating seven hundred
    claims to move one identifier would make the test slow and the cause of a failure harder to see.
    """
    governing = governing or _governing()
    directory.mkdir(parents=True, exist_ok=True)
    payloads: dict[str, JsonDict] = {
        "programs.json": {"programs": [{"program": {"program_id": p}} for p in (*HELD, *DEV)]},
        "claims.json": {
            "claims": [
                {"claim": {"claim_id": claim, "program_id": program}}
                for claim, program in sorted(_claim_programs().items())
            ]
        },
        "clauses.json": {
            "clauses": [
                {"clause": {"clause_id": clause, "program_id": program}}
                for clause, program in sorted(_clause_programs().items())
            ]
        },
        "truth.json": {
            "truth": {
                claim: {"governing_clause_id": clause}
                for claim, clause in sorted(governing.items())
            }
        },
    }
    for name, payload in payloads.items():
        (directory / name).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_freeze_writes_every_key_the_kill_test_reads(tmp_path: Path) -> None:
    corpus, artifacts = tmp_path / "corpus", tmp_path / "artifacts"
    _write_corpus(corpus)
    freeze(corpus, artifacts)

    payload = json.loads((artifacts / FREEZE_FILENAME).read_text(encoding="utf-8"))
    for key in (
        "programs_total",
        "split_unit",
        "leaked_programs",
        "leaked_cases",
        "digest_matches_corpus",
        "holdout_programs",
        "development_programs",
        "rule",
        "digest",
    ):
        assert key in payload, key
    assert payload["split_unit"] == "warranty_program"
    assert payload["programs_total"] == len(HELD) + len(DEV)
    assert payload["leaked_programs"] == 0
    assert payload["leaked_cases"] == 0
    assert payload["digest_matches_corpus"] is True
    assert payload["rule"] == HOLDOUT_RULE
    assert payload["is_synthetic"] is True
    assert sorted(payload["holdout_programs"]) == sorted(HELD)
    assert sorted(payload["development_programs"]) == sorted(DEV)


def test_refreezing_an_unchanged_corpus_is_a_no_op(tmp_path: Path) -> None:
    corpus, artifacts = tmp_path / "corpus", tmp_path / "artifacts"
    _write_corpus(corpus)
    freeze(corpus, artifacts)
    before = (artifacts / FREEZE_FILENAME).read_bytes()
    freeze(corpus, artifacts)
    assert (artifacts / FREEZE_FILENAME).read_bytes() == before


def test_a_corpus_that_moved_a_claim_across_the_line_raises_rather_than_following_it(
    tmp_path: Path,
) -> None:
    corpus, artifacts = tmp_path / "corpus", tmp_path / "artifacts"
    _write_corpus(corpus)
    freeze(corpus, artifacts)

    claims = json.loads((corpus / "claims.json").read_text(encoding="utf-8"))
    claims["claims"].append({"claim": {"claim_id": "late-arrival", "program_id": HELD[0]}})
    (corpus / "claims.json").write_text(json.dumps(claims, indent=2) + "\n", encoding="utf-8")
    truth = json.loads((corpus / "truth.json").read_text(encoding="utf-8"))
    truth["truth"]["late-arrival"] = {"governing_clause_id": f"{HELD[0]}-clause"}
    (corpus / "truth.json").write_text(json.dumps(truth, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(HoldoutDriftError):
        freeze(corpus, artifacts)

    # And the escape hatch exists, deliberately, and is not silent about what it costs.
    freeze(corpus, artifacts, allow_refreeze=True)
    frozen = load_frozen(artifacts)
    assert frozen is not None
    assert "late-arrival" in frozen.holdout_cases


def test_freezing_a_leaking_corpus_is_refused(tmp_path: Path) -> None:
    """The freeze must not materialise a leak.

    A committed `holdout.json` is the evidence every later score rests on, and one written
    over a corpus that leaks would make the leak permanent.
    """
    corpus, artifacts = tmp_path / "corpus", tmp_path / "artifacts"
    governing = _governing()
    victim = next(claim for claim, program in _claim_programs().items() if program in DEV)
    governing[victim] = f"{HELD[0]}-clause"
    _write_corpus(corpus, governing=governing)

    with pytest.raises(HoldoutLeakError) as raised:
        freeze(corpus, artifacts)
    assert victim in str(raised.value)
    assert not (artifacts / FREEZE_FILENAME).exists()


def test_load_corpus_says_what_is_missing_rather_than_raising_a_key_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as raised:
        load_corpus(tmp_path / "nothing")
    assert "generate_corpus" in str(raised.value)


def test_load_frozen_returns_none_when_nothing_is_frozen(tmp_path: Path) -> None:
    assert load_frozen(tmp_path) is None
