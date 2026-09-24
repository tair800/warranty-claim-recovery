"""Materialise the hold-out membership and commit it, before any score over it exists.

    python scripts/freeze_holdout.py                  # write artifacts/holdout.json
    python scripts/freeze_holdout.py --check          # fail if the corpus would now produce another
    python scripts/freeze_holdout.py --allow-refreeze # deliberately re-draw, and invalidate scores

ADR-001 §7 fixes the rule — a programme is held out iff `blake2b(program_id) % 100 < 30` — and a
rule is not yet a hold-out. A rule plus a corpus implies a membership only while both are
unchanged, so a corpus edit could move a programme across the line and every score either side of
it would then have been measured over a different set without anybody noticing.

This writes the membership out: every programme and every claim identifier, both sides, with a
digest. Once that file is **committed**, the hold-out is a fact in git history with a timestamp,
and a reader who does not trust the author can check any later score against it — including
checking that the freeze commit precedes the first commit carrying a score.

`--check` is what CI runs on every push. It recomputes and compares, and fails on any drift.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from warranty_claim_recovery.corpus.holdout import (  # noqa: E402
    FREEZE_FILENAME,
    HOLDOUT_RULE,
    SPLIT_UNIT,
    HoldoutDriftError,
    HoldoutLeakError,
    freeze,
    load_corpus,
    load_frozen,
    membership_of,
    partition_problems,
)

DEFAULT_CORPUS = REPO_ROOT / "data" / "generated"
DEFAULT_ARTIFACTS = REPO_ROOT / "artifacts"


def _check(corpus_dir: Path, artifacts_dir: Path) -> int:
    existing = load_frozen(artifacts_dir)
    if existing is None:
        print(f"artifacts/{FREEZE_FILENAME} does not exist; nothing is frozen", file=sys.stderr)
        return 1
    view = load_corpus(corpus_dir)
    computed = membership_of(view.program_ids, view.claim_programs)
    if existing.digest != computed.digest:
        print(
            "the committed hold-out no longer matches the corpus.\n"
            f"  frozen:   {len(existing.holdout_programs)} hold-out programmes, "
            f"{len(existing.holdout_cases)} cases, digest {existing.digest[:16]}\n"
            f"  computed: {len(computed.holdout_programs)} hold-out programmes, "
            f"{len(computed.holdout_cases)} cases, digest {computed.digest[:16]}\n"
            "Every score taken against the old membership is now measuring a different set.",
            file=sys.stderr,
        )
        return 1
    report = partition_problems(
        computed,
        claim_programs=view.claim_programs,
        clause_programs=view.clause_programs,
        governing_clause=view.governing_clause,
    )
    if not report.clean:
        print("the split no longer partitions cleanly:", file=sys.stderr)
        for line in report.problems[:10]:
            print(f"  {line}", file=sys.stderr)
        return 2
    print(
        f"the hold-out is unchanged: {len(existing.holdout_programs)} programmes, "
        f"{len(existing.holdout_cases)} cases, digest {existing.digest[:16]}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write; exit non-zero if the committed freeze no longer matches the corpus",
    )
    parser.add_argument(
        "--allow-refreeze",
        action="store_true",
        help=(
            "deliberately re-draw the hold-out. This invalidates every score taken against the old "
            "membership; ADR-001 §7 requires those to be re-run under a new recorded decision "
            "rather than carried over."
        ),
    )
    args = parser.parse_args(argv)

    if args.check:
        return _check(args.corpus, args.artifacts)

    try:
        freeze(args.corpus, args.artifacts, allow_refreeze=args.allow_refreeze)
    except FileNotFoundError as error:
        print(str(error), file=sys.stderr)
        return 1
    except HoldoutLeakError as error:
        print(str(error), file=sys.stderr)
        return 2
    except HoldoutDriftError as error:
        print(str(error), file=sys.stderr)
        return 1

    payload = json.loads((args.artifacts / FREEZE_FILENAME).read_text(encoding="utf-8"))
    print(f"hold-out frozen into artifacts/{FREEZE_FILENAME}")
    print(f"  rule        {HOLDOUT_RULE}")
    print(f"  split unit  {SPLIT_UNIT}")
    print(
        f"  programmes  {payload['counts']['holdout_programs']} held out of "
        f"{payload['programs_total']}"
    )
    for program_id in payload["holdout_programs"]:
        print(f"    {program_id}")
    held_cases = payload["counts"]["holdout_cases"]
    print(f"  cases       {held_cases} held out of {payload['cases_total']}")
    print(f"  leaks       {payload['leaked_programs']} programmes, {payload['leaked_cases']} cases")
    print(f"  digest      {payload['digest']}")
    print()
    print("Commit this file before running any evaluation. Its position in git history is the")
    print("evidence that the split was not chosen after seeing a result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
