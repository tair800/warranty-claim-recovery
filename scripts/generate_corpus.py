"""Build the synthetic corpus from its committed seed and write it, or refuse to.

    python scripts/generate_corpus.py                 # write the corpus and artifacts/corpus.json
    python scripts/generate_corpus.py --determinism   # build twice and compare every byte

The corpus is **not** kept in git. It is rebuilt from `corpus.rng.CORPUS_SEED`, which is, and a
seed plus a generator is a smaller thing to review than three megabytes of generated JSON — and the
only one of the two a reviewer can actually check. Regenerating must therefore be free of
surprises, which is what `--determinism` exists to prove: it builds the corpus twice in one process
and compares the serialised bytes of every file. A clock, a hostname, a path or a set iteration
deciding an order would all show up there and nowhere else.

The generator refuses to write a corpus that misses a contract floor. That refusal is the point:
a corpus one claim short of a floor is one the kill test would grade anyway, and it would grade it
over a population ADR-001 said was too small to mean anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from warranty_claim_recovery.corpus.generate import (  # noqa: E402
    FLOORS,
    CorpusContractError,
    build,
    generate_corpus,
)
from warranty_claim_recovery.corpus.rng import CORPUS_SEED, GENERATOR_VERSION  # noqa: E402

DEFAULT_CORPUS = REPO_ROOT / "data" / "generated"
DEFAULT_ARTIFACTS = REPO_ROOT / "artifacts"


def _serialise(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False)


def _shown(path: Path) -> str:
    """A path for the console, relative to the repository where it can be.

    `Path.relative_to` raises for a directory outside the repository, and `--out` is often exactly
    that — a scratch directory, when somebody is comparing two runs. Reporting where the files went
    must not be the thing that fails the run.
    """
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _determinism() -> int:
    first = build()
    second = build()
    differing: list[str] = []
    for (name, left), (_, right) in zip(first.files, second.files, strict=True):
        if _serialise(left) != _serialise(right):
            differing.append(name)
    if _serialise(first.artifact) != _serialise(second.artifact):
        differing.append("artifacts/corpus.json")
    if differing:
        print("two builds of the same seed disagree:", ", ".join(differing), file=sys.stderr)
        return 1
    total = sum(len(_serialise(payload)) for _, payload in first.files)
    print(f"two builds are byte-identical across {len(first.files)} files, {total:,} characters")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument(
        "--determinism",
        action="store_true",
        help="build twice in one process and compare every byte; write nothing",
    )
    args = parser.parse_args(argv)

    if args.determinism:
        return _determinism()

    try:
        generated = generate_corpus(args.out, artifact_path=args.artifacts / "corpus.json")
    except CorpusContractError as error:
        print(
            f"the corpus does not meet its contract, so nothing was written:\n  {error}",
            file=sys.stderr,
        )
        return 1

    artifact = generated.artifact
    counts = artifact["counts"]
    observed = artifact["observed"]

    print(f"corpus written to {_shown(args.out)}")
    print(f"  seed              {CORPUS_SEED}")
    print(f"  generator         {GENERATOR_VERSION}")
    print(
        f"  programmes {counts['programs']}   documents {counts['documents']}   "
        f"clauses {counts['clauses']}   claims {counts['claims']}   "
        f"coverage rows {counts['coverage_rows']}"
    )
    print(f"  clause offsets verified   {artifact['clause_offsets_verified']}")
    print(
        "  outcomes                  "
        + ", ".join(f"{name} {count}" for name, count in artifact["claims_by_outcome"].items())
    )
    print(
        "  rejection codes           "
        + ", ".join(
            f"{name} {count}" for name, count in artifact["claims_by_rejection_code"].items()
        )
    )
    print("  floors")
    for name, floor in FLOORS:
        print(f"    {name:<32} required {floor:>4}   observed {observed[name]:>4}")
    print(
        f"  hold-out programmes       {len(artifact['holdout_programs'])} "
        f"({', '.join(artifact['holdout_currencies'])})"
    )
    for program_id in artifact["holdout_programs"]:
        print(f"    {program_id}")
    print()
    print("Freeze the hold-out next: python scripts/freeze_holdout.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
