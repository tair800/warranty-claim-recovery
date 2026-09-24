"""The synthetic corpus, its ground truth and the frozen hold-out.

ADR-001 §1 records the blueprint's own risk about this project — that the warranty domain is less
familiar to the builder than the other domains in the slate, and that fixture realism would depend
on research nobody budgeted — and records how the risk is discharged: **the corpus is synthetic,
generated from a committed seed, and says so in the body of every file it writes.** No claim of
realism is made that a reader could check and find false. These policy texts do not resemble any
manufacturer's, the rejection codes are not an industry standard, and no recovery rate measured
here means anything outside this corpus.

What the corpus does have to be is *correct about itself*, and that is what these modules enforce:

- `rng`       every draw seeded by what it decides, so inserting a programme moves nothing else;
- `programs`  eighteen warranty programmes across six manufacturers and three currencies;
- `clauses`   the policy documents, and the clause offsets that make a citation checkable;
- `claims`    the claims, their evidence, and the ground truth as construction metadata;
- `holdout`   the split rule, the partition check that can fail, and the freeze;
- `generate`  assembly, the contract floors, and a writer that refuses a corpus that misses them.

Two runs of the generator produce byte-identical files. There is no timestamp, no hostname, no path
and no wall-clock reading anywhere in the output, iteration order is fixed by written sequences
rather than by set or dictionary hashing, and newlines are pinned so the bytes do not depend on the
operating system.
"""

from __future__ import annotations

__all__: list[str] = []
