"""Seeded randomness, keyed by what the stream is *for* rather than by how many draws preceded it.

Every random choice in this corpus comes from a generator seeded by a hash of the thing being
decided — the programme, the part, the claim slot — and never from a counter, never from a shared
generator threaded through the build, and never from the module-level `random` functions.

The difference shows up the first time the corpus changes. With one shared counter-ordered stream,
inserting a thirteenth warranty programme shifts every draw after it: every later programme's cap,
every later claim's invoice amount and every later serial number moves, the diff is unreadable, and
no figure measured against the old corpus is comparable with one measured against the new. Keying
by identity means the inserted programme draws its own values and nothing else moves, so a corpus
change can be reviewed as a change rather than as a regeneration.

`random.Random` is the Mersenne Twister, fixed by the language rather than by the platform, so the
same seed yields the same stream on any machine running the same CPython. That is the property the
byte-identical requirement needs. Cryptographic quality is irrelevant here and would be slower; an
unpredictable stream would be a defect, not a feature.

Rejected: seeding from `CORPUS_SEED` alone and drawing sequentially. It is shorter and it is what
broke in two earlier corpora in this portfolio. Also rejected: `hash()` as the key function, because
`hash()` of a `str` is salted per process and two runs would not agree.
"""

from __future__ import annotations

import hashlib
import random
from typing import Final

__all__ = ["CORPUS_SEED", "GENERATOR_VERSION", "derive_seed", "stream"]

#: The committed constant every stream descends from. Changing it regenerates the whole corpus and
#: invalidates every measurement taken against the old one, so it is written here and nowhere else.
CORPUS_SEED: Final = "warranty-claim-recovery/ADR-001/corpus/v1"

#: Bumped whenever the *shape* of the output changes — a new field, another clause section, a
#: different claim recipe. Recorded in every generated file so a stored number can be traced to the
#: corpus that produced it rather than to whatever the generator happens to emit today.
GENERATOR_VERSION: Final = "corpus-1.0.0"

#: A separator that cannot occur in any identifier used here, so ("ab", "c") and ("a", "bc") cannot
#: hash to the same key. Joining on "-" would collide, and a collision between two streams is a
#: silent duplicate value in the corpus — two programmes with the same cap, two claims with the same
#: invoice, and an evaluation that quietly grades one case twice.
_SEPARATOR: Final = "\x1f"


def derive_seed(*purpose: str) -> int:
    """A 64-bit seed from the purpose of the stream.

    blake2b rather than `hash()` because the built-in string hash is randomised per process, and a
    corpus whose values depend on the interpreter's start-up salt cannot be byte-identical across
    two runs — which is the one property this generator exists to guarantee.
    """
    key = _SEPARATOR.join((CORPUS_SEED, *purpose)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")


def stream(*purpose: str) -> random.Random:
    """An owned generator for one decision.

    Callers pass what the decision is about, most specific last, e.g.
    ``stream("claim-amounts", program_id, claim_id)``.
    """
    # S311: bandit flags the Mersenne Twister as unsuitable for cryptography, which is exactly the
    # point — a reproducible stream is required here and an unpredictable one would be the defect.
    return random.Random(derive_seed(*purpose))  # noqa: S311
