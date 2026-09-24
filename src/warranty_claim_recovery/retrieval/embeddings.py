"""The embedding policy, as one importable object, because the two ends of it are written months
apart and must agree exactly.

An embedding is only a coordinate in a space, and a space is defined by four things: the model, the
number of dimensions, whether the vectors are normalised, and what text was actually fed to the
model. Get any of the four different between the side that writes vectors and the side that reads
them and the system does not fail — it returns results. They are simply the wrong results, ranked
confidently, and nothing in a test suite that checks shapes and types will notice.

The three ways that happens here, each of which this module exists to make impossible:

1. **The loader and the retriever disagree about the instruction prefix.** BGE models are trained
   for asymmetric retrieval with an instruction prepended to the *query* and nothing prepended to
   the *passage*. Prepend it to both and every stored vector is shifted by the same constant
   direction; the ranking degrades quietly and uniformly, which is the hardest kind of degradation
   to attribute. Prepend it to neither and short queries match badly against long passages. Neither
   mistake raises. `EMBEDDING_POLICY.for_query` and `for_passage` are the only two places text is
   prepared, and both ends call them.
2. **The model is swapped and the vectors are not rebuilt.** A rerun of the seeding script that
   keys idempotency on the clause text alone would report every clause unchanged and leave the old
   model's vectors in the table, next to a retriever asking questions in a new space. That is why
   `fingerprint` exists and why `loader.content_fingerprint` folds it in: a model change changes
   every clause's key, so the rerun re-embeds everything rather than congratulating itself.
3. **The column width and the model's output width diverge.** `store.schema` builds its `Vector`
   column from `EMBEDDING_POLICY.dimensions` rather than from a literal, so a model with a different
   output size is a migration that fails at build time rather than a `DataError` raised once per row
   after the embedding cost has already been paid.

**These values are fixed now, before any retrieval score exists.** Changing the instruction or the
model after the hold-out has been scored would be tuning against the hold-out, which ADR-001 §7
forbids; it would require a new hold-out under a new recorded decision, not an edit here.

Rejected: calling `fastembed`'s own `query_embed`. It reads as the right method and, for this model
in this version of the library, it applies no instruction at all — it simply forwards to `embed`.
Depending on it would mean the query policy lives in a library's dispatch table rather than in this
repository, and a library upgrade that started applying an instruction would silently change the
query space without a diff anyone here could see.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import blake2b
from typing import Any, Final, Protocol

from warranty_claim_recovery.config import get_settings

__all__ = [
    "EMBEDDING_POLICY",
    "EmbeddingPolicy",
    "FastEmbedEncoder",
    "TextEncoder",
]


@dataclass(frozen=True, slots=True)
class EmbeddingPolicy:
    """Everything that defines the vector space, and the two functions that prepare text for it."""

    model_name: str
    dimensions: int
    #: The encoder returns unit vectors. Measured rather than assumed — `tests/test_retrieval.py`
    #: asserts the norm — because the index's operator class is chosen on the strength of it.
    normalised: bool
    #: Prepended to a query and to nothing else. See failure 1 in the module docstring.
    query_instruction: str
    #: Empty, and named anyway. A policy with a silent asymmetry is a policy someone symmetrises by
    #: accident; naming both halves makes the asymmetry the visible, deliberate thing it is.
    passage_instruction: str

    def for_query(self, text: str) -> str:
        return f"{self.query_instruction}{text}"

    def for_passage(self, text: str) -> str:
        return f"{self.passage_instruction}{text}"

    @property
    def fingerprint(self) -> str:
        """A short digest of the whole policy, for use inside an idempotency key.

        The fields are joined with a unit separator that cannot occur in any of them. Plain
        concatenation would make two different policies collide whenever one field's tail and the
        next field's head could be rearranged — the classic hash-concatenation defect — and a
        collision here means a stale vector that a rerun reports as unchanged.
        """
        digest = blake2b(digest_size=8)
        for field in (
            self.model_name,
            str(self.dimensions),
            str(self.normalised),
            self.query_instruction,
            self.passage_instruction,
        ):
            digest.update(field.encode("utf-8"))
            digest.update(b"\x1f")
        return digest.hexdigest()


#: English-only, 384 dimensions, 285MB resident at peak — the measurement is in
#: `artifacts/encoder_memory.json` and it is why this deployment can embed a free-text query live
#: instead of serving precomputed query vectors. The instruction is the one on the model card for
#: asymmetric passage retrieval, reproduced verbatim including the trailing space.
EMBEDDING_POLICY: Final = EmbeddingPolicy(
    model_name="BAAI/bge-small-en-v1.5",
    dimensions=384,
    normalised=True,
    query_instruction="Represent this sentence for searching relevant passages: ",
    passage_instruction="",
)


class TextEncoder(Protocol):
    """What the retriever and the loader need from an encoder, and nothing more.

    A protocol rather than a base class so that a test can supply a deterministic encoder without
    inheriting from anything, and so that the type of `Retriever.__init__`'s parameter is a
    description of a capability rather than a dependency on `fastembed`. The tests that check the
    *SQL* — that the metadata filter runs before the ranking, that the statement carries the
    distance operator — are about the database and not about the model, and forcing them to load a
    285MB ONNX session to ask a question about a `WHERE` clause would make them slow enough to be
    run less often, which is how a fast-running guarantee stops being checked.

    The two methods are separate rather than one method with a flag because the asymmetry is the
    point: a caller that has to pass `is_query=False` is a caller that can pass it wrongly.
    """

    @property
    def dimensions(self) -> int: ...

    def encode_passages(self, texts: Sequence[str]) -> list[list[float]]: ...

    def encode_query(self, text: str) -> list[float]: ...


class FastEmbedEncoder:
    """The shipped encoder: a local ONNX session, with no network call at query time.

    The model is loaded on first use and not in `__init__`. That is deliberate and it is not
    micro-optimisation: `Retriever()` is constructed by the console's start-up, by several scripts
    and by tests that never embed anything, and a constructor that pulls 285MB into residence makes
    all of them pay for a capability they may not exercise. It also means a machine with no model in
    its cache fails at the first embedding call, where the traceback names the model and the cache
    directory, rather than at import time where it names neither.

    Not thread-safe by construction, and it does not need to be: the worker holds one lease and
    embeds one query at a time. A lock here would be a claim about concurrency that nothing in this
    system tests.
    """

    __slots__ = ("_cache_dir", "_model", "_policy")

    def __init__(self, policy: EmbeddingPolicy = EMBEDDING_POLICY, cache_dir: str | None = None):
        self._policy = policy
        self._cache_dir = cache_dir if cache_dir is not None else get_settings().embedding_cache_dir
        self._model: Any | None = None

    @property
    def dimensions(self) -> int:
        return self._policy.dimensions

    @property
    def policy(self) -> EmbeddingPolicy:
        return self._policy

    def _session(self) -> Any:
        if self._model is None:
            from fastembed import TextEmbedding  # noqa: PLC0415

            self._model = TextEmbedding(
                model_name=self._policy.model_name, cache_dir=self._cache_dir
            )
        return self._model

    def _embed(self, prepared: Sequence[str]) -> list[list[float]]:
        vectors = [
            [float(component) for component in vector] for vector in self._session().embed(prepared)
        ]
        for index, vector in enumerate(vectors):
            if len(vector) != self._policy.dimensions:
                raise ValueError(
                    f"{self._policy.model_name} returned {len(vector)} dimensions for input "
                    f"{index} and the policy declares {self._policy.dimensions}. The vector column "
                    f"is built from the policy, so this would fail again at insert time with a "
                    f"message that named neither the model nor the input."
                )
        return vectors

    def encode_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed([self._policy.for_passage(text) for text in texts])

    def encode_query(self, text: str) -> list[float]:
        return self._embed([self._policy.for_query(text)])[0]
