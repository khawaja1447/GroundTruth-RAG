"""Embedders behind one protocol.

Two implementations, for two different jobs:

  * `HashingEmbedder` -- deterministic, stdlib-only, no download. It is
    **not semantic**: it is hashed bag-of-words with sub-linear term
    weighting, so it behaves like a lexical retriever wearing a vector
    interface. It exists so the pipeline, the index, and the tests run
    anywhere with no model and no network, and so CI is deterministic. It is
    honestly labeled everywhere so no number produced with it is ever
    mistaken for a semantic-retrieval result.

  * `SentenceTransformerEmbedder` -- the real one, loaded lazily so the
    import cost is only paid when it is used.

Phase 3 adds an API-backed embedder and measures all three against each
other. The protocol is what makes that a config change rather than a
rewrite.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

__all__ = ["Embedder", "HashingEmbedder", "SentenceTransformerEmbedder", "cosine", "l2_normalize"]

_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def _tokens(text: str) -> list[str]:
    """Lowercase tokens, keeping decimals whole.

    "42.1" stays one token rather than becoming "42" and "1" -- on a corpus
    of financial figures that distinction is most of the retrievable signal.
    """
    return _TOKEN.findall(text.lower())


def l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        return vector
    return [v / norm for v in vector]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Dot product; assumes both vectors are already L2-normalized.

    Every embedder here normalizes on output, so the index can use the
    cheaper dot product rather than recomputing norms per comparison.
    """
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    return sum(x * y for x, y in zip(a, b, strict=True))


class Embedder(Protocol):
    name: str
    dimension: int

    @property
    def config(self) -> dict[str, Any]: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


@dataclass
class HashingEmbedder:
    """Feature-hashed bag of words. Deterministic, offline, NOT semantic."""

    dimension: int = 512
    name: str = "hashing"
    seed: int = 0

    @property
    def config(self) -> dict[str, Any]:
        return {
            "embedder": self.name,
            "dimension": self.dimension,
            "seed": self.seed,
            "semantic": False,
        }

    def _bucket(self, token: str) -> tuple[int, float]:
        """Map a token to a bucket and a sign.

        The signed hash is the standard hashing-trick trick: collisions
        cancel in expectation instead of always adding, which keeps a
        collision from reliably inflating similarity.
        """
        digest = hashlib.blake2b(
            token.encode("utf-8"), digest_size=8, key=str(self.seed).encode()
        ).digest()
        value = int.from_bytes(digest, "big")
        return value % self.dimension, 1.0 if (value >> 63) & 1 else -1.0

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        counts = Counter(_tokens(text))
        for token, count in counts.items():
            index, sign = self._bucket(token)
            # Sub-linear term weighting: a term repeated 20 times is not 20x
            # as informative, and without damping long tables dominate.
            vector[index] += sign * (1.0 + math.log(count))
        return l2_normalize(vector)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


@dataclass
class SentenceTransformerEmbedder:
    """Real semantic embeddings via sentence-transformers.

    Loaded lazily: importing torch costs seconds, and the deterministic half
    of the pipeline must not pay that to run its tests.
    """

    model_name: str = "BAAI/bge-base-en-v1.5"
    name: str = "sentence-transformers"
    batch_size: int = 32
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    _model: Any = None
    _dimension: int = 0

    @property
    def config(self) -> dict[str, Any]:
        return {
            "embedder": self.name,
            "model": self.model_name,
            "batch_size": self.batch_size,
            "semantic": True,
        }

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "SentenceTransformerEmbedder needs sentence-transformers:\n"
                    "    pip install -e '.[embed]'\n"
                    "Or use HashingEmbedder, which is offline but not semantic."
                ) from exc
            self._model = SentenceTransformer(self.model_name)
            self._dimension = int(self._model.get_sentence_embedding_dimension())
        return self._model

    @property
    def dimension(self) -> int:
        if not self._dimension:
            self._load()
        return self._dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        model = self._load()
        vectors = model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [list(map(float, v)) for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        # BGE models expect an instruction prefix on queries but not on
        # passages. Omitting it costs several points of retrieval quality,
        # and the failure is silent.
        model = self._load()
        vector = model.encode(
            [self.query_prefix + text], normalize_embeddings=True, show_progress_bar=False
        )[0]
        return list(map(float, vector))
