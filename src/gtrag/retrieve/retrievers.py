"""Retrievers for the Phase 3 ablation (dimensions 3-5).

All of them satisfy one protocol and compose, so a configuration is an
expression rather than a code path:

    RerankingRetriever(
        FilteredRetriever(
            HybridRetriever([DenseRetriever(index), BM25Retriever(chunks)]),
            extractor=MetadataExtractor(),
        ),
        reranker=LexicalReranker(),
    )

Each layer is independently measurable, which is the entire point: the
ablation table's rows are these compositions, and its deltas are what each
layer is worth.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..chunking.base import SpannedChunk
from ..index.store import VectorIndex
from ..types import Chunk, RetrievedChunk

__all__ = [
    "Retriever",
    "DenseRetriever",
    "BM25Retriever",
    "HybridRetriever",
    "RerankingRetriever",
    "FilteredRetriever",
    "Reranker",
    "LexicalReranker",
    "CrossEncoderReranker",
    "MetadataExtractor",
    "reciprocal_rank_fusion",
]

_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "to",
        "was",
        "were",
        "what",
        "when",
        "which",
        "who",
        "why",
        "with",
    ]
)


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


class Retriever(Protocol):
    name: str

    @property
    def config(self) -> dict[str, Any]: ...

    def retrieve(
        self, query: str, *, top_k: int, where: Callable[[Chunk], bool] | None = None
    ) -> list[RetrievedChunk]: ...


# --------------------------------------------------------------------------
# Dense
# --------------------------------------------------------------------------


@dataclass
class DenseRetriever:
    """Vector similarity. The Phase 1 baseline retriever."""

    index: VectorIndex
    name: str = "dense"

    @property
    def config(self) -> dict[str, Any]:
        return {"retriever": self.name, **self.index.embedder.config}

    def retrieve(
        self, query: str, *, top_k: int, where: Callable[[Chunk], bool] | None = None
    ) -> list[RetrievedChunk]:
        return self.index.search(query, top_k=top_k, where=where)


# --------------------------------------------------------------------------
# BM25
# --------------------------------------------------------------------------


@dataclass
class BM25Retriever:
    """Okapi BM25 over the same chunks.

    The reason hybrid retrieval is expected to win on this corpus: queries
    containing ticker symbols, exact line-item names and specific figures are
    lexical problems. "42.1%" either appears in a chunk or it does not, and
    dense retrieval reduces that certainty to a similarity score.

    Expect a large gain on the numeric and table slices and close to none on
    conceptual ones. That *shape* of result is what a real analysis looks
    like, and reporting only the aggregate would hide it.
    """

    chunks: Sequence[SpannedChunk]
    k1: float = 1.5
    b: float = 0.75
    name: str = "bm25"
    _tf: dict[str, Counter[str]] = field(default_factory=dict, repr=False)
    _idf: dict[str, float] = field(default_factory=dict, repr=False)
    _lengths: dict[str, int] = field(default_factory=dict, repr=False)
    _avg_len: float = 0.0

    def __post_init__(self) -> None:
        self._by_id = {c.chunk_id: c for c in self.chunks}
        for spanned in self.chunks:
            toks = _tokens(spanned.chunk.text)
            self._tf[spanned.chunk_id] = Counter(toks)
            self._lengths[spanned.chunk_id] = len(toks)

        n_docs = len(self.chunks)
        self._avg_len = sum(self._lengths.values()) / n_docs if n_docs else 0.0
        df: Counter[str] = Counter()
        for counts in self._tf.values():
            df.update(counts.keys())
        # +0.5 smoothing keeps idf positive for terms appearing in every
        # document; without it a common term gets a negative weight and
        # actively penalises the chunks that contain it.
        self._idf = {
            term: math.log(1 + (n_docs - count + 0.5) / (count + 0.5)) for term, count in df.items()
        }

    @property
    def config(self) -> dict[str, Any]:
        return {"retriever": self.name, "k1": self.k1, "b": self.b}

    def _score(self, query_tokens: Sequence[str], chunk_id: str) -> float:
        tf = self._tf[chunk_id]
        length = self._lengths[chunk_id]
        norm = self.k1 * (1 - self.b + self.b * (length / self._avg_len if self._avg_len else 1))
        total = 0.0
        for term in query_tokens:
            freq = tf.get(term, 0)
            if freq:
                total += self._idf.get(term, 0.0) * (freq * (self.k1 + 1)) / (freq + norm)
        return total

    def retrieve(
        self, query: str, *, top_k: int, where: Callable[[Chunk], bool] | None = None
    ) -> list[RetrievedChunk]:
        query_tokens = _tokens(query)
        if not query_tokens:
            return []

        scored: list[tuple[float, SpannedChunk]] = []
        for spanned in self.chunks:
            if where is not None and not where(spanned.chunk):
                continue
            score = self._score(query_tokens, spanned.chunk_id)
            if score > 0:
                scored.append((score, spanned))

        scored.sort(key=lambda pair: (-pair[0], pair[1].chunk_id))
        return [
            RetrievedChunk(
                chunk_id=spanned.chunk_id,
                rank=rank,
                score=round(score, 6),
                text=spanned.chunk.text,
                metadata=dict(spanned.chunk.metadata),
            )
            for rank, (score, spanned) in enumerate(scored[:top_k], start=1)
        ]


# --------------------------------------------------------------------------
# Fusion
# --------------------------------------------------------------------------


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RetrievedChunk]],
    *,
    weights: Sequence[float] | None = None,
    k: int = 60,
    top_k: int = 10,
) -> list[RetrievedChunk]:
    """Fuse ranked lists by reciprocal rank.

        score(d) = sum_i  weight_i / (k + rank_i(d))

    Rank-based rather than score-based, because BM25 scores and cosine
    similarities live on incomparable scales -- normalising them into
    agreement requires a per-corpus calibration that is itself a hidden
    hyperparameter. Ranks need no calibration.

    `k=60` is the standard damping constant: large enough that the top few
    ranks do not dominate completely, small enough that rank still matters.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError(f"got {len(weights)} weights for {len(rankings)} rankings")
    if k <= 0:
        raise ValueError("k must be positive")

    scores: dict[str, float] = {}
    best: dict[str, RetrievedChunk] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for item in ranking:
            scores[item.chunk_id] = scores.get(item.chunk_id, 0.0) + weight / (k + item.rank)
            # Keep the copy that carries text, whichever list it came from.
            if item.chunk_id not in best or (item.text and not best[item.chunk_id].text):
                best[item.chunk_id] = item

    ordered = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    return [
        RetrievedChunk(
            chunk_id=chunk_id,
            rank=rank,
            score=round(score, 6),
            text=best[chunk_id].text,
            metadata=dict(best[chunk_id].metadata),
        )
        for rank, (chunk_id, score) in enumerate(ordered[:top_k], start=1)
    ]


@dataclass
class HybridRetriever:
    """Fuse several retrievers with reciprocal rank fusion.

    Each component retrieves `fetch_depth` candidates, deeper than the final
    `top_k`: a chunk ranked 15th by one retriever and 3rd by the other should
    be able to surface, and it cannot if each list is truncated at 5 first.
    """

    retrievers: Sequence[Retriever]
    weights: Sequence[float] | None = None
    rrf_k: int = 60
    fetch_depth: int = 50
    name: str = "hybrid"

    def __post_init__(self) -> None:
        if not self.retrievers:
            raise ValueError("HybridRetriever needs at least one retriever")
        if self.weights is not None and len(self.weights) != len(self.retrievers):
            raise ValueError("weights must match the number of retrievers")

    @property
    def config(self) -> dict[str, Any]:
        return {
            "retriever": self.name,
            "components": [r.name for r in self.retrievers],
            "weights": list(self.weights) if self.weights else None,
            "rrf_k": self.rrf_k,
            "fetch_depth": self.fetch_depth,
        }

    def retrieve(
        self, query: str, *, top_k: int, where: Callable[[Chunk], bool] | None = None
    ) -> list[RetrievedChunk]:
        rankings = [r.retrieve(query, top_k=self.fetch_depth, where=where) for r in self.retrievers]
        return reciprocal_rank_fusion(rankings, weights=self.weights, k=self.rrf_k, top_k=top_k)


# --------------------------------------------------------------------------
# Reranking
# --------------------------------------------------------------------------


class Reranker(Protocol):
    name: str

    @property
    def config(self) -> dict[str, Any]: ...

    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


@dataclass
class LexicalReranker:
    """Query-passage overlap scoring. Offline stand-in for a cross-encoder.

    Scores coverage of the *query's* terms, weighted by rarity, with a
    proximity bonus when matched terms appear close together. It is not a
    cross-encoder and does not pretend to be -- it exists so the reranking
    *stage* is exercised, measured and tested with no model download, and so
    the cross-encoder's true contribution can be reported as a delta against
    something rather than against nothing.
    """

    name: str = "lexical"
    proximity_window: int = 40

    @property
    def config(self) -> dict[str, Any]:
        return {"reranker": self.name, "neural": False}

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        query_terms = _tokens(query)
        if not query_terms:
            return [0.0] * len(passages)
        unique = list(dict.fromkeys(query_terms))

        # Rarity weighting over the candidate set: a term appearing in every
        # candidate cannot discriminate between them.
        tokenized = [_tokens(p) for p in passages]
        sets = [set(t) for t in tokenized]
        n = len(passages) or 1
        idf = {term: math.log(1 + n / (1 + sum(1 for s in sets if term in s))) for term in unique}

        out: list[float] = []
        for tokens, present in zip(tokenized, sets, strict=True):
            matched = [t for t in unique if t in present]
            if not matched:
                out.append(0.0)
                continue
            coverage = sum(idf[t] for t in matched) / sum(idf[t] for t in unique)

            positions = [i for i, tok in enumerate(tokens) if tok in set(matched)]
            proximity = 0.0
            if len(positions) > 1:
                spread = positions[-1] - positions[0]
                proximity = max(0.0, 1.0 - spread / max(self.proximity_window, 1)) * 0.25
            out.append(round(coverage + proximity, 6))
        return out


@dataclass
class CrossEncoderReranker:
    """A real cross-encoder. Loaded lazily.

    Usually the largest single quality win available, and it costs real
    milliseconds. Reporting the nDCG-against-latency curve matters more than
    the headline number.
    """

    model_name: str = "BAAI/bge-reranker-v2-m3"
    batch_size: int = 16
    name: str = "cross_encoder"
    _model: Any = None

    @property
    def config(self) -> dict[str, Any]:
        return {"reranker": self.name, "model": self.model_name, "neural": True}

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise ImportError(
                    "CrossEncoderReranker needs sentence-transformers:\n"
                    "    pip install -e '.[embed]'\n"
                    "Or use LexicalReranker, which runs offline."
                ) from exc
            self._model = CrossEncoder(self.model_name)
        return self._model

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        model = self._load()
        pairs = [(query, p) for p in passages]
        return [float(s) for s in model.predict(pairs, batch_size=self.batch_size)]


@dataclass
class RerankingRetriever:
    """Retrieve wide, rerank, return narrow.

    `retrieve_depth` is the recall ceiling: the reranker can only reorder
    what the first stage found, so a chunk missing from the top 50 can never
    be recovered no matter how good the reranker is. Sweeping the two depths
    jointly is the point -- the interesting output is the quality/latency
    curve, not one number.
    """

    base: Retriever
    reranker: Reranker
    retrieve_depth: int = 50
    name: str = "reranked"

    def __post_init__(self) -> None:
        if self.retrieve_depth < 1:
            raise ValueError("retrieve_depth must be >= 1")

    @property
    def config(self) -> dict[str, Any]:
        return {
            "retriever": self.name,
            "retrieve_depth": self.retrieve_depth,
            "base": self.base.config,
            **self.reranker.config,
        }

    def retrieve(
        self, query: str, *, top_k: int, where: Callable[[Chunk], bool] | None = None
    ) -> list[RetrievedChunk]:
        candidates = self.base.retrieve(query, top_k=self.retrieve_depth, where=where)
        if not candidates:
            return []

        scores = self.reranker.score(query, [c.text for c in candidates])
        paired = sorted(
            zip(scores, candidates, strict=True),
            key=lambda pair: (-pair[0], pair[1].chunk_id),
        )
        return [
            RetrievedChunk(
                chunk_id=candidate.chunk_id,
                rank=rank,
                score=round(float(score), 6),
                text=candidate.text,
                metadata=dict(candidate.metadata),
            )
            for rank, (score, candidate) in enumerate(paired[:top_k], start=1)
        ]


# --------------------------------------------------------------------------
# Query understanding: metadata pre-filtering
# --------------------------------------------------------------------------

_YEAR = re.compile(r"\b(?:fiscal\s+(?:year\s+)?|FY\s*)?((?:19|20)\d{2})\b", re.IGNORECASE)
_SUFFIX = re.compile(r"\b(inc|corp|corporation|co|ltd|llc|plc|company)\b\.?", re.IGNORECASE)


@dataclass
class MetadataExtractor:
    """Pull structured filters out of a natural-language question.

    Rule-based rather than model-based: company names and fiscal years are a
    closed vocabulary drawn from the corpus itself, so a lookup is more
    accurate than an LLM call, free, and deterministic. Reach for a model
    when the filters are open-ended -- these are not.

    On a multi-company, multi-year corpus this is often a larger win than any
    embedding upgrade, because it removes the near-duplicate boilerplate of
    every *other* company-year from contention before scoring begins.
    """

    companies: dict[str, str] = field(default_factory=dict)
    name: str = "rule_based"

    @classmethod
    def from_chunks(cls, chunks: Sequence[SpannedChunk]) -> MetadataExtractor:
        """Build the company vocabulary from the corpus itself."""
        companies: dict[str, str] = {}
        for spanned in chunks:
            full = str(spanned.chunk.metadata.get("company", "")).strip()
            if not full:
                continue
            companies[full.lower()] = full
            # Also index the distinctive head of the name, so "Northwind"
            # matches "Northwind Logistics, Inc."
            head = _SUFFIX.sub("", full).replace(",", " ").strip()
            first = head.split()[0].lower() if head.split() else ""
            if len(first) > 3:
                companies.setdefault(first, full)
            if head and head.lower() not in companies:
                companies[head.lower()] = full
        return cls(companies=companies)

    @property
    def config(self) -> dict[str, Any]:
        return {"extractor": self.name, "known_companies": len(set(self.companies.values()))}

    def extract(self, query: str) -> dict[str, Any]:
        filters: dict[str, Any] = {}
        lowered = query.lower()

        matched = [full for alias, full in self.companies.items() if alias in lowered]
        # Only filter on an unambiguous single match. A comparative question
        # naming two companies must not be filtered down to one of them --
        # that would silently make the question unanswerable.
        if len(set(matched)) == 1:
            filters["company"] = matched[0]

        years = {int(m.group(1)) for m in _YEAR.finditer(query)}
        if len(years) == 1:
            filters["fiscal_year"] = years.pop()
        return filters

    def predicate(self, query: str) -> Callable[[Chunk], bool] | None:
        filters = self.extract(query)
        if not filters:
            return None

        def matches(chunk: Chunk) -> bool:
            for key, value in filters.items():
                actual = chunk.metadata.get(key)
                if actual is None:
                    # Missing metadata is not a mismatch. Excluding chunks
                    # that simply lack the field would silently shrink the
                    # corpus on any incompletely-tagged document.
                    continue
                if str(actual).lower() != str(value).lower():
                    return False
            return True

        return matches


@dataclass
class FilteredRetriever:
    """Apply extracted metadata filters as a retrieval pre-filter.

    Pre-filter, never post-filter. Filtering after retrieval means the top-k
    was chosen from the wrong pool and the correct chunk may already have
    been dropped; it also cannot be relied on for access control, which is
    why Phase 6 builds on this rather than on a post-filter.

    Falls back to the unfiltered pool when a filter matches nothing, since an
    over-eager extraction should degrade the ranking rather than return an
    empty answer.
    """

    base: Retriever
    extractor: MetadataExtractor
    name: str = "filtered"
    fallback_on_empty: bool = True

    @property
    def config(self) -> dict[str, Any]:
        return {
            "retriever": self.name,
            "base": self.base.config,
            "fallback_on_empty": self.fallback_on_empty,
            **self.extractor.config,
        }

    def retrieve(
        self, query: str, *, top_k: int, where: Callable[[Chunk], bool] | None = None
    ) -> list[RetrievedChunk]:
        predicate = self.extractor.predicate(query)
        combined: Callable[[Chunk], bool] | None
        if predicate is None:
            combined = where
        elif where is None:
            combined = predicate
        else:
            combined = lambda chunk: where(chunk) and predicate(chunk)  # noqa: E731

        results = self.base.retrieve(query, top_k=top_k, where=combined)
        if not results and self.fallback_on_empty and predicate is not None:
            return self.base.retrieve(query, top_k=top_k, where=where)
        return results
