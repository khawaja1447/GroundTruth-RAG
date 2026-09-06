"""Declarative configurations for the Phase 3 ablation program.

A configuration is data, not code. That is the constraint Phase 1 was built
to satisfy -- "every knob Phase 3 will sweep is already a knob" -- and it is
what turns the sweep into a loop rather than a refactor.

    AblationConfig(label="+ bm25 hybrid", chunker="fixed", hybrid=True)

The sweep is a ladder: each rung adds one component to the rung below, so the
delta between adjacent rows is attributable to that component and nothing
else. Rungs that turn out not to help stay in the table, marked rejected --
negative results are the strongest credibility signal the repo has, because
they prove the numbers were not reverse-engineered from a conclusion.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from .chunking.base import SpannedChunk
from .chunking.strategies import build_chunker
from .generate.generator import ExtractiveGenerator, Generator
from .index.embed import Embedder, HashingEmbedder
from .index.store import VectorIndex
from .ingest.document import Document
from .retrieve.retrievers import (
    BM25Retriever,
    DenseRetriever,
    FilteredRetriever,
    HybridRetriever,
    LexicalReranker,
    MetadataExtractor,
    Reranker,
    RerankingRetriever,
    Retriever,
)
from .types import SystemResponse

__all__ = [
    "AblationConfig",
    "AblationSystem",
    "build_system",
    "ABLATION_LADDER",
    "CHUNKING_SWEEP",
]


@dataclass(frozen=True, slots=True)
class AblationConfig:
    """One point in the ablation space.

    Every field is a knob the sweep varies. Nothing here is a code path --
    two configurations differ only in these values, which is what makes the
    measured delta attributable to the change rather than to a refactor that
    rode along with it.
    """

    label: str
    chunker: str = "fixed"
    chunker_params: dict[str, Any] = field(default_factory=dict)
    embedder: str = "hashing"
    dense: bool = True
    bm25: bool = False
    rrf_k: int = 60
    fetch_depth: int = 50
    rerank: str = ""  # "" | "lexical" | "cross_encoder"
    retrieve_depth: int = 50
    metadata_filter: bool = False
    top_k: int = 5
    rejected: bool = False
    note: str = ""

    def __post_init__(self) -> None:
        if not (self.dense or self.bm25):
            raise ValueError(
                f"{self.label!r}: a configuration needs at least one retriever "
                f"(dense, bm25, or both)"
            )
        if self.top_k < 1:
            raise ValueError(f"{self.label!r}: top_k must be >= 1")
        if self.rerank and self.retrieve_depth < self.top_k:
            # The reranker can only reorder what the first stage found, so a
            # retrieve_depth below top_k silently caps the result count.
            raise ValueError(
                f"{self.label!r}: retrieve_depth ({self.retrieve_depth}) must be at "
                f"least top_k ({self.top_k}); reranking cannot invent candidates"
            )

    @property
    def retriever_name(self) -> str:
        if self.dense and self.bm25:
            return "hybrid"
        return "dense" if self.dense else "bm25"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# System
# --------------------------------------------------------------------------


@dataclass
class AblationSystem:
    """A configured retrieval system satisfying the harness `RagSystem` protocol."""

    retriever: Retriever
    generator: Generator
    chunks: Sequence[SpannedChunk]
    top_k: int = 5
    name: str = "ablation"
    _config: dict[str, Any] = field(default_factory=dict)

    @property
    def config(self) -> dict[str, Any]:
        return dict(self._config)

    @property
    def spanned_chunks(self) -> list[SpannedChunk]:
        """Exposed so the harness can resolve span-anchored gold labels
        against *this* configuration's chunking."""
        return list(self.chunks)

    def answer(
        self, question: str, *, history: list[tuple[str, str]] | None = None
    ) -> SystemResponse:
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        retrieved = self.retriever.retrieve(question, top_k=self.top_k)
        timings["retrieval"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        generated = self.generator.generate(question, retrieved, history=history)
        timings["generation"] = (time.perf_counter() - t0) * 1000.0
        timings["total"] = sum(timings.values())

        return SystemResponse(
            answer=generated.answer,
            retrieved=tuple(retrieved),
            citations=generated.citations,
            refused=generated.refused,
            usage=generated.usage,
            timings=timings,
            error=generated.error,
        )


def _build_embedder(name: str) -> Embedder:
    if name == "hashing":
        return HashingEmbedder()
    if name == "sentence-transformers":
        from .index.embed import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder()
    raise ValueError(f"unknown embedder {name!r}")


def _build_reranker(name: str) -> Reranker:
    if name == "lexical":
        return LexicalReranker()
    if name == "cross_encoder":
        from .retrieve.retrievers import CrossEncoderReranker

        return CrossEncoderReranker()
    raise ValueError(f"unknown reranker {name!r} (have: lexical, cross_encoder)")


def build_system(
    config: AblationConfig,
    documents: Sequence[Document],
    *,
    generator: Generator | None = None,
    index_cache: dict[str, VectorIndex] | None = None,
) -> AblationSystem:
    """Assemble a system from a configuration.

    `index_cache` is keyed on (chunker config, embedder) so a sweep that
    varies only the retriever does not re-embed the corpus for every row.
    Embedding dominates sweep runtime, and re-doing it per row would make the
    full ladder impractical to run often -- which in practice means it stops
    being run at all.
    """
    chunker = build_chunker(config.chunker, **config.chunker_params)
    chunks: list[SpannedChunk] = []
    for document in documents:
        chunks.extend(chunker.chunk(document))

    embedder = _build_embedder(config.embedder)
    index: VectorIndex | None = None

    if config.dense:
        cache_key = f"{sorted(chunker.config.items())}|{config.embedder}"
        if index_cache is not None and cache_key in index_cache:
            index = index_cache[cache_key]
        else:
            index = VectorIndex(embedder=embedder)
            index.add(chunks)
            if index_cache is not None:
                index_cache[cache_key] = index

    components: list[Retriever] = []
    if config.dense and index is not None:
        components.append(DenseRetriever(index=index))
    if config.bm25:
        components.append(BM25Retriever(chunks=chunks))

    retriever: Retriever
    if len(components) == 1:
        retriever = components[0]
    else:
        retriever = HybridRetriever(
            retrievers=components, rrf_k=config.rrf_k, fetch_depth=config.fetch_depth
        )

    if config.metadata_filter:
        retriever = FilteredRetriever(
            base=retriever, extractor=MetadataExtractor.from_chunks(chunks)
        )

    if config.rerank:
        retriever = RerankingRetriever(
            base=retriever,
            reranker=_build_reranker(config.rerank),
            retrieve_depth=config.retrieve_depth,
        )

    resolved = {
        "system": "ablation",
        "label": config.label,
        "top_k": config.top_k,
        "corpus_chunks": len(chunks),
        **chunker.config,
        **retriever.config,
    }
    generator = generator or ExtractiveGenerator()
    resolved.update(generator.config)

    return AblationSystem(
        retriever=retriever,
        generator=generator,
        chunks=chunks,
        top_k=config.top_k,
        name=config.label or "ablation",
        _config=resolved,
    )


# --------------------------------------------------------------------------
# The sweeps
# --------------------------------------------------------------------------

# Dimension 1, run first and alone: chunking is the only dimension whose
# change invalidates the index, and its winner becomes the substrate every
# later dimension is measured on.
CHUNKING_SWEEP: tuple[AblationConfig, ...] = (
    AblationConfig(label="fixed 512/50 (baseline)", chunker="fixed"),
    AblationConfig(label="recursive", chunker="recursive"),
    AblationConfig(label="structure-aware", chunker="structure_aware"),
    AblationConfig(label="sentence-window w=2", chunker="sentence_window"),
    AblationConfig(label="parent-document", chunker="parent_document"),
    AblationConfig(
        label="semantic p25",
        chunker="semantic",
        note="only meaningful with a semantic embedder; see docs/ablation.md",
    ),
)

# Dimensions 3-5, as a ladder. Each rung adds exactly one component to the
# rung above it, so the delta is attributable.
ABLATION_LADDER: tuple[AblationConfig, ...] = (
    AblationConfig(label="baseline: fixed + dense", chunker="fixed"),
    AblationConfig(label="+ structure-aware chunking", chunker="structure_aware"),
    AblationConfig(label="+ bm25 hybrid (RRF)", chunker="structure_aware", bm25=True),
    AblationConfig(
        label="+ reranking",
        chunker="structure_aware",
        bm25=True,
        rerank="lexical",
        retrieve_depth=30,
    ),
    AblationConfig(
        label="+ metadata pre-filtering",
        chunker="structure_aware",
        bm25=True,
        rerank="lexical",
        retrieve_depth=30,
        metadata_filter=True,
    ),
)
