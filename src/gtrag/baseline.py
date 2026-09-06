"""The Phase 1 baseline system.

Deliberately unsophisticated, and that is the whole point: fixed-size
chunks, one dense retriever, top-k, everything stuffed into a prompt. No
reranking, no hybrid retrieval, no metadata filtering, no query rewriting.

It exists to be beaten. Every Phase 3 improvement is measured against the
number this produces, and without a measured control those improvements
would be assertions.

What it is *not* is throwaway. It satisfies the same `RagSystem` protocol as
every later configuration, it is fully config-driven, and it stays in the
repo as `configs/baseline.yaml` forever. The constraint that every knob
Phase 3 will sweep is already a knob here is what turns the ablation sweep
into a loop rather than a refactor.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .chunking.base import Chunker, FixedTokenChunker, SpannedChunk
from .generate.generator import ExtractiveGenerator, Generator
from .index.embed import Embedder, HashingEmbedder
from .index.store import VectorIndex
from .ingest.document import Document
from .types import SystemResponse

__all__ = ["BaselineRagSystem", "build_baseline"]


@dataclass
class BaselineRagSystem:
    """Dense retrieval + generation, with per-stage timing."""

    index: VectorIndex
    generator: Generator
    top_k: int = 5
    name: str = "baseline"
    chunker_config: dict[str, Any] = field(default_factory=dict)

    @property
    def config(self) -> dict[str, Any]:
        """Fully-resolved configuration.

        This is hashed into the run id, so it must contain every knob that
        can change results -- and nothing that cannot, or two identical runs
        would look like different configurations.
        """
        return {
            "system": self.name,
            "top_k": self.top_k,
            "corpus_chunks": len(self.index),
            **self.chunker_config,
            **self.index.embedder.config,
            **self.generator.config,
        }

    @property
    def spanned_chunks(self) -> list[SpannedChunk]:
        """Indexed chunks, for resolving span-anchored gold labels."""
        return self.index.spanned_chunks

    def answer(
        self, question: str, *, history: list[tuple[str, str]] | None = None
    ) -> SystemResponse:
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        retrieved = self.index.search(question, top_k=self.top_k)
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


def build_baseline(
    documents: Sequence[Document] | None = None,
    *,
    index: VectorIndex | None = None,
    embedder: Embedder | None = None,
    chunker: Chunker | None = None,
    generator: Generator | None = None,
    top_k: int = 5,
    name: str = "baseline",
) -> BaselineRagSystem:
    """Assemble a baseline system, building the index if one is not supplied.

    Defaults are offline on purpose (hashing embedder, extractive generator)
    so the system is constructible in any environment. The offline defaults
    are honestly labeled in `config`, so a run made with them can never be
    mistaken for a semantic-retrieval result.
    """
    embedder = embedder or HashingEmbedder()
    chunker = chunker or FixedTokenChunker()
    generator = generator or ExtractiveGenerator()

    if index is None:
        if documents is None:
            raise ValueError("build_baseline needs either `documents` or a prebuilt `index`")
        index = VectorIndex(embedder=embedder)
        chunks: list[SpannedChunk] = []
        for document in documents:
            chunks.extend(chunker.chunk(document))
        index.add(chunks)

    return BaselineRagSystem(
        index=index,
        generator=generator,
        top_k=top_k,
        name=name,
        chunker_config=dict(chunker.config),
    )
