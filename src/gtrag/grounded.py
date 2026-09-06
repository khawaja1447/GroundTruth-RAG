"""The Phase 4 system: retrieval plus everything that happens after it.

Phase 3 stopped at "the right chunks were retrieved". This adds the stages
between that and an answer a reader can check:

    query -> rewrite -> retrieve -> assemble -> refuse? -> generate -> verify

Each stage is optional and configured, so the ablation can measure what each
is worth rather than assuming. The per-question trace records what every
stage did, which is what makes the failure taxonomy buildable from result
files instead of by re-running things by hand.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .chunking.base import SpannedChunk
from .generate.context import AssembledContext, ContextAssembler
from .generate.generator import ExtractiveGenerator, Generator
from .generate.refusal import RefusalPolicy, confidence_of
from .generate.verify import VerificationResult, Verifier, annotate_unsupported
from .retrieve.retrievers import Retriever
from .retrieve.rewrite import NullRewriter, QueryRewriter
from .types import Citation, SystemResponse

__all__ = ["GroundedRagSystem", "GenerationTrace"]


@dataclass(frozen=True, slots=True)
class GenerationTrace:
    """What each post-retrieval stage did, for one question."""

    original_query: str
    rewritten_query: str
    context: dict[str, Any] = field(default_factory=dict)
    confidence: dict[str, float | int] = field(default_factory=dict)
    refused_by_policy: bool = False
    verification: dict[str, Any] = field(default_factory=dict)

    @property
    def was_rewritten(self) -> bool:
        return self.original_query != self.rewritten_query

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_query": self.original_query,
            "rewritten_query": self.rewritten_query,
            "was_rewritten": self.was_rewritten,
            "context": self.context,
            "confidence": self.confidence,
            "refused_by_policy": self.refused_by_policy,
            "verification": self.verification,
        }


@dataclass
class GroundedRagSystem:
    """Retrieval, context assembly, refusal, generation, verification."""

    retriever: Retriever
    generator: Generator
    chunks: Sequence[SpannedChunk]
    assembler: ContextAssembler = field(default_factory=ContextAssembler)
    rewriter: QueryRewriter = field(default_factory=NullRewriter)
    refusal: RefusalPolicy | None = None
    verifier: Verifier | None = None
    annotate: bool = True
    top_k: int = 5
    name: str = "grounded"
    _extra_config: dict[str, Any] = field(default_factory=dict)

    @property
    def config(self) -> dict[str, Any]:
        config: dict[str, Any] = {
            "system": self.name,
            "top_k": self.top_k,
            "corpus_chunks": len(self.chunks),
            **self.retriever.config,
            **self.assembler.config,
            **self.rewriter.config,
            **self.generator.config,
            **self._extra_config,
        }
        config.update(self.refusal.config if self.refusal else {"refusal": "generator"})
        config.update(self.verifier.config if self.verifier else {"verifier": "none"})
        return config

    @property
    def spanned_chunks(self) -> list[SpannedChunk]:
        return list(self.chunks)

    # -- the pipeline -----------------------------------------------------

    def answer(
        self, question: str, *, history: list[tuple[str, str]] | None = None
    ) -> SystemResponse:
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        query = self.rewriter.rewrite(question, history or ())
        timings["rewrite"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        retrieved = self.retriever.retrieve(query, top_k=self.top_k)
        timings["retrieval"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        assembled: AssembledContext = self.assembler.assemble(retrieved)
        timings["assembly"] = (time.perf_counter() - t0) * 1000.0

        confidence = confidence_of(retrieved)
        trace = GenerationTrace(
            original_query=question,
            rewritten_query=query,
            context=assembled.to_dict(),
            confidence=confidence.to_dict(),
        )

        # Refuse before generating: a system that must produce an answer
        # before deciding whether to withhold it has already paid for it.
        if self.refusal is not None and self.refusal.should_refuse(retrieved):
            timings["total"] = sum(timings.values())
            return SystemResponse(
                answer="",
                retrieved=assembled.chunks,
                refused=True,
                timings=timings,
                metadata={"trace": {**trace.to_dict(), "refused_by_policy": True}},
            )

        t0 = time.perf_counter()
        generated = self.generator.generate(query, assembled.chunks, history=history)
        timings["generation"] = (time.perf_counter() - t0) * 1000.0

        answer = generated.answer
        citations: tuple[Citation, ...] = generated.citations
        verification: VerificationResult | None = None

        if self.verifier is not None and answer.strip() and not generated.refused:
            t0 = time.perf_counter()
            claims = [c.text for c in citations if c.text] or _sentences(answer)
            verification = self.verifier.verify(claims, [c.text for c in assembled.chunks])
            timings["verification"] = (time.perf_counter() - t0) * 1000.0
            if self.annotate:
                answer = annotate_unsupported(answer, verification)

        timings["total"] = sum(timings.values())

        return SystemResponse(
            answer=answer,
            retrieved=assembled.chunks,
            citations=citations,
            refused=generated.refused,
            usage=generated.usage,
            timings=timings,
            error=generated.error,
            metadata={
                "trace": {
                    **trace.to_dict(),
                    "verification": verification.to_dict() if verification else {},
                }
            },
        )


def _sentences(text: str) -> list[str]:
    """Fall back to sentences when the generator produced no citations."""
    from .chunking.strategies import split_sentences

    return [text[s:e] for s, e in split_sentences(text)]


def build_grounded(
    retriever: Retriever,
    chunks: Sequence[SpannedChunk],
    *,
    generator: Generator | None = None,
    assembler: ContextAssembler | None = None,
    rewriter: QueryRewriter | None = None,
    refusal: RefusalPolicy | None = None,
    verifier: Verifier | None = None,
    top_k: int = 5,
    name: str = "grounded",
) -> GroundedRagSystem:
    return GroundedRagSystem(
        retriever=retriever,
        generator=generator or ExtractiveGenerator(),
        chunks=chunks,
        assembler=assembler or ContextAssembler(),
        rewriter=rewriter or NullRewriter(),
        refusal=refusal,
        verifier=verifier,
        top_k=top_k,
        name=name,
    )
