"""A minimal, dependency-free RAG system for exercising the harness.

This is not the Phase 1 baseline -- that one indexes a real corpus and calls a
real model. This is a stand-in with the same interface, so the harness can be
run, tested, and demonstrated before any of that exists.

It is still a genuine retriever (BM25-style lexical scoring) and a genuine
extractive generator with a refusal threshold, which means the metrics it
produces are real measurements of a real, weak system rather than canned
numbers. That matters: a smoke test against a stub that always returns the
gold chunks would pass while the metrics were wrong.
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

from ..types import Chunk, Citation, RetrievedChunk, SystemResponse, Usage
from .corpus import FIXTURE_CHUNKS

__all__ = ["FixtureRagSystem", "tokenize"]

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


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, keeping decimals intact.

    Decimals are kept whole so "42.1" matches "42.1" rather than becoming
    two useless tokens -- exactly the lexical-match case that motivates
    hybrid retrieval on this corpus in Phase 3.
    """
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


class FixtureRagSystem:
    """BM25 retrieval over the fixture corpus + extractive answering."""

    def __init__(
        self,
        *,
        chunks: Sequence[Chunk] | None = None,
        top_k: int = 5,
        k1: float = 1.5,
        b: float = 0.75,
        refusal_threshold: float = 2.0,
        name: str = "fixture-bm25",
    ) -> None:
        self.name = name
        self.top_k = top_k
        self.k1 = k1
        self.b = b
        self.refusal_threshold = refusal_threshold
        self._chunks: tuple[Chunk, ...] = tuple(chunks if chunks is not None else FIXTURE_CHUNKS)

        self._tokens: dict[str, list[str]] = {c.chunk_id: tokenize(c.text) for c in self._chunks}
        self._tf: dict[str, Counter[str]] = {
            cid: Counter(toks) for cid, toks in self._tokens.items()
        }
        lengths = [len(t) for t in self._tokens.values()]
        self._avg_len = sum(lengths) / len(lengths) if lengths else 0.0

        n_docs = len(self._chunks)
        df: Counter[str] = Counter()
        for toks in self._tokens.values():
            df.update(set(toks))
        # BM25 idf with the +0.5 smoothing that keeps it positive for terms
        # appearing in every document.
        self._idf = {
            term: math.log(1 + (n_docs - count + 0.5) / (count + 0.5)) for term, count in df.items()
        }

    @property
    def config(self) -> dict[str, Any]:
        return {
            "retriever": "bm25",
            "top_k": self.top_k,
            "k1": self.k1,
            "b": self.b,
            "refusal_threshold": self.refusal_threshold,
            "corpus_size": len(self._chunks),
            "generator": "extractive",
        }

    # -- retrieval --------------------------------------------------------

    def _score(self, query_tokens: Iterable[str], chunk_id: str) -> float:
        tf = self._tf[chunk_id]
        length = len(self._tokens[chunk_id])
        norm = self.k1 * (1 - self.b + self.b * (length / self._avg_len if self._avg_len else 1))
        total = 0.0
        for term in query_tokens:
            freq = tf.get(term, 0)
            if not freq:
                continue
            total += self._idf.get(term, 0.0) * (freq * (self.k1 + 1)) / (freq + norm)
        return total

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        tokens = tokenize(question)
        scored = [(self._score(tokens, c.chunk_id), c) for c in self._chunks]
        # Sort by score desc, then chunk_id asc -- the tiebreak keeps the
        # output deterministic, which the reproducibility tests depend on.
        scored.sort(key=lambda pair: (-pair[0], pair[1].chunk_id))
        out: list[RetrievedChunk] = []
        for rank, (score, chunk) in enumerate(scored[: self.top_k], start=1):
            if score <= 0.0:
                break
            out.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    rank=rank,
                    score=round(score, 6),
                    text=chunk.text,
                    metadata=dict(chunk.metadata),
                )
            )
        return out

    # -- generation -------------------------------------------------------

    def answer(
        self, question: str, *, history: list[tuple[str, str]] | None = None
    ) -> SystemResponse:
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        retrieved = self.retrieve(question)
        timings["retrieval"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        top_score = retrieved[0].score if retrieved else 0.0
        if not retrieved or top_score < self.refusal_threshold:
            timings["generation"] = (time.perf_counter() - t0) * 1000.0
            timings["total"] = sum(timings.values())
            return SystemResponse(
                answer="",
                retrieved=tuple(retrieved),
                refused=True,
                timings=timings,
                usage=Usage(),
            )

        best = retrieved[0]
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", best.text) if s.strip()]
        query_terms = set(tokenize(question))
        ranked = sorted(
            sentences,
            key=lambda s: -len(query_terms & set(tokenize(s))),
        )
        answer = " ".join(ranked[:2]) if ranked else best.text
        timings["generation"] = (time.perf_counter() - t0) * 1000.0
        timings["total"] = sum(timings.values())

        return SystemResponse(
            answer=answer,
            retrieved=tuple(retrieved),
            citations=(Citation(claim_index=0, chunk_ids=(best.chunk_id,), text=answer),),
            refused=False,
            timings=timings,
            usage=Usage(input_tokens=len(tokenize(best.text)), output_tokens=len(tokenize(answer))),
        )
