"""Resolving span-anchored gold evidence into per-chunking relevance labels.

This is the bridge between Phase 1 and Phase 3, and the reason the chunking
ablation is possible.

The eval set labels evidence once, as a document character span. A chunking
strategy produces chunks with their own spans. This module derives the
`chunk_id -> relevance` map that the Phase 2 metrics consume, for whatever
chunking is currently in play. Change the chunker, re-resolve, and the same
human labeling still applies.

## The grading rule

A chunk's relevance to a gold span is decided by how much of *the span* the
chunk covers, not how much of the chunk is the span:

  * `>= FULL_COVERAGE` of the span  -> relevance 2 (contains the answer)
  * `>= PARTIAL_COVERAGE`           -> relevance 1 (supporting context)
  * below that                      -> 0 (not labeled)

Asymmetric on purpose. A 512-token chunk that happens to contain a
one-sentence answer is a retrieval success even though 95% of its text is
unrelated; grading on the chunk's own composition would punish exactly the
large-chunk strategies Phase 3 needs to evaluate fairly.

The partial band exists because a chunk boundary can cut an answer in half.
Neither half alone answers the question, and calling both irrelevant would
make fixed-size chunking look better than it is by hiding its worst failure.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from gtrag.chunking.base import SpannedChunk
from gtrag.ingest.document import Span

__all__ = [
    "GoldSpan",
    "resolve_relevance",
    "resolve_dataset_spans",
    "FULL_COVERAGE",
    "PARTIAL_COVERAGE",
]

# A chunk containing this much of the gold span is treated as answer-bearing.
# Not 1.0: a chunk that clips a trailing period should not be demoted.
FULL_COVERAGE = 0.90
# Enough of the evidence to be genuinely useful context.
PARTIAL_COVERAGE = 0.25


@dataclass(frozen=True, slots=True)
class GoldSpan:
    """Human-labeled evidence, anchored to a document rather than a chunk.

    `weight` distinguishes evidence that is sufficient on its own (1.0, and
    a covering chunk earns relevance 2) from evidence that merely supports
    (0.5, capping a covering chunk at relevance 1) -- the multi-hop case
    where a passage is necessary but not sufficient.
    """

    span: Span
    weight: float = 1.0
    note: str = ""

    def __post_init__(self) -> None:
        if not 0.0 < self.weight <= 1.0:
            raise ValueError(f"weight must be in (0, 1], got {self.weight}")

    @property
    def max_relevance(self) -> int:
        return 2 if self.weight >= 1.0 else 1

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {**self.span.to_dict(), "weight": self.weight}
        if self.note:
            out["note"] = self.note
        return out

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> GoldSpan:
        return cls(
            span=Span.from_dict(raw),
            weight=float(raw.get("weight", 1.0)),
            note=raw.get("note", ""),
        )


def resolve_relevance(
    gold_spans: Sequence[GoldSpan],
    chunks: Iterable[SpannedChunk],
    *,
    full_coverage: float = FULL_COVERAGE,
    partial_coverage: float = PARTIAL_COVERAGE,
) -> dict[str, int]:
    """Derive `chunk_id -> relevance` for one question under one chunking.

    A chunk overlapping several gold spans takes its highest earned
    relevance. Only non-zero entries are returned; unlabeled chunks are
    relevance 0 by omission, which is what the metrics expect.
    """
    if full_coverage <= partial_coverage:
        raise ValueError(
            f"full_coverage ({full_coverage}) must exceed partial_coverage ({partial_coverage})"
        )

    relevance: dict[str, int] = {}
    for chunk in chunks:
        best = 0
        for gold in gold_spans:
            covered = chunk.span.overlap_fraction(gold.span)
            if covered >= full_coverage:
                earned = gold.max_relevance
            elif covered >= partial_coverage:
                earned = 1
            else:
                earned = 0
            best = max(best, earned)
        if best:
            relevance[chunk.chunk_id] = best
    return relevance


def resolve_dataset_spans(
    questions_spans: dict[str, Sequence[GoldSpan]],
    chunks: Sequence[SpannedChunk],
    **kwargs: Any,
) -> dict[str, dict[str, int]]:
    """Resolve every question's spans against one chunking.

    Returns `question_id -> {chunk_id: relevance}`. Questions whose spans
    resolve to nothing are still present with an empty map, so a chunking
    that destroys the evidence for a question is visible as an explicit
    empty rather than a missing key.
    """
    return {
        question_id: resolve_relevance(spans, chunks, **kwargs)
        for question_id, spans in questions_spans.items()
    }


def coverage_report(
    questions_spans: dict[str, Sequence[GoldSpan]],
    chunks: Sequence[SpannedChunk],
    **kwargs: Any,
) -> dict[str, Any]:
    """Diagnose how well a chunking preserves the labeled evidence.

    Run this whenever the chunking changes. A question that loses its
    answer-bearing chunk under a new strategy is not a retrieval result -- it
    is the chunking having split the evidence, and it needs to be visible as
    a property of the chunking rather than mistaken for a worse retriever.
    """
    resolved = resolve_dataset_spans(questions_spans, chunks, **kwargs)
    lost: list[str] = []
    degraded: list[str] = []
    for question_id, spans in questions_spans.items():
        mapping = resolved[question_id]
        if not mapping:
            lost.append(question_id)
            continue
        expects_answer_bearing = any(g.max_relevance == 2 for g in spans)
        if expects_answer_bearing and 2 not in mapping.values():
            degraded.append(question_id)

    return {
        "questions": len(questions_spans),
        "resolved": len(questions_spans) - len(lost),
        "lost": lost,
        "degraded_to_partial": degraded,
        "mean_relevant_chunks": (
            sum(len(m) for m in resolved.values()) / len(resolved) if resolved else 0.0
        ),
    }
