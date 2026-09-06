"""Context assembly: what actually reaches the model, in what order.

Phase 2 gave us `context_sufficiency`, so we can now isolate the cases where
retrieval found the right chunks and the answer was still wrong. Those are
generation failures, and they have their own distinct fixes -- all three of
which live here:

  1. **Deduplication.** Filings repeat boilerplate almost verbatim across
     fiscal years and across peer companies. Two near-identical chunks spend
     the context budget twice and skew the model toward whatever got
     repeated. This corpus makes the problem acute rather than theoretical.

  2. **Ordering.** Models attend unevenly across a long context; the middle
     is where information goes to die. Placing the strongest evidence at both
     ends is cheap and measurable.

  3. **Budget.** A hard token ceiling with a stated drop policy, not
     "whatever fits". More context frequently makes answers worse, and the
     ablation should be able to show that on your own numbers.

Every decision here is recorded on the returned `AssembledContext`, so a
per-question result file can explain why a chunk the retriever found never
reached the model.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..types import RetrievedChunk

__all__ = [
    "AssembledContext",
    "ContextAssembler",
    "shingles",
    "jaccard",
    "lost_in_the_middle_order",
]

_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def shingles(text: str, size: int = 4) -> frozenset[tuple[str, ...]]:
    """Overlapping n-grams, for near-duplicate detection.

    Shingles rather than a bag of words: filings from two different years
    share almost all of their vocabulary while differing in the figures that
    matter, and a bag-of-words comparison calls those duplicates. Word order
    is what distinguishes "revenue was $4,218 million" from "revenue was
    $3,800 million" once the numbers are just two more tokens.
    """
    toks = _tokens(text)
    if len(toks) < size:
        return frozenset({tuple(toks)}) if toks else frozenset()
    return frozenset(tuple(toks[i : i + size]) for i in range(len(toks) - size + 1))


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def lost_in_the_middle_order(chunks: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
    """Place the strongest evidence at both ends, weakest in the middle.

    Takes the relevance-ordered list and deals alternately to the front and
    the back, so rank 1 opens the context, rank 2 closes it, and the weakest
    chunk lands dead centre.

        [1, 2, 3, 4, 5]  ->  [1, 3, 5, 4, 2]

    Whether this helps is an empirical question on your own eval set, not a
    law -- which is why it is a toggle rather than a hardcoded reordering.
    """
    ordered = sorted(chunks, key=lambda c: c.rank)
    front: list[RetrievedChunk] = []
    back: list[RetrievedChunk] = []
    for i, chunk in enumerate(ordered):
        (front if i % 2 == 0 else back).append(chunk)
    return front + list(reversed(back))


@dataclass(frozen=True, slots=True)
class AssembledContext:
    """The passages that reached the model, and an account of the rest."""

    chunks: tuple[RetrievedChunk, ...]
    dropped_duplicate: tuple[tuple[str, str, float], ...] = ()  # (dropped, kept, similarity)
    dropped_budget: tuple[str, ...] = ()
    tokens_used: int = 0
    tokens_available: int = 0

    @property
    def n_dropped(self) -> int:
        return len(self.dropped_duplicate) + len(self.dropped_budget)

    @property
    def utilisation(self) -> float | None:
        if not self.tokens_available:
            return None
        return self.tokens_used / self.tokens_available

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_chunks": len(self.chunks),
            "chunk_ids": [c.chunk_id for c in self.chunks],
            "dropped_duplicate": [
                {"dropped": d, "kept": k, "similarity": round(s, 4)}
                for d, k, s in self.dropped_duplicate
            ],
            "dropped_budget": list(self.dropped_budget),
            "tokens_used": self.tokens_used,
            "tokens_available": self.tokens_available,
            "utilisation": self.utilisation,
        }


@dataclass
class ContextAssembler:
    """Turns retrieved chunks into the passage list the model sees.

    Order of operations is deliberate: deduplicate, then budget, then order.

      * Dedup first, so the budget is not spent on a repeat.
      * Budget before ordering, because dropping happens by relevance and the
        reorder destroys relevance order by construction.
    """

    max_tokens: int = 4000
    deduplicate: bool = True
    duplicate_threshold: float = 0.8
    shingle_size: int = 4
    reorder: bool = True
    name: str = "assembler"

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if not 0.0 < self.duplicate_threshold <= 1.0:
            raise ValueError("duplicate_threshold must be in (0, 1]")

    @property
    def config(self) -> dict[str, Any]:
        return {
            "context_max_tokens": self.max_tokens,
            "context_deduplicate": self.deduplicate,
            "context_duplicate_threshold": self.duplicate_threshold,
            "context_reorder": self.reorder,
        }

    def assemble(self, chunks: Sequence[RetrievedChunk]) -> AssembledContext:
        if not chunks:
            return AssembledContext(chunks=(), tokens_available=self.max_tokens)

        ranked = sorted(chunks, key=lambda c: c.rank)

        # 1. Deduplicate, keeping the higher-ranked member of each pair.
        kept: list[RetrievedChunk] = []
        kept_shingles: list[frozenset] = []
        dropped_dupes: list[tuple[str, str, float]] = []
        if self.deduplicate:
            for chunk in ranked:
                sig = shingles(chunk.text, self.shingle_size)
                match: tuple[str, float] | None = None
                for existing, existing_sig in zip(kept, kept_shingles, strict=True):
                    similarity = jaccard(sig, existing_sig)
                    if similarity >= self.duplicate_threshold:
                        match = (existing.chunk_id, similarity)
                        break
                if match is not None:
                    dropped_dupes.append((chunk.chunk_id, match[0], match[1]))
                else:
                    kept.append(chunk)
                    kept_shingles.append(sig)
        else:
            kept = list(ranked)

        # 2. Budget, dropping the least relevant first.
        selected: list[RetrievedChunk] = []
        dropped_budget: list[str] = []
        used = 0
        for chunk in kept:
            cost = len(_tokens(chunk.text))
            if used + cost > self.max_tokens and selected:
                dropped_budget.append(chunk.chunk_id)
                continue
            selected.append(chunk)
            used += cost

        # 3. Order.
        final = lost_in_the_middle_order(selected) if self.reorder else selected

        # Ranks are renumbered to presentation order, because the citation
        # indices the model returns are positions in what it was shown. Any
        # other numbering makes every citation off by an unpredictable amount.
        renumbered = tuple(
            RetrievedChunk(
                chunk_id=c.chunk_id,
                rank=i,
                score=c.score,
                text=c.text,
                metadata=dict(c.metadata),
            )
            for i, c in enumerate(final, start=1)
        )

        return AssembledContext(
            chunks=renumbered,
            dropped_duplicate=tuple(dropped_dupes),
            dropped_budget=tuple(dropped_budget),
            tokens_used=used,
            tokens_available=self.max_tokens,
        )
