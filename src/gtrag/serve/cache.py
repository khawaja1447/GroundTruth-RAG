"""Semantic response cache.

Exact-match caching is nearly useless for a question-answering service: two
users asking "what was Northwind's FY24 revenue?" and "Northwind fiscal 2024
net revenue?" want the same answer and share almost no characters. Caching on
embedding similarity catches both.

The hard part is not the lookup, it is **invalidation**. A cached answer is
derived from specific chunks; when the corpus is re-ingested and those chunks
change, the answer is stale and there is nothing in the question to tell you
so. Every entry therefore records the chunk ids it was derived from and the
corpus version it was built against, and both are checked on read. A cache
that cannot explain when it goes stale is a correctness bug with a latency
benefit.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..index.embed import Embedder, cosine
from ..types import SystemResponse

__all__ = ["SemanticCache", "CacheEntry", "CacheStats"]


@dataclass(frozen=True, slots=True)
class CacheEntry:
    """One cached answer, with everything needed to invalidate it."""

    question: str
    vector: tuple[float, ...]
    response: SystemResponse
    chunk_ids: frozenset[str]
    corpus_version: str
    created_at: float
    hits: int = 0

    def age_seconds(self, now: float | None = None) -> float:
        return (now if now is not None else time.time()) - self.created_at


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    stale_invalidations: int = 0
    cost_saved_usd: float = 0.0
    latency_saved_ms: float = 0.0

    @property
    def lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float | None:
        return self.hits / self.lookups if self.lookups else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "lookups": self.lookups,
            "hit_rate": self.hit_rate,
            "evictions": self.evictions,
            "stale_invalidations": self.stale_invalidations,
            "cost_saved_usd": round(self.cost_saved_usd, 6),
            "latency_saved_ms": round(self.latency_saved_ms, 2),
        }


class SemanticCache:
    """Similarity-keyed cache with chunk-level and version-level invalidation.

    Thread-safe: the server serves concurrently and a torn read of the entry
    list would return a half-updated cache.

    Exhaustive scan over entries. At a few thousand entries that is
    microseconds, and an ANN index here would add an approximation to a
    component whose whole job is returning *the same answer* -- a recall miss
    in the cache is invisible (it just costs a regeneration), but a false
    positive returns the wrong answer to a user.
    """

    def __init__(
        self,
        embedder: Embedder,
        *,
        threshold: float = 0.95,
        max_entries: int = 1000,
        ttl_seconds: float | None = 3600.0,
        corpus_version: str = "",
        enabled: bool = True,
    ) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")

        self.embedder = embedder
        # High by default. A loose threshold returns a confidently wrong
        # answer to a question nobody asked, which is far worse than a miss.
        self.threshold = threshold
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.corpus_version = corpus_version
        self.enabled = enabled
        self.stats = CacheStats()
        self._entries: list[CacheEntry] = []
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def config(self) -> dict[str, Any]:
        return {
            "cache": "semantic",
            "cache_enabled": self.enabled,
            "cache_threshold": self.threshold,
            "cache_max_entries": self.max_entries,
            "cache_ttl_seconds": self.ttl_seconds,
        }

    # -- read -------------------------------------------------------------

    def get(self, question: str) -> SystemResponse | None:
        if not self.enabled:
            return None

        vector = tuple(self.embedder.embed_query(question))
        now = time.time()

        with self._lock:
            best: CacheEntry | None = None
            best_score = 0.0
            expired: list[CacheEntry] = []

            for entry in self._entries:
                if self.ttl_seconds is not None and entry.age_seconds(now) > self.ttl_seconds:
                    expired.append(entry)
                    continue
                if entry.corpus_version != self.corpus_version:
                    # The corpus moved underneath this answer.
                    expired.append(entry)
                    continue
                score = cosine(vector, entry.vector)
                if score >= self.threshold and score > best_score:
                    best, best_score = entry, score

            for entry in expired:
                self._entries.remove(entry)
                self.stats.stale_invalidations += 1

            if best is None:
                self.stats.misses += 1
                return None

            index = self._entries.index(best)
            self._entries[index] = CacheEntry(
                question=best.question,
                vector=best.vector,
                response=best.response,
                chunk_ids=best.chunk_ids,
                corpus_version=best.corpus_version,
                created_at=best.created_at,
                hits=best.hits + 1,
            )
            self.stats.hits += 1
            self.stats.cost_saved_usd += best.response.usage.cost_usd
            self.stats.latency_saved_ms += best.response.timings.get("total", 0.0)

        # Mark the response so a caller can tell a cached answer from a fresh
        # one. Timings are zeroed because reporting the original generation
        # latency on a cache hit would make p95 meaningless.
        return SystemResponse(
            answer=best.response.answer,
            retrieved=best.response.retrieved,
            citations=best.response.citations,
            refused=best.response.refused,
            usage=best.response.usage,
            timings={"total": 0.0, "cache_lookup": 0.0},
            error=best.response.error,
            metadata={**best.response.metadata, "cache": {"hit": True, "similarity": best_score}},
        )

    # -- write ------------------------------------------------------------

    def put(self, question: str, response: SystemResponse) -> None:
        if not self.enabled or response.error:
            # Never cache an errored response: the next request would get the
            # same failure for free, for an hour.
            return

        vector = tuple(self.embedder.embed_query(question))
        entry = CacheEntry(
            question=question,
            vector=vector,
            response=response,
            chunk_ids=frozenset(response.retrieved_ids),
            corpus_version=self.corpus_version,
            created_at=time.time(),
        )

        with self._lock:
            if len(self._entries) >= self.max_entries:
                # Evict least-recently-created among the least-hit. Pure LRU
                # would need access ordering; this is close enough and keeps
                # the popular answers.
                victim = min(self._entries, key=lambda e: (e.hits, e.created_at))
                self._entries.remove(victim)
                self.stats.evictions += 1
            self._entries.append(entry)

    # -- invalidation -----------------------------------------------------

    def invalidate_chunks(self, chunk_ids: Sequence[str]) -> int:
        """Drop every entry derived from any of these chunks.

        Called after a partial re-ingest. Without it, an answer citing a
        figure that has since been restated stays served until its TTL
        expires -- correct-looking, sourced, and wrong.
        """
        changed = set(chunk_ids)
        with self._lock:
            keep = [e for e in self._entries if not (e.chunk_ids & changed)]
            removed = len(self._entries) - len(keep)
            self._entries = keep
            self.stats.stale_invalidations += removed
        return removed

    def set_corpus_version(self, version: str) -> int:
        """Bump the corpus version, invalidating everything built before it."""
        with self._lock:
            self.corpus_version = version
            stale = [e for e in self._entries if e.corpus_version != version]
            for entry in stale:
                self._entries.remove(entry)
            self.stats.stale_invalidations += len(stale)
            return len(stale)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
