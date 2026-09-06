"""A small persistent vector index.

Exhaustive cosine search. At the Phase 1 corpus size (tens of thousands of
chunks) an exact scan is milliseconds and always correct, whereas an ANN
index adds a recall parameter that would silently confound every retrieval
measurement in Phase 3 -- you would not know whether a change moved
retrieval quality or just the approximation. Qdrant arrives in Phase 3, when
metadata pre-filtering is actually needed and the recall/latency tradeoff is
itself something to measure.

Persistence is plain JSONL so an index can be inspected, diffed and version
controlled.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..chunking.base import SpannedChunk
from ..ingest.document import Span
from ..types import Chunk, RetrievedChunk
from .embed import Embedder, cosine

__all__ = ["VectorIndex", "IndexEntry"]


@dataclass(frozen=True, slots=True)
class IndexEntry:
    chunk: Chunk
    span: Span
    vector: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk.chunk_id,
            "doc_id": self.chunk.doc_id,
            "text": self.chunk.text,
            "metadata": self.chunk.metadata,
            "span": self.span.to_dict(),
            "vector": list(self.vector),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> IndexEntry:
        return cls(
            chunk=Chunk(
                chunk_id=raw["chunk_id"],
                text=raw["text"],
                doc_id=raw.get("doc_id", ""),
                metadata=dict(raw.get("metadata", {})),
            ),
            span=Span.from_dict(raw["span"]),
            vector=tuple(raw["vector"]),
        )


class VectorIndex:
    """Exhaustive-search vector index with metadata filtering."""

    def __init__(self, embedder: Embedder, entries: Sequence[IndexEntry] = ()) -> None:
        self.embedder = embedder
        self._entries: list[IndexEntry] = list(entries)

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> list[IndexEntry]:
        return list(self._entries)

    @property
    def chunk_ids(self) -> list[str]:
        return [e.chunk.chunk_id for e in self._entries]

    @property
    def spanned_chunks(self) -> list[SpannedChunk]:
        """The indexed chunks as `SpannedChunk`s, for span resolution."""
        return [SpannedChunk(chunk=e.chunk, span=e.span) for e in self._entries]

    # -- building ---------------------------------------------------------

    def add(self, chunks: Iterable[SpannedChunk], *, batch_size: int = 64) -> int:
        """Embed and add chunks. Returns the number added.

        Duplicate chunk ids are skipped rather than overwritten: a duplicate
        means the same span was chunked twice, and silently replacing it
        would hide a real bug in the ingestion pipeline.
        """
        known = set(self.chunk_ids)
        pending = [c for c in chunks if c.chunk_id not in known]
        if not pending:
            return 0

        added = 0
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            vectors = self.embedder.embed([c.chunk.text for c in batch])
            for spanned, vector in zip(batch, vectors, strict=True):
                self._entries.append(
                    IndexEntry(chunk=spanned.chunk, span=spanned.span, vector=tuple(vector))
                )
                added += 1
        return added

    # -- searching --------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        where: Callable[[Chunk], bool] | None = None,
    ) -> list[RetrievedChunk]:
        """Return the top_k most similar chunks, ranked from 1."""
        if not self._entries:
            return []
        query_vector = self.embedder.embed_query(query)

        scored: list[tuple[float, IndexEntry]] = []
        for entry in self._entries:
            if where is not None and not where(entry.chunk):
                continue
            scored.append((cosine(query_vector, entry.vector), entry))

        # Tiebreak on chunk_id so equal scores rank deterministically -- the
        # reproducibility tests depend on it.
        scored.sort(key=lambda pair: (-pair[0], pair[1].chunk.chunk_id))

        return [
            RetrievedChunk(
                chunk_id=entry.chunk.chunk_id,
                rank=rank,
                score=round(score, 6),
                text=entry.chunk.text,
                metadata=dict(entry.chunk.metadata),
            )
            for rank, (score, entry) in enumerate(scored[:top_k], start=1)
        ]

    # -- persistence ------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            fh.write(
                json.dumps({"_meta": {"embedder": self.embedder.config, "count": len(self)}}) + "\n"
            )
            for entry in self._entries:
                fh.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
        return p

    @classmethod
    def load(cls, path: str | Path, embedder: Embedder) -> VectorIndex:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"index not found: {p}\nRun `make ingest` first.")

        entries: list[IndexEntry] = []
        meta: dict[str, Any] = {}
        with p.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                if lineno == 1 and "_meta" in raw:
                    meta = raw["_meta"]
                    continue
                entries.append(IndexEntry.from_dict(raw))

        stored = meta.get("embedder", {})
        if stored and stored != embedder.config:
            # Vectors from a different model are not comparable to this
            # model's query vectors. Searching anyway returns plausible
            # nonsense, which is far worse than refusing.
            raise ValueError(
                f"index at {p} was built with a different embedder.\n"
                f"  stored:  {stored}\n"
                f"  current: {embedder.config}\n"
                f"Vectors from different models are not comparable -- re-run `make ingest`."
            )
        return cls(embedder=embedder, entries=entries)
