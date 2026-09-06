"""Chunking, with every chunk carrying the document span it covers.

The span is not bookkeeping -- it is what makes Phase 3's chunking ablation
possible. Gold evidence is labeled once as a document span; each chunking
strategy produces chunks with their own spans; relevance is derived by
overlap. Swap the strategy and the labels still apply.

`Chunker` is a protocol so Phase 3 can add sentence-window, semantic,
structure-aware and parent-document strategies without touching the harness.
`FixedTokenChunker` is the Phase 1 baseline: deliberately unsophisticated,
because it exists to be beaten.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from ..ingest.document import Document, Span, stable_id
from ..types import Chunk

__all__ = ["Chunker", "FixedTokenChunker", "SpannedChunk", "approx_tokens", "TOKEN_RE"]

# Approximate word-piece boundaries well enough for budgeting without pulling
# in a tokenizer. Chunk sizes are a knob to be swept, not a quantity that
# needs to be exact -- and being wrong by a consistent factor is harmless
# because every configuration is measured, not assumed.
TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def approx_tokens(text: str) -> int:
    return len(TOKEN_RE.findall(text))


@dataclass(frozen=True, slots=True)
class SpannedChunk:
    """A chunk plus the document span it covers."""

    chunk: Chunk
    span: Span

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id


class Chunker(Protocol):
    name: str

    @property
    def config(self) -> dict[str, Any]: ...

    def chunk(self, document: Document) -> list[SpannedChunk]: ...


@dataclass
class FixedTokenChunker:
    """Fixed-size token windows with overlap. The Phase 1 baseline.

    Intentionally structure-blind: it splits mid-sentence, mid-table and
    mid-section. That is the point -- Phase 3 measures what fixing each of
    those is worth, and it can only do that against a control that does none
    of it.
    """

    chunk_tokens: int = 512
    overlap_tokens: int = 50
    name: str = "fixed"

    def __post_init__(self) -> None:
        if self.chunk_tokens <= 0:
            raise ValueError("chunk_tokens must be positive")
        if self.overlap_tokens < 0:
            raise ValueError("overlap_tokens must be >= 0")
        if self.overlap_tokens >= self.chunk_tokens:
            # Otherwise the window never advances and chunking does not
            # terminate. Catch it here rather than as a hang during ingest.
            raise ValueError(
                f"overlap_tokens ({self.overlap_tokens}) must be less than "
                f"chunk_tokens ({self.chunk_tokens}), or the window never advances"
            )

    @property
    def config(self) -> dict[str, Any]:
        return {
            "chunker": self.name,
            "chunk_tokens": self.chunk_tokens,
            "overlap_tokens": self.overlap_tokens,
        }

    def chunk(self, document: Document) -> list[SpannedChunk]:
        tokens = list(TOKEN_RE.finditer(document.text))
        if not tokens:
            return []

        out: list[SpannedChunk] = []
        step = self.chunk_tokens - self.overlap_tokens
        index = 0
        position = 0

        while position < len(tokens):
            window = tokens[position : position + self.chunk_tokens]
            if not window:
                break
            start = window[0].start()
            end = window[-1].end()
            text = document.text[start:end]

            if text.strip():
                section = document.section_at(start)
                metadata: dict[str, Any] = dict(document.metadata)
                metadata.update(
                    {
                        "section": section.name if section else "",
                        "char_start": start,
                        "char_end": end,
                        "chunk_index": index,
                        "n_tokens": len(window),
                    }
                )
                # The id is derived from the document, the chunking config and
                # the span -- so re-ingesting unchanged content reproduces it
                # exactly, and a different chunking produces different ids
                # without colliding.
                chunk_id = stable_id(
                    document.doc_id, self.name, str(self.chunk_tokens), str(start), str(end)
                )
                out.append(
                    SpannedChunk(
                        chunk=Chunk(
                            chunk_id=chunk_id,
                            text=text,
                            doc_id=document.doc_id,
                            metadata=metadata,
                        ),
                        span=Span(document.doc_id, start, end),
                    )
                )
                index += 1

            position += step

        return out


def chunk_documents(documents: Sequence[Document], chunker: Chunker) -> list[SpannedChunk]:
    out: list[SpannedChunk] = []
    for document in documents:
        out.extend(chunker.chunk(document))
    return out
