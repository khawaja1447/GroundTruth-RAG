"""The normalized document model, and the span anchoring that makes the
Phase 3 ablation program possible at all.

## Why spans rather than chunk ids

Phase 2 labels gold evidence by `chunk_id`. That is fine while the chunking
is fixed -- and it silently destroys the project the moment Phase 3 starts,
because the first ablation dimension *is* the chunking. Re-chunk the corpus
and every `chunk_id` changes, so every label points at something that no
longer exists. You would have to re-label the entire eval set once per
chunking strategy, which nobody does, so in practice people quietly stop
comparing chunking strategies and the most valuable ablation never happens.

The fix is to anchor evidence to something chunking cannot move: a character
span in the *document*. A chunk then records which span of its document it
covers, and relevance is computed by overlap at eval time. One labeling
effort survives every chunking strategy you will ever try.

    document text  ......[=== gold span ===]...........
    chunking A     [ chunk 1 ][ chunk 2 ][ chunk 3 ]      -> chunk 2 relevant
    chunking B     [   chunk 1   ][   chunk 2   ]         -> chunk 1 relevant

Both labelings are derived from the same span. Neither needed a human.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Span",
    "Section",
    "Table",
    "Document",
    "stable_id",
    "normalize_whitespace",
]


def stable_id(*parts: str) -> str:
    """A short, deterministic id derived from content.

    Used for document and chunk ids. Deterministic across machines and runs,
    so re-ingesting unchanged content produces identical ids and the eval
    labels keep pointing at the right thing.
    """
    joined = "\x00".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


_WS = re.compile(r"[ \t\r\f\v]+")
_BLANKS = re.compile(r"\n{3,}")


def normalize_whitespace(text: str) -> str:
    """Collapse runs of spaces and excess blank lines.

    Run **once**, at ingestion, before any offset is recorded. Every span in
    the system indexes into the normalized text, so normalizing again later
    would shift every offset and silently corrupt every label.
    """
    text = text.replace(" ", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = _WS.sub(" ", text)
    text = _BLANKS.sub("\n\n", text)
    return text.strip()


@dataclass(frozen=True, slots=True)
class Span:
    """A half-open character range `[start, end)` in a document's text."""

    doc_id: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"span start must be >= 0, got {self.start}")
        if self.end < self.start:
            raise ValueError(f"span end {self.end} precedes start {self.start}")

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlap(self, other: Span) -> int:
        """Number of characters shared with `other`; 0 if different documents."""
        if self.doc_id != other.doc_id:
            return 0
        return max(0, min(self.end, other.end) - max(self.start, other.start))

    def overlap_fraction(self, other: Span) -> float:
        """Fraction of `other` that this span covers.

        Asymmetric on purpose. The question at eval time is "how much of the
        *gold evidence* does this chunk contain?", not "how much of the chunk
        is gold" -- a large chunk containing the whole answer is a retrieval
        success even though most of its text is unrelated.
        """
        if other.length == 0:
            return 1.0 if self.overlap(other) > 0 or self.start <= other.start < self.end else 0.0
        return self.overlap(other) / other.length

    def to_dict(self) -> dict[str, Any]:
        return {"doc_id": self.doc_id, "start": self.start, "end": self.end}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Span:
        return cls(doc_id=raw["doc_id"], start=int(raw["start"]), end=int(raw["end"]))


@dataclass(frozen=True, slots=True)
class Table:
    """A table extracted from a filing, kept structured *and* linearized.

    Both forms are retained because they serve different stages: the
    linearized text is what gets embedded and shown to the model, while the
    structure is what lets Phase 3's structure-aware chunker refuse to split
    a table down the middle.
    """

    rows: tuple[tuple[str, ...], ...]
    span: Span
    caption: str = ""

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_cols(self) -> int:
        return max((len(r) for r in self.rows), default=0)

    def linearize(self) -> str:
        """Render as aligned pipe-separated text.

        Column alignment is not cosmetic: it keeps a row's cells adjacent in
        the token stream, which is what makes a figure retrievable together
        with its row label instead of drifting apart.
        """
        if not self.rows:
            return ""
        widths = [0] * self.n_cols
        for row in self.rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell))
        lines = [
            " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip()
            for row in self.rows
        ]
        return (f"{self.caption}\n" if self.caption else "") + "\n".join(lines)


@dataclass(frozen=True, slots=True)
class Section:
    """A named region of a filing (an SEC "Item")."""

    name: str
    span: Span
    order: int = 0

    def text_of(self, document: Document) -> str:
        return document.text[self.span.start : self.span.end]


@dataclass(frozen=True, slots=True)
class Document:
    """One filing, normalized.

    `text` is the single source of truth for offsets. Sections, tables, chunks
    and gold spans all index into it, so it must not be mutated after
    construction -- hence the frozen dataclass.
    """

    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    sections: tuple[Section, ...] = ()
    tables: tuple[Table, ...] = ()
    source_url: str = ""

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    def span(self, start: int, end: int) -> Span:
        return Span(doc_id=self.doc_id, start=start, end=end)

    def slice(self, span: Span) -> str:
        if span.doc_id != self.doc_id:
            raise ValueError(f"span belongs to document {span.doc_id!r}, not {self.doc_id!r}")
        return self.text[span.start : span.end]

    def section_at(self, offset: int) -> Section | None:
        """The section containing `offset`, if any.

        Used to stamp each chunk with its section, which Phase 3 filters on.
        """
        for section in self.sections:
            if section.span.start <= offset < section.span.end:
                return section
        return None

    def tables_overlapping(self, span: Span) -> Iterator[Table]:
        for table in self.tables:
            if table.span.overlap(span) > 0:
                yield table

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "metadata": self.metadata,
            "source_url": self.source_url,
            "content_hash": self.content_hash,
            "sections": [
                {"name": s.name, "order": s.order, **s.span.to_dict()} for s in self.sections
            ],
            "tables": [
                {"rows": [list(r) for r in t.rows], "caption": t.caption, **t.span.to_dict()}
                for t in self.tables
            ],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Document:
        doc_id = raw["doc_id"]
        return cls(
            doc_id=doc_id,
            text=raw["text"],
            metadata=dict(raw.get("metadata", {})),
            source_url=raw.get("source_url", ""),
            sections=tuple(
                Section(
                    name=s["name"],
                    order=int(s.get("order", 0)),
                    span=Span(doc_id, int(s["start"]), int(s["end"])),
                )
                for s in raw.get("sections", [])
            ),
            tables=tuple(
                Table(
                    rows=tuple(tuple(c for c in row) for row in t["rows"]),
                    caption=t.get("caption", ""),
                    span=Span(doc_id, int(t["start"]), int(t["end"])),
                )
                for t in raw.get("tables", [])
            ),
        )
