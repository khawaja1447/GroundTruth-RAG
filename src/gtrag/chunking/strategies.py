"""Chunking strategies for the Phase 3 ablation (dimension 1).

Five strategies competing against the Phase 1 fixed-size baseline. Each one
exists to fix a specific, nameable failure of the strategy before it:

  * `RecursiveChunker`      -- fixed-size splits mid-sentence. Split on
                               structural separators instead.
  * `SentenceWindowChunker` -- large chunks dilute the embedding. Embed one
                               sentence, return its neighbours.
  * `StructureAwareChunker` -- fixed-size splits tables in half and merges
                               unrelated sections. Respect both.
  * `ParentDocumentChunker` -- small chunks retrieve precisely but starve the
                               generator. Embed small, return the parent.
  * `SemanticChunker`       -- structural boundaries are not always topical
                               ones. Split where the meaning shifts.

Every strategy produces spans over the returned text, so the span-anchored
gold labels grade all of them from one human labeling.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..index.embed import Embedder, cosine
from ..ingest.document import Document, Span, stable_id
from ..types import Chunk
from .base import SpannedChunk, approx_tokens

__all__ = [
    "RecursiveChunker",
    "SentenceWindowChunker",
    "StructureAwareChunker",
    "ParentDocumentChunker",
    "SemanticChunker",
    "split_sentences",
    "CHUNKERS",
]

# Sentence boundaries, protecting the abbreviations and decimals that fill
# filing prose. "$1.2 million" and "Item 1A." both defeat a naive split, and a
# bad split silently changes what every downstream strategy operates on.
_PROTECT = re.compile(
    r"\b(?:Inc|Corp|Ltd|LLC|Co|No|Nos|vs|etc|approx|est|Mr|Ms|Dr|Jr|Sr|U\.S|Item|Note|Fig)\.",
    re.IGNORECASE,
)
_DECIMAL = re.compile(r"(?<=\d)\.(?=\d)")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_PLACEHOLDER = "\x00"


def split_sentences(text: str, offset: int = 0) -> list[tuple[int, int]]:
    """Sentence boundaries as `(start, end)` offsets into the original text.

    Returns offsets rather than strings so callers can build spans without
    re-locating substrings -- searching for a sentence in the document is
    both slow and ambiguous when filings repeat boilerplate verbatim.
    """
    if not text.strip():
        return []

    protected = _PROTECT.sub(lambda m: m.group(0).replace(".", _PLACEHOLDER), text)
    protected = _DECIMAL.sub(_PLACEHOLDER, protected)

    spans: list[tuple[int, int]] = []
    cursor = 0
    for match in _SENTENCE_END.finditer(protected):
        end = match.start()
        if end > cursor and text[cursor:end].strip():
            spans.append((offset + cursor, offset + end))
        cursor = match.end()
    if cursor < len(text) and text[cursor:].strip():
        spans.append((offset + cursor, offset + len(text)))
    return spans


def _make(
    document: Document,
    strategy: str,
    config_key: str,
    start: int,
    end: int,
    index: int,
    *,
    embed_text: str = "",
    extra: dict[str, Any] | None = None,
) -> SpannedChunk:
    """Build a SpannedChunk with a content-derived, collision-free id."""
    text = document.text[start:end]
    section = document.section_at(start)
    metadata: dict[str, Any] = dict(document.metadata)
    metadata.update(
        {
            "section": section.name if section else "",
            "char_start": start,
            "char_end": end,
            "chunk_index": index,
            "n_tokens": approx_tokens(text),
            "chunker": strategy,
        }
    )
    if extra:
        metadata.update(extra)
    return SpannedChunk(
        chunk=Chunk(
            chunk_id=stable_id(document.doc_id, strategy, config_key, str(start), str(end)),
            text=text,
            doc_id=document.doc_id,
            metadata=metadata,
        ),
        span=Span(document.doc_id, start, end),
        embed_text=embed_text,
    )


# --------------------------------------------------------------------------
# 1. Recursive
# --------------------------------------------------------------------------


@dataclass
class RecursiveChunker:
    """Split on structural separators, largest first, merging up to a budget.

    Fixes the fixed-size chunker's most visible flaw: cutting mid-sentence.
    Tries paragraph breaks before line breaks before sentence ends before
    words, so a chunk boundary lands at the coarsest structure that fits.
    """

    chunk_tokens: int = 512
    overlap_tokens: int = 50
    separators: tuple[str, ...] = ("\n\n", "\n", ". ", " ")
    name: str = "recursive"

    def __post_init__(self) -> None:
        if self.overlap_tokens >= self.chunk_tokens:
            raise ValueError("overlap_tokens must be less than chunk_tokens")

    @property
    def config(self) -> dict[str, Any]:
        return {
            "chunker": self.name,
            "chunk_tokens": self.chunk_tokens,
            "overlap_tokens": self.overlap_tokens,
            "separators": list(self.separators),
        }

    def _split(self, text: str, offset: int, depth: int = 0) -> list[tuple[int, int]]:
        if approx_tokens(text) <= self.chunk_tokens or depth >= len(self.separators):
            return [(offset, offset + len(text))] if text.strip() else []

        separator = self.separators[depth]
        pieces: list[tuple[int, int]] = []
        cursor = 0
        for part in text.split(separator):
            start = cursor
            end = cursor + len(part)
            if part.strip():
                pieces.extend(self._split(part, offset + start, depth + 1))
            cursor = end + len(separator)
        return pieces

    def chunk(self, document: Document) -> list[SpannedChunk]:
        pieces = self._split(document.text, 0)
        if not pieces:
            return []

        key = f"{self.chunk_tokens}-{self.overlap_tokens}"
        out: list[SpannedChunk] = []
        buffer: list[tuple[int, int]] = []
        index = 0

        def flush() -> None:
            nonlocal buffer, index
            if not buffer:
                return
            start, end = buffer[0][0], buffer[-1][1]
            if document.text[start:end].strip():
                out.append(_make(document, self.name, key, start, end, index))
                index += 1
            # Carry the tail forward as overlap, so a fact split across the
            # boundary appears whole in at least one chunk.
            carried: list[tuple[int, int]] = []
            budget = self.overlap_tokens
            for piece in reversed(buffer):
                cost = approx_tokens(document.text[piece[0] : piece[1]])
                if budget - cost < 0:
                    break
                carried.insert(0, piece)
                budget -= cost
            buffer = carried

        for piece in pieces:
            candidate = buffer + [piece]
            size = approx_tokens(document.text[candidate[0][0] : candidate[-1][1]])
            if size > self.chunk_tokens and buffer:
                flush()
                buffer = buffer + [piece] if buffer else [piece]
            else:
                buffer = candidate
        flush()
        # `flush` leaves the overlap tail in the buffer; emit it only if it
        # holds content not already covered by the last emitted chunk.
        if buffer and (not out or buffer[-1][1] > out[-1].span.end):
            start, end = buffer[0][0], buffer[-1][1]
            if document.text[start:end].strip():
                out.append(_make(document, self.name, key, start, end, index))
        return out


# --------------------------------------------------------------------------
# 2. Sentence window
# --------------------------------------------------------------------------


@dataclass
class SentenceWindowChunker:
    """Embed one sentence; return it with its neighbours.

    The bet: a single sentence embeds precisely (no dilution from surrounding
    text), while the generator still needs context to interpret it. Returning
    a window is how you get both.

    The cost is index size -- one entry per sentence rather than per 512
    tokens -- which is exactly the kind of tradeoff the ablation table should
    make visible rather than argue about.
    """

    window: int = 2
    name: str = "sentence_window"

    def __post_init__(self) -> None:
        if self.window < 0:
            raise ValueError("window must be >= 0")

    @property
    def config(self) -> dict[str, Any]:
        return {"chunker": self.name, "window": self.window}

    def chunk(self, document: Document) -> list[SpannedChunk]:
        sentences = split_sentences(document.text)
        if not sentences:
            return []

        key = str(self.window)
        out: list[SpannedChunk] = []
        for i, (start, end) in enumerate(sentences):
            low = max(0, i - self.window)
            high = min(len(sentences) - 1, i + self.window)
            window_start = sentences[low][0]
            window_end = sentences[high][1]
            if not document.text[window_start:window_end].strip():
                continue
            out.append(
                _make(
                    document,
                    self.name,
                    key,
                    window_start,
                    window_end,
                    len(out),
                    # The embedded string is the focus sentence alone.
                    embed_text=document.text[start:end],
                    extra={"focus_start": start, "focus_end": end},
                )
            )
        return out


# --------------------------------------------------------------------------
# 3. Structure aware
# --------------------------------------------------------------------------


@dataclass
class StructureAwareChunker:
    """Respect section boundaries and never split a table.

    The strategy this corpus was chosen to reward. Two rules:

      1. **A chunk never crosses a section boundary.** Merging the tail of
         Risk Factors with the head of MD&A produces a chunk that is about
         neither, and it retrieves for queries about both.
      2. **A table is atomic.** A financial table cut in half is the single
         largest source of retrieval failure on filings: the row labels end
         up in one chunk and the figures in another, and neither is
         answerable.

    An oversized table becomes its own chunk rather than being split. That is
    a deliberate budget violation -- half a table is worth less than an
    over-long one.
    """

    chunk_tokens: int = 512
    overlap_tokens: int = 50
    name: str = "structure_aware"

    def __post_init__(self) -> None:
        if self.overlap_tokens >= self.chunk_tokens:
            raise ValueError("overlap_tokens must be less than chunk_tokens")

    @property
    def config(self) -> dict[str, Any]:
        return {
            "chunker": self.name,
            "chunk_tokens": self.chunk_tokens,
            "overlap_tokens": self.overlap_tokens,
        }

    def _regions(self, document: Document) -> list[tuple[int, int]]:
        """Top-level regions: the sections, or the whole document if none."""
        if not document.sections:
            return [(0, len(document.text))]
        regions = [(s.span.start, s.span.end) for s in document.sections]
        first = regions[0][0]
        if first > 0:
            regions.insert(0, (0, first))
        return regions

    def _units(self, document: Document, start: int, end: int) -> list[tuple[int, int, bool]]:
        """Split a region into atomic units: `(start, end, is_table)`.

        Tables are lifted out whole first, and the prose between them is
        split at paragraph boundaries.
        """
        tables = sorted(
            (t for t in document.tables if t.span.start >= start and t.span.end <= end),
            key=lambda t: t.span.start,
        )
        units: list[tuple[int, int, bool]] = []
        cursor = start
        for table in tables:
            if table.span.start > cursor:
                units.extend(self._paragraphs(document, cursor, table.span.start))
            units.append((table.span.start, table.span.end, True))
            cursor = table.span.end
        if cursor < end:
            units.extend(self._paragraphs(document, cursor, end))
        return units

    def _paragraphs(self, document: Document, start: int, end: int) -> list[tuple[int, int, bool]]:
        text = document.text[start:end]
        out: list[tuple[int, int, bool]] = []
        cursor = 0
        for part in text.split("\n\n"):
            if part.strip():
                out.append((start + cursor, start + cursor + len(part), False))
            cursor += len(part) + 2
        return out

    def chunk(self, document: Document) -> list[SpannedChunk]:
        key = f"{self.chunk_tokens}-{self.overlap_tokens}"
        out: list[SpannedChunk] = []

        for region_start, region_end in self._regions(document):
            units = self._units(document, region_start, region_end)
            buffer: list[tuple[int, int, bool]] = []

            def flush(buf: list[tuple[int, int, bool]]) -> None:
                if not buf:
                    return
                start, end = buf[0][0], buf[-1][1]
                if document.text[start:end].strip():
                    out.append(
                        _make(
                            document,
                            self.name,
                            key,
                            start,
                            end,
                            len(out),
                            extra={"has_table": any(u[2] for u in buf)},
                        )
                    )

            for unit in units:
                unit_tokens = approx_tokens(document.text[unit[0] : unit[1]])

                # An oversized table is emitted alone rather than split.
                if unit[2] and unit_tokens > self.chunk_tokens:
                    flush(buffer)
                    buffer = []
                    flush([unit])
                    continue

                candidate = buffer + [unit]
                size = approx_tokens(document.text[candidate[0][0] : candidate[-1][1]])
                if size > self.chunk_tokens and buffer:
                    flush(buffer)
                    buffer = [unit]
                else:
                    buffer = candidate
            flush(buffer)

        return out


# --------------------------------------------------------------------------
# 4. Parent document
# --------------------------------------------------------------------------


@dataclass
class ParentDocumentChunker:
    """Embed a small child; return its parent.

    Small chunks retrieve precisely and starve the generator; large chunks
    feed the generator and retrieve imprecisely. This takes both: the index
    holds fine-grained vectors, retrieval returns the coarse parent.

    Distinct from sentence-window only in the parent's construction --
    sections here rather than a fixed neighbour count -- which is the sort of
    near-duplicate pair worth measuring rather than reasoning about.
    """

    child_tokens: int = 128
    parent_tokens: int = 1024
    name: str = "parent_document"

    def __post_init__(self) -> None:
        if self.child_tokens >= self.parent_tokens:
            raise ValueError("child_tokens must be smaller than parent_tokens")

    @property
    def config(self) -> dict[str, Any]:
        return {
            "chunker": self.name,
            "child_tokens": self.child_tokens,
            "parent_tokens": self.parent_tokens,
        }

    def chunk(self, document: Document) -> list[SpannedChunk]:
        parents = StructureAwareChunker(chunk_tokens=self.parent_tokens, overlap_tokens=0).chunk(
            document
        )
        if not parents:
            return []

        key = f"{self.child_tokens}-{self.parent_tokens}"
        out: list[SpannedChunk] = []
        for parent in parents:
            sentences = split_sentences(
                document.text[parent.span.start : parent.span.end], offset=parent.span.start
            )
            if not sentences:
                continue

            # Group sentences into child-sized units, each embedded on its
            # own but all returning the same parent span.
            group: list[tuple[int, int]] = []
            groups: list[list[tuple[int, int]]] = []
            for sentence in sentences:
                candidate = group + [sentence]
                size = approx_tokens(document.text[candidate[0][0] : candidate[-1][1]])
                if size > self.child_tokens and group:
                    groups.append(group)
                    group = [sentence]
                else:
                    group = candidate
            if group:
                groups.append(group)

            for child in groups:
                child_start, child_end = child[0][0], child[-1][1]
                out.append(
                    SpannedChunk(
                        chunk=Chunk(
                            # Id keyed on the child, since two children of the
                            # same parent must be distinct index entries.
                            chunk_id=stable_id(
                                document.doc_id, self.name, key, str(child_start), str(child_end)
                            ),
                            text=parent.chunk.text,
                            doc_id=document.doc_id,
                            metadata={
                                **parent.chunk.metadata,
                                "chunker": self.name,
                                "child_start": child_start,
                                "child_end": child_end,
                                "chunk_index": len(out),
                            },
                        ),
                        span=parent.span,
                        embed_text=document.text[child_start:child_end],
                    )
                )
        return out


# --------------------------------------------------------------------------
# 5. Semantic
# --------------------------------------------------------------------------


@dataclass
class SemanticChunker:
    """Split where consecutive sentences stop being about the same thing.

    Embeds each sentence, walks the sequence, and cuts where similarity to
    the running context drops below a percentile threshold.

    **Only meaningful with a semantic embedder.** Run with `HashingEmbedder`
    it splits on vocabulary overlap, not meaning, and the resulting number
    measures the embedder rather than the strategy. The config records which
    embedder produced it so a run made that way is identifiable in the
    ablation table rather than quietly comparable.
    """

    embedder: Embedder = field(default=None)  # type: ignore[assignment]
    percentile: float = 25.0
    max_tokens: int = 1024
    name: str = "semantic"

    def __post_init__(self) -> None:
        if self.embedder is None:
            from ..index.embed import HashingEmbedder

            self.embedder = HashingEmbedder()
        if not 0.0 < self.percentile < 100.0:
            raise ValueError("percentile must be in (0, 100)")

    @property
    def config(self) -> dict[str, Any]:
        return {
            "chunker": self.name,
            "percentile": self.percentile,
            "max_tokens": self.max_tokens,
            "split_embedder": self.embedder.config.get("embedder", "?"),
            "split_embedder_semantic": self.embedder.config.get("semantic", False),
        }

    def chunk(self, document: Document) -> list[SpannedChunk]:
        sentences = split_sentences(document.text)
        if len(sentences) < 2:
            return (
                [_make(document, self.name, str(self.percentile), 0, len(document.text), 0)]
                if document.text.strip()
                else []
            )

        vectors = self.embedder.embed([document.text[s:e] for s, e in sentences])
        similarities = [cosine(vectors[i], vectors[i + 1]) for i in range(len(vectors) - 1)]

        ordered = sorted(similarities)
        cut = ordered[min(len(ordered) - 1, int(len(ordered) * self.percentile / 100.0))]

        key = str(self.percentile)
        out: list[SpannedChunk] = []
        group: list[tuple[int, int]] = [sentences[0]]

        for i in range(1, len(sentences)):
            size = approx_tokens(document.text[group[0][0] : sentences[i][1]])
            boundary = similarities[i - 1] <= cut or size > self.max_tokens
            if boundary and group:
                start, end = group[0][0], group[-1][1]
                if document.text[start:end].strip():
                    out.append(_make(document, self.name, key, start, end, len(out)))
                group = [sentences[i]]
            else:
                group.append(sentences[i])

        if group:
            start, end = group[0][0], group[-1][1]
            if document.text[start:end].strip():
                out.append(_make(document, self.name, key, start, end, len(out)))
        return out


# Registry for the sweep runner, so a config file names a strategy by string.
CHUNKERS: dict[str, Any] = {
    "recursive": RecursiveChunker,
    "sentence_window": SentenceWindowChunker,
    "structure_aware": StructureAwareChunker,
    "parent_document": ParentDocumentChunker,
    "semantic": SemanticChunker,
}


def build_chunker(name: str, **kwargs: Any) -> Any:
    """Construct a chunker by name, including the Phase 1 baseline."""
    from .base import FixedTokenChunker

    if name == "fixed":
        return FixedTokenChunker(**kwargs)
    if name not in CHUNKERS:
        raise ValueError(f"unknown chunker {name!r} (have: fixed, {', '.join(sorted(CHUNKERS))})")
    return CHUNKERS[name](**kwargs)


def all_chunker_names() -> Sequence[str]:
    return ("fixed", *sorted(CHUNKERS))
