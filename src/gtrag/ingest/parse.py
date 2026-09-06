"""Filing HTML -> normalized text, sections, and tables.

Three problems this module exists to solve, in increasing order of how much
they cost you if you get them wrong:

1. **HTML to text without losing table structure.** A generic tag-stripper
   turns a financial table into a stream of unlabeled numbers, which is
   unretrievable and unreadable. Tables are extracted structurally first,
   then linearized in place.

2. **The table-of-contents trap.** "Item 1A. Risk Factors" appears at least
   twice in every 10-K: once in the TOC near the top, once at the real
   section. A naive first-match parser anchors every section to the TOC and
   produces sections a few characters long. The symptom is not a crash --
   it is a corpus that looks fine and retrieves nothing.

3. **Offsets must survive.** Every span in the system indexes into the
   normalized text, so normalization happens exactly once, before any offset
   is recorded.

Uses only `html.parser` from the stdlib. BeautifulSoup would be marginally
more robust on malformed markup, but the zero-dependency core is worth more
than the margin, and filing HTML is machine-generated and well-formed enough.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

from .document import Document, Section, Span, Table, normalize_whitespace, stable_id

__all__ = ["html_to_text", "find_sections", "parse_filing", "ITEM_PATTERNS", "TextExtractor"]

# Tags whose content is never body text.
_SKIP_CONTENT = frozenset({"script", "style", "head", "title", "meta", "link"})
# Tags that imply a line break when they close.
_BLOCK = frozenset(
    {
        "p",
        "div",
        "br",
        "tr",
        "li",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "table",
        "thead",
        "tbody",
        "section",
        "article",
        "hr",
    }
)


@dataclass
class _RawTable:
    rows: list[list[str]] = field(default_factory=list)
    current: list[str] = field(default_factory=list)
    cell: list[str] = field(default_factory=list)
    start_offset: int = 0


class TextExtractor(HTMLParser):
    """Streams HTML to text while recording where each table landed.

    Table spans are recorded against the *output* text as it is built, so a
    table's span points at its linearized form in the final document. That is
    what lets the structure-aware chunker in Phase 3 refuse to split it.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._out: list[str] = []
        self._len = 0
        self._skip_depth = 0
        self._table_stack: list[_RawTable] = []
        self.tables: list[tuple[list[list[str]], int, int]] = []

    # -- helpers ----------------------------------------------------------

    def _emit(self, text: str) -> None:
        if not text:
            return
        self._out.append(text)
        self._len += len(text)

    def _emit_break(self) -> None:
        if self._out and not self._out[-1].endswith("\n"):
            self._emit("\n")

    # -- HTMLParser hooks -------------------------------------------------

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_CONTENT:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return

        if tag == "table":
            self._emit_break()
            self._table_stack.append(_RawTable(start_offset=self._len))
        elif tag == "tr" and self._table_stack:
            self._table_stack[-1].current = []
        elif tag in ("td", "th") and self._table_stack:
            self._table_stack[-1].cell = []
        elif tag in _BLOCK:
            self._emit_break()

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_CONTENT:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return

        if tag in ("td", "th") and self._table_stack:
            table = self._table_stack[-1]
            table.current.append(normalize_whitespace("".join(table.cell)))
            table.cell = []
        elif tag == "tr" and self._table_stack:
            table = self._table_stack[-1]
            # Drop rows that are entirely empty -- filings use spacer rows
            # heavily for layout, and they add noise without information.
            if any(c.strip() for c in table.current):
                table.rows.append(list(table.current))
            table.current = []
        elif tag == "table" and self._table_stack:
            table = self._table_stack.pop()
            if table.rows:
                linearized = _linearize_rows(table.rows)
                start = self._len
                self._emit(linearized)
                self._emit("\n")
                self.tables.append((table.rows, start, start + len(linearized)))
        elif tag in _BLOCK:
            self._emit_break()

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._table_stack:
            # Inside a table, text belongs to the current cell, not the body.
            self._table_stack[-1].cell.append(data)
            return
        self._emit(data)

    def get_text(self) -> str:
        return "".join(self._out)


def _linearize_rows(rows: list[list[str]]) -> str:
    width = max((len(r) for r in rows), default=0)
    padded = [r + [""] * (width - len(r)) for r in rows]
    col_widths = [max(len(row[i]) for row in padded) for i in range(width)] if width else []
    return "\n".join(
        " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row)).rstrip()
        for row in padded
    )


def html_to_text(html: str) -> tuple[str, list[tuple[list[list[str]], int, int]]]:
    """Convert filing HTML to normalized text plus table spans.

    Returns `(text, [(rows, start, end), ...])` where offsets index into the
    returned text. Whitespace normalization is applied to the *assembled*
    string and the table offsets are re-derived from it, so both stay
    consistent -- normalizing first and hoping offsets survive is the bug
    this ordering avoids.
    """
    extractor = TextExtractor()
    extractor.feed(html)
    extractor.close()
    raw = extractor.get_text()

    # Re-locate tables after normalization by searching for their linearized
    # form. Whitespace collapsing shifts every offset, and a table's rendered
    # text is distinctive enough to relocate unambiguously.
    text = normalize_whitespace(raw)
    tables: list[tuple[list[list[str]], int, int]] = []
    cursor = 0
    for rows, _, _ in extractor.tables:
        needle = normalize_whitespace(_linearize_rows(rows))
        if not needle:
            continue
        idx = text.find(needle, cursor)
        if idx == -1:
            idx = text.find(needle)
        if idx == -1:
            continue
        tables.append((rows, idx, idx + len(needle)))
        cursor = idx + len(needle)
    return text, tables


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------

# The SEC items worth separating in a 10-K. Ordered as they appear in the
# filing, which the boundary logic relies on.
ITEM_PATTERNS: tuple[tuple[str, str], ...] = (
    ("Item 1 Business", r"item\s*1\s*[.\-–—:]?\s*business"),
    ("Item 1A Risk Factors", r"item\s*1a\s*[.\-–—:]?\s*risk\s*factors"),
    ("Item 1B Unresolved Staff Comments", r"item\s*1b\s*[.\-–—:]?\s*unresolved"),
    ("Item 2 Properties", r"item\s*2\s*[.\-–—:]?\s*propert"),
    ("Item 3 Legal Proceedings", r"item\s*3\s*[.\-–—:]?\s*legal"),
    ("Item 5 Market for Common Equity", r"item\s*5\s*[.\-–—:]?\s*market\s*for"),
    ("Item 7 MD&A", r"item\s*7\s*[.\-–—:]?\s*management.{0,5}s\s*discussion"),
    ("Item 7A Market Risk", r"item\s*7a\s*[.\-–—:]?\s*quantitative"),
    ("Item 8 Financial Statements", r"item\s*8\s*[.\-–—:]?\s*financial\s*statements"),
    ("Item 9A Controls and Procedures", r"item\s*9a\s*[.\-–—:]?\s*controls"),
)

# A real section heading is followed by substantive text. The TOC entry is
# followed by a page number and the next TOC line. This is the threshold that
# separates them.
MIN_SECTION_CHARS = 500


def find_sections(text: str) -> list[Section]:
    """Locate SEC item sections in normalized filing text.

    The TOC problem is handled by taking, for each item, the **last** match
    that is followed by enough text to be a real section. Last rather than
    first because the TOC always precedes the body; "enough text" because a
    filing that genuinely omits an item (Item 1B is often "None.") must not
    swallow the rest of the document.
    """
    candidates: dict[str, list[int]] = {}
    for name, pattern in ITEM_PATTERNS:
        matches = [m.start() for m in re.finditer(pattern, text, re.IGNORECASE)]
        if matches:
            candidates[name] = matches

    if not candidates:
        return []

    # Choose one offset per item: the last candidate, since the body follows
    # the table of contents.
    chosen: list[tuple[str, int, int]] = []
    for order, (name, _) in enumerate(ITEM_PATTERNS):
        if name not in candidates:
            continue
        chosen.append((name, candidates[name][-1], order))
    chosen.sort(key=lambda t: t[1])

    sections: list[Section] = []
    for i, (name, start, order) in enumerate(chosen):
        end = chosen[i + 1][1] if i + 1 < len(chosen) else len(text)
        if end - start < MIN_SECTION_CHARS and i + 1 < len(chosen):
            # Too short to be a real section: almost certainly a stray TOC
            # match that survived, or a genuinely empty item. Either way it
            # is not worth a section boundary.
            continue
        sections.append(Section(name=name, span=Span("", start, end), order=order))
    return sections


def parse_filing(
    html: str,
    *,
    doc_id: str | None = None,
    metadata: dict | None = None,
    source_url: str = "",
) -> Document:
    """Parse filing HTML into a `Document` with sections and tables."""
    text, raw_tables = html_to_text(html)
    metadata = dict(metadata or {})

    if doc_id is None:
        doc_id = stable_id(
            str(metadata.get("cik", "")),
            str(metadata.get("accession", "")),
            str(metadata.get("form_type", "")),
        )

    sections = tuple(
        Section(name=s.name, order=s.order, span=Span(doc_id, s.span.start, s.span.end))
        for s in find_sections(text)
    )
    tables = tuple(
        Table(
            rows=tuple(tuple(cell for cell in row) for row in rows),
            span=Span(doc_id, start, end),
        )
        for rows, start, end in raw_tables
    )
    return Document(
        doc_id=doc_id,
        text=text,
        metadata=metadata,
        sections=sections,
        tables=tables,
        source_url=source_url,
    )
