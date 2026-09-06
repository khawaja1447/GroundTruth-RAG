"""Ingestion: HTML parsing, the TOC trap, tables, spans, and the EDGAR client."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gtrag.ingest.document import Document, Span, Table, normalize_whitespace, stable_id
from gtrag.ingest.edgar import (
    EdgarClient,
    EdgarError,
    HttpFetcher,
    RateLimiter,
    RecordedFetcher,
)
from gtrag.ingest.parse import find_sections, html_to_text, parse_filing

FIXTURE = Path(__file__).parent / "fixtures" / "filing_sample.html"


@pytest.fixture(scope="module")
def filing_html() -> str:
    return FIXTURE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def document(filing_html: str) -> Document:
    return parse_filing(
        filing_html,
        metadata={"company": "Northwind Logistics, Inc.", "cik": 1234567, "fiscal_year": 2024},
        source_url="https://example.invalid/nwl-10k-2024.htm",
    )


# --------------------------------------------------------------------------
# Spans
# --------------------------------------------------------------------------


class TestSpan:
    def test_overlap_basic(self):
        a = Span("d", 0, 10)
        b = Span("d", 5, 15)
        assert a.overlap(b) == 5

    def test_no_overlap(self):
        assert Span("d", 0, 5).overlap(Span("d", 10, 20)) == 0

    def test_adjacent_spans_do_not_overlap(self):
        # Half-open ranges: [0,5) and [5,10) share no character.
        assert Span("d", 0, 5).overlap(Span("d", 5, 10)) == 0

    def test_different_documents_never_overlap(self):
        assert Span("a", 0, 100).overlap(Span("b", 0, 100)) == 0

    def test_overlap_fraction_is_asymmetric(self):
        big = Span("d", 0, 100)
        small = Span("d", 10, 20)
        # The big chunk contains all of the small gold span.
        assert big.overlap_fraction(small) == pytest.approx(1.0)
        # But the small span covers only a tenth of the big one.
        assert small.overlap_fraction(big) == pytest.approx(0.1)

    def test_rejects_inverted_span(self):
        with pytest.raises(ValueError, match="precedes start"):
            Span("d", 10, 5)

    def test_rejects_negative_start(self):
        with pytest.raises(ValueError, match=">= 0"):
            Span("d", -1, 5)


class TestStableId:
    def test_deterministic(self):
        assert stable_id("a", "b") == stable_id("a", "b")

    def test_distinct_inputs_differ(self):
        assert stable_id("a", "b") != stable_id("b", "a")

    def test_separator_prevents_collision(self):
        # Without a separator, ("ab","c") and ("a","bc") would collide.
        assert stable_id("ab", "c") != stable_id("a", "bc")


class TestNormalizeWhitespace:
    def test_collapses_spaces(self):
        assert normalize_whitespace("a    b") == "a b"

    def test_collapses_excess_blank_lines(self):
        assert normalize_whitespace("a\n\n\n\n\nb") == "a\n\nb"

    def test_preserves_paragraph_breaks(self):
        assert normalize_whitespace("a\n\nb") == "a\n\nb"

    def test_handles_nbsp(self):
        assert normalize_whitespace("a  b") == "a b"

    def test_idempotent(self):
        # Load-bearing: offsets are recorded against normalized text, so a
        # second pass must not shift anything.
        once = normalize_whitespace("a  \n\n\n b c ")
        assert normalize_whitespace(once) == once


# --------------------------------------------------------------------------
# HTML extraction
# --------------------------------------------------------------------------


class TestHtmlToText:
    def test_drops_script_and_style(self, filing_html):
        text, _ = html_to_text(filing_html)
        assert "should never appear" not in text
        assert "display: none" not in text

    def test_keeps_body_prose(self, filing_html):
        text, _ = html_to_text(filing_html)
        assert "Northwind Logistics provides ground freight" in text

    def test_extracts_tables(self, filing_html):
        _, tables = html_to_text(filing_html)
        # The TOC table plus the segment-results table.
        assert len(tables) >= 2

    def test_table_rows_keep_cells_together(self, filing_html):
        _, tables = html_to_text(filing_html)
        segment = next(t for t in tables if any("Ground" in c for row in t[0] for c in row))
        rows = segment[0]
        ground = next(r for r in rows if r and r[0] == "Ground")
        assert ground[1] == "2,910"
        assert ground[3] == "+12.8%"

    def test_table_spans_point_at_linearized_text(self, filing_html):
        text, tables = html_to_text(filing_html)
        for rows, start, end in tables:
            snippet = text[start:end]
            first_cell = next((c for r in rows for c in r if c.strip()), "")
            assert first_cell in snippet

    def test_offsets_are_within_text(self, filing_html):
        text, tables = html_to_text(filing_html)
        for _, start, end in tables:
            assert 0 <= start < end <= len(text)


# --------------------------------------------------------------------------
# Sections: the table-of-contents trap
# --------------------------------------------------------------------------


class TestFindSections:
    def test_finds_the_expected_items(self, document):
        names = [s.name for s in document.sections]
        assert "Item 1 Business" in names
        assert "Item 1A Risk Factors" in names
        assert "Item 7 MD&A" in names

    def test_does_not_anchor_to_the_table_of_contents(self, document):
        """The central parsing test.

        Every Item heading appears in the TOC before it appears in the body.
        A first-match parser anchors here and produces sections a few
        characters long -- a corpus that looks fine and retrieves nothing.
        """
        business = next(s for s in document.sections if s.name == "Item 1 Business")
        body = document.slice(business.span)
        assert len(body) > 500, "section anchored to the TOC entry, not the body"
        assert "distribution centres" in body

    def test_sections_do_not_overlap(self, document):
        spans = sorted(document.sections, key=lambda s: s.span.start)
        for earlier, later in zip(spans, spans[1:], strict=False):
            assert earlier.span.end <= later.span.start

    def test_section_content_is_correctly_bounded(self, document):
        risk = next(s for s in document.sections if s.name == "Item 1A Risk Factors")
        body = document.slice(risk.span)
        assert "three largest customers accounted" in body
        # Must not bleed into the following section.
        assert "Results of Operations" not in body

    def test_mda_contains_the_revenue_figure(self, document):
        mda = next(s for s in document.sections if s.name == "Item 7 MD&A")
        assert "$4,218 million" in document.slice(mda.span)

    def test_empty_text_yields_no_sections(self):
        assert find_sections("") == []

    def test_text_without_items_yields_no_sections(self):
        assert find_sections("Just some prose with no item headings at all.") == []


# --------------------------------------------------------------------------
# Document
# --------------------------------------------------------------------------


class TestDocument:
    def test_section_at_offset(self, document):
        risk = next(s for s in document.sections if s.name == "Item 1A Risk Factors")
        found = document.section_at(risk.span.start + 10)
        assert found is not None and found.name == "Item 1A Risk Factors"

    def test_section_at_offset_outside_any_section(self, document):
        assert document.section_at(0) is None or document.section_at(0).span.start <= 0

    def test_slice_rejects_foreign_span(self, document):
        with pytest.raises(ValueError, match="belongs to document"):
            document.slice(Span("some-other-doc", 0, 10))

    def test_content_hash_is_stable(self, document):
        assert document.content_hash == document.content_hash

    def test_roundtrip_through_dict(self, document):
        restored = Document.from_dict(json.loads(json.dumps(document.to_dict())))
        assert restored.doc_id == document.doc_id
        assert restored.text == document.text
        assert [s.name for s in restored.sections] == [s.name for s in document.sections]
        assert len(restored.tables) == len(document.tables)

    def test_reparsing_gives_the_same_doc_id(self, filing_html):
        meta = {"cik": 1234567, "accession": "0001234567-24-000001", "form_type": "10-K"}
        first = parse_filing(filing_html, metadata=meta)
        second = parse_filing(filing_html, metadata=meta)
        assert first.doc_id == second.doc_id

    def test_tables_overlapping(self, document):
        if not document.tables:
            pytest.skip("fixture produced no tables")
        table = document.tables[0]
        found = list(document.tables_overlapping(table.span))
        assert table in found


class TestTable:
    def test_linearize_aligns_columns(self):
        table = Table(
            rows=(("Segment", "FY2024"), ("Ground", "2,910")),
            span=Span("d", 0, 1),
        )
        lines = table.linearize().splitlines()
        assert lines[0].startswith("Segment")
        assert "|" in lines[0]

    def test_dimensions(self):
        table = Table(rows=(("a", "b", "c"), ("d", "e")), span=Span("d", 0, 1))
        assert table.n_rows == 2
        assert table.n_cols == 3

    def test_empty_table_linearizes_to_empty(self):
        assert Table(rows=(), span=Span("d", 0, 0)).linearize() == ""


# --------------------------------------------------------------------------
# EDGAR client
# --------------------------------------------------------------------------


SUBMISSIONS = {
    "name": "NORTHWIND LOGISTICS INC",
    "filings": {
        "recent": {
            "form": ["10-K", "8-K", "10-K", "10-Q"],
            "accessionNumber": [
                "0001234567-25-000001",
                "0001234567-24-000009",
                "0001234567-24-000001",
                "0001234567-24-000005",
            ],
            "filingDate": ["2025-02-14", "2024-11-01", "2024-02-16", "2024-05-03"],
            "reportDate": ["2024-12-28", "2024-10-30", "2023-12-30", "2024-03-30"],
            "primaryDocument": [
                "nwl-10k_2024.htm",
                "nwl-8k.htm",
                "nwl-10k_2023.htm",
                "nwl-10q.htm",
            ],
        }
    },
}


@pytest.fixture
def client() -> EdgarClient:
    url = "https://data.sec.gov/submissions/CIK0001234567.json"
    return EdgarClient(RecordedFetcher({url: json.dumps(SUBMISSIONS).encode()}))


class TestEdgarClient:
    def test_lists_only_requested_forms(self, client):
        filings = client.list_filings(1234567, form_types=("10-K",))
        assert len(filings) == 2
        assert all(f.form_type == "10-K" for f in filings)

    def test_multiple_form_types(self, client):
        filings = client.list_filings(1234567, form_types=("10-K", "10-Q"))
        assert len(filings) == 3

    def test_limit(self, client):
        assert len(client.list_filings(1234567, form_types=("10-K",), limit=1)) == 1

    def test_fiscal_year_comes_from_report_date_not_filing_date(self, client):
        """A 10-K for fiscal 2024 is filed in 2025.

        Using the filing date mislabels the year on a corpus whose whole
        point includes fiscal-year disambiguation.
        """
        latest = client.list_filings(1234567, form_types=("10-K",))[0]
        assert latest.filing_date.startswith("2025")
        assert latest.fiscal_year == 2024

    def test_url_construction_strips_dashes_from_accession(self, client):
        filing = client.list_filings(1234567, form_types=("10-K",))[0]
        assert "000123456725000001" in filing.url
        assert "nwl-10k_2024.htm" in filing.url

    def test_company_name_propagated(self, client):
        assert client.list_filings(1234567)[0].company == "NORTHWIND LOGISTICS INC"

    def test_no_filings_returns_empty(self):
        url = "https://data.sec.gov/submissions/CIK0000000001.json"
        empty = EdgarClient(
            RecordedFetcher({url: json.dumps({"filings": {"recent": {}}}).encode()})
        )
        assert empty.list_filings(1) == []

    def test_invalid_json_raises_clearly(self):
        url = "https://data.sec.gov/submissions/CIK0000000001.json"
        bad = EdgarClient(RecordedFetcher({url: b"<html>not json</html>"}))
        with pytest.raises(EdgarError, match="not valid JSON"):
            bad.list_filings(1)


class TestRecordedFetcher:
    def test_strict_mode_rejects_unknown_url(self):
        with pytest.raises(EdgarError, match="no recorded response"):
            RecordedFetcher({}, strict=True).get("https://example.invalid/x")

    def test_lenient_mode_returns_empty(self):
        assert RecordedFetcher({}, strict=False).get("https://example.invalid/x") == b""


class TestUserAgentEnforcement:
    def test_rejects_missing_contact_address(self):
        with pytest.raises(EdgarError, match="contact email"):
            HttpFetcher(user_agent="python-urllib/3.11")

    def test_rejects_empty(self):
        with pytest.raises(EdgarError, match="User-Agent"):
            HttpFetcher(user_agent="")

    def test_accepts_compliant_agent(self):
        fetcher = HttpFetcher(user_agent="groundtruth-rag research someone@example.com")
        assert fetcher.limiter is not None


class TestRateLimiter:
    def test_enforces_minimum_interval(self):
        import time

        limiter = RateLimiter(rate_per_sec=50.0)  # 20ms apart
        start = time.monotonic()
        for _ in range(4):
            limiter.acquire()
        elapsed = time.monotonic() - start
        # Four acquisitions => at least three intervals of 20ms.
        assert elapsed >= 0.055

    def test_first_call_does_not_block(self):
        assert RateLimiter(rate_per_sec=1.0).acquire() == 0.0

    def test_rejects_non_positive_rate(self):
        with pytest.raises(ValueError):
            RateLimiter(rate_per_sec=0)

    def test_is_thread_safe(self):
        """A per-call sleep does not bound concurrent throughput.

        Ten threads each sleeping would still issue ten requests at once;
        only serialising the grant actually enforces the limit.
        """
        import threading
        import time

        limiter = RateLimiter(rate_per_sec=100.0)  # 10ms apart
        start = time.monotonic()
        threads = [threading.Thread(target=limiter.acquire) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.monotonic() - start
        assert elapsed >= 0.06, "concurrent acquires were not serialised"
