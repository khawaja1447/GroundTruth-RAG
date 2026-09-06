"""A tiny synthetic filing-shaped corpus.

**These are invented companies and invented figures.** Nothing here is real
financial data and none of it should be presented as such. The corpus exists
so the eval harness has something to run against before Phase 1 delivers a
real EDGAR ingestion, and so the test suite has fixed, hand-checkable inputs.

It is deliberately shaped like the real thing in the ways that matter for
retrieval: two similar companies with overlapping vocabulary, two fiscal
years with near-identical boilerplate, figures that live in tables, and a
footnote that qualifies the number above it.
"""

from __future__ import annotations

from ..types import Chunk

__all__ = ["FIXTURE_CHUNKS", "chunk_ids", "by_id"]


FIXTURE_CHUNKS: tuple[Chunk, ...] = (
    Chunk(
        chunk_id="nwl-2024-item7-001",
        doc_id="nwl-10k-2024",
        metadata={
            "company": "Northwind Logistics, Inc.",
            "ticker": "NWL",
            "fiscal_year": 2024,
            "form_type": "10-K",
            "section": "Item 7 MD&A",
        },
        text=(
            "Results of Operations. Total net revenue for fiscal 2024 was $4,218 million, "
            "an increase of 11.0% compared with $3,800 million in fiscal 2023. The increase "
            "was driven primarily by higher freight volumes in the Ground segment and by "
            "contractual rate escalators that took effect in the second quarter."
        ),
    ),
    Chunk(
        chunk_id="nwl-2024-item7-002",
        doc_id="nwl-10k-2024",
        metadata={
            "company": "Northwind Logistics, Inc.",
            "ticker": "NWL",
            "fiscal_year": 2024,
            "form_type": "10-K",
            "section": "Item 7 MD&A",
            "contains_table": True,
        },
        text=(
            "Segment results (in millions):\n"
            "Segment        | FY2024  | FY2023  | Change\n"
            "Ground         | 2,910   | 2,580   | +12.8%\n"
            "Air Freight    |   884   |   842   |  +5.0%\n"
            "Warehousing    |   424   |   378   | +12.2%\n"
            "Total          | 4,218   | 3,800   | +11.0%"
        ),
    ),
    Chunk(
        chunk_id="nwl-2024-item7-003",
        doc_id="nwl-10k-2024",
        metadata={
            "company": "Northwind Logistics, Inc.",
            "ticker": "NWL",
            "fiscal_year": 2024,
            "form_type": "10-K",
            "section": "Item 7 MD&A",
            "is_footnote": True,
        },
        text=(
            "(1) Warehousing segment revenue for fiscal 2024 includes $31 million of "
            "non-recurring termination fees related to the exit of two leased facilities "
            "in the Midwest. Excluding these fees, Warehousing revenue grew 4.0%."
        ),
    ),
    Chunk(
        chunk_id="nwl-2023-item7-001",
        doc_id="nwl-10k-2023",
        metadata={
            "company": "Northwind Logistics, Inc.",
            "ticker": "NWL",
            "fiscal_year": 2023,
            "form_type": "10-K",
            "section": "Item 7 MD&A",
        },
        text=(
            "Results of Operations. Total net revenue for fiscal 2023 was $3,800 million, "
            "an increase of 3.2% compared with $3,682 million in fiscal 2022. Growth was "
            "constrained by softer industrial demand in the first half of the year."
        ),
    ),
    Chunk(
        chunk_id="nwl-2024-item1a-001",
        doc_id="nwl-10k-2024",
        metadata={
            "company": "Northwind Logistics, Inc.",
            "ticker": "NWL",
            "fiscal_year": 2024,
            "form_type": "10-K",
            "section": "Item 1A Risk Factors",
        },
        text=(
            "Risk Factors. A significant portion of our Ground segment revenue is "
            "concentrated among a small number of customers. In fiscal 2024, our three "
            "largest customers accounted for approximately 28% of consolidated net revenue. "
            "The loss of any one of these customers could materially affect our results."
        ),
    ),
    Chunk(
        chunk_id="csc-2024-item7-001",
        doc_id="csc-10k-2024",
        metadata={
            "company": "Cascade Semiconductor Corp.",
            "ticker": "CSC",
            "fiscal_year": 2024,
            "form_type": "10-K",
            "section": "Item 7 MD&A",
        },
        text=(
            "Results of Operations. Total net revenue for fiscal 2024 was $2,640 million, "
            "a decrease of 6.4% compared with $2,821 million in fiscal 2023. The decline "
            "reflected inventory correction among distribution partners in the "
            "industrial end market."
        ),
    ),
    Chunk(
        chunk_id="csc-2024-item7-002",
        doc_id="csc-10k-2024",
        metadata={
            "company": "Cascade Semiconductor Corp.",
            "ticker": "CSC",
            "fiscal_year": 2024,
            "form_type": "10-K",
            "section": "Item 7 MD&A",
            "contains_table": True,
        },
        text=(
            "Selected financial data (in millions, except percentages):\n"
            "Measure            | FY2024  | FY2023\n"
            "Net revenue        | 2,640   | 2,821\n"
            "Gross profit       | 1,113   | 1,254\n"
            "Gross margin       |  42.1%  |  44.5%\n"
            "Research and dev.  |   488   |   455"
        ),
    ),
    Chunk(
        chunk_id="csc-2024-item1-001",
        doc_id="csc-10k-2024",
        metadata={
            "company": "Cascade Semiconductor Corp.",
            "ticker": "CSC",
            "fiscal_year": 2024,
            "form_type": "10-K",
            "section": "Item 1 Business",
        },
        text=(
            "Business. Cascade Semiconductor designs and sells analog and mixed-signal "
            "integrated circuits for industrial automation, automotive, and energy "
            "infrastructure customers. We operate one wafer fabrication facility in "
            "Oregon and rely on third-party foundries for advanced process nodes."
        ),
    ),
)


def chunk_ids() -> list[str]:
    return [c.chunk_id for c in FIXTURE_CHUNKS]


def by_id(chunk_id: str) -> Chunk:
    for c in FIXTURE_CHUNKS:
        if c.chunk_id == chunk_id:
            return c
    raise KeyError(f"no fixture chunk {chunk_id!r}")
