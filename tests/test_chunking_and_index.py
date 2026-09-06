"""Chunking, span-anchored relevance resolution, the index, and the baseline.

The span-resolution tests carry the most weight here: they are what prove the
Phase 3 chunking ablation is possible, by showing one human labeling survives
a change of chunking strategy.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from evals.spans import (
    GoldSpan,
    coverage_report,
    resolve_relevance,
)

from gtrag.baseline import build_baseline
from gtrag.chunking.base import FixedTokenChunker, SpannedChunk, approx_tokens
from gtrag.generate.generator import ExtractiveGenerator, format_passages
from gtrag.index.embed import HashingEmbedder, cosine, l2_normalize
from gtrag.index.store import VectorIndex
from gtrag.ingest.document import Span
from gtrag.ingest.parse import parse_filing
from gtrag.types import Chunk, RetrievedChunk

FIXTURE = Path(__file__).parent / "fixtures" / "filing_sample.html"


@pytest.fixture(scope="module")
def document():
    return parse_filing(
        FIXTURE.read_text(encoding="utf-8"),
        metadata={"company": "Northwind Logistics, Inc.", "fiscal_year": 2024},
    )


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------


class TestFixedTokenChunker:
    def test_produces_chunks(self, document):
        chunks = FixedTokenChunker(chunk_tokens=100, overlap_tokens=10).chunk(document)
        assert len(chunks) > 1

    def test_chunk_ids_are_unique(self, document):
        chunks = FixedTokenChunker(chunk_tokens=80, overlap_tokens=10).chunk(document)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_deterministic_across_runs(self, document):
        a = FixedTokenChunker(chunk_tokens=100, overlap_tokens=10).chunk(document)
        b = FixedTokenChunker(chunk_tokens=100, overlap_tokens=10).chunk(document)
        assert [c.chunk_id for c in a] == [c.chunk_id for c in b]

    def test_different_config_gives_different_ids(self, document):
        a = FixedTokenChunker(chunk_tokens=100).chunk(document)
        b = FixedTokenChunker(chunk_tokens=200).chunk(document)
        assert set(c.chunk_id for c in a).isdisjoint(c.chunk_id for c in b)

    def test_spans_point_at_the_chunk_text(self, document):
        for chunk in FixedTokenChunker(chunk_tokens=60, overlap_tokens=5).chunk(document):
            assert document.slice(chunk.span) == chunk.chunk.text

    def test_spans_are_ordered_and_within_document(self, document):
        chunks = FixedTokenChunker(chunk_tokens=60, overlap_tokens=5).chunk(document)
        for chunk in chunks:
            assert 0 <= chunk.span.start < chunk.span.end <= len(document.text)
        starts = [c.span.start for c in chunks]
        assert starts == sorted(starts)

    def test_chunks_cover_the_document(self, document):
        # With overlap, consecutive chunks must not leave a gap -- a gap is
        # text that can never be retrieved.
        chunks = FixedTokenChunker(chunk_tokens=60, overlap_tokens=10).chunk(document)
        for earlier, later in zip(chunks, chunks[1:], strict=False):
            assert later.span.start <= earlier.span.end

    def test_overlap_is_real(self, document):
        chunks = FixedTokenChunker(chunk_tokens=60, overlap_tokens=20).chunk(document)
        assert any(c2.span.start < c1.span.end for c1, c2 in zip(chunks, chunks[1:], strict=False))

    def test_chunks_carry_section_metadata(self, document):
        chunks = FixedTokenChunker(chunk_tokens=60, overlap_tokens=5).chunk(document)
        assert any(c.chunk.metadata.get("section") for c in chunks)

    def test_rejects_overlap_at_or_above_chunk_size(self):
        # Otherwise the window never advances and chunking hangs.
        with pytest.raises(ValueError, match="never advances"):
            FixedTokenChunker(chunk_tokens=100, overlap_tokens=100)

    def test_rejects_nonpositive_size(self):
        with pytest.raises(ValueError):
            FixedTokenChunker(chunk_tokens=0)

    def test_empty_document_yields_nothing(self, document):
        from gtrag.ingest.document import Document

        empty = Document(doc_id="empty", text="")
        assert FixedTokenChunker().chunk(empty) == []

    def test_approx_tokens(self):
        assert approx_tokens("hello world") == 2
        assert approx_tokens("") == 0
        assert approx_tokens("a, b.") == 4  # two words, two punctuation marks


# --------------------------------------------------------------------------
# Span resolution -- the Phase 1 <-> Phase 3 bridge
# --------------------------------------------------------------------------


def _chunk(doc_id: str, cid: str, start: int, end: int) -> SpannedChunk:
    return SpannedChunk(
        chunk=Chunk(chunk_id=cid, text="x" * (end - start), doc_id=doc_id),
        span=Span(doc_id, start, end),
    )


class TestResolveRelevance:
    def test_full_containment_is_answer_bearing(self):
        gold = [GoldSpan(Span("d", 100, 150))]
        chunks = [_chunk("d", "c1", 0, 200)]
        assert resolve_relevance(gold, chunks) == {"c1": 2}

    def test_no_overlap_is_unlabeled(self):
        gold = [GoldSpan(Span("d", 100, 150))]
        chunks = [_chunk("d", "c1", 0, 50)]
        assert resolve_relevance(gold, chunks) == {}

    def test_partial_overlap_is_supporting(self):
        # Chunk covers half the gold span: enough to help, not to answer.
        gold = [GoldSpan(Span("d", 100, 200))]
        chunks = [_chunk("d", "c1", 0, 150)]
        assert resolve_relevance(gold, chunks) == {"c1": 1}

    def test_tiny_overlap_is_ignored(self):
        # 5% of the span -- below PARTIAL_COVERAGE, so noise rather than context.
        gold = [GoldSpan(Span("d", 100, 200))]
        chunks = [_chunk("d", "c1", 0, 105)]
        assert resolve_relevance(gold, chunks) == {}

    def test_a_large_chunk_is_not_penalised_for_being_large(self):
        """The asymmetry that makes the chunking ablation fair.

        A 10,000-char chunk containing a 50-char answer is a retrieval
        success. Grading on how much of the *chunk* is gold would report
        0.005 and make large-chunk strategies look terrible for the wrong
        reason.
        """
        gold = [GoldSpan(Span("d", 5000, 5050))]
        assert resolve_relevance(gold, [_chunk("d", "big", 0, 10000)]) == {"big": 2}

    def test_supporting_weight_caps_relevance_at_one(self):
        gold = [GoldSpan(Span("d", 100, 150), weight=0.5)]
        assert resolve_relevance(gold, [_chunk("d", "c1", 0, 200)]) == {"c1": 1}

    def test_chunk_takes_its_highest_earned_relevance(self):
        gold = [
            GoldSpan(Span("d", 10, 20), weight=0.5),
            GoldSpan(Span("d", 30, 40), weight=1.0),
        ]
        assert resolve_relevance(gold, [_chunk("d", "c1", 0, 100)]) == {"c1": 2}

    def test_spans_in_other_documents_are_ignored(self):
        gold = [GoldSpan(Span("other", 0, 100))]
        assert resolve_relevance(gold, [_chunk("d", "c1", 0, 100)]) == {}

    def test_rejects_inverted_thresholds(self):
        with pytest.raises(ValueError, match="must exceed"):
            resolve_relevance([], [], full_coverage=0.1, partial_coverage=0.9)

    def test_weight_must_be_in_unit_interval(self):
        with pytest.raises(ValueError, match="weight must be"):
            GoldSpan(Span("d", 0, 10), weight=0.0)


class TestLabelsSurviveRechunking:
    """The property the whole span design exists to provide."""

    def test_same_labels_resolve_under_two_chunkings(self, document):
        # Anchor gold evidence at the revenue sentence, once.
        idx = document.text.index("$4,218 million")
        gold = [GoldSpan(Span(document.doc_id, idx - 60, idx + 40))]

        small = FixedTokenChunker(chunk_tokens=64, overlap_tokens=8).chunk(document)
        large = FixedTokenChunker(chunk_tokens=256, overlap_tokens=32).chunk(document)

        under_small = resolve_relevance(gold, small)
        under_large = resolve_relevance(gold, large)

        # Both chunkings find the evidence; the chunk ids differ entirely.
        assert under_small, "evidence lost under small chunking"
        assert under_large, "evidence lost under large chunking"
        assert set(under_small).isdisjoint(under_large)

    def test_resolved_chunks_actually_contain_the_evidence(self, document):
        idx = document.text.index("$4,218 million")
        gold = [GoldSpan(Span(document.doc_id, idx, idx + 14))]
        chunks = FixedTokenChunker(chunk_tokens=128, overlap_tokens=16).chunk(document)
        for chunk_id, relevance in resolve_relevance(gold, chunks).items():
            if relevance == 2:
                chunk = next(c for c in chunks if c.chunk_id == chunk_id)
                assert "$4,218 million" in chunk.chunk.text

    def test_coverage_report_flags_lost_evidence(self, document):
        chunks = FixedTokenChunker(chunk_tokens=128).chunk(document)
        spans = {
            "q-present": [GoldSpan(Span(document.doc_id, 100, 200))],
            "q-absent": [GoldSpan(Span("some-other-doc", 0, 100))],
        }
        report = coverage_report(spans, chunks)
        assert report["lost"] == ["q-absent"]
        assert report["resolved"] == 1


# --------------------------------------------------------------------------
# Embedding and index
# --------------------------------------------------------------------------


class TestHashingEmbedder:
    def test_deterministic(self):
        e = HashingEmbedder(dimension=64)
        assert e.embed_query("net revenue") == e.embed_query("net revenue")

    def test_output_is_normalized(self):
        vector = HashingEmbedder(dimension=64).embed_query("total net revenue fiscal 2024")
        assert cosine(vector, vector) == pytest.approx(1.0, abs=1e-9)

    def test_similar_text_scores_higher_than_unrelated(self):
        e = HashingEmbedder(dimension=512)
        query = e.embed_query("total net revenue fiscal 2024")
        near = e.embed(["Total net revenue for fiscal 2024 was $4,218 million."])[0]
        far = e.embed(["We operate one wafer fabrication facility in Oregon."])[0]
        assert cosine(query, near) > cosine(query, far)

    def test_declares_itself_non_semantic(self):
        # Honesty guard: a run made with this embedder must never be
        # mistaken for a semantic-retrieval result.
        assert HashingEmbedder().config["semantic"] is False

    def test_empty_text_does_not_crash(self):
        assert len(HashingEmbedder(dimension=32).embed_query("")) == 32

    def test_dimension_respected(self):
        assert len(HashingEmbedder(dimension=128).embed_query("x")) == 128

    def test_l2_normalize_handles_zero_vector(self):
        assert l2_normalize([0.0, 0.0]) == [0.0, 0.0]

    def test_cosine_rejects_dimension_mismatch(self):
        with pytest.raises(ValueError, match="dimension mismatch"):
            cosine([1.0], [1.0, 2.0])


class TestVectorIndex:
    @pytest.fixture
    def index(self, document):
        idx = VectorIndex(embedder=HashingEmbedder(dimension=512))
        idx.add(FixedTokenChunker(chunk_tokens=80, overlap_tokens=10).chunk(document))
        return idx

    def test_builds(self, index):
        assert len(index) > 0

    def test_search_returns_ranked_results(self, index):
        results = index.search("total net revenue fiscal 2024", top_k=3)
        assert [r.rank for r in results] == [1, 2, 3]
        assert results[0].score >= results[1].score

    def test_search_finds_the_revenue_chunk(self, index):
        results = index.search("total net revenue fiscal 2024", top_k=5)
        assert any("4,218" in r.text for r in results)

    def test_search_is_deterministic(self, index):
        a = [r.chunk_id for r in index.search("net revenue", top_k=5)]
        b = [r.chunk_id for r in index.search("net revenue", top_k=5)]
        assert a == b

    def test_top_k_respected(self, index):
        assert len(index.search("revenue", top_k=2)) == 2

    def test_metadata_filter(self, index):
        results = index.search(
            "revenue",
            top_k=10,
            where=lambda c: c.metadata.get("section") == "Item 1A Risk Factors",
        )
        assert results
        assert all(r.metadata.get("section") == "Item 1A Risk Factors" for r in results)

    def test_empty_index_returns_nothing(self):
        assert VectorIndex(embedder=HashingEmbedder()).search("anything") == []

    def test_duplicate_chunks_are_not_added_twice(self, document):
        chunks = FixedTokenChunker(chunk_tokens=80).chunk(document)
        index = VectorIndex(embedder=HashingEmbedder(dimension=64))
        assert index.add(chunks) == len(chunks)
        assert index.add(chunks) == 0

    def test_roundtrip_through_disk(self, index, tmp_path):
        path = index.save(tmp_path / "index.jsonl")
        reloaded = VectorIndex.load(path, HashingEmbedder(dimension=512))
        assert len(reloaded) == len(index)
        assert [r.chunk_id for r in reloaded.search("revenue", top_k=3)] == [
            r.chunk_id for r in index.search("revenue", top_k=3)
        ]

    def test_refuses_to_load_with_a_different_embedder(self, index, tmp_path):
        """Vectors from different models are not comparable.

        Searching anyway returns plausible nonsense, which is worse than an
        error because nothing about the output looks wrong.
        """
        path = index.save(tmp_path / "index.jsonl")
        with pytest.raises(ValueError, match="different embedder"):
            VectorIndex.load(path, HashingEmbedder(dimension=64))

    def test_missing_index_file_says_what_to_run(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="make ingest"):
            VectorIndex.load(tmp_path / "nope.jsonl", HashingEmbedder())

    def test_spanned_chunks_exposed_for_resolution(self, index):
        spanned = index.spanned_chunks
        assert len(spanned) == len(index)
        assert all(isinstance(c, SpannedChunk) for c in spanned)


# --------------------------------------------------------------------------
# Generation and the assembled baseline
# --------------------------------------------------------------------------


class TestExtractiveGenerator:
    def test_answers_from_the_top_passage(self):
        passages = [
            RetrievedChunk(
                chunk_id="c1",
                rank=1,
                score=0.9,
                text="Total net revenue for fiscal 2024 was $4,218 million. Unrelated sentence.",
            )
        ]
        result = ExtractiveGenerator(max_sentences=1).generate("What was net revenue?", passages)
        assert not result.refused
        assert "4,218" in result.answer
        assert result.citations[0].chunk_ids == ("c1",)

    def test_refuses_with_no_passages(self):
        assert ExtractiveGenerator().generate("anything", []).refused

    def test_refuses_below_score_threshold(self):
        passages = [RetrievedChunk(chunk_id="c1", rank=1, score=0.001, text="text")]
        assert ExtractiveGenerator(min_score=0.5).generate("q", passages).refused

    def test_citations_reference_only_retrieved_chunks(self):
        passages = [RetrievedChunk(chunk_id="c1", rank=1, score=0.9, text="A fact. Another.")]
        result = ExtractiveGenerator().generate("fact", passages)
        assert all(cid == "c1" for c in result.citations for cid in c.chunk_ids)


class TestFormatPassages:
    def test_includes_index_and_chunk_id(self):
        rendered = format_passages([RetrievedChunk(chunk_id="abc", rank=1, text="hello")])
        assert 'index="1"' in rendered
        assert 'chunk_id="abc"' in rendered

    def test_empty(self):
        assert "no passages" in format_passages([])


class TestBaselineSystem:
    def test_answers_end_to_end(self, document):
        system = build_baseline([document], top_k=3)
        response = system.answer("What was total net revenue in fiscal 2024?")
        assert response.retrieved
        assert response.timings["total"] > 0

    def test_config_is_complete_and_serialisable(self, document):
        import json

        config = build_baseline([document]).config
        json.dumps(config)
        for key in ("top_k", "chunker", "embedder", "generator", "corpus_chunks"):
            assert key in config

    def test_config_reflects_top_k(self, document):
        assert build_baseline([document], top_k=7).config["top_k"] == 7

    def test_satisfies_the_harness_protocol(self, document):
        from evals.runner import run_eval
        from evals.types import Dataset

        system = build_baseline([document], top_k=3)
        # An empty-but-valid dataset would be rejected, so just confirm the
        # protocol surface the runner requires is present and callable.
        assert hasattr(system, "name") and hasattr(system, "config")
        assert callable(system.answer)
        assert isinstance(Dataset, type) and callable(run_eval)

    def test_requires_documents_or_index(self):
        with pytest.raises(ValueError, match="documents.*or a prebuilt"):
            build_baseline()

    def test_deterministic(self, document):
        first = build_baseline([document], top_k=3).answer("net revenue fiscal 2024")
        second = build_baseline([document], top_k=3).answer("net revenue fiscal 2024")
        assert first.retrieved_ids == second.retrieved_ids
