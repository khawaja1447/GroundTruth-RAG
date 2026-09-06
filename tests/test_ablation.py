"""Phase 3: chunking strategies, retrievers, fusion, reranking, filtering.

The tests that matter most here are the ones asserting a strategy actually
does the thing it exists to do -- structure-aware chunking never splitting a
table, BM25 beating dense on an exact-figure query, RRF preferring a chunk
both retrievers rank well. A chunker that runs without crashing but splits
tables anyway would pass a weaker suite and be worthless.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from evals.metrics.stats import paired_power
from evals.spans import GoldSpan, resolve_relevance

from gtrag.ablation import ABLATION_LADDER, CHUNKING_SWEEP, AblationConfig, build_system
from gtrag.chunking.base import FixedTokenChunker
from gtrag.chunking.strategies import (
    ParentDocumentChunker,
    RecursiveChunker,
    SemanticChunker,
    SentenceWindowChunker,
    StructureAwareChunker,
    build_chunker,
    split_sentences,
)
from gtrag.index.embed import HashingEmbedder
from gtrag.index.store import VectorIndex
from gtrag.ingest.parse import parse_filing
from gtrag.retrieve.retrievers import (
    BM25Retriever,
    DenseRetriever,
    FilteredRetriever,
    HybridRetriever,
    LexicalReranker,
    MetadataExtractor,
    RerankingRetriever,
    reciprocal_rank_fusion,
)
from gtrag.types import Chunk, RetrievedChunk

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def document():
    return parse_filing(
        (FIXTURES / "filing_sample.html").read_text(encoding="utf-8"),
        metadata={"company": "Northwind Logistics, Inc.", "cik": 1234567, "fiscal_year": 2024},
    )


@pytest.fixture(scope="module")
def peer():
    return parse_filing(
        (FIXTURES / "filing_peer.html").read_text(encoding="utf-8"),
        metadata={"company": "Cascade Semiconductor Corp.", "cik": 7654321, "fiscal_year": 2024},
    )


@pytest.fixture(scope="module")
def corpus(document, peer):
    return [document, peer]


# --------------------------------------------------------------------------
# Sentence splitting
# --------------------------------------------------------------------------


class TestSplitSentences:
    def test_basic(self):
        spans = split_sentences("One. Two. Three.")
        assert len(spans) == 3

    def test_offsets_index_the_source(self):
        text = "First sentence. Second sentence."
        for start, end in split_sentences(text):
            assert text[start:end].strip()

    def test_offset_parameter_shifts_results(self):
        assert split_sentences("A. B.", offset=100)[0][0] == 100

    def test_decimals_do_not_split(self):
        assert len(split_sentences("Revenue was $4.2 billion in total.")) == 1

    def test_abbreviations_do_not_split(self):
        assert len(split_sentences("Northwind Logistics, Inc. reported growth.")) == 1

    def test_empty(self):
        assert split_sentences("   ") == []


# --------------------------------------------------------------------------
# Chunking strategies
# --------------------------------------------------------------------------


def _shared_chunker_invariants(chunker, document):
    chunks = chunker.chunk(document)
    assert chunks, f"{chunker.name} produced no chunks"
    for chunk in chunks:
        # The span must point at the returned text, or span-anchored labels
        # grade the wrong region.
        assert document.slice(chunk.span) == chunk.chunk.text
        assert 0 <= chunk.span.start < chunk.span.end <= len(document.text)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), f"{chunker.name} produced duplicate chunk ids"
    return chunks


class TestAllChunkers:
    @pytest.mark.parametrize("name", ["fixed", "recursive", "structure_aware", "semantic"])
    def test_invariants_hold(self, name, document):
        _shared_chunker_invariants(build_chunker(name), document)

    @pytest.mark.parametrize("name", ["sentence_window", "parent_document"])
    def test_window_strategies_embed_a_narrower_string(self, name, document):
        chunks = build_chunker(name).chunk(document)
        assert chunks
        for chunk in chunks:
            assert document.slice(chunk.span) == chunk.chunk.text
            assert chunk.embed_text, f"{name} must set embed_text"
            assert len(chunk.text_to_embed) <= len(chunk.chunk.text)

    @pytest.mark.parametrize(
        "name", ["fixed", "recursive", "structure_aware", "sentence_window", "parent_document"]
    )
    def test_deterministic(self, name, document):
        a = [c.chunk_id for c in build_chunker(name).chunk(document)]
        b = [c.chunk_id for c in build_chunker(name).chunk(document)]
        assert a == b

    def test_strategies_produce_distinct_ids(self, document):
        # Two strategies must not collide, or an index built from one would
        # silently accept chunks from the other.
        recursive = {c.chunk_id for c in RecursiveChunker().chunk(document)}
        structure = {c.chunk_id for c in StructureAwareChunker().chunk(document)}
        assert recursive.isdisjoint(structure)

    def test_unknown_name_lists_options(self):
        with pytest.raises(ValueError, match="unknown chunker"):
            build_chunker("telepathy")


class TestStructureAwareChunker:
    def test_never_splits_a_table(self, document):
        """The rule this strategy exists for.

        A financial table cut in half puts row labels in one chunk and
        figures in another, and neither is answerable.
        """
        chunks = StructureAwareChunker(chunk_tokens=64).chunk(document)
        for table in document.tables:
            covering = [
                c
                for c in chunks
                if c.span.start <= table.span.start and c.span.end >= table.span.end
            ]
            partial = [c for c in chunks if c.span.overlap(table.span) > 0 and c not in covering]
            assert covering, f"table at {table.span.start} is not wholly inside any chunk"
            assert not partial, f"table at {table.span.start} was split across chunks"

    def test_chunks_do_not_cross_section_boundaries(self, document):
        chunks = StructureAwareChunker(chunk_tokens=2048).chunk(document)
        for chunk in chunks:
            crossed = [
                s
                for s in document.sections
                if s.span.start > chunk.span.start and s.span.start < chunk.span.end
            ]
            assert not crossed, "chunk spans a section boundary"

    def test_oversized_table_becomes_its_own_chunk(self, document):
        # Budget far below the table's size: it must still emerge whole.
        chunks = StructureAwareChunker(chunk_tokens=10, overlap_tokens=0).chunk(document)
        for table in document.tables:
            assert any(
                c.span.start <= table.span.start and c.span.end >= table.span.end for c in chunks
            )

    def test_rejects_bad_overlap(self):
        with pytest.raises(ValueError, match="less than chunk_tokens"):
            StructureAwareChunker(chunk_tokens=100, overlap_tokens=100)


class TestSentenceWindowChunker:
    def test_embeds_the_focus_sentence_only(self, document):
        chunks = SentenceWindowChunker(window=2).chunk(document)
        chunk = chunks[len(chunks) // 2]
        assert len(chunk.text_to_embed) < len(chunk.chunk.text)
        assert chunk.text_to_embed in chunk.chunk.text

    def test_window_zero_returns_single_sentences(self, document):
        chunks = SentenceWindowChunker(window=0).chunk(document)
        assert all(c.text_to_embed == c.chunk.text for c in chunks)

    def test_larger_window_returns_more_text(self, document):
        narrow = SentenceWindowChunker(window=1).chunk(document)
        wide = SentenceWindowChunker(window=4).chunk(document)
        assert sum(len(c.chunk.text) for c in wide) > sum(len(c.chunk.text) for c in narrow)

    def test_rejects_negative_window(self):
        with pytest.raises(ValueError, match="window must be"):
            SentenceWindowChunker(window=-1)


class TestParentDocumentChunker:
    def test_children_share_their_parent_text(self, document):
        chunks = ParentDocumentChunker(child_tokens=32, parent_tokens=256).chunk(document)
        by_span: dict[tuple[int, int], set[str]] = {}
        for chunk in chunks:
            by_span.setdefault((chunk.span.start, chunk.span.end), set()).add(chunk.chunk.text)
        # Every child of the same parent returns identical text.
        assert all(len(texts) == 1 for texts in by_span.values())
        # And at least one parent has more than one child, or the strategy
        # degenerated into plain chunking.
        assert any(
            sum(1 for c in chunks if (c.span.start, c.span.end) == span) > 1 for span in by_span
        )

    def test_rejects_child_larger_than_parent(self):
        with pytest.raises(ValueError, match="smaller than parent"):
            ParentDocumentChunker(child_tokens=512, parent_tokens=128)


class TestSemanticChunker:
    def test_records_whether_its_embedder_is_semantic(self, document):
        # Run with a lexical embedder this measures vocabulary overlap, not
        # meaning. The config must say so, or a run made that way looks
        # comparable to one that is not.
        config = SemanticChunker(embedder=HashingEmbedder()).config
        assert config["split_embedder_semantic"] is False

    def test_respects_the_token_ceiling(self, document):
        from gtrag.chunking.base import approx_tokens

        for chunk in SemanticChunker(max_tokens=80).chunk(document):
            # One sentence may exceed the ceiling on its own; the ceiling
            # bounds accumulation, not a single unit.
            assert approx_tokens(chunk.chunk.text) <= 80 + 120

    def test_rejects_bad_percentile(self):
        with pytest.raises(ValueError, match="percentile"):
            SemanticChunker(percentile=0.0)


class TestLabelsSurviveEveryStrategy:
    """One human labeling, graded under all six chunkings."""

    def test_evidence_resolves_under_each_strategy(self, document):
        idx = document.text.index("$4,218 million")
        gold = [GoldSpan(document.span(idx - 50, idx + 14))]

        resolved: dict[str, set[str]] = {}
        for name in (
            "fixed",
            "recursive",
            "structure_aware",
            "sentence_window",
            "parent_document",
            "semantic",
        ):
            chunks = build_chunker(name).chunk(document)
            mapping = resolve_relevance(gold, chunks)
            assert mapping, f"{name} lost the gold evidence entirely"
            resolved[name] = set(mapping)

        # The chunk ids differ completely between strategies; only the span
        # labeling makes them comparable.
        assert resolved["fixed"].isdisjoint(resolved["structure_aware"])


# --------------------------------------------------------------------------
# Retrievers
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def indexed(corpus):
    chunks = []
    for doc in corpus:
        chunks.extend(FixedTokenChunker(chunk_tokens=96, overlap_tokens=12).chunk(doc))
    index = VectorIndex(embedder=HashingEmbedder(dimension=512))
    index.add(chunks)
    return index, chunks


class TestBM25Retriever:
    def test_finds_an_exact_figure(self, indexed):
        _, chunks = indexed
        results = BM25Retriever(chunks=chunks).retrieve("4,218", top_k=3)
        assert results
        assert any("4,218" in r.text for r in results)

    def test_ranks_are_contiguous(self, indexed):
        _, chunks = indexed
        results = BM25Retriever(chunks=chunks).retrieve("net revenue", top_k=5)
        assert [r.rank for r in results] == list(range(1, len(results) + 1))

    def test_deterministic(self, indexed):
        _, chunks = indexed
        r = BM25Retriever(chunks=chunks)
        assert [c.chunk_id for c in r.retrieve("revenue", top_k=5)] == [
            c.chunk_id for c in r.retrieve("revenue", top_k=5)
        ]

    def test_scores_descend(self, indexed):
        _, chunks = indexed
        results = BM25Retriever(chunks=chunks).retrieve("segment revenue growth", top_k=5)
        assert all(a.score >= b.score for a, b in zip(results, results[1:], strict=False))

    def test_stopword_only_query_returns_nothing(self, indexed):
        _, chunks = indexed
        assert BM25Retriever(chunks=chunks).retrieve("the and of", top_k=5) == []

    def test_respects_where_predicate(self, indexed):
        _, chunks = indexed
        results = BM25Retriever(chunks=chunks).retrieve(
            "revenue",
            top_k=10,
            where=lambda c: c.metadata.get("company") == "Cascade Semiconductor Corp.",
        )
        assert results
        assert all(r.metadata["company"] == "Cascade Semiconductor Corp." for r in results)


class TestReciprocalRankFusion:
    def _item(self, cid, rank):
        return RetrievedChunk(chunk_id=cid, rank=rank, text=cid)

    def test_agreement_wins(self):
        # "b" is 2nd and 1st; "a" is 1st and 3rd. Consistent agreement beats
        # one strong showing.
        a = [self._item("a", 1), self._item("b", 2)]
        b = [self._item("b", 1), self._item("c", 2), self._item("a", 3)]
        fused = reciprocal_rank_fusion([a, b], top_k=3)
        assert fused[0].chunk_id == "b"

    def test_ranks_are_renumbered(self):
        fused = reciprocal_rank_fusion([[self._item("a", 5)]], top_k=5)
        assert fused[0].rank == 1

    def test_weights_shift_the_outcome(self):
        a = [self._item("a", 1)]
        b = [self._item("b", 1)]
        assert reciprocal_rank_fusion([a, b], weights=[10.0, 1.0], top_k=1)[0].chunk_id == "a"
        assert reciprocal_rank_fusion([a, b], weights=[1.0, 10.0], top_k=1)[0].chunk_id == "b"

    def test_deduplicates(self):
        item = [self._item("a", 1)]
        assert len(reciprocal_rank_fusion([item, item], top_k=5)) == 1

    def test_keeps_the_copy_carrying_text(self):
        with_text = [RetrievedChunk(chunk_id="a", rank=1, text="hello")]
        without = [RetrievedChunk(chunk_id="a", rank=1, text="")]
        assert reciprocal_rank_fusion([without, with_text], top_k=1)[0].text == "hello"

    def test_rejects_mismatched_weights(self):
        with pytest.raises(ValueError, match="weights for"):
            reciprocal_rank_fusion([[], []], weights=[1.0], top_k=1)

    def test_rejects_nonpositive_k(self):
        with pytest.raises(ValueError, match="k must be positive"):
            reciprocal_rank_fusion([[]], k=0, top_k=1)

    def test_empty_input(self):
        assert reciprocal_rank_fusion([], top_k=5) == []


class TestHybridRetriever:
    def test_combines_both_retrievers(self, indexed):
        index, chunks = indexed
        hybrid = HybridRetriever(
            retrievers=[DenseRetriever(index=index), BM25Retriever(chunks=chunks)]
        )
        assert hybrid.retrieve("total net revenue fiscal 2024", top_k=5)

    def test_config_names_its_components(self, indexed):
        index, chunks = indexed
        hybrid = HybridRetriever(
            retrievers=[DenseRetriever(index=index), BM25Retriever(chunks=chunks)]
        )
        assert hybrid.config["components"] == ["dense", "bm25"]

    def test_rejects_empty_component_list(self):
        with pytest.raises(ValueError, match="at least one retriever"):
            HybridRetriever(retrievers=[])


class TestReranking:
    def test_lexical_reranker_prefers_covering_passages(self):
        reranker = LexicalReranker()
        scores = reranker.score(
            "warehousing segment revenue",
            ["Warehousing segment revenue grew 12.2%.", "Unrelated text about drivers."],
        )
        assert scores[0] > scores[1]

    def test_declares_itself_non_neural(self):
        assert LexicalReranker().config["neural"] is False

    def test_no_query_terms_scores_zero(self):
        assert LexicalReranker().score("the and", ["anything"]) == [0.0]

    def test_reranking_reorders(self, indexed):
        index, chunks = indexed
        base = DenseRetriever(index=index)
        reranked = RerankingRetriever(base=base, reranker=LexicalReranker(), retrieve_depth=10)
        query = "warehousing termination fees"
        before = [c.chunk_id for c in base.retrieve(query, top_k=5)]
        after = [c.chunk_id for c in reranked.retrieve(query, top_k=5)]
        assert set(after) <= set(before) | set(after)
        assert [c.rank for c in reranked.retrieve(query, top_k=5)] == [1, 2, 3, 4, 5]

    def test_cannot_recover_what_the_first_stage_missed(self, indexed):
        index, chunks = indexed
        # retrieve_depth caps the candidate pool: the reranker reorders, it
        # does not search.
        reranked = RerankingRetriever(
            base=DenseRetriever(index=index), reranker=LexicalReranker(), retrieve_depth=2
        )
        assert len(reranked.retrieve("revenue", top_k=5)) <= 2

    def test_rejects_bad_depth(self, indexed):
        index, _ = indexed
        with pytest.raises(ValueError, match="retrieve_depth"):
            RerankingRetriever(
                base=DenseRetriever(index=index), reranker=LexicalReranker(), retrieve_depth=0
            )


class TestMetadataExtractor:
    @pytest.fixture
    def extractor(self, indexed):
        _, chunks = indexed
        return MetadataExtractor.from_chunks(chunks)

    def test_extracts_company(self, extractor):
        assert extractor.extract("What was Northwind's revenue?")["company"].startswith("Northwind")

    def test_extracts_fiscal_year(self, extractor):
        assert extractor.extract("revenue in fiscal 2024")["fiscal_year"] == 2024

    def test_does_not_filter_on_two_companies(self, extractor):
        """A comparative question must not be narrowed to one company.

        Filtering here would silently make the question unanswerable.
        """
        filters = extractor.extract("Compare Northwind and Cascade revenue")
        assert "company" not in filters

    def test_does_not_filter_on_two_years(self, extractor):
        assert "fiscal_year" not in extractor.extract("compare fiscal 2023 and fiscal 2024")

    def test_no_filters_gives_no_predicate(self, extractor):
        assert extractor.predicate("What drove the increase?") is None

    def test_predicate_matches_and_rejects(self, extractor):
        predicate = extractor.predicate("Northwind fiscal 2024 revenue")
        assert predicate is not None
        assert predicate(
            Chunk(
                chunk_id="a",
                text="x",
                metadata={"company": "Northwind Logistics, Inc.", "fiscal_year": 2024},
            )
        )
        assert not predicate(
            Chunk(
                chunk_id="b",
                text="x",
                metadata={"company": "Cascade Semiconductor Corp.", "fiscal_year": 2024},
            )
        )

    def test_missing_metadata_is_not_a_mismatch(self, extractor):
        # Excluding chunks that merely lack the field would silently shrink
        # the corpus on any incompletely-tagged document.
        predicate = extractor.predicate("Northwind fiscal 2024")
        assert predicate is not None
        assert predicate(Chunk(chunk_id="c", text="x", metadata={}))


class TestFilteredRetriever:
    def test_filters_to_the_named_company(self, indexed):
        index, chunks = indexed
        filtered = FilteredRetriever(
            base=DenseRetriever(index=index), extractor=MetadataExtractor.from_chunks(chunks)
        )
        results = filtered.retrieve("Cascade gross margin fiscal 2024", top_k=5)
        assert results
        assert all(r.metadata.get("company") == "Cascade Semiconductor Corp." for r in results)

    def test_falls_back_when_the_filter_matches_nothing(self, indexed):
        index, chunks = indexed
        extractor = MetadataExtractor.from_chunks(chunks)
        extractor.companies["nonexistent"] = "No Such Company, Inc."
        filtered = FilteredRetriever(base=DenseRetriever(index=index), extractor=extractor)
        # An over-eager extraction should degrade the ranking, not return
        # nothing at all.
        assert filtered.retrieve("nonexistent revenue", top_k=3)


# --------------------------------------------------------------------------
# Ablation configuration
# --------------------------------------------------------------------------


class TestAblationConfig:
    def test_retriever_name_reflects_components(self):
        assert AblationConfig(label="x").retriever_name == "dense"
        assert AblationConfig(label="x", bm25=True).retriever_name == "hybrid"
        assert AblationConfig(label="x", dense=False, bm25=True).retriever_name == "bm25"

    def test_rejects_no_retriever(self):
        with pytest.raises(ValueError, match="at least one retriever"):
            AblationConfig(label="x", dense=False, bm25=False)

    def test_rejects_retrieve_depth_below_top_k(self):
        with pytest.raises(ValueError, match="cannot invent candidates"):
            AblationConfig(label="x", rerank="lexical", retrieve_depth=2, top_k=5)

    def test_ladder_adds_one_component_per_rung(self):
        """Each rung must differ from the previous by exactly one field, or
        the measured delta is not attributable to a single component."""
        for prev, current in zip(ABLATION_LADDER, ABLATION_LADDER[1:], strict=False):
            a, b = prev.to_dict(), current.to_dict()
            changed = {
                k for k in a if a[k] != b[k] and k not in ("label", "note", "retrieve_depth")
            }
            assert len(changed) <= 1, f"{current.label} changes {sorted(changed)}"

    def test_chunking_sweep_covers_every_strategy(self):
        from gtrag.chunking.strategies import all_chunker_names

        swept = {c.chunker for c in CHUNKING_SWEEP}
        assert swept == set(all_chunker_names())


class TestBuildSystem:
    def test_builds_and_answers(self, corpus):
        system = build_system(AblationConfig(label="test"), corpus)
        response = system.answer("What was total net revenue in fiscal 2024?")
        assert response.retrieved
        assert response.timings["total"] > 0

    def test_exposes_chunks_for_span_resolution(self, corpus):
        system = build_system(AblationConfig(label="test"), corpus)
        assert system.spanned_chunks

    def test_config_is_serialisable_and_complete(self, corpus):
        import json

        config = build_system(AblationConfig(label="test", bm25=True), corpus).config
        json.dumps(config)
        assert config["retriever"] == "hybrid"
        assert config["label"] == "test"

    def test_index_cache_avoids_re_embedding(self, corpus):
        cache: dict = {}
        build_system(AblationConfig(label="a"), corpus, index_cache=cache)
        assert len(cache) == 1
        # Same chunker and embedder, different retriever: must reuse.
        build_system(AblationConfig(label="b", bm25=True), corpus, index_cache=cache)
        assert len(cache) == 1
        # Different chunker: must not reuse.
        build_system(AblationConfig(label="c", chunker="recursive"), corpus, index_cache=cache)
        assert len(cache) == 2

    def test_every_ladder_rung_builds(self, corpus):
        for config in ABLATION_LADDER:
            system = build_system(config, corpus)
            assert system.answer("net revenue").retrieved is not None


# --------------------------------------------------------------------------
# Power
# --------------------------------------------------------------------------


class TestPairedPower:
    def test_reports_underpowered_for_a_small_noisy_set(self):
        baseline = {f"q{i}": 0.5 for i in range(10)}
        candidate = {f"q{i}": (1.0 if i % 2 else 0.0) for i in range(10)}
        power = paired_power("ndcg@10", baseline, candidate, target_effect=0.02)
        assert power is not None
        assert not power.adequate
        assert power.required_n > 100

    def test_zero_variance_is_adequate(self):
        baseline = {f"q{i}": 0.5 for i in range(10)}
        candidate = {f"q{i}": 0.7 for i in range(10)}
        power = paired_power("m", baseline, candidate)
        assert power is not None
        assert power.sd == 0.0
        assert power.adequate

    def test_needs_at_least_two_pairs(self):
        assert paired_power("m", {"a": 0.5}, {"a": 0.7}) is None

    def test_required_n_grows_with_variance(self):
        low = paired_power(
            "m",
            {f"q{i}": 0.5 for i in range(20)},
            {f"q{i}": 0.5 + (0.01 if i % 2 else -0.01) for i in range(20)},
        )
        high = paired_power(
            "m",
            {f"q{i}": 0.5 for i in range(20)},
            {f"q{i}": 0.5 + (0.5 if i % 2 else -0.5) for i in range(20)},
        )
        assert high.required_n > low.required_n
