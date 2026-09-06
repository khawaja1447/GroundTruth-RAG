"""Phase 4: context assembly, refusal calibration, verification, rewriting.

The load-bearing tests here are the ones that assert a stage does the thing
it exists for -- dedup catching near-identical filings, the reorder placing
strong evidence at both ends, the refusal criterion refusing to accept a
degenerate operating point, and the verifier failing a claim whose figure is
absent from the context.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gtrag.ablation import AblationConfig, build_system
from gtrag.generate.context import (
    ContextAssembler,
    jaccard,
    lost_in_the_middle_order,
    shingles,
)
from gtrag.generate.refusal import (
    RefusalObservation,
    RefusalPoint,
    RefusalPolicy,
    choose_operating_point,
    confidence_of,
    refusal_curve,
)
from gtrag.generate.verify import (
    JudgeVerifier,
    LexicalVerifier,
    annotate_unsupported,
)
from gtrag.grounded import GroundedRagSystem
from gtrag.ingest.parse import parse_filing
from gtrag.retrieve.rewrite import HeuristicRewriter, NullRewriter, is_dependent
from gtrag.types import RetrievedChunk

FIXTURES = Path(__file__).parent / "fixtures"


def chunk(cid: str, rank: int, text: str, score: float = 1.0) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=cid, rank=rank, score=score, text=text)


@pytest.fixture(scope="module")
def documents():
    return [
        parse_filing(
            (FIXTURES / name).read_text(encoding="utf-8"),
            metadata={"company": company, "fiscal_year": year},
        )
        for name, company, year in (
            ("filing_sample.html", "Northwind Logistics, Inc.", 2024),
            ("filing_prior_year.html", "Northwind Logistics, Inc.", 2023),
            ("filing_peer.html", "Cascade Semiconductor Corp.", 2024),
        )
    ]


# --------------------------------------------------------------------------
# Shingles and similarity
# --------------------------------------------------------------------------


class TestShingles:
    def test_identical_text_is_identical(self):
        a = shingles("the quick brown fox jumps")
        assert jaccard(a, a) == 1.0

    def test_word_order_matters(self):
        """Bag-of-words would call these identical.

        Two filings from different years share nearly all vocabulary and
        differ in the figures that matter, so order-insensitive comparison
        is exactly the wrong tool.
        """
        a = shingles("revenue was 4218 million in fiscal 2024")
        b = shingles("in fiscal 2024 million 4218 was revenue")
        assert jaccard(a, b) < 0.5

    def test_short_text_degrades_gracefully(self):
        assert shingles("two words", size=8)

    def test_empty(self):
        assert shingles("") == frozenset()
        assert jaccard(frozenset(), frozenset()) == 1.0
        assert jaccard(shingles("a b c d"), frozenset()) == 0.0


# --------------------------------------------------------------------------
# Ordering
# --------------------------------------------------------------------------


class TestLostInTheMiddleOrder:
    def test_strongest_at_both_ends(self):
        chunks = [chunk(f"c{i}", i, f"text {i}") for i in range(1, 6)]
        ordered = lost_in_the_middle_order(chunks)
        assert [c.chunk_id for c in ordered] == ["c1", "c3", "c5", "c4", "c2"]

    def test_weakest_lands_in_the_middle(self):
        chunks = [chunk(f"c{i}", i, "x") for i in range(1, 6)]
        ordered = lost_in_the_middle_order(chunks)
        assert ordered[len(ordered) // 2].chunk_id == "c5"

    def test_preserves_every_chunk(self):
        chunks = [chunk(f"c{i}", i, "x") for i in range(1, 8)]
        assert {c.chunk_id for c in lost_in_the_middle_order(chunks)} == {
            c.chunk_id for c in chunks
        }

    def test_single_and_empty(self):
        assert len(lost_in_the_middle_order([chunk("a", 1, "x")])) == 1
        assert lost_in_the_middle_order([]) == []


# --------------------------------------------------------------------------
# Context assembly
# --------------------------------------------------------------------------


class TestContextAssembler:
    def test_drops_near_duplicates(self):
        text = "Total net revenue for fiscal 2024 was 4218 million dollars in the period."
        assembled = ContextAssembler(reorder=False).assemble(
            [chunk("a", 1, text), chunk("b", 2, text)]
        )
        assert len(assembled.chunks) == 1
        assert assembled.dropped_duplicate[0][:2] == ("b", "a")

    def test_keeps_the_higher_ranked_duplicate(self):
        text = "identical boilerplate language repeated verbatim across filings here"
        assembled = ContextAssembler(reorder=False).assemble(
            [chunk("keep", 1, text), chunk("drop", 2, text)]
        )
        assert [c.chunk_id for c in assembled.chunks] == ["keep"]

    def test_distinct_text_is_not_deduplicated(self):
        assembled = ContextAssembler(reorder=False).assemble(
            [
                chunk("a", 1, "Northwind revenue was 4218 million in fiscal 2024 overall"),
                chunk("b", 2, "Cascade gross margin was 42.1 percent in fiscal 2024 overall"),
            ]
        )
        assert len(assembled.chunks) == 2

    def test_deduplication_can_be_disabled(self):
        text = "the same words repeated exactly for the purposes of this test case"
        assembled = ContextAssembler(deduplicate=False, reorder=False).assemble(
            [chunk("a", 1, text), chunk("b", 2, text)]
        )
        assert len(assembled.chunks) == 2

    def test_budget_drops_least_relevant_first(self):
        chunks = [chunk(f"c{i}", i, " ".join(["word"] * 50)) for i in range(1, 5)]
        assembled = ContextAssembler(max_tokens=120, deduplicate=False, reorder=False).assemble(
            chunks
        )
        assert [c.chunk_id for c in assembled.chunks] == ["c1", "c2"]
        assert set(assembled.dropped_budget) == {"c3", "c4"}

    def test_always_keeps_at_least_one_chunk(self):
        # A single oversized chunk must not produce an empty context.
        big = chunk("big", 1, " ".join(["word"] * 500))
        assembled = ContextAssembler(max_tokens=10).assemble([big])
        assert len(assembled.chunks) == 1

    def test_ranks_are_renumbered_to_presentation_order(self):
        """Citation indices are positions in what the model was shown.

        Any other numbering makes every returned citation off by an
        unpredictable amount.
        """
        chunks = [chunk(f"c{i}", i, f"distinct text number {i} here") for i in range(1, 6)]
        assembled = ContextAssembler(deduplicate=False).assemble(chunks)
        assert [c.rank for c in assembled.chunks] == [1, 2, 3, 4, 5]

    def test_reporting_accounts_for_every_dropped_chunk(self):
        text = "shared boilerplate text appearing in more than one filing verbatim"
        chunks = [chunk("a", 1, text), chunk("b", 2, text)] + [
            chunk(f"c{i}", i, " ".join([f"unique{i}"] * 80)) for i in range(3, 6)
        ]
        assembled = ContextAssembler(max_tokens=100).assemble(chunks)
        accounted = len(assembled.chunks) + assembled.n_dropped
        assert accounted == len(chunks)

    def test_empty_input(self):
        assembled = ContextAssembler().assemble([])
        assert assembled.chunks == ()
        assert assembled.tokens_used == 0
        assert assembled.utilisation == 0.0

    def test_rejects_bad_config(self):
        with pytest.raises(ValueError, match="max_tokens"):
            ContextAssembler(max_tokens=0)
        with pytest.raises(ValueError, match="duplicate_threshold"):
            ContextAssembler(duplicate_threshold=0.0)


# --------------------------------------------------------------------------
# Refusal
# --------------------------------------------------------------------------


class TestConfidence:
    def test_empty_retrieval_is_zero(self):
        assert confidence_of([]).top_score == 0.0

    def test_margin_separates_peaked_from_flat(self):
        peaked = confidence_of([chunk("a", 1, "x", 1.0), chunk("b", 2, "x", 0.1)])
        flat = confidence_of([chunk("a", 1, "x", 0.55), chunk("b", 2, "x", 0.55)])
        assert peaked.margin > flat.margin

    def test_single_chunk_margin_is_its_score(self):
        assert confidence_of([chunk("a", 1, "x", 0.7)]).margin == pytest.approx(0.7)


class TestRefusalPolicy:
    def test_refuses_below_threshold(self):
        policy = RefusalPolicy(threshold=0.5)
        assert policy.should_refuse([chunk("a", 1, "x", 0.4)])
        assert not policy.should_refuse([chunk("a", 1, "x", 0.6)])

    def test_empty_retrieval_always_refuses(self):
        assert RefusalPolicy(threshold=0.0).should_refuse([])

    def test_signal_is_selectable(self):
        chunks = [chunk("a", 1, "x", 1.0), chunk("b", 2, "x", 0.0)]
        assert RefusalPolicy(threshold=0.9, signal="top_score").should_refuse(chunks) is False
        assert RefusalPolicy(threshold=0.9, signal="mean_score").should_refuse(chunks) is True

    def test_rejects_unknown_signal(self):
        with pytest.raises(ValueError, match="unknown refusal signal"):
            RefusalPolicy(signal="vibes")


class TestRefusalCurve:
    def _observations(self):
        # Perfectly separable: unanswerable questions score below answerable.
        return [RefusalObservation(f"u{i}", 0.1 + i * 0.01, False) for i in range(4)] + [
            RefusalObservation(f"a{i}", 0.8 + i * 0.01, True) for i in range(10)
        ]

    def test_perfect_separation_yields_a_clean_point(self):
        curve = refusal_curve(self._observations())
        chosen = choose_operating_point(curve, max_false_refusal=0.05)
        assert chosen is not None
        assert chosen.correct_refusal_rate == 1.0
        assert chosen.false_refusal_rate == 0.0

    def test_curve_spans_refuse_nothing_to_refuse_everything(self):
        curve = refusal_curve(self._observations())
        assert curve[0].correct_refusals == 0 and curve[0].false_refusals == 0
        assert curve[-1].correct_refusals == curve[-1].n_unanswerable
        assert curve[-1].false_refusals == curve[-1].n_answerable

    def test_degenerate_point_is_rejected(self):
        """A threshold that refuses nothing is the null policy wearing a number.

        It satisfies any ceiling trivially, so accepting it would let the
        criterion be met by doing nothing.
        """
        point = RefusalPoint(
            threshold=0.0,
            correct_refusals=0,
            n_unanswerable=4,
            false_refusals=0,
            n_answerable=10,
        )
        assert point.degenerate
        assert choose_operating_point([point], max_false_refusal=0.05) is None
        # Explicitly opting out returns it.
        assert choose_operating_point([point], require_useful=False) is point

    def test_no_separation_returns_none(self):
        # Both classes at the same confidence: nothing to threshold on.
        observations = [RefusalObservation(f"q{i}", 0.5, i % 2 == 0) for i in range(10)]
        assert choose_operating_point(refusal_curve(observations)) is None

    def test_raising_the_ceiling_can_admit_a_point(self):
        observations = [
            RefusalObservation("u1", 0.1, False),
            RefusalObservation("a1", 0.1, True),
            RefusalObservation("a2", 0.9, True),
        ]
        curve = refusal_curve(observations)
        assert choose_operating_point(curve, max_false_refusal=0.05) is None
        assert choose_operating_point(curve, max_false_refusal=0.50) is not None

    def test_youden_j_bounds(self):
        curve = refusal_curve(self._observations())
        for point in curve:
            assert -1.0 <= (point.youden_j or 0.0) <= 1.0

    def test_empty_observations(self):
        assert refusal_curve([]) == []


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------


class TestLexicalVerifier:
    CONTEXT = ["Total net revenue for fiscal 2024 was $4,218 million, up 11.0%."]

    def test_supports_a_grounded_claim(self):
        result = LexicalVerifier().verify(
            ["Total net revenue for fiscal 2024 was $4,218 million."], self.CONTEXT
        )
        assert result.verdicts[0].supported
        assert result.groundedness == 1.0

    def test_fails_a_fabricated_figure(self):
        """The failure that matters most on this corpus.

        A wrong figure shares almost all of its vocabulary with the right
        one, so a single blended similarity score would call it supported.
        Numbers are therefore checked strictly and separately.
        """
        result = LexicalVerifier().verify(
            ["Total net revenue for fiscal 2024 was $9,999 million."], self.CONTEXT
        )
        assert not result.verdicts[0].supported
        assert "not present" in result.verdicts[0].reason

    def test_discourse_is_not_a_claim(self):
        result = LexicalVerifier().verify(["In summary:"], self.CONTEXT)
        assert result.verdicts[0].supported

    def test_off_topic_claim_fails(self):
        result = LexicalVerifier().verify(
            ["The company operates fourteen distribution centres."], self.CONTEXT
        )
        assert not result.verdicts[0].supported

    def test_groundedness_is_none_for_no_claims(self):
        assert LexicalVerifier().verify([], self.CONTEXT).groundedness is None

    def test_unsupported_are_listed(self):
        result = LexicalVerifier().verify(
            ["Revenue was $4,218 million.", "Revenue was $1 million."], self.CONTEXT
        )
        assert len(result.unsupported) == 1

    def test_rejects_bad_threshold(self):
        with pytest.raises(ValueError, match="threshold"):
            LexicalVerifier(threshold=1.5)


class TestJudgeVerifier:
    def test_falls_back_without_a_judge(self):
        result = JudgeVerifier(judge=None).verify(
            ["Revenue was $5 million."], ["Revenue was $5 million."]
        )
        assert result.verdicts[0].supported

    def test_judge_error_falls_back_rather_than_passing(self):
        class BrokenJudge:
            model = "broken"

            def score(self, *args, **kwargs):
                from evals.judges.base import Judgment

                return Judgment(metric="claim_support", score=0, error="boom")

        result = JudgeVerifier(judge=BrokenJudge()).verify(
            ["Revenue was $9,999 million."], ["Revenue was $5 million."]
        )
        # A failed check must not silently become "supported".
        assert not result.verdicts[0].supported
        assert "judge unavailable" in result.verdicts[0].reason


class TestAnnotateUnsupported:
    def test_appends_a_caveat(self):
        result = LexicalVerifier().verify(["Revenue was $9,999 million."], ["Revenue was $5m."])
        annotated = annotate_unsupported("Revenue was $9,999 million.", result)
        assert "Not supported" in annotated

    def test_clean_answer_is_untouched(self):
        result = LexicalVerifier().verify(["Revenue was $5 million."], ["Revenue was $5 million."])
        assert annotate_unsupported("Revenue was $5 million.", result) == "Revenue was $5 million."


# --------------------------------------------------------------------------
# Query rewriting
# --------------------------------------------------------------------------


class TestIsDependent:
    @pytest.mark.parametrize(
        "query",
        [
            "And what drove that increase?",
            "What about the year before?",
            "Why did it fall?",
            "And the prior year?",
        ],
    )
    def test_detects_dependent_queries(self, query):
        assert is_dependent(query)

    @pytest.mark.parametrize(
        "query",
        [
            "What was Northwind Logistics' total net revenue in fiscal 2024?",
            "How many distribution centres does Cascade Semiconductor operate?",
        ],
    )
    def test_standalone_queries_are_not_dependent(self, query):
        assert not is_dependent(query)

    def test_empty(self):
        assert not is_dependent("   ")


class TestHeuristicRewriter:
    HISTORY = [
        (
            "What was Northwind's net revenue in fiscal 2024?",
            "Total net revenue for fiscal 2024 was $4,218 million, up 11.0%.",
        )
    ]

    def test_splices_in_the_missing_entity(self):
        rewritten = HeuristicRewriter().rewrite("And what drove that increase?", self.HISTORY)
        assert "Northwind" in rewritten
        assert "drove that increase" in rewritten

    def test_adds_the_period(self):
        rewritten = HeuristicRewriter().rewrite("And what drove that?", self.HISTORY)
        assert "2024" in rewritten

    def test_standalone_query_is_untouched(self):
        query = "What was Cascade Semiconductor's gross margin in fiscal 2024?"
        assert HeuristicRewriter().rewrite(query, self.HISTORY) == query

    def test_no_history_is_a_no_op(self):
        assert HeuristicRewriter().rewrite("And what about that?", []) == "And what about that?"

    def test_null_rewriter_never_changes_anything(self):
        assert NullRewriter().rewrite("And what about that?", self.HISTORY) == (
            "And what about that?"
        )

    def test_rewriting_improves_retrieval(self, documents):
        """The point of the whole component.

        The bare query has no company, no period and no subject; it retrieves
        on stopwords.
        """
        system = build_system(
            AblationConfig(label="rw", chunker="structure_aware", bm25=True), documents
        )
        history = [
            (
                "What was Northwind's net revenue in fiscal 2024?",
                "Total net revenue for fiscal 2024 was $4,218 million, up 11.0%.",
            )
        ]
        bare = system.retriever.retrieve("And what drove that increase?", top_k=3)
        rewritten_query = HeuristicRewriter().rewrite("And what drove that increase?", history)
        rewritten = system.retriever.retrieve(rewritten_query, top_k=3)

        def hits(results):
            return sum(1 for r in results if "freight volumes" in r.text)

        assert hits(rewritten) >= hits(bare)


# --------------------------------------------------------------------------
# The assembled Phase 4 system
# --------------------------------------------------------------------------


class TestGroundedRagSystem:
    def test_answers_with_every_stage_enabled(self, documents):
        system = build_system(
            AblationConfig(
                label="full",
                chunker="structure_aware",
                bm25=True,
                rewriter="heuristic",
                verifier="lexical",
            ),
            documents,
        )
        response = system.answer("What was total net revenue in fiscal 2024?")
        assert isinstance(system, GroundedRagSystem)
        assert response.retrieved
        for stage in ("rewrite", "retrieval", "assembly", "generation", "verification"):
            assert stage in response.timings

    def test_trace_is_recorded(self, documents):
        system = build_system(
            AblationConfig(label="t", verifier="lexical", rewriter="heuristic"), documents
        )
        trace = system.answer("What was net revenue?").metadata["trace"]
        assert "context" in trace and "confidence" in trace and "verification" in trace

    def test_refusal_policy_short_circuits_generation(self, documents):
        # An unreachable threshold must refuse without paying for generation.
        system = build_system(
            AblationConfig(label="r", refusal_signal="top_score", refusal_threshold=999.0),
            documents,
        )
        response = system.answer("What was total net revenue?")
        assert response.refused
        assert "generation" not in response.timings
        assert response.metadata["trace"]["refused_by_policy"]

    def test_all_stages_off_matches_phase_three_behaviour(self, documents):
        """With every Phase 4 stage disabled the system must behave as before,
        or the ladder stops being comparable across phases."""
        config = AblationConfig(label="off", deduplicate=False, reorder=False)
        system = build_system(config, documents)
        assert system.rewriter.name == "none"
        assert system.refusal is None
        assert system.verifier is None
        assert system.answer("net revenue fiscal 2024").retrieved

    def test_config_is_serialisable_and_names_every_stage(self, documents):
        import json

        config = build_system(
            AblationConfig(label="c", verifier="lexical", rewriter="heuristic"), documents
        ).config
        json.dumps(config)
        for key in ("rewriter", "verifier", "context_max_tokens", "context_deduplicate"):
            assert key in config

    def test_deterministic(self, documents):
        config = AblationConfig(label="d", chunker="structure_aware", bm25=True, verifier="lexical")
        first = build_system(config, documents).answer("net revenue fiscal 2024")
        second = build_system(config, documents).answer("net revenue fiscal 2024")
        assert first.retrieved_ids == second.retrieved_ids
        assert first.answer == second.answer

    def test_no_fabricated_citations(self, documents):
        """Phase 4's hard gate: every cited chunk must have been retrieved."""
        from evals.dataset import load_dataset
        from evals.metrics.generation import citation_validity

        system = build_system(
            AblationConfig(label="cite", chunker="structure_aware", bm25=True), documents
        )
        for question in load_dataset("evals/datasets/qa_filing.jsonl"):
            validity = citation_validity(system.answer(question.question))
            assert validity.is_clean, f"{question.id}: {validity.fabricated}"
