"""Retrieval metric tests against hand-computed values.

Every expected number here is worked out by hand in the test itself rather
than captured from a previous run. A regression test that asserts whatever
the code produced last time cannot catch a metric that was wrong from the
start -- which is the failure that matters, because a subtly wrong nDCG
silently reorders an entire ablation table.
"""

from __future__ import annotations

from math import isclose, log2

import pytest
from evals.metrics.retrieval import (
    answer_bearing_recall_at_k,
    dcg,
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

GOLD = {"a", "b", "c"}


class TestRecall:
    def test_all_relevant_retrieved(self):
        assert recall_at_k(["a", "b", "c"], GOLD, 3) == 1.0

    def test_partial(self):
        assert recall_at_k(["a", "x", "b"], GOLD, 3) == pytest.approx(2 / 3)

    def test_truncation_excludes_later_hits(self):
        # "c" sits at rank 4 and must not count at k=3.
        assert recall_at_k(["a", "x", "y", "c"], GOLD, 3) == pytest.approx(1 / 3)

    def test_none_retrieved(self):
        assert recall_at_k(["x", "y"], GOLD, 5) == 0.0

    def test_undefined_without_gold(self):
        # The central convention: undefined, not zero.
        assert recall_at_k(["a"], set(), 5) is None

    def test_empty_retrieval(self):
        assert recall_at_k([], GOLD, 5) == 0.0

    def test_invalid_k(self):
        with pytest.raises(ValueError):
            recall_at_k(["a"], GOLD, 0)


class TestAnswerBearingRecall:
    def test_stricter_than_recall(self):
        retrieved = ["support", "x"]
        # "support" is relevance 1; the answer-bearing chunk was missed.
        assert recall_at_k(retrieved, {"support", "answer"}, 5) == pytest.approx(0.5)
        assert answer_bearing_recall_at_k(retrieved, {"answer"}, 5) == 0.0

    def test_undefined_without_answer_chunks(self):
        assert answer_bearing_recall_at_k(["a"], set(), 5) is None


class TestPrecision:
    def test_all_relevant(self):
        assert precision_at_k(["a", "b"], GOLD, 2) == 1.0

    def test_half(self):
        assert precision_at_k(["a", "x"], GOLD, 2) == 0.5

    def test_denominator_is_returned_count_not_k(self):
        # Three returned, all relevant, k=10. Dividing by k would report 0.3
        # for a system that got everything right.
        assert precision_at_k(["a", "b", "c"], GOLD, 10) == 1.0

    def test_empty_retrieval_is_zero(self):
        assert precision_at_k([], GOLD, 5) == 0.0


class TestHitRateAndMRR:
    def test_hit_rate_binary(self):
        assert hit_rate_at_k(["x", "a"], GOLD, 2) == 1.0
        assert hit_rate_at_k(["x", "y"], GOLD, 2) == 0.0

    def test_hit_rate_respects_k(self):
        assert hit_rate_at_k(["x", "y", "a"], GOLD, 2) == 0.0

    def test_mrr_first_position(self):
        assert mrr(["a", "x"], GOLD) == 1.0

    def test_mrr_third_position(self):
        assert mrr(["x", "y", "b"], GOLD) == pytest.approx(1 / 3)

    def test_mrr_miss_is_zero_not_none(self):
        # Distinct from "undefined": the system retrieved, and missed.
        assert mrr(["x", "y"], GOLD) == 0.0

    def test_mrr_undefined_without_gold(self):
        assert mrr(["a"], set()) is None


class TestDCG:
    def test_single_relevance_two(self):
        # (2^2 - 1) / log2(2) = 3 / 1 = 3
        assert dcg([2]) == pytest.approx(3.0)

    def test_hand_computed_sequence(self):
        # rel = [2, 0, 1]
        #   rank 1: (2^2-1)/log2(2) = 3/1     = 3.0
        #   rank 2: (2^0-1)/log2(3) = 0
        #   rank 3: (2^1-1)/log2(4) = 1/2     = 0.5
        assert dcg([2, 0, 1]) == pytest.approx(3.5)

    def test_exponential_gain_beats_linear(self):
        # A relevance-2 chunk is worth 3x a relevance-1 chunk, not 2x.
        assert dcg([2]) == pytest.approx(3 * dcg([1]))

    def test_empty(self):
        assert dcg([]) == 0.0


class TestNDCG:
    def test_perfect_ranking(self):
        relevance = {"a": 2, "b": 1}
        assert ndcg_at_k(["a", "b"], relevance, 10) == pytest.approx(1.0)

    def test_inverted_ranking_hand_computed(self):
        relevance = {"a": 2, "b": 1}
        # actual ["b", "a"]: 1/log2(2) + 3/log2(3) = 1.0 + 1.892789... = 2.892789
        # ideal  ["a", "b"]: 3/log2(2) + 1/log2(3) = 3.0 + 0.630930   = 3.630930
        actual = 1.0 + 3 / log2(3)
        ideal = 3.0 + 1 / log2(3)
        assert ndcg_at_k(["b", "a"], relevance, 10) == pytest.approx(actual / ideal)

    def test_ranking_order_matters(self):
        relevance = {"a": 2, "b": 1}
        good = ndcg_at_k(["a", "b"], relevance, 10)
        bad = ndcg_at_k(["b", "a"], relevance, 10)
        assert good > bad

    def test_irrelevant_chunks_score_zero(self):
        relevance = {"a": 2}
        assert ndcg_at_k(["x", "y", "a"], relevance, 10) == pytest.approx(3 / log2(4) / 3.0)

    def test_ideal_truncated_at_k(self):
        # Three relevant chunks but k=1: the ideal is the single best chunk,
        # so retrieving it scores 1.0 rather than 1/3.
        relevance = {"a": 2, "b": 2, "c": 2}
        assert ndcg_at_k(["a"], relevance, 1) == pytest.approx(1.0)

    def test_undefined_when_no_relevant_chunks(self):
        assert ndcg_at_k(["a"], {}, 10) is None
        assert ndcg_at_k(["a"], {"a": 0}, 10) is None

    def test_bounded_in_unit_interval(self):
        relevance = {"a": 2, "b": 1, "c": 1}
        for retrieved in (["a", "b", "c"], ["c", "b", "a"], ["x"], []):
            score = ndcg_at_k(retrieved, relevance, 10)
            assert score is not None
            assert 0.0 <= score <= 1.0 or isclose(score, 1.0)
