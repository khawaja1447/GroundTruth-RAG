"""Bootstrap aggregation and paired comparison."""

from __future__ import annotations

import pytest
from evals.metrics.stats import aggregate, bootstrap_ci, paired_bootstrap, percentile

FAST = {"resamples": 400, "seed": 7}


class TestPercentile:
    def test_median_odd(self):
        assert percentile([1, 2, 3], 50) == 2

    def test_median_even_interpolates(self):
        assert percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)

    def test_bounds(self):
        assert percentile([1, 5, 9], 0) == 1
        assert percentile([1, 5, 9], 100) == 9

    def test_single_value(self):
        assert percentile([4.2], 50) == 4.2

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            percentile([], 50)

    def test_rejects_out_of_range(self):
        with pytest.raises(ValueError):
            percentile([1, 2], 101)


class TestBootstrapCI:
    def test_constant_values_give_zero_width(self):
        low, high = bootstrap_ci([0.5] * 20, **FAST)
        assert low == pytest.approx(0.5)
        assert high == pytest.approx(0.5)

    def test_interval_brackets_the_mean(self):
        values = [0.1, 0.4, 0.5, 0.6, 0.9, 0.3, 0.7, 0.2]
        mean = sum(values) / len(values)
        low, high = bootstrap_ci(values, **FAST)
        assert low <= mean <= high

    def test_deterministic_under_same_seed(self):
        values = [0.1, 0.9, 0.3, 0.7]
        assert bootstrap_ci(values, **FAST) == bootstrap_ci(values, **FAST)

    def test_single_observation_collapses(self):
        assert bootstrap_ci([0.42], **FAST) == (0.42, 0.42)

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            bootstrap_ci([], **FAST)


class TestAggregate:
    def test_none_values_are_excluded_not_zeroed(self):
        # The single most important behaviour in this module: three defined
        # 1.0s average to 1.0, not to 0.5 because two were undefined.
        agg = aggregate("m", [1.0, 1.0, 1.0, None, None], **FAST)
        assert agg.mean == pytest.approx(1.0)
        assert agg.n == 3
        assert agg.n_undefined == 2

    def test_all_none_is_undefined(self):
        agg = aggregate("m", [None, None], **FAST)
        assert agg.mean is None
        assert not agg.defined
        assert agg.n == 0
        assert agg.n_undefined == 2

    def test_empty_input(self):
        agg = aggregate("m", [], **FAST)
        assert agg.mean is None
        assert agg.n == 0

    def test_format_undefined_does_not_crash(self):
        assert "undefined" in aggregate("m", [None], **FAST).format()


class TestPairedBootstrap:
    def test_uniform_improvement_is_significant(self):
        baseline = {f"q{i}": 0.5 for i in range(30)}
        candidate = {f"q{i}": 0.7 for i in range(30)}
        cmp = paired_bootstrap("ndcg@10", baseline, candidate, **FAST)
        assert cmp is not None
        assert cmp.delta == pytest.approx(0.2)
        assert cmp.significant
        assert cmp.direction == "improvement"

    def test_noise_is_inconclusive(self):
        # Half improve, half regress, by the same amount: a real delta of
        # zero. The CI must straddle zero.
        baseline = {f"q{i}": 0.5 for i in range(40)}
        candidate = {f"q{i}": (0.9 if i % 2 else 0.1) for i in range(40)}
        cmp = paired_bootstrap("ndcg@10", baseline, candidate, **FAST)
        assert cmp is not None
        assert cmp.delta == pytest.approx(0.0, abs=1e-9)
        assert not cmp.significant
        assert cmp.direction == "inconclusive"

    def test_regression_detected(self):
        baseline = {f"q{i}": 0.8 for i in range(25)}
        candidate = {f"q{i}": 0.6 for i in range(25)}
        cmp = paired_bootstrap("ndcg@10", baseline, candidate, **FAST)
        assert cmp is not None
        assert cmp.significant
        assert cmp.direction == "regression"

    def test_pairs_on_question_id_not_position(self):
        # Same questions, different insertion order. Pairing by id must give
        # a clean +0.2; pairing by position would give nonsense.
        baseline = {"a": 0.1, "b": 0.5, "c": 0.9}
        candidate = {"c": 1.0, "a": 0.2, "b": 0.7}
        cmp = paired_bootstrap("m", baseline, candidate, **FAST)
        assert cmp is not None
        assert cmp.n_paired == 3
        assert cmp.delta == pytest.approx((0.1 + 0.2 + 0.1) / 3)

    def test_unpairable_questions_are_dropped_and_counted(self):
        baseline = {"a": 0.5, "b": 0.5, "c": None}
        candidate = {"a": 0.7, "d": 0.9}
        cmp = paired_bootstrap("m", baseline, candidate, **FAST)
        assert cmp is not None
        assert cmp.n_paired == 1
        assert cmp.n_dropped == 3

    def test_no_overlap_returns_none(self):
        assert paired_bootstrap("m", {"a": 0.5}, {"b": 0.5}, **FAST) is None

    def test_deterministic_under_same_seed(self):
        baseline = {f"q{i}": i / 10 for i in range(10)}
        candidate = {f"q{i}": (i + 1) / 10 for i in range(10)}
        first = paired_bootstrap("m", baseline, candidate, **FAST)
        second = paired_bootstrap("m", baseline, candidate, **FAST)
        assert first == second
