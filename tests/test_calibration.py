"""Cohen's kappa tests, including the undefined cases.

The undefined cases get as much attention as the arithmetic. A kappa
implementation that returns 0.0 or NaN when one rater used a single category
will silently report "the judge is useless" for a sample that simply had no
disagreement to measure -- and someone will then rewrite a perfectly good
rubric chasing it.
"""

from __future__ import annotations

import pytest
from evals.calibration import (
    KAPPA_GATE,
    CalibrationSample,
    agreement_report,
    cohens_kappa,
    interpret_kappa,
)


class TestPerfectAndChance:
    def test_perfect_agreement_is_one(self):
        result = cohens_kappa([0, 1, 2, 1, 0], [0, 1, 2, 1, 0], categories=[0, 1, 2])
        assert result.kappa == pytest.approx(1.0)
        assert result.raw_agreement == 1.0
        assert result.passes_gate

    def test_total_disagreement_is_negative(self):
        result = cohens_kappa([0, 0, 2, 2], [2, 2, 0, 0], categories=[0, 1, 2])
        assert result.kappa is not None
        assert result.kappa < 0
        assert not result.passes_gate

    def test_hand_computed_unweighted_binary(self):
        # 2x2 confusion: [[20, 5], [10, 15]], n = 50
        #   po = (20 + 15) / 50 = 0.70
        #   row marginals = [0.5, 0.5]; col marginals = [0.6, 0.4]
        #   pe = 0.5*0.6 + 0.5*0.4 = 0.50
        #   kappa = (0.70 - 0.50) / (1 - 0.50) = 0.40
        human = [0] * 25 + [1] * 25
        judge = [0] * 20 + [1] * 5 + [0] * 10 + [1] * 15
        result = cohens_kappa(human, judge, weighting="unweighted", categories=[0, 1])
        assert result.raw_agreement == pytest.approx(0.70)
        assert result.kappa == pytest.approx(0.40)


class TestWeighting:
    def test_quadratic_more_forgiving_of_near_misses(self):
        # Every disagreement is one step on a 3-point scale. Weighted kappa
        # should exceed unweighted, which treats a one-step miss as a total
        # miss.
        human = [0, 1, 2, 1, 0, 2, 1, 0]
        judge = [1, 1, 1, 2, 0, 2, 0, 0]
        unweighted = cohens_kappa(human, judge, weighting="unweighted", categories=[0, 1, 2])
        quadratic = cohens_kappa(human, judge, weighting="quadratic", categories=[0, 1, 2])
        assert quadratic.kappa > unweighted.kappa

    def test_quadratic_and_linear_agree_when_perfect(self):
        for weighting in ("unweighted", "linear", "quadratic"):
            r = cohens_kappa([0, 1, 2], [0, 1, 2], weighting=weighting, categories=[0, 1, 2])
            assert r.kappa == pytest.approx(1.0)

    def test_far_miss_penalised_more_than_near_miss(self):
        near = cohens_kappa([2, 2, 1, 0], [1, 2, 1, 0], weighting="quadratic", categories=[0, 1, 2])
        far = cohens_kappa([2, 2, 1, 0], [0, 2, 1, 0], weighting="quadratic", categories=[0, 1, 2])
        assert near.kappa > far.kappa


class TestUndefined:
    def test_single_category_is_undefined_not_zero(self):
        result = cohens_kappa([1, 1, 1, 1], [1, 1, 1, 1], categories=[1])
        assert result.kappa is None
        assert result.raw_agreement == 1.0
        assert "one category" in (result.undefined_reason or "")
        assert not result.passes_gate

    def test_constant_judge_scores_exactly_zero(self):
        # The judge said "2" for everything while the human varied. Observed
        # and expected disagreement coincide, so kappa is exactly 0 -- the
        # correct and informative answer: this judge carries no information
        # beyond chance. It fails the gate, which is the point.
        result = cohens_kappa([0, 1, 2, 1], [2, 2, 2, 2], categories=[0, 1, 2])
        assert result.kappa == pytest.approx(0.0)
        assert not result.passes_gate

    def test_both_raters_constant_within_wider_scale_is_undefined(self):
        # Every item scored 2 by both raters, on a 0-2 scale. Chance agreement
        # is 1.0, so chance-corrected agreement genuinely cannot be computed.
        # Returning 0.0 here would report "useless judge" for a sample that
        # simply had no disagreement to measure.
        result = cohens_kappa([2, 2, 2], [2, 2, 2], categories=[0, 1, 2])
        assert result.kappa is None
        assert result.raw_agreement == 1.0
        assert "single category" in (result.undefined_reason or "")

    def test_empty_input(self):
        result = cohens_kappa([], [], categories=[0, 1])
        assert result.kappa is None
        assert result.n == 0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="lengths differ"):
            cohens_kappa([0, 1], [0], categories=[0, 1])

    def test_out_of_scale_label_raises(self):
        with pytest.raises(ValueError, match="outside the category set"):
            cohens_kappa([0, 5], [0, 1], categories=[0, 1])


class TestConfusionMatrix:
    def test_orientation_is_rows_human_cols_judge(self):
        # One item: human said 0, judge said 2.
        result = cohens_kappa([0], [2], categories=[0, 1, 2])
        assert result.confusion[0][2] == 1
        assert result.confusion[2][0] == 0


class TestInterpretation:
    @pytest.mark.parametrize(
        "kappa,expected",
        [
            (-0.1, "worse than chance"),
            (0.15, "poor"),
            (0.35, "fair"),
            (0.55, "moderate"),
            (0.75, "substantial"),
            (0.95, "almost perfect"),
        ],
    )
    def test_bands(self, kappa, expected):
        assert interpret_kappa(kappa) == expected

    def test_gate_boundary(self):
        assert cohens_kappa([0] * 10 + [1] * 10, [0] * 10 + [1] * 10, categories=[0, 1]).passes_gate
        assert KAPPA_GATE == 0.60


class TestAgreementReport:
    def _sample(self, qid, human, judge):
        return CalibrationSample(
            question_id=qid,
            question="q",
            answer="a",
            gold_answer="g",
            judge_scores=judge,
            human_scores=human,
            context_preview=[],
        )

    def test_per_metric_agreement(self):
        samples = [
            self._sample("q1", {"answer_correctness": 2}, {"answer_correctness": 2}),
            self._sample("q2", {"answer_correctness": 0}, {"answer_correctness": 0}),
            self._sample("q3", {"answer_correctness": 1}, {"answer_correctness": 1}),
        ]
        report = agreement_report(samples, scales={"answer_correctness": [0, 1, 2]})
        assert report["answer_correctness"].kappa == pytest.approx(1.0)

    def test_binary_metric_forced_unweighted(self):
        # With two categories every weighting is identical; labelling the
        # result "quadratic" would misrepresent what was computed.
        samples = [
            self._sample("q1", {"context_sufficiency": 1}, {"context_sufficiency": 1}),
            self._sample("q2", {"context_sufficiency": 0}, {"context_sufficiency": 0}),
        ]
        report = agreement_report(
            samples, scales={"context_sufficiency": [0, 1]}, weighting="quadratic"
        )
        assert report["context_sufficiency"].weighting == "unweighted"

    def test_missing_metric_reports_no_pairs(self):
        samples = [self._sample("q1", {"answer_correctness": 2}, {"answer_correctness": 2})]
        report = agreement_report(
            samples, scales={"answer_correctness": [0, 1, 2], "groundedness": [0, 1]}
        )
        assert report["groundedness"].n == 0
        assert report["groundedness"].kappa is None

    def test_only_labeled_samples_counted(self):
        samples = [
            self._sample("q1", {"answer_correctness": 2}, {"answer_correctness": 2}),
            self._sample("q2", {}, {"answer_correctness": 0}),
        ]
        report = agreement_report(samples, scales={"answer_correctness": [0, 1, 2]})
        assert report["answer_correctness"].n == 1
