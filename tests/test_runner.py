"""End-to-end runner behaviour, using stub systems with known outputs.

The stubs matter: each one is built to produce a specific failure mode, so
the assertions check that the harness *detects* that failure rather than that
it runs without crashing.
"""

from __future__ import annotations

import pytest
from evals.dataset import load_dataset
from evals.gate import GateThresholds, check_gate
from evals.judges.base import NullJudge
from evals.metrics.generation import citation_validity, split_claims
from evals.runner import RunConfig, config_hash, evaluate_question, load_system, run_eval
from evals.types import QuestionType

from gtrag.fixtures.corpus import FIXTURE_CHUNKS
from gtrag.fixtures.system import FixtureRagSystem
from gtrag.types import Citation, RetrievedChunk, SystemResponse, Usage

SEED_PATH = "evals/datasets/qa_seed.jsonl"
K = (1, 5, 10)


@pytest.fixture
def dataset():
    return load_dataset(SEED_PATH)


# --------------------------------------------------------------------------
# Stubs
# --------------------------------------------------------------------------


class PerfectSystem:
    """Returns exactly the gold chunks, in ideal order."""

    name = "perfect"

    def __init__(self, dataset):
        self._gold = {x.id: sorted(x.gold_chunks, key=lambda g: -g.relevance) for x in dataset}
        self._by_question = {x.question: x.id for x in dataset}

    @property
    def config(self):
        return {"stub": "perfect"}

    def answer(self, question, *, history=None):
        qid = self._by_question[question]
        gold = self._gold[qid]
        if not gold:
            return SystemResponse(answer="", refused=True, timings={"total": 1.0})
        retrieved = tuple(
            RetrievedChunk(chunk_id=g.chunk_id, rank=i, score=1.0 / i, text="ctx")
            for i, g in enumerate(gold, start=1)
        )
        return SystemResponse(
            answer="An answer.",
            retrieved=retrieved,
            citations=(Citation(claim_index=0, chunk_ids=(gold[0].chunk_id,)),),
            timings={"total": 1.0},
            usage=Usage(input_tokens=10, output_tokens=5, cost_usd=0.001),
        )


class FabricatingSystem:
    """Cites a chunk it never retrieved -- the failure Phase 4 gates on."""

    name = "fabricator"

    @property
    def config(self):
        return {"stub": "fabricator"}

    def answer(self, question, *, history=None):
        return SystemResponse(
            answer="Revenue was $9,999 million.",
            retrieved=(RetrievedChunk(chunk_id="real-chunk", rank=1, text="ctx"),),
            citations=(Citation(claim_index=0, chunk_ids=("chunk-that-was-never-retrieved",)),),
            timings={"total": 1.0},
        )


class AlwaysRefusingSystem:
    name = "refuser"

    @property
    def config(self):
        return {"stub": "refuser"}

    def answer(self, question, *, history=None):
        return SystemResponse(answer="", refused=True, timings={"total": 1.0})


class ExplodingSystem:
    name = "exploder"

    @property
    def config(self):
        return {"stub": "exploder"}

    def answer(self, question, *, history=None):
        raise RuntimeError("retriever unavailable")


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


class TestConfigHash:
    def test_stable_across_calls(self, dataset):
        cfg = RunConfig("d", "s", {"k": 1}, "m", True)
        assert config_hash(cfg, dataset) == config_hash(cfg, dataset)

    def test_changes_with_system_config(self, dataset):
        a = RunConfig("d", "s", {"top_k": 5}, "m", True)
        b = RunConfig("d", "s", {"top_k": 10}, "m", True)
        assert config_hash(a, dataset) != config_hash(b, dataset)

    def test_changes_with_dataset_content(self, dataset):
        cfg = RunConfig("d", "s", {}, "m", True)
        trimmed = dataset.filter(limit=len(dataset) - 1)
        assert config_hash(cfg, dataset) != config_hash(cfg, trimmed)

    def test_label_does_not_change_hash(self, dataset):
        a = RunConfig("d", "s", {}, "m", True, label="one")
        b = RunConfig("d", "s", {}, "m", True, label="two")
        assert config_hash(a, dataset) == config_hash(b, dataset)


class TestLoadSystem:
    def test_loads_by_dotted_path(self):
        system = load_system("gtrag.fixtures.system:FixtureRagSystem")
        assert isinstance(system, FixtureRagSystem)

    def test_rejects_malformed_spec(self):
        with pytest.raises(ValueError, match="module.path:attribute"):
            load_system("no_colon_here")

    def test_missing_attribute(self):
        with pytest.raises(AttributeError):
            load_system("gtrag.fixtures.system:NotAThing")


class TestPerQuestionScoring:
    def test_perfect_retrieval_scores_one(self, dataset):
        question = next(x for x in dataset if x.id == "seed-001")
        system = PerfectSystem(dataset)
        result = evaluate_question(question, system, NullJudge(), k_values=K, primary_k=10)
        assert result.retrieval["recall@10"] == pytest.approx(1.0)
        assert result.retrieval["ndcg@10"] == pytest.approx(1.0)
        assert result.retrieval["mrr"] == pytest.approx(1.0)

    def test_unanswerable_has_undefined_retrieval(self, dataset):
        question = next(x for x in dataset if x.id == "seed-012")
        result = evaluate_question(
            question, AlwaysRefusingSystem(), NullJudge(), k_values=K, primary_k=10
        )
        # Undefined, not zero -- the whole convention in one assertion.
        assert result.retrieval["recall@10"] is None
        assert result.retrieval["ndcg@10"] is None
        assert result.refusal["correct_refusal"] is True
        assert result.refusal["false_refusal"] is None

    def test_false_refusal_on_answerable(self, dataset):
        question = next(x for x in dataset if x.id == "seed-001")
        result = evaluate_question(
            question, AlwaysRefusingSystem(), NullJudge(), k_values=K, primary_k=10
        )
        assert result.refusal["false_refusal"] is True
        assert result.refusal["correct_refusal"] is None

    def test_system_exception_is_captured_not_raised(self, dataset):
        question = next(iter(dataset))
        result = evaluate_question(
            question, ExplodingSystem(), NullJudge(), k_values=K, primary_k=10
        )
        assert result.error is not None
        assert "retriever unavailable" in result.error

    def test_fabricated_citation_detected(self, dataset):
        question = next(x for x in dataset if x.id == "seed-001")
        result = evaluate_question(
            question, FabricatingSystem(), NullJudge(), k_values=K, primary_k=10
        )
        assert result.citations["fabricated"] == ["chunk-that-was-never-retrieved"]
        assert result.citations["fabrication_rate"] == 1.0


class TestRunEval:
    def test_refuses_unverified_dataset_by_default(self, dataset, monkeypatch):
        import dataclasses

        tampered = dataset.filter(limit=2)
        tampered = type(tampered)(
            questions=tuple(dataclasses.replace(x, verified_by=None) for x in tampered.questions),
            path=tampered.path,
        )
        with pytest.raises(ValueError, match="unverified"):
            run_eval(tampered, FixtureRagSystem(), workers=1)

    def test_runs_with_fixture_system(self, dataset):
        result = run_eval(dataset, FixtureRagSystem(), workers=2)
        assert len(result.per_question) == len(dataset)
        assert result.run_id
        assert "ndcg@10" in result.aggregates

    def test_slices_present(self, dataset):
        result = run_eval(dataset, FixtureRagSystem(), workers=1)
        assert QuestionType.UNANSWERABLE.value in result.slices
        assert QuestionType.SINGLE_HOP.value in result.slices

    def test_warns_when_no_judge(self, dataset):
        result = run_eval(dataset, FixtureRagSystem(), workers=1)
        assert any("no judge configured" in w for w in result.warnings)

    def test_deterministic_across_runs(self, dataset):
        first = run_eval(dataset, FixtureRagSystem(), workers=1)
        second = run_eval(dataset, FixtureRagSystem(), workers=4)
        assert first.run_id == second.run_id
        assert first.aggregates["ndcg@10"]["mean"] == second.aggregates["ndcg@10"]["mean"]

    def test_perfect_system_beats_fixture_system(self, dataset):
        perfect = run_eval(dataset, PerfectSystem(dataset), workers=1)
        actual = run_eval(dataset, FixtureRagSystem(), workers=1)
        assert perfect.aggregates["ndcg@10"]["mean"] > actual.aggregates["ndcg@10"]["mean"]

    def test_results_roundtrip_through_disk(self, dataset, tmp_path):
        from evals.runner import RunResult

        result = run_eval(dataset, FixtureRagSystem(), workers=1)
        path = result.save(tmp_path)
        reloaded = RunResult.load(path)
        assert reloaded.run_id == result.run_id
        assert reloaded.aggregates == result.aggregates

    def test_per_question_scores_keyed_by_id(self, dataset):
        result = run_eval(dataset, FixtureRagSystem(), workers=1)
        scores = result.per_question_scores("ndcg@10")
        assert set(scores) == {x.id for x in dataset}


class TestGate:
    def test_fabrication_fails_gate(self, dataset):
        result = run_eval(dataset.filter(limit=4), FabricatingSystem(), workers=1)
        gate = check_gate(result, None)
        assert not gate.passed
        assert any("fabrication" in f for f in gate.failures)

    def test_clean_run_passes_without_baseline(self, dataset):
        result = run_eval(dataset, FixtureRagSystem(), workers=1)
        gate = check_gate(result, None)
        assert gate.passed
        assert any("no baseline" in s for s in gate.skipped)

    def test_drop_against_baseline_fails(self, dataset):
        baseline = run_eval(dataset, PerfectSystem(dataset), workers=1)
        candidate = run_eval(dataset, FixtureRagSystem(), workers=1)
        gate = check_gate(candidate, baseline)
        assert not gate.passed
        assert any("ndcg@10 dropped" in f for f in gate.failures)

    def test_identical_run_passes(self, dataset):
        result = run_eval(dataset, FixtureRagSystem(), workers=1)
        assert check_gate(result, result).passed

    def test_unscored_metric_is_skipped_not_passed(self, dataset):
        result = run_eval(dataset, FixtureRagSystem(), workers=1)
        gate = check_gate(result, result)
        # Judged metrics are unscored without a judge; they must appear as
        # skipped rather than quietly counting as passing checks.
        assert any("groundedness" in s for s in gate.skipped)

    def test_thresholds_are_configurable(self, dataset):
        baseline = run_eval(dataset, PerfectSystem(dataset), workers=1)
        candidate = run_eval(dataset, FixtureRagSystem(), workers=1)
        lenient = GateThresholds(max_drop={"ndcg@10": 1.0}, max_rise={})
        assert check_gate(candidate, baseline, lenient).passed


class TestGenerationHelpers:
    def test_split_claims_protects_decimals(self):
        claims = split_claims("Revenue was $4.2 billion. Margin was 42.1%.")
        assert claims == ["Revenue was $4.2 billion.", "Margin was 42.1%."]

    def test_split_claims_protects_abbreviations(self):
        claims = split_claims("Northwind Logistics, Inc. reported growth. It was 11%.")
        assert len(claims) == 2
        assert claims[0].endswith("growth.")

    def test_split_claims_empty(self):
        assert split_claims("   ") == []

    def test_citation_validity_clean(self):
        response = SystemResponse(
            answer="x",
            retrieved=(RetrievedChunk(chunk_id="c1", rank=1),),
            citations=(Citation(claim_index=0, chunk_ids=("c1",)),),
        )
        validity = citation_validity(response)
        assert validity.is_clean
        assert validity.fabrication_rate == 0.0

    def test_uncited_answer_flagged(self):
        response = SystemResponse(
            answer="An unsourced claim.",
            retrieved=(RetrievedChunk(chunk_id="c1", rank=1),),
        )
        assert citation_validity(response).uncited_answer

    def test_refusal_has_nothing_to_cite(self):
        response = SystemResponse(answer="", refused=True)
        assert not citation_validity(response).uncited_answer


class TestFixtureSystem:
    def test_retrieval_is_deterministic(self):
        system = FixtureRagSystem()
        first = system.retrieve("Northwind net revenue fiscal 2024")
        second = system.retrieve("Northwind net revenue fiscal 2024")
        assert [c.chunk_id for c in first] == [c.chunk_id for c in second]

    def test_finds_the_obvious_answer(self):
        system = FixtureRagSystem()
        retrieved = system.retrieve("Northwind total net revenue fiscal 2024")
        assert "nwl-2024-item7-001" in [c.chunk_id for c in retrieved]

    def test_refuses_on_unrelated_query(self):
        system = FixtureRagSystem(refusal_threshold=100.0)
        assert system.answer("something completely unrelated").refused

    def test_ranks_are_contiguous_from_one(self):
        retrieved = FixtureRagSystem().retrieve("revenue")
        assert [c.rank for c in retrieved] == list(range(1, len(retrieved) + 1))

    def test_config_is_json_serialisable(self):
        import json

        json.dumps(FixtureRagSystem().config)

    def test_corpus_ids_are_unique(self):
        ids = [c.chunk_id for c in FIXTURE_CHUNKS]
        assert len(ids) == len(set(ids))
