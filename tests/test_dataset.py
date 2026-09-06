"""Dataset validation: the invariants that keep labels trustworthy."""

from __future__ import annotations

import pytest
from evals.dataset import check_against_corpus, composition_report, load_dataset
from evals.types import Dataset, DatasetError, EvalQuestion, QuestionType

SEED_PATH = "evals/datasets/qa_seed.jsonl"


def q(**overrides):
    base = {
        "id": "t1",
        "question": "What was revenue?",
        "question_type": "single_hop",
        "gold_answer": "$100m",
        "gold_chunks": [{"chunk_id": "c1", "relevance": 2}],
        "verified_by": "tester",
    }
    base.update(overrides)
    return base


class TestQuestionValidation:
    def test_valid_question(self):
        parsed = EvalQuestion.from_dict(q())
        assert parsed.answerable
        assert parsed.verified
        assert parsed.gold_ids == {"c1"}
        assert parsed.answer_bearing_ids == {"c1"}

    def test_missing_id_rejected(self):
        with pytest.raises(DatasetError, match="non-empty id"):
            EvalQuestion.from_dict(q(id=""))

    def test_empty_question_text_rejected(self):
        with pytest.raises(DatasetError, match="question text is empty"):
            EvalQuestion.from_dict(q(question="   "))

    def test_unknown_type_lists_valid_options(self):
        with pytest.raises(DatasetError, match="unknown question_type"):
            EvalQuestion.from_dict(q(question_type="vibes"))

    def test_bad_relevance_rejected(self):
        with pytest.raises(DatasetError, match="relevance must be"):
            EvalQuestion.from_dict(q(gold_chunks=[{"chunk_id": "c1", "relevance": 3}]))

    def test_duplicate_gold_chunks_rejected(self):
        with pytest.raises(DatasetError, match="duplicate gold chunk"):
            EvalQuestion.from_dict(
                q(
                    gold_chunks=[
                        {"chunk_id": "c1", "relevance": 2},
                        {"chunk_id": "c1", "relevance": 1},
                    ]
                )
            )

    def test_supporting_only_labels_rejected(self):
        # Relevance-1 chunks alone mean nothing in the corpus actually answers
        # the question, so recall is unmeasurable.
        with pytest.raises(DatasetError, match="relevance 2"):
            EvalQuestion.from_dict(q(gold_chunks=[{"chunk_id": "c1", "relevance": 1}]))

    def test_gold_chunks_without_answer_rejected(self):
        with pytest.raises(DatasetError, match="need a gold_answer"):
            EvalQuestion.from_dict(q(gold_answer=None))

    def test_answerable_type_without_gold_chunks_rejected(self):
        with pytest.raises(DatasetError, match="has no gold chunks"):
            EvalQuestion.from_dict(q(gold_chunks=[], gold_answer=None))


class TestUnanswerable:
    def test_valid(self):
        parsed = EvalQuestion.from_dict(
            q(question_type="unanswerable", gold_chunks=[], gold_answer=None)
        )
        assert not parsed.answerable

    def test_gold_chunks_rejected(self):
        with pytest.raises(DatasetError, match="must have no gold chunks"):
            EvalQuestion.from_dict(q(question_type="unanswerable", gold_answer=None))

    def test_gold_answer_rejected(self):
        with pytest.raises(DatasetError, match="must not have a gold_answer"):
            EvalQuestion.from_dict(
                q(question_type="unanswerable", gold_chunks=[], gold_answer="$100m")
            )


class TestAmbiguous:
    def test_clarification_shape_allowed(self):
        parsed = EvalQuestion.from_dict(
            q(question_type="ambiguous", gold_chunks=[], gold_answer=None)
        )
        assert not parsed.answerable

    def test_may_still_carry_labels(self):
        parsed = EvalQuestion.from_dict(q(question_type="ambiguous"))
        assert parsed.answerable

    def test_clarification_shape_rejects_gold_answer(self):
        with pytest.raises(DatasetError, match="must not have a\n?\\s*gold_answer|gold_answer"):
            EvalQuestion.from_dict(
                q(question_type="ambiguous", gold_chunks=[], gold_answer="$100m")
            )


class TestAnswerableIsDerived:
    def test_derived_from_labels_not_type(self):
        answerable = EvalQuestion.from_dict(q())
        unanswerable = EvalQuestion.from_dict(
            q(question_type="unanswerable", gold_chunks=[], gold_answer=None)
        )
        assert answerable.answerable is True
        assert unanswerable.answerable is False

    def test_relevance_lookup(self):
        parsed = EvalQuestion.from_dict(
            q(gold_chunks=[{"chunk_id": "c1", "relevance": 2}, {"chunk_id": "c2", "relevance": 1}])
        )
        assert parsed.relevance_of("c1") == 2
        assert parsed.relevance_of("c2") == 1
        assert parsed.relevance_of("nope") == 0


class TestDatasetLevel:
    def test_duplicate_ids_rejected(self):
        one = EvalQuestion.from_dict(q(id="dup"))
        two = EvalQuestion.from_dict(q(id="dup"))
        with pytest.raises(DatasetError, match="duplicate question id"):
            Dataset(questions=(one, two))

    def test_filter_by_type(self):
        dataset = load_dataset(SEED_PATH)
        filtered = dataset.filter(types=[QuestionType.UNANSWERABLE])
        assert len(filtered) > 0
        assert all(x.question_type is QuestionType.UNANSWERABLE for x in filtered)

    def test_filter_by_limit(self):
        dataset = load_dataset(SEED_PATH)
        assert len(dataset.filter(limit=3)) == 3

    def test_roundtrip_jsonl(self):
        dataset = load_dataset(SEED_PATH)
        reparsed = [
            EvalQuestion.from_dict(__import__("json").loads(line))
            for line in dataset.to_jsonl().splitlines()
        ]
        assert [x.id for x in reparsed] == [x.id for x in dataset]


class TestSeedDataset:
    def test_loads_and_validates(self):
        dataset = load_dataset(SEED_PATH)
        assert len(dataset) >= 16

    def test_every_question_is_verified(self):
        assert load_dataset(SEED_PATH).unverified() == []

    def test_covers_every_slice(self):
        report = composition_report(load_dataset(SEED_PATH))
        missing = [
            t.value
            for t in QuestionType
            if t not in (QuestionType.ADVERSARIAL,) and report.by_type.get(t, 0) == 0
        ]
        assert missing == [], f"seed set has no questions for: {missing}"

    def test_gold_ids_all_exist_in_fixture_corpus(self):
        from gtrag.fixtures.corpus import chunk_ids

        problems = check_against_corpus(load_dataset(SEED_PATH), chunk_ids())
        assert problems == []

    def test_stale_label_is_detected(self):
        problems = check_against_corpus(load_dataset(SEED_PATH), ["not-a-real-chunk"])
        assert problems
        assert "not in corpus" in problems[0]


class TestFileErrors:
    def test_missing_file(self):
        with pytest.raises(DatasetError, match="not found"):
            load_dataset("evals/datasets/does-not-exist.jsonl")

    def test_bad_json_reports_line_number(self, tmp_path):
        import json

        path = tmp_path / "bad.jsonl"
        path.write_text(json.dumps(q()) + "\nnot json\n", encoding="utf-8")
        with pytest.raises(DatasetError, match=r":2: invalid JSON"):
            load_dataset(path)

    def test_invalid_record_reports_line_number(self, tmp_path):
        import json

        path = tmp_path / "bad.jsonl"
        path.write_text(
            json.dumps(q()) + "\n" + json.dumps(q(id="t2", question_type="nope")) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(DatasetError, match=r":2:"):
            load_dataset(path)

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("\n\n", encoding="utf-8")
        with pytest.raises(DatasetError, match="empty"):
            load_dataset(path)
