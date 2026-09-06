"""Eval-set types and the validation rules that keep labels trustworthy.

The rules encoded here are the ones that stop an eval set from quietly
degrading into decoration:

  * graded relevance, so nDCG is computable rather than approximated;
  * `answerable` derived from the label, never from the question text;
  * `verified` provenance, so an unreviewed machine-written question cannot
    slip into a published number.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class QuestionType(StrEnum):
    """Eval slices. Reporting per slice is what makes the eval diagnostic:
    an aggregate that hides a collapse on `numeric_table` is worse than no
    metric at all."""

    SINGLE_HOP = "single_hop"
    NUMERIC_TABLE = "numeric_table"
    MULTI_HOP = "multi_hop"
    COMPARATIVE_TEMPORAL = "comparative_temporal"
    UNANSWERABLE = "unanswerable"
    AMBIGUOUS = "ambiguous"
    ADVERSARIAL = "adversarial"


class Difficulty(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Provenance(StrEnum):
    """How the question was written.

    `llm_generated` questions are legitimate -- generating candidates and
    verifying them by hand is the efficient way to build a set of this size --
    but they are only usable once `verified_by` is filled in. An eval set the
    model wrote and grades unsupervised is circular.
    """

    HAND_WRITTEN = "hand_written"
    LLM_GENERATED = "llm_generated"


# Graded relevance. Binary labels make nDCG meaningless, so the dataset
# carries three levels and metrics that need a binary view derive it.
RELEVANCE_LEVELS = (0, 1, 2)
RELEVANCE_MEANING = {
    0: "irrelevant",
    1: "supporting context, insufficient alone",
    2: "contains the answer",
}


class DatasetError(ValueError):
    """Raised when a dataset file violates a structural invariant."""


@dataclass(frozen=True, slots=True)
class GoldChunk:
    chunk_id: str
    relevance: int

    def __post_init__(self) -> None:
        if not self.chunk_id:
            raise DatasetError("gold chunk_id must be non-empty")
        if self.relevance not in RELEVANCE_LEVELS:
            raise DatasetError(
                f"relevance must be one of {RELEVANCE_LEVELS}, got {self.relevance!r} "
                f"for chunk {self.chunk_id!r}"
            )


@dataclass(frozen=True, slots=True)
class EvalQuestion:
    """One labeled question.

    Invariants enforced at construction (see `_validate`):
      * an answerable question has >=1 gold chunk at relevance 2 and a gold answer
      * an unanswerable question has no gold chunks and no gold answer
      * gold chunk ids are unique within a question
    """

    id: str
    question: str
    question_type: QuestionType
    gold_chunks: tuple[GoldChunk, ...] = ()
    # Span-anchored evidence. Preferred over `gold_chunks` for any corpus
    # whose chunking will change -- which is every corpus that reaches
    # Phase 3. Chunk-id labels remain supported for fixed corpora.
    gold_spans: tuple[Any, ...] = ()
    gold_answer: str | None = None
    difficulty: Difficulty = Difficulty.MEDIUM
    provenance: Provenance = Provenance.HAND_WRITTEN
    verified_by: str | None = None
    verified_at: str | None = None
    history: tuple[tuple[str, str], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    # -- derived ----------------------------------------------------------

    @property
    def answerable(self) -> bool:
        """Derived from the labels, never from the question text or its type.

        A question is answerable exactly when the corpus contains chunks that
        answer it. That covers three cases with one rule:

          * `unanswerable` -- nothing in the corpus answers it; refusing is
            correct.
          * `ambiguous` -- the question is underspecified (no company named
            when two are in the corpus), so no single passage answers it *as
            asked*; asking for clarification rather than picking one is
            correct, and confidently answering is the failure. Labeled with
            no gold chunks for the same reason.
          * everything else -- gold evidence exists and an answer is expected.

        Refusal scoring and the "is this retrieval metric defined" check both
        key off this, so it must not be independently settable.
        """
        return bool(self.gold_chunks or self.gold_spans)

    @property
    def verified(self) -> bool:
        return bool(self.verified_by)

    @property
    def gold_ids(self) -> set[str]:
        """Chunks at relevance >= 1: the binary-relevant set used by recall/MRR."""
        return {g.chunk_id for g in self.gold_chunks if g.relevance >= 1}

    @property
    def answer_bearing_ids(self) -> set[str]:
        """Chunks at relevance 2: sufficient on their own to answer."""
        return {g.chunk_id for g in self.gold_chunks if g.relevance == 2}

    def relevance_of(self, chunk_id: str) -> int:
        for g in self.gold_chunks:
            if g.chunk_id == chunk_id:
                return g.relevance
        return 0

    def relevance_map(self, chunks: Any = None) -> dict[str, int]:
        """`chunk_id -> relevance` for this question under a given chunking.

        Span-labeled questions resolve against `chunks`; chunk-labeled ones
        ignore it. Passing chunks for a span-labeled question and getting an
        empty map back is meaningful: this chunking destroyed the evidence.
        """
        if self.gold_spans and chunks is not None:
            from .spans import resolve_relevance

            return resolve_relevance(self.gold_spans, chunks)
        return {g.chunk_id: g.relevance for g in self.gold_chunks}

    # -- (de)serialization -------------------------------------------------

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> EvalQuestion:
        try:
            qtype = QuestionType(raw["question_type"])
        except KeyError:
            raise DatasetError(
                f"question {raw.get('id', '<no id>')!r}: missing question_type"
            ) from None
        except ValueError as exc:
            raise DatasetError(
                f"question {raw.get('id', '<no id>')!r}: unknown question_type "
                f"{raw['question_type']!r}; expected one of "
                f"{[t.value for t in QuestionType]}"
            ) from exc

        gold = tuple(
            GoldChunk(chunk_id=g["chunk_id"], relevance=int(g["relevance"]))
            for g in raw.get("gold_chunks", [])
        )
        from .spans import GoldSpan

        spans = tuple(GoldSpan.from_dict(g) for g in raw.get("gold_spans", []))
        history = tuple((turn["question"], turn["answer"]) for turn in raw.get("history", []))

        q = cls(
            id=str(raw.get("id", "")),
            question=str(raw.get("question", "")),
            question_type=qtype,
            gold_chunks=gold,
            gold_spans=spans,
            gold_answer=raw.get("gold_answer"),
            difficulty=Difficulty(raw.get("difficulty", "medium")),
            provenance=Provenance(raw.get("provenance", "hand_written")),
            verified_by=raw.get("verified_by"),
            verified_at=raw.get("verified_at"),
            history=history,
            metadata=dict(raw.get("metadata", {})),
            notes=str(raw.get("notes", "")),
        )
        q._validate()
        return q

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "question": self.question,
            "question_type": self.question_type.value,
            "difficulty": self.difficulty.value,
            "provenance": self.provenance.value,
            "gold_chunks": [
                {"chunk_id": g.chunk_id, "relevance": g.relevance} for g in self.gold_chunks
            ],
        }
        if self.gold_spans:
            out["gold_spans"] = [g.to_dict() for g in self.gold_spans]
        if self.gold_answer is not None:
            out["gold_answer"] = self.gold_answer
        if self.verified_by:
            out["verified_by"] = self.verified_by
        if self.verified_at:
            out["verified_at"] = self.verified_at
        if self.history:
            out["history"] = [{"question": q, "answer": a} for q, a in self.history]
        if self.metadata:
            out["metadata"] = self.metadata
        if self.notes:
            out["notes"] = self.notes
        return out

    # -- validation --------------------------------------------------------

    def _validate(self) -> None:
        where = f"question {self.id!r}"
        if not self.id:
            raise DatasetError("every question needs a non-empty id")
        if not self.question.strip():
            raise DatasetError(f"{where}: question text is empty")

        ids = [g.chunk_id for g in self.gold_chunks]
        dupes = {c for c in ids if ids.count(c) > 1}
        if dupes:
            raise DatasetError(f"{where}: duplicate gold chunk ids {sorted(dupes)}")

        # An `unanswerable` question must have no labels -- if the corpus does
        # answer it, the type is wrong.
        if self.question_type is QuestionType.UNANSWERABLE:
            if self.gold_chunks or self.gold_spans:
                raise DatasetError(
                    f"{where}: question_type 'unanswerable' must have no gold evidence "
                    f"(got {len(self.gold_chunks)} chunks, {len(self.gold_spans)} spans). "
                    f"If the corpus does answer it, relabel the type."
                )
            if self.gold_answer:
                raise DatasetError(
                    f"{where}: question_type 'unanswerable' must not have a gold_answer"
                )
            return

        if self.gold_spans:
            # Span labels carry their own weight, so the relevance-2
            # requirement is expressed as "at least one sufficient span".
            if not any(g.max_relevance == 2 for g in self.gold_spans):
                raise DatasetError(
                    f"{where}: no gold span with weight 1.0. At least one span must be "
                    f"sufficient to answer, or recall is unmeasurable."
                )
            if not (self.gold_answer or "").strip():
                raise DatasetError(f"{where}: questions with gold spans need a gold_answer")
            return

        if self.gold_chunks:
            if not self.answer_bearing_ids:
                raise DatasetError(
                    f"{where}: no gold chunk at relevance 2. At least one chunk must "
                    f"contain the answer, or the question is not answerable from the "
                    f"corpus and recall is unmeasurable."
                )
            if not (self.gold_answer or "").strip():
                raise DatasetError(f"{where}: questions with gold chunks need a gold_answer")
            return

        # No gold chunks and not typed 'unanswerable'. Legitimate only for
        # `ambiguous`, where the expected behaviour is a clarifying question
        # rather than an answer.
        if self.question_type is not QuestionType.AMBIGUOUS:
            raise DatasetError(
                f"{where}: question_type '{self.question_type.value}' has no gold evidence. "
                f"Label it 'unanswerable' if the corpus cannot answer it, 'ambiguous' if "
                f"it is underspecified and clarification is the correct response, or add "
                f"the gold chunks."
            )
        if self.gold_answer:
            raise DatasetError(
                f"{where}: an 'ambiguous' question with no gold chunks must not have a "
                f"gold_answer -- the expected behaviour is clarification, not an answer"
            )


@dataclass(frozen=True, slots=True)
class Dataset:
    """An ordered, id-unique collection of eval questions."""

    questions: tuple[EvalQuestion, ...]
    path: Path | None = None

    def __post_init__(self) -> None:
        seen: dict[str, int] = {}
        for i, q in enumerate(self.questions):
            if q.id in seen:
                raise DatasetError(
                    f"duplicate question id {q.id!r} at positions {seen[q.id]} and {i}"
                )
            seen[q.id] = i

    def __iter__(self) -> Iterator[EvalQuestion]:
        return iter(self.questions)

    def __len__(self) -> int:
        return len(self.questions)

    def filter(
        self,
        *,
        types: Iterable[QuestionType] | None = None,
        ids: Iterable[str] | None = None,
        verified_only: bool = False,
        limit: int | None = None,
    ) -> Dataset:
        qs = list(self.questions)
        if types is not None:
            wanted = set(types)
            qs = [q for q in qs if q.question_type in wanted]
        if ids is not None:
            wanted_ids = set(ids)
            qs = [q for q in qs if q.id in wanted_ids]
        if verified_only:
            qs = [q for q in qs if q.verified]
        if limit is not None:
            qs = qs[:limit]
        return Dataset(questions=tuple(qs), path=self.path)

    def by_type(self) -> dict[QuestionType, list[EvalQuestion]]:
        out: dict[QuestionType, list[EvalQuestion]] = {}
        for q in self.questions:
            out.setdefault(q.question_type, []).append(q)
        return out

    def unverified(self) -> list[EvalQuestion]:
        return [q for q in self.questions if not q.verified]

    def to_jsonl(self) -> str:
        return "".join(
            json.dumps(q.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
            for q in self.questions
        )
