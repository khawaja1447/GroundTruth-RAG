"""Loading, cross-checking and summarising eval datasets.

Two checks live here rather than on the question itself, because they need
context a single record does not have:

  * `check_against_corpus` -- every gold chunk id must exist. A renamed or
    re-chunked corpus silently invalidates labels, and the failure mode is
    not a crash but a quietly falling recall score.
  * `composition_report` -- slice counts against targets, so gaps in coverage
    are visible before the numbers get published.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .types import Dataset, DatasetError, EvalQuestion, Provenance, QuestionType

# Slice targets from the Phase 2 plan. Used for reporting coverage, not enforced.
COMPOSITION_TARGETS: dict[QuestionType, int] = {
    QuestionType.SINGLE_HOP: 55,
    QuestionType.NUMERIC_TABLE: 40,
    QuestionType.MULTI_HOP: 35,
    QuestionType.COMPARATIVE_TEMPORAL: 25,
    QuestionType.UNANSWERABLE: 30,
    QuestionType.AMBIGUOUS: 20,
    QuestionType.ADVERSARIAL: 15,
}


def load_dataset(path: str | Path) -> Dataset:
    """Read a JSONL eval set, validating every record.

    Errors carry the line number -- with a few hundred hand-labeled records,
    "line 137" is the difference between a one-minute fix and a hunt.
    """
    p = Path(path)
    if not p.exists():
        raise DatasetError(f"dataset not found: {p}")

    questions: list[EvalQuestion] = []
    with p.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{p}:{lineno}: invalid JSON: {exc}") from exc
            try:
                questions.append(EvalQuestion.from_dict(raw))
            except DatasetError as exc:
                raise DatasetError(f"{p}:{lineno}: {exc}") from exc

    if not questions:
        raise DatasetError(f"{p}: dataset is empty")
    return Dataset(questions=tuple(questions), path=p)


def check_against_corpus(dataset: Dataset, known_chunk_ids: Iterable[str]) -> list[str]:
    """Return human-readable problems found by joining labels to the corpus.

    Empty list means the dataset is consistent with this corpus.
    """
    known = set(known_chunk_ids)
    problems: list[str] = []
    for q in dataset:
        missing = sorted({g.chunk_id for g in q.gold_chunks} - known)
        if missing:
            problems.append(
                f"{q.id}: gold chunk(s) not in corpus: {', '.join(missing)} "
                f"(labels are stale, or the corpus was re-chunked without re-labeling)"
            )
    return problems


@dataclass(frozen=True, slots=True)
class CompositionReport:
    total: int
    by_type: dict[QuestionType, int]
    by_difficulty: dict[str, int]
    verified: int
    unverified: int
    llm_generated: int
    llm_generated_verified: int
    answerable: int
    unanswerable: int
    multi_turn: int
    mean_gold_chunks: float

    @property
    def verification_rate(self) -> float:
        return self.verified / self.total if self.total else 0.0

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "by_type": {t.value: n for t, n in self.by_type.items()},
            "by_difficulty": self.by_difficulty,
            "verified": self.verified,
            "unverified": self.unverified,
            "verification_rate": round(self.verification_rate, 4),
            "llm_generated": self.llm_generated,
            "llm_generated_verified": self.llm_generated_verified,
            "answerable": self.answerable,
            "unanswerable": self.unanswerable,
            "multi_turn": self.multi_turn,
            "mean_gold_chunks_per_answerable": round(self.mean_gold_chunks, 2),
        }


def composition_report(dataset: Dataset) -> CompositionReport:
    types = Counter(q.question_type for q in dataset)
    diffs = Counter(q.difficulty.value for q in dataset)
    answerable = [q for q in dataset if q.answerable]
    llm_gen = [q for q in dataset if q.provenance is Provenance.LLM_GENERATED]
    mean_gold = sum(len(q.gold_chunks) for q in answerable) / len(answerable) if answerable else 0.0
    return CompositionReport(
        total=len(dataset),
        by_type={t: types.get(t, 0) for t in QuestionType},
        by_difficulty=dict(diffs),
        verified=sum(1 for q in dataset if q.verified),
        unverified=sum(1 for q in dataset if not q.verified),
        llm_generated=len(llm_gen),
        llm_generated_verified=sum(1 for q in llm_gen if q.verified),
        answerable=len(answerable),
        unanswerable=len(dataset) - len(answerable),
        multi_turn=sum(1 for q in dataset if q.history),
        mean_gold_chunks=mean_gold,
    )


def format_composition(report: CompositionReport, *, targets: bool = True) -> str:
    """Render the composition report as a fixed-width table."""
    lines = [
        f"{'slice':<24} {'count':>6} {'target':>7} {'gap':>6}",
        "-" * 46,
    ]
    for qtype in QuestionType:
        count = report.by_type.get(qtype, 0)
        if targets:
            target = COMPOSITION_TARGETS.get(qtype, 0)
            gap = count - target
            marker = "" if gap >= 0 else f"{gap:+d}"
            lines.append(f"{qtype.value:<24} {count:>6} {target:>7} {marker:>6}")
        else:
            lines.append(f"{qtype.value:<24} {count:>6}")
    lines.append("-" * 46)
    lines.append(f"{'TOTAL':<24} {report.total:>6}")
    lines.append("")
    lines.append(f"answerable / unanswerable : {report.answerable} / {report.unanswerable}")
    lines.append(
        f"verified                  : {report.verified}/{report.total} "
        f"({report.verification_rate:.0%})"
    )
    if report.llm_generated:
        lines.append(
            f"llm-generated             : {report.llm_generated} "
            f"({report.llm_generated_verified} hand-verified)"
        )
    lines.append(f"multi-turn                : {report.multi_turn}")
    lines.append(f"mean gold chunks (answerable): {report.mean_gold_chunks:.2f}")
    return "\n".join(lines)
