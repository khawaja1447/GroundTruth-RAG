"""Judge calibration: does the LLM judge agree with a human?

An uncalibrated LLM judge produces numbers that look like measurements and
are not. This module is the check: label a sample by hand, compare, and
publish the agreement statistic alongside every judged metric.

Cohen's kappa rather than raw agreement, because raw agreement is inflated by
the base rate -- on a set where 85% of answers are correct, a judge that
always says "correct" scores 85% agreement and has measured nothing. Kappa
corrects for agreement expected by chance.

Weighted kappa for the ordinal scales (correctness is 0/1/2): confusing
"correct" with "partially correct" is a smaller error than confusing it with
"incorrect", and unweighted kappa treats those identically. Quadratic weights
are the convention for ordinal agreement and are the default here.

Interpretation used by the gate (Landis & Koch bands):
    < 0.20  poor        0.21-0.40  fair       0.41-0.60  moderate
    0.61-0.80  substantial        0.81-1.00  almost perfect

The plan's exit gate is kappa >= 0.60 on correctness and groundedness. Below
that, the rubric is measuring something other than what you intended -- add
few-shot anchors, tighten the boundary definitions, re-run.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

__all__ = [
    "Agreement",
    "cohens_kappa",
    "agreement_report",
    "CalibrationSample",
    "load_human_labels",
    "interpret_kappa",
]

Weighting = Literal["unweighted", "linear", "quadratic"]

KAPPA_GATE = 0.60


def interpret_kappa(kappa: float) -> str:
    if kappa < 0.0:
        return "worse than chance"
    if kappa <= 0.20:
        return "poor"
    if kappa <= 0.40:
        return "fair"
    if kappa <= 0.60:
        return "moderate"
    if kappa <= 0.80:
        return "substantial"
    return "almost perfect"


@dataclass(frozen=True, slots=True)
class Agreement:
    """Agreement between a judge and a human rater on one metric."""

    metric: str
    kappa: float | None
    weighting: Weighting
    raw_agreement: float
    n: int
    categories: tuple[int, ...]
    confusion: tuple[tuple[int, ...], ...]
    undefined_reason: str | None = None

    @property
    def passes_gate(self) -> bool:
        return self.kappa is not None and self.kappa >= KAPPA_GATE

    @property
    def interpretation(self) -> str:
        if self.kappa is None:
            return "undefined"
        return interpret_kappa(self.kappa)

    def format(self) -> str:
        lines = [f"metric: {self.metric}   (n={self.n}, weighting={self.weighting})"]
        if self.kappa is None:
            lines.append(f"  kappa: undefined -- {self.undefined_reason}")
            lines.append(f"  raw agreement: {self.raw_agreement:.1%}")
        else:
            verdict = "PASS" if self.passes_gate else "FAIL"
            lines.append(
                f"  kappa: {self.kappa:.3f} ({self.interpretation})   "
                f"gate >= {KAPPA_GATE:.2f}: {verdict}"
            )
            lines.append(f"  raw agreement: {self.raw_agreement:.1%}")

        header = "      " + "".join(f"{c:>6}" for c in self.categories)
        lines.append("  confusion (rows=human, cols=judge)")
        lines.append("  " + header)
        for cat, row in zip(self.categories, self.confusion, strict=True):
            lines.append(f"  {cat:>4}" + "  " + "".join(f"{v:>6}" for v in row))
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "kappa": self.kappa,
            "weighting": self.weighting,
            "raw_agreement": self.raw_agreement,
            "n": self.n,
            "categories": list(self.categories),
            "confusion": [list(r) for r in self.confusion],
            "interpretation": self.interpretation,
            "passes_gate": self.passes_gate,
            "undefined_reason": self.undefined_reason,
        }


def _weight(i: int, j: int, k: int, weighting: Weighting) -> float:
    """Disagreement weight between category positions i and j.

    0 on the diagonal (full agreement), 1 at maximum distance.
    """
    if weighting == "unweighted":
        return 0.0 if i == j else 1.0
    if k <= 1:
        return 0.0
    d = abs(i - j) / (k - 1)
    return d * d if weighting == "quadratic" else d


def cohens_kappa(
    human: Sequence[int],
    judge: Sequence[int],
    *,
    weighting: Weighting = "quadratic",
    categories: Sequence[int] | None = None,
    metric: str = "",
) -> Agreement:
    """Cohen's kappa between two raters over the same items.

    `categories` fixes the label space explicitly. Inferring it from observed
    values makes kappa depend on which labels happen to appear in the sample,
    so callers should pass the rubric's full scale.
    """
    if len(human) != len(judge):
        raise ValueError(f"rater lengths differ: {len(human)} vs {len(judge)}")
    n = len(human)
    if n == 0:
        return Agreement(metric, None, weighting, 0.0, 0, (), (), "no paired labels")

    cats = tuple(sorted(set(categories) if categories is not None else set(human) | set(judge)))
    index = {c: i for i, c in enumerate(cats)}
    k = len(cats)

    unknown = {v for v in (*human, *judge) if v not in index}
    if unknown:
        raise ValueError(f"labels {sorted(unknown)} are outside the category set {list(cats)}")

    confusion = [[0] * k for _ in range(k)]
    for h, j in zip(human, judge, strict=True):
        confusion[index[h]][index[j]] += 1

    raw = sum(confusion[i][i] for i in range(k)) / n

    if k == 1:
        # Both raters used a single category. Kappa is undefined (chance
        # agreement is 1.0), but this is a real finding, not a crash: the
        # sample carries no discriminative signal and needs harder items.
        return Agreement(
            metric,
            None,
            weighting,
            raw,
            n,
            cats,
            tuple(tuple(r) for r in confusion),
            "only one category present -- sample cannot discriminate; "
            "re-sample to include items the judge and human might score differently",
        )

    row_marg = [sum(confusion[i]) / n for i in range(k)]
    col_marg = [sum(confusion[i][j] for i in range(k)) / n for j in range(k)]

    observed_disagree = 0.0
    expected_disagree = 0.0
    for i in range(k):
        for j in range(k):
            w = _weight(i, j, k, weighting)
            observed_disagree += w * (confusion[i][j] / n)
            expected_disagree += w * row_marg[i] * col_marg[j]

    if expected_disagree == 0.0:
        return Agreement(
            metric,
            None,
            weighting,
            raw,
            n,
            cats,
            tuple(tuple(r) for r in confusion),
            "expected disagreement is zero -- both raters used the same single "
            "category, so chance agreement is 1.0 and the correction is undefined. "
            "This is not a failing judge: the sample had no disagreement to measure. "
            "Re-sample to include items the raters might score differently.",
        )

    kappa = 1.0 - (observed_disagree / expected_disagree)
    return Agreement(
        metric=metric,
        kappa=kappa,
        weighting=weighting,
        raw_agreement=raw,
        n=n,
        categories=cats,
        confusion=tuple(tuple(r) for r in confusion),
    )


# --------------------------------------------------------------------------
# Human-label round trip
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CalibrationSample:
    """One item in the hand-labeling set.

    `judge_scores` are filled by a run; `human_scores` are filled by you.
    The file is written with `human_scores` empty so it can be edited in
    place -- the workflow is: export, label, ingest, report.
    """

    question_id: str
    question: str
    answer: str
    gold_answer: str | None
    judge_scores: dict[str, int]
    human_scores: dict[str, int]
    context_preview: list[str]

    @property
    def labeled(self) -> bool:
        return bool(self.human_scores)

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "answer": self.answer,
            "gold_answer": self.gold_answer,
            "judge_scores": self.judge_scores,
            "human_scores": self.human_scores,
            "context_preview": self.context_preview,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> CalibrationSample:
        return cls(
            question_id=raw["question_id"],
            question=raw.get("question", ""),
            answer=raw.get("answer", ""),
            gold_answer=raw.get("gold_answer"),
            judge_scores={k: int(v) for k, v in (raw.get("judge_scores") or {}).items()},
            human_scores={k: int(v) for k, v in (raw.get("human_scores") or {}).items()},
            context_preview=list(raw.get("context_preview", [])),
        )


def load_human_labels(path: str | Path) -> list[CalibrationSample]:
    """Load a calibration file, keeping only items a human has labeled."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"calibration file not found: {p}\n"
            f"Run `make calibrate-export` first, label the samples by hand, then re-run."
        )
    data = json.loads(p.read_text(encoding="utf-8"))
    samples = [CalibrationSample.from_dict(row) for row in data.get("samples", [])]
    return [s for s in samples if s.labeled]


def agreement_report(
    samples: Sequence[CalibrationSample],
    *,
    scales: dict[str, Sequence[int]],
    weighting: Weighting = "quadratic",
) -> dict[str, Agreement]:
    """Compute agreement per metric across a labeled sample.

    `scales` maps metric name -> the rubric's full category list, e.g.
    `{"answer_correctness": [0, 1, 2], "context_sufficiency": [0, 1]}`.
    Binary metrics are scored unweighted regardless of the requested
    weighting -- with two categories every weighting scheme is identical, and
    naming it "quadratic" in the report would be misleading.
    """
    out: dict[str, Agreement] = {}
    for metric, categories in scales.items():
        pairs = [
            (s.human_scores[metric], s.judge_scores[metric])
            for s in samples
            if metric in s.human_scores and metric in s.judge_scores
        ]
        effective: Weighting = "unweighted" if len(set(categories)) <= 2 else weighting
        if not pairs:
            out[metric] = Agreement(
                metric, None, effective, 0.0, 0, tuple(categories), (), "no paired labels"
            )
            continue
        human, judge = zip(*pairs, strict=True)
        out[metric] = cohens_kappa(
            human, judge, weighting=effective, categories=categories, metric=metric
        )
    return out
