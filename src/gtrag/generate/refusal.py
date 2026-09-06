"""Refusal: when the system should decline, and how to choose the threshold.

A system that refuses nothing hallucinates. A system that refuses everything
is useless and scores 100% on the unanswerable slice while being worthless.
The interesting engineering is neither extreme -- it is the operating point
you chose between them, and why.

This module makes that choice measurable rather than assumed:

    curve = refusal_curve(observations)
    point = choose_operating_point(curve, max_false_refusal=0.05)

`choose_operating_point` implements a stated criterion -- maximise correct
refusals subject to a ceiling on false refusals -- rather than an unstated
one. The criterion is an argument, so a different deployment can pick a
different tradeoff and say so.

The confidence signal is deliberately retrieval-derived and computable with
no model call: a system that has to generate an answer before deciding
whether to refuse has already paid for the answer.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..types import RetrievedChunk

__all__ = [
    "RetrievalConfidence",
    "confidence_of",
    "RefusalPolicy",
    "RefusalObservation",
    "RefusalPoint",
    "refusal_curve",
    "choose_operating_point",
]


@dataclass(frozen=True, slots=True)
class RetrievalConfidence:
    """Signals available before generation.

    `margin` matters as much as `top_score`: a retriever that returns five
    equally mediocre chunks is in a different state from one that returns a
    single strong match, even when the top scores agree. On an unanswerable
    question the corpus usually offers several equally-weak candidates, so a
    flat distribution is itself evidence that nothing here answers it.
    """

    top_score: float
    mean_score: float
    margin: float
    n_retrieved: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "top_score": self.top_score,
            "mean_score": self.mean_score,
            "margin": self.margin,
            "n_retrieved": self.n_retrieved,
        }


def confidence_of(chunks: Sequence[RetrievedChunk]) -> RetrievalConfidence:
    if not chunks:
        return RetrievalConfidence(0.0, 0.0, 0.0, 0)
    scores = [c.score for c in sorted(chunks, key=lambda c: c.rank)]
    top = scores[0]
    mean = sum(scores) / len(scores)
    rest = scores[1:]
    margin = top - (sum(rest) / len(rest)) if rest else top
    return RetrievalConfidence(
        top_score=top, mean_score=mean, margin=margin, n_retrieved=len(scores)
    )


@dataclass
class RefusalPolicy:
    """Decide whether to answer, from retrieval confidence alone.

    `signal` names which confidence field the threshold applies to, so the
    sweep can compare signals rather than assuming `top_score` is the right
    one -- which it often is not, since score scales differ by retriever.
    """

    threshold: float = 0.0
    signal: str = "top_score"
    name: str = "threshold"

    def __post_init__(self) -> None:
        if self.signal not in ("top_score", "mean_score", "margin"):
            raise ValueError(
                f"unknown refusal signal {self.signal!r} (have: top_score, mean_score, margin)"
            )

    @property
    def config(self) -> dict[str, Any]:
        return {
            "refusal": self.name,
            "refusal_signal": self.signal,
            "refusal_threshold": self.threshold,
        }

    def value(self, confidence: RetrievalConfidence) -> float:
        return float(getattr(confidence, self.signal))

    def should_refuse(self, chunks: Sequence[RetrievedChunk]) -> bool:
        if not chunks:
            return True
        return self.value(confidence_of(chunks)) < self.threshold


# --------------------------------------------------------------------------
# The curve
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RefusalObservation:
    """One question's confidence, paired with whether it was answerable."""

    question_id: str
    confidence: float
    answerable: bool


@dataclass(frozen=True, slots=True)
class RefusalPoint:
    """Behaviour at one threshold."""

    threshold: float
    correct_refusals: int
    n_unanswerable: int
    false_refusals: int
    n_answerable: int

    @property
    def correct_refusal_rate(self) -> float | None:
        """Of the unanswerable questions, how many were correctly declined."""
        if not self.n_unanswerable:
            return None
        return self.correct_refusals / self.n_unanswerable

    @property
    def false_refusal_rate(self) -> float | None:
        """Of the answerable questions, how many were wrongly declined."""
        if not self.n_answerable:
            return None
        return self.false_refusals / self.n_answerable

    @property
    def degenerate(self) -> bool:
        """True when this point refuses nothing it should.

        A threshold with zero correct refusals is the null policy wearing a
        number. It satisfies any false-refusal ceiling trivially, so a
        selection criterion that accepts it can be satisfied by doing
        nothing -- which is not a criterion.
        """
        return not self.correct_refusals

    @property
    def youden_j(self) -> float | None:
        """Sensitivity + specificity - 1: the threshold-free summary.

        Weights both error types equally, which is a defensible default and
        rarely the right deployment choice -- most systems care more about
        one direction. It is reported for comparison, not used to choose.
        """
        correct = self.correct_refusal_rate
        false = self.false_refusal_rate
        if correct is None or false is None:
            return None
        return correct - false

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "correct_refusals": self.correct_refusals,
            "n_unanswerable": self.n_unanswerable,
            "correct_refusal_rate": self.correct_refusal_rate,
            "false_refusals": self.false_refusals,
            "n_answerable": self.n_answerable,
            "false_refusal_rate": self.false_refusal_rate,
            "youden_j": self.youden_j,
        }


def refusal_curve(
    observations: Sequence[RefusalObservation], *, thresholds: Sequence[float] | None = None
) -> list[RefusalPoint]:
    """Sweep the refusal threshold over the observed confidence range.

    Thresholds default to the observed values themselves plus the midpoints
    between them, which is the standard construction: every distinct
    behaviour of the classifier appears exactly once, and no resolution is
    wasted on ranges where nothing changes.
    """
    if not observations:
        return []

    if thresholds is None:
        values = sorted({o.confidence for o in observations})
        midpoints = [(a + b) / 2.0 for a, b in zip(values, values[1:], strict=False)]
        span = (values[-1] - values[0]) or 1.0
        thresholds = sorted(
            {values[0] - span * 0.01, *values, *midpoints, values[-1] + span * 0.01}
        )

    n_answerable = sum(1 for o in observations if o.answerable)
    n_unanswerable = len(observations) - n_answerable

    points: list[RefusalPoint] = []
    for threshold in thresholds:
        refused = [o for o in observations if o.confidence < threshold]
        points.append(
            RefusalPoint(
                threshold=threshold,
                correct_refusals=sum(1 for o in refused if not o.answerable),
                n_unanswerable=n_unanswerable,
                false_refusals=sum(1 for o in refused if o.answerable),
                n_answerable=n_answerable,
            )
        )
    return points


def choose_operating_point(
    curve: Sequence[RefusalPoint],
    *,
    max_false_refusal: float = 0.05,
    require_useful: bool = True,
) -> RefusalPoint | None:
    """Highest correct-refusal rate subject to a false-refusal ceiling.

    A stated criterion, not an implied one. The ceiling is the deployment
    decision: how often are you willing to decline a question you could have
    answered, in exchange for declining the ones you could not?

    Returns None when nothing satisfies the criterion. Two distinct ways that
    happens, and both are findings rather than errors:

      * no threshold meets the ceiling at all;
      * every threshold that meets it refuses nothing (`require_useful`).

    The second case is the subtle one. A threshold with zero correct refusals
    trivially satisfies any ceiling, so returning it would report an
    "operating point" that is indistinguishable from having no refusal policy
    -- the signal simply does not separate the classes at this ceiling, and
    saying so is the honest output.
    """
    eligible = [
        p
        for p in curve
        if p.false_refusal_rate is not None and p.false_refusal_rate <= max_false_refusal
    ]
    if require_useful:
        eligible = [p for p in eligible if not p.degenerate]
    if not eligible:
        return None
    # Tie-break toward the lower threshold: among thresholds that refuse the
    # same unanswerable questions, prefer the one that answers more.
    return max(
        eligible,
        key=lambda p: (p.correct_refusal_rate or 0.0, -p.threshold),
    )


def format_curve(curve: Sequence[RefusalPoint], *, chosen: RefusalPoint | None = None) -> str:
    """Render the curve as a table, marking the chosen operating point."""
    if not curve:
        return "(no observations)"
    lines = [
        f"{'threshold':>12}  {'correct refusal':>16}  {'false refusal':>15}  {'J':>7}",
        "-" * 58,
    ]
    for point in curve:
        marker = " <-- chosen" if chosen is not None and point is chosen else ""
        correct = point.correct_refusal_rate
        false = point.false_refusal_rate
        j = point.youden_j
        lines.append(
            f"{point.threshold:>12.4f}  "
            f"{'n/a' if correct is None else f'{correct:>7.1%}':>16}  "
            f"{'n/a' if false is None else f'{false:>7.1%}':>15}  "
            f"{'n/a' if j is None else f'{j:>+.3f}':>7}{marker}"
        )
    return "\n".join(lines)
