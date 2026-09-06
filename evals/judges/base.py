"""Judge protocol, rubric loading, and the scoring scales.

The scales are declared here, in one place, because three things must agree
on them or the calibration report is nonsense: the rubric prompt, the JSON
schema the model is constrained to, and the category set passed to Cohen's
kappa. Declaring them once and deriving the other two removes the class of
bug where a rubric says 0-3 and kappa is computed over 0-2.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "Judgment",
    "Judge",
    "Scale",
    "SCALES",
    "load_rubric",
    "rubric_version",
    "NullJudge",
]

RUBRIC_DIR = Path(__file__).parent / "rubrics"


@dataclass(frozen=True, slots=True)
class Scale:
    """A judged metric's label space and meaning."""

    name: str
    categories: tuple[int, ...]
    labels: dict[int, str]
    higher_is_better: bool = True

    @property
    def is_binary(self) -> bool:
        return len(self.categories) == 2

    def normalise(self, score: int) -> float:
        """Map a raw category onto [0, 1] for aggregation."""
        lo, hi = min(self.categories), max(self.categories)
        if hi == lo:
            return 1.0
        return (score - lo) / (hi - lo)

    def validate(self, score: int) -> int:
        if score not in self.categories:
            raise ValueError(f"{self.name}: score {score!r} outside scale {list(self.categories)}")
        return score


SCALES: dict[str, Scale] = {
    "answer_correctness": Scale(
        name="answer_correctness",
        categories=(0, 1, 2),
        labels={0: "incorrect", 1: "partially correct", 2: "correct"},
    ),
    "context_sufficiency": Scale(
        name="context_sufficiency",
        categories=(0, 1),
        labels={0: "insufficient", 1: "sufficient"},
    ),
    "claim_support": Scale(
        name="claim_support",
        categories=(0, 1, 2),
        labels={0: "contradicted", 1: "not stated", 2: "supported"},
    ),
}


@dataclass(frozen=True, slots=True)
class Judgment:
    """One judged score, carrying enough provenance to audit it later.

    `raw` keeps the model's own reasoning text. When calibration disagrees
    with a human, reading the judge's stated reason is how you find out
    whether the rubric or the judge is at fault.
    """

    metric: str
    score: int
    reasoning: str = ""
    rubric_version: str = ""
    model: str = ""
    cached: bool = False
    raw: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "score": self.score,
            "reasoning": self.reasoning,
            "rubric_version": self.rubric_version,
            "model": self.model,
            "cached": self.cached,
            "error": self.error,
        }


@runtime_checkable
class Judge(Protocol):
    """Scores one answer against one rubric.

    Implementations must be safe to call from multiple threads: the runner
    fans questions out across a pool.
    """

    model: str

    def score(
        self,
        metric: str,
        *,
        question: str,
        answer: str,
        context: Sequence[str],
        gold_answer: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Judgment: ...

    def decompose_claims(self, answer: str) -> list[str]: ...


@cache
def _read_rubric(name: str) -> str:
    path = RUBRIC_DIR / f"{name}.md"
    if not path.exists():
        available = sorted(p.stem for p in RUBRIC_DIR.glob("*.md"))
        raise FileNotFoundError(
            f"no rubric named {name!r} in {RUBRIC_DIR} (have: {', '.join(available)})"
        )
    return path.read_text(encoding="utf-8")


def load_rubric(name: str) -> str:
    return _read_rubric(name)


@cache
def rubric_version(name: str) -> str:
    """Content hash of the rubric text, used in the cache key.

    Deriving the version from the content rather than a hand-maintained
    constant means an edited rubric cannot reuse scores from the old wording
    even if you forget to bump anything.
    """
    import hashlib

    return hashlib.sha256(_read_rubric(name).encode("utf-8")).hexdigest()[:12]


class NullJudge:
    """A judge that refuses to guess.

    Used when no API key is configured. It returns an errored `Judgment` for
    every call rather than a plausible-looking zero, so the runner records
    "not judged" instead of silently publishing a groundedness score of 0.0
    that nobody measured.
    """

    model = "null"

    def score(
        self,
        metric: str,
        *,
        question: str,
        answer: str,
        context: Sequence[str],
        gold_answer: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Judgment:
        return Judgment(
            metric=metric,
            score=SCALES[metric].categories[0],
            rubric_version=rubric_version(metric) if (RUBRIC_DIR / f"{metric}.md").exists() else "",
            model=self.model,
            error="no judge configured (set ANTHROPIC_API_KEY or pass --no-judge explicitly)",
        )

    def decompose_claims(self, answer: str) -> list[str]:
        from ..metrics.generation import split_claims

        return split_claims(answer)
