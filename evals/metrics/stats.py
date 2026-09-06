"""Aggregation with uncertainty.

A single mean over ~220 questions is a noisy estimate, and RAG ablations
routinely produce differences smaller than that noise. Reporting a bare
"+2.1 nDCG" invites you to keep a change that did nothing.

Two tools address that:

  * `bootstrap_ci` -- a percentile confidence interval on one run's mean, so
    the number is reported as a range rather than a point.
  * `paired_bootstrap` -- the correct comparison between two configurations
    evaluated on the *same* questions. Resampling questions (not runs)
    and differencing within each resample cancels per-question difficulty,
    which is the dominant source of variance. An unpaired comparison of the
    same data needs a much larger gap to reach the same conclusion.

Stdlib only: `random.Random` with an explicit seed, so a reported interval is
reproducible from the results file alone.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "Power",
    "paired_power",
    "Aggregate",
    "aggregate",
    "bootstrap_ci",
    "paired_bootstrap",
    "Comparison",
    "percentile",
]

DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 20260906


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile. `q` in [0, 100]."""
    if not values:
        raise ValueError("percentile of empty sequence")
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"q must be in [0, 100], got {q}")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * (q / 100.0)
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return ordered[low] * (1 - frac) + ordered[high] * frac


def _clean(values: Sequence[float | None]) -> list[float]:
    """Drop undefined (None) scores.

    Undefined means the metric does not apply to that question -- recall on an
    unanswerable question, groundedness on a refusal. Dropping is correct;
    coercing to 0.0 would make the aggregate depend on dataset composition.
    """
    return [v for v in values if v is not None]


@dataclass(frozen=True, slots=True)
class Aggregate:
    """A metric's mean with a bootstrap interval and its denominator."""

    metric: str
    mean: float | None
    ci_low: float | None
    ci_high: float | None
    n: int
    n_undefined: int

    @property
    def defined(self) -> bool:
        return self.mean is not None

    def format(self, places: int = 4) -> str:
        if self.mean is None:
            return f"{self.metric}: undefined (0/{self.n_undefined} applicable)"
        if self.ci_low is None:
            return f"{self.metric}: {self.mean:.{places}f} (n={self.n})"
        return (
            f"{self.metric}: {self.mean:.{places}f} "
            f"[{self.ci_low:.{places}f}, {self.ci_high:.{places}f}] (n={self.n})"
        )

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "mean": self.mean,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n": self.n,
            "n_undefined": self.n_undefined,
        }


def bootstrap_ci(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean.

    Resamples the observed values with replacement and takes percentiles of
    the resampled means. With n < 2 there is nothing to resample, so the CI
    collapses to the point estimate.
    """
    if not values:
        raise ValueError("bootstrap of empty sequence")
    if len(values) == 1:
        return (values[0], values[0])
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    rng = random.Random(seed)
    n = len(values)
    means: list[float] = []
    for _ in range(resamples):
        total = 0.0
        for _ in range(n):
            total += values[rng.randrange(n)]
        means.append(total / n)

    alpha = (1.0 - confidence) / 2.0
    return (percentile(means, alpha * 100), percentile(means, (1.0 - alpha) * 100))


def aggregate(
    metric: str,
    values: Sequence[float | None],
    *,
    confidence: float = 0.95,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> Aggregate:
    """Mean + CI over the defined values of one metric."""
    defined = _clean(values)
    n_undefined = len(values) - len(defined)
    if not defined:
        return Aggregate(metric, None, None, None, 0, n_undefined)

    mean = sum(defined) / len(defined)
    low, high = bootstrap_ci(defined, confidence=confidence, resamples=resamples, seed=seed)
    return Aggregate(metric, mean, low, high, len(defined), n_undefined)


@dataclass(frozen=True, slots=True)
class Comparison:
    """Paired comparison of one metric between two runs."""

    metric: str
    baseline_mean: float
    candidate_mean: float
    delta: float
    ci_low: float
    ci_high: float
    n_paired: int
    n_dropped: int

    @property
    def significant(self) -> bool:
        """True when the CI of the difference excludes zero.

        This is the guard against shipping noise: a change whose interval
        straddles zero has not been shown to do anything on this eval set,
        however inviting the point estimate looks.
        """
        return self.ci_low > 0.0 or self.ci_high < 0.0

    @property
    def direction(self) -> str:
        if not self.significant:
            return "inconclusive"
        return "improvement" if self.delta > 0 else "regression"

    def format(self, places: int = 4) -> str:
        mark = "*" if self.significant else " "
        return (
            f"{mark} {self.metric}: {self.baseline_mean:.{places}f} -> "
            f"{self.candidate_mean:.{places}f} "
            f"(delta {self.delta:+.{places}f} "
            f"[{self.ci_low:+.{places}f}, {self.ci_high:+.{places}f}], "
            f"{self.direction}, n={self.n_paired})"
        )

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "baseline_mean": self.baseline_mean,
            "candidate_mean": self.candidate_mean,
            "delta": self.delta,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "significant": self.significant,
            "direction": self.direction,
            "n_paired": self.n_paired,
            "n_dropped": self.n_dropped,
        }


def paired_bootstrap(
    metric: str,
    baseline: dict[str, float | None],
    candidate: dict[str, float | None],
    *,
    confidence: float = 0.95,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> Comparison | None:
    """Paired percentile bootstrap on per-question scores keyed by question id.

    Only questions where *both* runs produced a defined score are used; the
    count of dropped pairs is reported so a comparison silently computed over
    a handful of questions is visible rather than implied.

    Returns None when no pair survives.
    """
    shared = [
        qid
        for qid in baseline
        if qid in candidate and baseline[qid] is not None and candidate[qid] is not None
    ]
    shared.sort()  # deterministic ordering before seeded resampling
    n_dropped = len(set(baseline) | set(candidate)) - len(shared)
    if not shared:
        return None

    base_vals = [float(baseline[q]) for q in shared]  # type: ignore[arg-type]
    cand_vals = [float(candidate[q]) for q in shared]  # type: ignore[arg-type]
    diffs = [c - b for c, b in zip(cand_vals, base_vals, strict=True)]

    base_mean = sum(base_vals) / len(base_vals)
    cand_mean = sum(cand_vals) / len(cand_vals)
    delta = cand_mean - base_mean

    if len(shared) == 1:
        return Comparison(metric, base_mean, cand_mean, delta, delta, delta, 1, n_dropped)

    rng = random.Random(seed)
    n = len(diffs)
    boot: list[float] = []
    for _ in range(resamples):
        total = 0.0
        for _ in range(n):
            total += diffs[rng.randrange(n)]
        boot.append(total / n)

    alpha = (1.0 - confidence) / 2.0
    return Comparison(
        metric=metric,
        baseline_mean=base_mean,
        candidate_mean=cand_mean,
        delta=delta,
        ci_low=percentile(boot, alpha * 100),
        ci_high=percentile(boot, (1.0 - alpha) * 100),
        n_paired=n,
        n_dropped=n_dropped,
    )


@dataclass(frozen=True, slots=True)
class Power:
    """How large an effect this eval set can actually resolve.

    The diagnostic that turns "inconclusive" from a dead end into a number.
    An ablation on too few questions returns inconclusive for *every* row,
    and without this you cannot tell whether the components do nothing or the
    eval set is simply too small to say -- which are opposite conclusions
    with opposite next actions.
    """

    metric: str
    n: int
    sd: float
    detectable_effect: float
    target_effect: float
    required_n: int

    @property
    def adequate(self) -> bool:
        return self.n >= self.required_n

    def format(self) -> str:
        verdict = "adequate" if self.adequate else f"UNDERPOWERED (need ~{self.required_n})"
        return (
            f"{self.metric}: n={self.n}, sd={self.sd:.4f} -> can resolve "
            f"~{self.detectable_effect:.4f}; to detect {self.target_effect:.4f}: {verdict}"
        )

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "n": self.n,
            "sd": self.sd,
            "detectable_effect": self.detectable_effect,
            "target_effect": self.target_effect,
            "required_n": self.required_n,
            "adequate": self.adequate,
        }


def paired_power(
    metric: str,
    baseline: dict[str, float | None],
    candidate: dict[str, float | None],
    *,
    target_effect: float = 0.02,
    z: float = 1.96,
) -> Power | None:
    """Estimate the smallest paired difference this set could detect.

    Normal approximation on the paired differences: the 95% interval is
    roughly +/- z * sd / sqrt(n), so an effect is resolvable when it exceeds
    that half-width, and the n needed for a target effect is
    `(z * sd / target)^2`.

    Deliberately an estimate, not a guarantee -- the bootstrap remains the
    thing that decides any individual comparison. This is for answering "how
    many more questions do I need to label?", which is a planning question.
    """
    shared = sorted(
        qid
        for qid in baseline
        if qid in candidate and baseline[qid] is not None and candidate[qid] is not None
    )
    if len(shared) < 2:
        return None

    diffs = [float(candidate[q]) - float(baseline[q]) for q in shared]  # type: ignore[arg-type]
    n = len(diffs)
    mean = sum(diffs) / n
    variance = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    sd = variance**0.5

    if sd == 0.0:
        # Every question moved identically; any non-zero effect is resolvable.
        return Power(metric, n, 0.0, 0.0, target_effect, 2)

    detectable = z * sd / (n**0.5)
    required = int((z * sd / target_effect) ** 2) + 1
    return Power(metric, n, sd, detectable, target_effect, required)
