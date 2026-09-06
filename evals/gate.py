"""The CI regression gate.

Runs a smoke subset on every push and fails the build when quality drops
against a stored baseline. Thresholds are absolute rather than statistical
because CI needs a fast, stable, yes/no answer -- the paired bootstrap in
`report.compare_runs` is the tool for deciding whether a change is real; this
is the tool for stopping an obvious regression from merging.

Two gates that are not about drift at all, and are the ones most likely to
actually catch something:

  * `max_citation_fabrication` defaults to 0. A citation pointing at a chunk
    that was never retrieved is a bug in every case, not a quality tradeoff.
  * `require_no_judge_errors` stops a run whose judge silently failed from
    being recorded as a passing baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .runner import RunResult

__all__ = ["GateThresholds", "GateResult", "check_gate", "DEFAULT_THRESHOLDS"]


@dataclass(frozen=True, slots=True)
class GateThresholds:
    """Maximum tolerated drop per metric, in absolute points.

    Defaults come from the Phase 2 plan: 2 points of nDCG, 3 of groundedness.
    Deliberately loose enough to absorb judge jitter on a 60-question subset
    and tight enough that a real regression trips it.
    """

    max_drop: dict[str, float] = field(
        default_factory=lambda: {
            "ndcg@10": 0.02,
            "recall@10": 0.02,
            "groundedness": 0.03,
            "answer_correctness": 0.03,
            "context_sufficiency": 0.03,
        }
    )
    max_rise: dict[str, float] = field(
        default_factory=lambda: {
            "false_refusal": 0.05,
            "answered_unanswerable": 0.05,
            "latency_ms": 500.0,
        }
    )
    max_citation_fabrication: float = 0.0
    require_no_judge_errors: bool = True


DEFAULT_THRESHOLDS = GateThresholds()


@dataclass(frozen=True, slots=True)
class GateResult:
    passed: bool
    failures: tuple[str, ...]
    checks: tuple[str, ...]
    skipped: tuple[str, ...]

    def format(self) -> str:
        lines = [f"regression gate: {'PASS' if self.passed else 'FAIL'}", ""]
        lines.extend(f"  ok    {c}" for c in self.checks)
        lines.extend(f"  SKIP  {s}" for s in self.skipped)
        lines.extend(f"  FAIL  {f}" for f in self.failures)
        return "\n".join(lines)


def check_gate(
    candidate: RunResult,
    baseline: RunResult | None,
    thresholds: GateThresholds = DEFAULT_THRESHOLDS,
) -> GateResult:
    """Compare a run against a stored baseline.

    A metric unscored in either run is skipped and reported as skipped, never
    treated as a pass. Silently passing on an unscored metric is how a broken
    judge gets through CI green.
    """
    failures: list[str] = []
    checks: list[str] = []
    skipped: list[str] = []

    fabrication = candidate.aggregates.get("citation_fabrication_rate", {}).get("mean")
    if fabrication is None:
        skipped.append("citation_fabrication_rate: no citations produced")
    elif fabrication > thresholds.max_citation_fabrication:
        failures.append(
            f"citation_fabrication_rate = {fabrication:.4f}, "
            f"limit {thresholds.max_citation_fabrication:.4f} "
            f"(a citation resolved to a chunk that was never retrieved)"
        )
    else:
        checks.append(f"citation_fabrication_rate = {fabrication:.4f} (limit 0)")

    if thresholds.require_no_judge_errors:
        errored = sum(len(row.get("judge_errors") or []) for row in candidate.per_question)
        if errored:
            failures.append(
                f"{errored} judge call(s) failed; a run with unscored metrics must not "
                f"be recorded as passing"
            )
        else:
            checks.append("no judge errors")

    if baseline is None:
        skipped.append("drift checks: no baseline stored yet (run `make baseline` to set one)")
        return GateResult(not failures, tuple(failures), tuple(checks), tuple(skipped))

    for metric, limit in thresholds.max_drop.items():
        base = baseline.aggregates.get(metric, {}).get("mean")
        cand = candidate.aggregates.get(metric, {}).get("mean")
        if base is None or cand is None:
            skipped.append(f"{metric}: unscored in {'baseline' if base is None else 'candidate'}")
            continue
        delta = cand - base
        if delta < -limit:
            failures.append(
                f"{metric} dropped {abs(delta):.4f} ({base:.4f} -> {cand:.4f}), limit {limit:.4f}"
            )
        else:
            checks.append(f"{metric} {base:.4f} -> {cand:.4f} ({delta:+.4f}, limit -{limit:.4f})")

    for metric, limit in thresholds.max_rise.items():
        base = baseline.aggregates.get(metric, {}).get("mean")
        cand = candidate.aggregates.get(metric, {}).get("mean")
        if base is None or cand is None:
            skipped.append(f"{metric}: unscored in {'baseline' if base is None else 'candidate'}")
            continue
        delta = cand - base
        if delta > limit:
            failures.append(
                f"{metric} rose {delta:.4f} ({base:.4f} -> {cand:.4f}), limit {limit:.4f}"
            )
        else:
            checks.append(f"{metric} {base:.4f} -> {cand:.4f} ({delta:+.4f}, limit +{limit:.4f})")

    return GateResult(not failures, tuple(failures), tuple(checks), tuple(skipped))


def load_baseline(path: str | Path) -> RunResult | None:
    p = Path(path)
    if not p.exists():
        return None
    return RunResult.load(p)
