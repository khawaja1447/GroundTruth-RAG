"""Rendering runs and comparisons.

Every table the repo publishes is generated from result files by this module.
Nothing is typed by hand, which is the only way the README's ablation table
cannot drift from what the harness actually measured.
"""

from __future__ import annotations

from collections.abc import Sequence

from .metrics.stats import Comparison, paired_bootstrap
from .runner import HEADLINE_METRICS, RunResult

__all__ = [
    "HEADLINE_METRICS",
    "headline_metrics",
    "format_run",
    "format_slices",
    "compare_runs",
    "format_comparison",
    "markdown_ablation_row",
]

# Metrics where lower is better -- used to colour deltas correctly.
LOWER_IS_BETTER: frozenset[str] = frozenset(
    {
        "false_refusal",
        "answered_unanswerable",
        "citation_fabrication_rate",
        "latency_ms",
        "cost_usd",
    }
)


def headline_metrics(run: RunResult) -> dict[str, dict]:
    return {m: run.aggregates[m] for m in HEADLINE_METRICS if m in run.aggregates}


def _fmt(value: float | None, metric: str) -> str:
    if value is None:
        return "n/a"
    if metric == "latency_ms":
        # Sub-millisecond stages are real (an in-process retriever), and
        # rounding them to "0ms" hides the difference between fast and free.
        if value < 10:
            return f"{value:.2f}ms"
        return f"{value:.0f}ms"
    if metric == "cost_usd":
        return f"${value:.5f}"
    return f"{value:.4f}"


def format_run(run: RunResult, *, show_ci: bool = True) -> str:
    """Human-readable summary of a single run."""
    lines: list[str] = [
        f"run {run.run_id}   {run.config.get('label') or run.config['system']['name']}",
        f"dataset: {run.dataset['path']}  ({run.dataset['total']} questions, "
        f"{run.dataset['verified']} verified)",
        f"judge:   {run.config['judge']['model']} "
        f"({'enabled' if run.config['judge']['enabled'] else 'disabled'})",
        "",
    ]

    width = max(len(m) for m in HEADLINE_METRICS)
    for metric, agg in headline_metrics(run).items():
        mean = agg["mean"]
        if mean is None:
            lines.append(f"  {metric:<{width}}  {'unscored':>10}  (n=0)")
            continue
        cell = _fmt(mean, metric)
        if show_ci and agg["ci_low"] is not None:
            cell += f"  [{_fmt(agg['ci_low'], metric)}, {_fmt(agg['ci_high'], metric)}]"
        suffix = f"(n={agg['n']}"
        if agg["n_undefined"]:
            suffix += f", {agg['n_undefined']} n/a"
        suffix += ")"
        lines.append(f"  {metric:<{width}}  {cell:>28}  {suffix}")

    if run.cache.get("lookups"):
        hit_rate = run.cache.get("hit_rate")
        lines.append("")
        lines.append(
            f"  judge cache: {run.cache['hits']}/{run.cache['hits'] + run.cache['misses']} hits"
            + (f" ({hit_rate:.0%})" if hit_rate is not None else "")
        )

    if run.warnings:
        lines.append("")
        lines.append("warnings:")
        lines.extend(f"  ! {w}" for w in run.warnings)
    return "\n".join(lines)


def format_slices(run: RunResult, metrics: Sequence[str] = ("ndcg@10", "recall@10")) -> str:
    """Per-slice table.

    The reason this exists: an aggregate that improved while the numeric_table
    slice collapsed is a regression wearing a disguise, and only the
    breakdown shows it.
    """
    if not run.slices:
        return "(no slices)"

    metrics = [m for m in metrics if any(m in s for s in run.slices.values())]
    name_w = max(len(s) for s in run.slices)
    header = f"{'slice':<{name_w}} {'n':>4}  " + "  ".join(f"{m:>16}" for m in metrics)
    lines = [header, "-" * len(header)]
    for slice_name, aggs in run.slices.items():
        n = max((aggs[m]["n"] for m in metrics if m in aggs), default=0)
        cells = []
        for m in metrics:
            agg = aggs.get(m)
            cells.append(f"{_fmt(agg['mean'], m) if agg else 'n/a':>16}")
        lines.append(f"{slice_name:<{name_w}} {n:>4}  " + "  ".join(cells))
    return "\n".join(lines)


def compare_runs(
    baseline: RunResult,
    candidate: RunResult,
    *,
    metrics: Sequence[str] = HEADLINE_METRICS,
) -> dict[str, Comparison]:
    """Paired comparison across metrics.

    Paired on question id, so both runs must have been evaluated on the same
    dataset. A mismatch is reported rather than silently producing a
    comparison over whatever ids happen to overlap.
    """
    if baseline.dataset.get("path") != candidate.dataset.get("path"):
        raise ValueError(
            f"runs used different datasets ({baseline.dataset.get('path')!r} vs "
            f"{candidate.dataset.get('path')!r}); a paired comparison would be meaningless"
        )

    out: dict[str, Comparison] = {}
    for metric in metrics:
        cmp = paired_bootstrap(
            metric,
            baseline.per_question_scores(metric),
            candidate.per_question_scores(metric),
        )
        if cmp is not None:
            out[metric] = cmp
    return out


def format_comparison(comparisons: dict[str, Comparison]) -> str:
    """Render a comparison, marking which deltas survive the CI."""
    if not comparisons:
        return "(no comparable metrics)"

    lines = [
        "  * = confidence interval of the difference excludes zero",
        "",
    ]
    width = max(len(m) for m in comparisons)
    for metric, cmp in comparisons.items():
        better = "lower" if metric in LOWER_IS_BETTER else "higher"
        if cmp.significant:
            improved = (cmp.delta < 0) if metric in LOWER_IS_BETTER else (cmp.delta > 0)
            verdict = "improvement" if improved else "REGRESSION"
        else:
            verdict = "inconclusive"
        lines.append(
            f"{'*' if cmp.significant else ' '} {metric:<{width}}  "
            f"{_fmt(cmp.baseline_mean, metric):>10} -> {_fmt(cmp.candidate_mean, metric):>10}  "
            f"delta {cmp.delta:+.4f} [{cmp.ci_low:+.4f}, {cmp.ci_high:+.4f}]  "
            f"{verdict} ({better} is better, n={cmp.n_paired})"
        )
    return "\n".join(lines)


def markdown_ablation_row(run: RunResult, label: str | None = None) -> str:
    """One row of the Phase 3 ablation table, in Markdown.

    `scripts/build_ablation_table.py` concatenates these over the result
    directory so the README table is a build artifact, never a hand edit.
    """
    name = label or run.config.get("label") or run.config["system"]["name"]
    cells = [name]
    for metric in ("ndcg@10", "recall@10", "groundedness", "latency_ms", "cost_usd"):
        agg = run.aggregates.get(metric)
        cells.append(_fmt(agg["mean"] if agg else None, metric))
    return "| " + " | ".join(cells) + " |"
