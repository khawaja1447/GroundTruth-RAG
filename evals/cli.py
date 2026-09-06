"""Command line entry point: `python -m evals.cli <command>`.

Commands map onto the Phase 2 workflow:

    validate            check a dataset's structure and its join to the corpus
    stats               slice composition against targets
    run                 evaluate a system, write a result file
    compare             paired comparison of two runs
    gate                check a run against the stored baseline
    baseline            promote a run to the stored baseline
    calibrate-export    sample a run for hand-labeling
    calibrate-report    compute judge/human agreement
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Sequence
from pathlib import Path

from .cache import ResponseCache
from .calibration import KAPPA_GATE, CalibrationSample, agreement_report, load_human_labels
from .dataset import check_against_corpus, composition_report, format_composition, load_dataset
from .gate import DEFAULT_THRESHOLDS, check_gate, load_baseline
from .judges.base import SCALES, NullJudge
from .judges.llm_judge import DEFAULT_JUDGE_MODEL, build_judge
from .report import compare_runs, format_comparison, format_run, format_slices
from .runner import RunConfig, RunResult, load_system, run_eval
from .types import DatasetError

DEFAULT_DATASET = "evals/datasets/qa_seed.jsonl"
DEFAULT_SYSTEM = "gtrag.fixtures.system:FixtureRagSystem"
DEFAULT_BASELINE = "evals/baselines/main.json"
DEFAULT_CALIBRATION = "evals/calibration/samples.json"


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def cmd_validate(args: argparse.Namespace) -> int:
    dataset = load_dataset(args.dataset)
    print(f"structure ok: {len(dataset)} questions parsed from {args.dataset}")

    if args.corpus:
        from gtrag.fixtures.corpus import chunk_ids

        problems = check_against_corpus(dataset, chunk_ids())
        if problems:
            print(f"\n{len(problems)} label/corpus mismatch(es):", file=sys.stderr)
            for p in problems:
                print(f"  - {p}", file=sys.stderr)
            return 1
        print("corpus join ok: every gold chunk id exists")

    unverified = dataset.unverified()
    if unverified:
        print(f"\nwarning: {len(unverified)} unverified question(s):")
        for q in unverified[:10]:
            print(f"  - {q.id} ({q.provenance.value})")
        if args.strict:
            print("\nfailing because --strict was passed", file=sys.stderr)
            return 1
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    dataset = load_dataset(args.dataset)
    print(format_composition(composition_report(dataset), targets=not args.no_targets))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    dataset = load_dataset(args.dataset)
    if args.types:
        from .types import QuestionType

        dataset = dataset.filter(types=[QuestionType(t) for t in args.types])
    if args.limit:
        dataset = dataset.filter(limit=args.limit)
    if not len(dataset):
        print("no questions selected after filtering", file=sys.stderr)
        return 1

    system = load_system(args.system)
    cache = ResponseCache(args.cache_path, enabled=not args.no_cache)
    judge = build_judge(model=args.judge_model, cache=cache, enabled=not args.no_judge)

    if isinstance(judge, NullJudge) and not args.no_judge:
        print(
            "note: no Anthropic credentials found -- running deterministic metrics only.\n"
            "      Set ANTHROPIC_API_KEY (or run `ant auth login`) for judged metrics,\n"
            "      or pass --no-judge to silence this.\n",
            file=sys.stderr,
        )

    config = RunConfig(
        dataset_path=str(dataset.path or args.dataset),
        system_name=getattr(system, "name", type(system).__name__),
        system_config=dict(getattr(system, "config", {}) or {}),
        judge_model=getattr(judge, "model", "null"),
        judge_enabled=not isinstance(judge, NullJudge),
        label=args.label,
    )

    def progress(done: int, total: int) -> None:
        if not args.quiet:
            print(f"\r  {done}/{total} questions", end="", file=sys.stderr, flush=True)

    try:
        result = run_eval(
            dataset,
            system,
            judge=judge,
            config=config,
            workers=args.workers,
            cache=cache,
            allow_unverified=args.allow_unverified,
            progress=progress,
        )
    except ValueError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1
    finally:
        cache.close()

    if not args.quiet:
        print("", file=sys.stderr)

    path = result.save(args.results_dir)
    print(format_run(result))
    print()
    print(format_slices(result))
    print(f"\nwritten to {path}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    baseline = RunResult.load(args.baseline)
    candidate = RunResult.load(args.candidate)
    try:
        comparisons = compare_runs(baseline, candidate)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"baseline:  {baseline.run_id}  {baseline.config.get('label', '')}")
    print(f"candidate: {candidate.run_id}  {candidate.config.get('label', '')}")
    print()
    print(format_comparison(comparisons))
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    candidate = RunResult.load(args.run)
    baseline = load_baseline(args.baseline)
    result = check_gate(candidate, baseline, DEFAULT_THRESHOLDS)
    print(result.format())
    return 0 if result.passed else 1


def cmd_baseline(args: argparse.Namespace) -> int:
    src = Path(args.run)
    dst = Path(args.baseline)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"baseline set from {src} -> {dst}")
    return 0


def cmd_calibrate_export(args: argparse.Namespace) -> int:
    """Sample judged questions into a file for hand-labeling.

    Sampling is seeded and stratified by question type so the calibration set
    reflects the eval set rather than whichever slice happens to be largest.
    """
    run = RunResult.load(args.run)
    judged = [
        row
        for row in run.per_question
        if any(row.get(m) is not None for m in ("answer_correctness", "context_sufficiency"))
    ]
    if not judged:
        print(
            "this run has no judged questions -- calibration needs a run with the judge enabled",
            file=sys.stderr,
        )
        return 1

    by_type: dict[str, list[dict]] = {}
    for row in judged:
        by_type.setdefault(row["question_type"], []).append(row)

    rng = random.Random(args.seed)
    per_type = max(1, args.n // max(1, len(by_type)))
    chosen: list[dict] = []
    for rows in by_type.values():
        rows_sorted = sorted(rows, key=lambda r: r["question_id"])
        chosen.extend(rng.sample(rows_sorted, min(per_type, len(rows_sorted))))

    chosen_ids = {r["question_id"] for r in chosen}
    remaining = sorted(
        (r for r in judged if r["question_id"] not in chosen_ids),
        key=lambda r: r["question_id"],
    )
    rng.shuffle(remaining)
    chosen.extend(remaining[: max(0, args.n - len(chosen))])
    chosen.sort(key=lambda r: r["question_id"])

    samples: list[CalibrationSample] = []
    for row in chosen:
        judge_scores: dict[str, int] = {}
        for j in row.get("judgments", []):
            if j["metric"] in SCALES and j.get("error") is None:
                judge_scores.setdefault(j["metric"], int(j["score"]))
        samples.append(
            CalibrationSample(
                question_id=row["question_id"],
                question=row.get("question", ""),
                answer=row.get("answer", ""),
                gold_answer=row.get("gold_answer"),
                judge_scores=judge_scores,
                human_scores={},
                context_preview=row.get("retrieved_ids", [])[:5],
            )
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run.run_id,
        "judge_model": run.config["judge"]["model"],
        "instructions": (
            "Fill human_scores for each sample WITHOUT reading judge_scores first. "
            "Scales: " + "; ".join(f"{name} = {scale.labels}" for name, scale in SCALES.items())
        ),
        "samples": [s.to_dict() for s in samples],
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"exported {len(samples)} samples to {out}")
    print("Label `human_scores` by hand, then run: make calibrate-report")
    return 0


def cmd_calibrate_report(args: argparse.Namespace) -> int:
    samples = load_human_labels(args.samples)
    if not samples:
        print(
            f"no hand-labeled samples in {args.samples} -- fill `human_scores` first",
            file=sys.stderr,
        )
        return 1

    scales = {name: scale.categories for name, scale in SCALES.items()}
    report = agreement_report(samples, scales=scales, weighting=args.weighting)

    print(f"judge calibration  (n={len(samples)} hand-labeled samples)\n")
    gated = []
    for metric, agreement in report.items():
        if agreement.n == 0:
            continue
        print(agreement.format())
        print()
        if metric in ("answer_correctness", "context_sufficiency"):
            gated.append(agreement)

    if not gated:
        print("no gated metrics were labeled", file=sys.stderr)
        return 1

    failed = [a for a in gated if not a.passes_gate]
    if failed:
        print(
            f"GATE FAIL: {', '.join(a.metric for a in failed)} below kappa {KAPPA_GATE:.2f}.\n"
            f"The judge is not measuring what the rubric intends. Tighten the rubric's "
            f"boundary definitions, add worked examples at the disagreement points, "
            f"then re-run the eval and re-label.",
            file=sys.stderr,
        )
        return 1

    print(f"GATE PASS: all gated metrics at kappa >= {KAPPA_GATE:.2f}")
    return 0


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="evals", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    v = sub.add_parser("validate", help="check dataset structure and corpus join")
    v.add_argument("--dataset", default=DEFAULT_DATASET)
    v.add_argument("--corpus", action="store_true", help="also check gold ids against the corpus")
    v.add_argument("--strict", action="store_true", help="fail if any question is unverified")
    v.set_defaults(func=cmd_validate)

    s = sub.add_parser("stats", help="slice composition")
    s.add_argument("--dataset", default=DEFAULT_DATASET)
    s.add_argument("--no-targets", action="store_true")
    s.set_defaults(func=cmd_stats)

    r = sub.add_parser("run", help="evaluate a system")
    r.add_argument("--dataset", default=DEFAULT_DATASET)
    r.add_argument("--system", default=DEFAULT_SYSTEM, help="module.path:attribute")
    r.add_argument("--label", default="", help="human-readable name for this configuration")
    r.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    r.add_argument("--no-judge", action="store_true", help="deterministic metrics only")
    r.add_argument("--no-cache", action="store_true")
    r.add_argument("--cache-path", default=None)
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--types", nargs="*", default=None, help="restrict to these question types")
    r.add_argument("--results-dir", default="evals/results")
    r.add_argument("--allow-unverified", action="store_true")
    r.add_argument("--quiet", action="store_true")
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("compare", help="paired comparison of two runs")
    c.add_argument("baseline")
    c.add_argument("candidate")
    c.set_defaults(func=cmd_compare)

    g = sub.add_parser("gate", help="check a run against the stored baseline")
    g.add_argument("run")
    g.add_argument("--baseline", default=DEFAULT_BASELINE)
    g.set_defaults(func=cmd_gate)

    b = sub.add_parser("baseline", help="promote a run to the stored baseline")
    b.add_argument("run")
    b.add_argument("--baseline", default=DEFAULT_BASELINE)
    b.set_defaults(func=cmd_baseline)

    ce = sub.add_parser("calibrate-export", help="sample a run for hand-labeling")
    ce.add_argument("run")
    ce.add_argument("--dataset", default=DEFAULT_DATASET)
    ce.add_argument("--out", default=DEFAULT_CALIBRATION)
    ce.add_argument("-n", type=int, default=100)
    ce.add_argument("--seed", type=int, default=20260906)
    ce.set_defaults(func=cmd_calibrate_export)

    cr = sub.add_parser("calibrate-report", help="judge/human agreement")
    cr.add_argument("--samples", default=DEFAULT_CALIBRATION)
    cr.add_argument(
        "--weighting", default="quadratic", choices=["unweighted", "linear", "quadratic"]
    )
    cr.set_defaults(func=cmd_calibrate_report)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except DatasetError as exc:
        print(f"dataset error: {exc}", file=sys.stderr)
        return 1
    except (FileNotFoundError, ImportError, AttributeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
