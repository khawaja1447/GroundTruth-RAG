#!/usr/bin/env python3
"""Run an ablation sweep and write one result file per configuration.

    python scripts/run_sweep.py --sweep ladder
    python scripts/run_sweep.py --sweep chunking
    python scripts/run_sweep.py --sweep both --docs corpus/documents

Each configuration is evaluated on the same questions, so the paired
bootstrap in `evals.report` can compare adjacent rows and say whether a
delta survives. The table is generated from the result files afterwards --
never typed by hand.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from evals.dataset import load_dataset  # noqa: E402
from evals.metrics.stats import paired_power  # noqa: E402
from evals.report import compare_runs, format_comparison  # noqa: E402
from evals.runner import RunConfig, run_eval  # noqa: E402
from evals.spans import coverage_report  # noqa: E402

from gtrag.ablation import (  # noqa: E402
    ABLATION_LADDER,
    CHUNKING_SWEEP,
    AblationConfig,
    build_system,
)
from gtrag.generate.generator import build_generator  # noqa: E402
from gtrag.ingest.document import Document  # noqa: E402
from gtrag.ingest.parse import parse_filing  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"

# The smoke corpus. The gold spans live in the first document; the other two
# are distractors with heavily overlapping vocabulary -- a prior year of the
# same company (near-identical boilerplate, different figures) and a peer in
# another sector. Without them the corpus is small enough that top-k returns
# everything, recall is 1.0 by construction, and the ablation measures nothing.
SMOKE_CORPUS: tuple[tuple[str, dict], ...] = (
    (
        "filing_sample.html",
        {
            "company": "Northwind Logistics, Inc.",
            "cik": 1234567,
            "form_type": "10-K",
            "fiscal_year": 2024,
            "accession": "0001234567-25-000001",
        },
    ),
    (
        "filing_prior_year.html",
        {
            "company": "Northwind Logistics, Inc.",
            "cik": 1234567,
            "form_type": "10-K",
            "fiscal_year": 2023,
            "accession": "0001234567-24-000001",
        },
    ),
    (
        "filing_peer.html",
        {
            "company": "Cascade Semiconductor Corp.",
            "cik": 7654321,
            "form_type": "10-K",
            "fiscal_year": 2024,
            "accession": "0007654321-25-000001",
        },
    ),
)


def load_documents(docs_dir: str | None) -> list[Document]:
    """Load the ingested corpus, or fall back to the test fixture.

    The fallback exists so the sweep is runnable before `make ingest` has
    been run -- and it prints which one it used, because a sweep over one
    fixture filing is a smoke test, not a result.
    """
    if docs_dir:
        files = sorted(Path(docs_dir).glob("*.json"))
        if files:
            print(f"corpus: {len(files)} ingested document(s) from {docs_dir}")
            return [Document.from_dict(json.loads(f.read_text(encoding="utf-8"))) for f in files]
        print(f"no documents in {docs_dir}; falling back to the test fixture", file=sys.stderr)

    print(
        f"corpus: {len(SMOKE_CORPUS)} test fixture filings "
        f"(smoke test -- run `make ingest` for real numbers)"
    )
    return [
        parse_filing((FIXTURES / name).read_text(encoding="utf-8"), metadata=meta)
        for name, meta in SMOKE_CORPUS
    ]


def run_one(
    config: AblationConfig,
    documents: list[Document],
    dataset,
    *,
    results_dir: str,
    workers: int,
    index_cache: dict,
    use_model: bool,
) -> object:
    system = build_system(
        config,
        documents,
        generator=build_generator(prefer_model=use_model),
        index_cache=index_cache,
    )

    # How well did this chunking preserve the labeled evidence? A question
    # that loses its evidence here is a property of the chunking, not a
    # retrieval failure, and it must be reported as such.
    spans = {q.id: q.gold_spans for q in dataset if q.gold_spans}
    coverage = coverage_report(spans, system.spanned_chunks) if spans else {}

    run_config = RunConfig(
        dataset_path=str(dataset.path),
        system_name=system.name,
        system_config=system.config,
        judge_model="null",
        judge_enabled=False,
        label=config.label,
    )
    result = run_eval(dataset, system, config=run_config, workers=workers)
    path = result.save(results_dir)

    ndcg = result.aggregates.get("ndcg@10", {}).get("mean")
    recall = result.aggregates.get("recall@10", {}).get("mean")
    lost = len(coverage.get("lost", []))
    print(
        f"  {config.label:<34} chunks={len(system.spanned_chunks):>4}  "
        f"nDCG@10={ndcg if ndcg is None else f'{ndcg:.4f}'}  "
        f"recall@10={recall if recall is None else f'{recall:.4f}'}"
        + (f"  [{lost} question(s) lost evidence]" if lost else "")
    )
    return result, coverage, path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", default="ladder", choices=["ladder", "chunking", "both"])
    parser.add_argument("--dataset", default="evals/datasets/qa_filing.jsonl")
    parser.add_argument("--docs", default=None, help="ingested document store")
    parser.add_argument("--results-dir", default="evals/results/sweep")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--use-model", action="store_true", help="real generation, not extractive")
    args = parser.parse_args()

    documents = load_documents(args.docs)
    dataset = load_dataset(args.dataset)
    print(f"dataset: {len(dataset)} questions from {args.dataset}\n")

    sweeps: list[tuple[str, tuple[AblationConfig, ...]]] = []
    if args.sweep in ("chunking", "both"):
        sweeps.append(("chunking (dimension 1)", CHUNKING_SWEEP))
    if args.sweep in ("ladder", "both"):
        sweeps.append(("ablation ladder (dimensions 1-5)", ABLATION_LADDER))

    index_cache: dict = {}
    all_results: list[tuple[AblationConfig, object]] = []

    for title, configs in sweeps:
        print(f"=== {title} ===")
        for config in configs:
            result, coverage, _ = run_one(
                config,
                documents,
                dataset,
                results_dir=args.results_dir,
                workers=args.workers,
                index_cache=index_cache,
                use_model=args.use_model,
            )
            all_results.append((config, result))
        print()

    # Adjacent-row comparisons. The ladder is built so each rung differs from
    # the one above by exactly one component, which is what makes the delta
    # attributable to that component.
    ladder = [(c, r) for c, r in all_results if c in ABLATION_LADDER]
    if len(ladder) > 1:
        print("=== attributable deltas (paired bootstrap, adjacent rungs) ===\n")
        for (prev_config, prev), (config, current) in zip(ladder, ladder[1:], strict=False):
            print(f"{prev_config.label}  ->  {config.label}")
            comparisons = compare_runs(prev, current, metrics=["ndcg@10", "recall@10", "mrr"])
            print(format_comparison(comparisons))
            print()

        # If every delta came back inconclusive, the next question is whether
        # the components do nothing or the eval set is too small to say --
        # opposite conclusions with opposite next actions.
        print("=== statistical power ===\n")
        first, last = ladder[0][1], ladder[-1][1]
        for metric in ("ndcg@10", "recall@10"):
            power = paired_power(
                metric,
                first.per_question_scores(metric),
                last.per_question_scores(metric),
                target_effect=0.02,
            )
            if power is not None:
                print(f"  {power.format()}")
        print()

    print(f"{len(all_results)} result file(s) in {args.results_dir}")
    print(
        "Regenerate the table with: python scripts/build_ablation_table.py "
        f"--results {args.results_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
