#!/usr/bin/env python3
"""Regenerate every number the repository publishes.

    python scripts/reproduce.py            # run everything, print a report
    python scripts/reproduce.py --check    # also fail if a figure or a claim moved

This is what makes the README's figures claims rather than decoration: one
command reproduces all of them from source, and `--check` compares what it
gets against what is written down.

There are two contracts here, because there are two ways for a published
number to become a lie:

- `PUBLISHED` is what the code must still produce. When a number in it
  legitimately changes, the change is a visible edit to this file next to the
  code change that caused it -- which is the point. A number that can drift
  silently is not a result.
- `DOC_CLAIMS` is how each number is written down. Correcting `PUBLISHED`
  while a paragraph three files away still quotes the old figure is exactly
  how the superseded refusal conclusion in `docs/generation.md` survived; the
  second contract closes that.
"""

from __future__ import annotations

import argparse
import io
import json
import platform
import sys
import time
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evals.dataset import load_dataset  # noqa: E402
from evals.metrics.stats import paired_power  # noqa: E402
from evals.runner import RunConfig, run_eval  # noqa: E402

from gtrag.ablation import (  # noqa: E402
    ABLATION_LADDER,
    CHUNKING_SWEEP,
    GENERATION_LADDER,
    build_system,
)
from gtrag.generate.refusal import (  # noqa: E402
    RefusalObservation,
    confidence_of,
    refusal_curve,
)
from run_sweep import load_documents  # noqa: E402

DATASET = "evals/datasets/qa_filing.jsonl"

# Every figure the README and docs state. Tolerances are tight because the
# whole pipeline is deterministic -- a moved number means a changed behaviour,
# not noise.
PUBLISHED: dict[str, tuple[float, float]] = {
    # docs/ablation.md, retrieval ladder
    "ladder.baseline.ndcg@10": (0.8196, 0.0005),
    "ladder.structure_aware.ndcg@10": (0.7527, 0.0005),
    "ladder.bm25.ndcg@10": (0.8297, 0.0005),
    "ladder.reranking.ndcg@10": (0.7859, 0.0005),
    "ladder.metadata.ndcg@10": (0.8663, 0.0005),
    # docs/ablation.md, chunking sweep
    "chunking.fixed.ndcg@10": (0.8196, 0.0005),
    "chunking.recursive.ndcg@10": (0.8326, 0.0005),
    "chunking.structure_aware.ndcg@10": (0.7527, 0.0005),
    "chunking.sentence_window.ndcg@10": (0.2678, 0.0005),
    "chunking.parent_document.ndcg@10": (0.5482, 0.0005),
    "chunking.semantic.ndcg@10": (0.7225, 0.0005),
    # docs/ablation.md, statistical power
    "power.ndcg@10.required_n": (723, 5),
    "power.recall@10.required_n": (739, 5),
    # docs/generation.md, generation ladder
    "generation.no_refusal.answered_unanswerable": (1.0, 0.0001),
    "generation.refusal.false_refusal": (0.0769, 0.0005),
    # docs/generation.md, refusal signal separability
    "refusal.margin.best_j": (0.3462, 0.001),
    "refusal.mean_score.best_j": (0.4423, 0.001),
    "refusal.top_score.best_j": (0.6731, 0.001),
}

# How each figure is written down, and where. Re-deriving a number proves the
# code still produces it; it says nothing about whether the prose was updated
# to match. These are the exact strings the documents must contain -- so a
# figure that moves fails the check twice, once for the value and once for
# every sentence that still quotes the old one.
#
# Prose is written for a reader, so one figure can appear in several renderings
# (0.0769 as "0.0769" and as "7.7%"). Every rendering listed must be present.
DOC_CLAIMS: dict[str, tuple[str, ...]] = {
    "ladder.baseline.ndcg@10": ("0.8196",),
    "ladder.structure_aware.ndcg@10": ("0.7527",),
    "ladder.bm25.ndcg@10": ("0.8297",),
    "ladder.reranking.ndcg@10": ("0.7859",),
    "ladder.metadata.ndcg@10": ("0.8663",),
    "chunking.fixed.ndcg@10": ("0.8196",),
    "chunking.recursive.ndcg@10": ("0.8326",),
    "chunking.structure_aware.ndcg@10": ("0.7527",),
    "chunking.sentence_window.ndcg@10": ("0.2678",),
    "chunking.parent_document.ndcg@10": ("0.5482",),
    "chunking.semantic.ndcg@10": ("0.7225",),
    "power.ndcg@10.required_n": ("need ~723", "**720 labeled questions**"),
    "power.recall@10.required_n": ("need ~739",),
    "generation.no_refusal.answered_unanswerable": ("100.0%",),
    "generation.refusal.false_refusal": ("7.7%",),
    "refusal.margin.best_j": ("+0.346",),
    "refusal.mean_score.best_j": ("+0.442",),
    "refusal.top_score.best_j": ("+0.673",),
}

# The operating point is chosen by `refusal_curve.py` from the curve above
# rather than being an aggregate, so it is checked as prose only.
DOC_CLAIMS_UNMEASURED: tuple[str, ...] = ("threshold=0.0324 on top_score",)

DOCS = ("README.md", "docs/ablation.md", "docs/generation.md", "docs/serving.md")


def check_docs() -> list[str]:
    """Return the claims no document makes. Empty means the prose is current."""
    corpus = "\n".join((ROOT / name).read_text(encoding="utf-8") for name in DOCS)
    missing = [
        f"{key}: {claim!r}"
        for key, claims in DOC_CLAIMS.items()
        for claim in claims
        if claim not in corpus
    ]
    missing += [f"operating point: {c!r}" for c in DOC_CLAIMS_UNMEASURED if c not in corpus]
    return missing


@dataclass
class Measurement:
    key: str
    value: float
    published: float | None
    tolerance: float | None

    @property
    def status(self) -> str:
        if self.published is None:
            return "new"
        return "ok" if abs(self.value - self.published) <= (self.tolerance or 0.0) else "MOVED"


def evaluate(configs, documents, dataset, cache: dict) -> dict[str, object]:
    out = {}
    for config in configs:
        system = build_system(config, documents, index_cache=cache)
        run_config = RunConfig(
            str(dataset.path), system.name, system.config, "null", False, label=config.label
        )
        out[config.label] = run_eval(dataset, system, config=run_config, workers=4)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if a published number moved")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    started = time.time()
    with redirect_stdout(io.StringIO()):
        documents = load_documents(None)
    dataset = load_dataset(DATASET)
    cache: dict = {}

    print("groundtruth-rag: reproducing published figures")
    print(f"  python   {platform.python_version()} on {platform.system()}")
    print(f"  corpus   {len(documents)} documents")
    print(f"  dataset  {len(dataset)} questions\n")

    measurements: list[Measurement] = []

    def record(key: str, value: float) -> None:
        published, tolerance = PUBLISHED.get(key, (None, None))
        measurements.append(Measurement(key, float(value), published, tolerance))

    # -- retrieval ladder ------------------------------------------------
    ladder = evaluate(ABLATION_LADDER, documents, dataset, cache)
    labels = [c.label for c in ABLATION_LADDER]
    record("ladder.baseline.ndcg@10", ladder[labels[0]].aggregates["ndcg@10"]["mean"])
    record("ladder.structure_aware.ndcg@10", ladder[labels[1]].aggregates["ndcg@10"]["mean"])
    record("ladder.bm25.ndcg@10", ladder[labels[2]].aggregates["ndcg@10"]["mean"])
    record("ladder.reranking.ndcg@10", ladder[labels[3]].aggregates["ndcg@10"]["mean"])
    record("ladder.metadata.ndcg@10", ladder[labels[4]].aggregates["ndcg@10"]["mean"])

    # The same pair `run_sweep.py` reports power over: the bottom and top of
    # the ladder. Pinning a different pair here would publish a number no
    # command prints.
    for metric in ("ndcg@10", "recall@10"):
        power = paired_power(
            metric,
            ladder[labels[0]].per_question_scores(metric),
            ladder[labels[-1]].per_question_scores(metric),
            target_effect=0.02,
        )
        if power is not None:
            record(f"power.{metric}.required_n", power.required_n)

    # -- chunking sweep --------------------------------------------------
    chunking = evaluate(CHUNKING_SWEEP, documents, dataset, cache)
    for config in CHUNKING_SWEEP:
        key = f"chunking.{config.chunker}.ndcg@10"
        if key in PUBLISHED:
            record(key, chunking[config.label].aggregates["ndcg@10"]["mean"])

    # -- generation ladder -----------------------------------------------
    generation = evaluate(GENERATION_LADDER, documents, dataset, cache)
    gen_labels = [c.label for c in GENERATION_LADDER]
    record(
        "generation.no_refusal.answered_unanswerable",
        generation[gen_labels[0]].aggregates["answered_unanswerable"]["mean"],
    )
    record(
        "generation.refusal.false_refusal",
        generation[gen_labels[-1]].aggregates["false_refusal"]["mean"],
    )

    # -- refusal separability --------------------------------------------
    system = build_system(ABLATION_LADDER[2], documents, index_cache=cache)
    for signal in ("margin", "mean_score", "top_score"):
        observations = [
            RefusalObservation(
                q.id,
                float(
                    getattr(confidence_of(system.retriever.retrieve(q.question, top_k=5)), signal)
                ),
                q.answerable,
            )
            for q in dataset
        ]
        useful = [
            p for p in refusal_curve(observations) if not p.degenerate and p.youden_j is not None
        ]
        if useful:
            record(f"refusal.{signal}.best_j", max(p.youden_j for p in useful))

    # -- report ------------------------------------------------------------
    width = max(len(m.key) for m in measurements)
    print(f"{'figure':<{width}} {'reproduced':>12} {'published':>11} {'':>7}")
    print("-" * (width + 34))
    moved = 0
    for m in measurements:
        published = "—" if m.published is None else f"{m.published:.4f}"
        if m.status == "MOVED":
            moved += 1
        print(f"{m.key:<{width}} {m.value:>12.4f} {published:>11} {m.status:>7}")

    stale = check_docs()
    elapsed = time.time() - started
    print(f"\n{len(measurements)} figures reproduced in {elapsed:.1f}s, {moved} moved.")
    if stale:
        print(f"{len(stale)} documented claim(s) no longer appear in the docs:")
        for claim in stale:
            print(f"  {claim}")
    else:
        print(f"{len(DOCS)} documents check out: every figure is quoted as measured.")
    print("Cost: $0.00 — every published figure is produced without a model call.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(
                {
                    "python": platform.python_version(),
                    "elapsed_seconds": round(elapsed, 2),
                    "stale_doc_claims": stale,
                    "figures": [
                        {
                            "key": m.key,
                            "value": m.value,
                            "published": m.published,
                            "status": m.status,
                        }
                        for m in measurements
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"written to {args.out}")

    if args.check and (moved or stale):
        if moved:
            print(
                "\nA published figure moved. Either the change is a regression, or it "
                "is intended -- in which case update PUBLISHED in this file and the "
                "docs together, in the same commit as the code change.",
                file=sys.stderr,
            )
        if stale:
            print(
                "\nA documented claim is no longer in the docs. The code and the prose "
                "have diverged; correct the sentence rather than deleting the claim.",
                file=sys.stderr,
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
