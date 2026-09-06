#!/usr/bin/env python3
"""Measure the refusal tradeoff and pick an operating point.

    python scripts/refusal_curve.py
    python scripts/refusal_curve.py --signal margin --max-false-refusal 0.10

Sweeps the refusal threshold over the observed confidence range and reports,
at each point, how many unanswerable questions were correctly declined
against how many answerable ones were wrongly declined. The operating point
is then chosen by a stated criterion rather than an implied one.

Phase 4's exit gate asks for exactly this: a refusal threshold chosen from a
measured curve and documented, not a number someone liked the look of.
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

from gtrag.ablation import AblationConfig, build_system  # noqa: E402
from gtrag.generate.refusal import (  # noqa: E402
    RefusalObservation,
    choose_operating_point,
    confidence_of,
    format_curve,
    refusal_curve,
)

sys.path.insert(0, str(ROOT / "scripts"))
from run_sweep import load_documents  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="evals/datasets/qa_filing.jsonl")
    parser.add_argument("--docs", default=None)
    parser.add_argument(
        "--signal", default="top_score", choices=["top_score", "mean_score", "margin"]
    )
    parser.add_argument("--max-false-refusal", type=float, default=0.05)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out", default=None, help="write the curve as JSON here")
    args = parser.parse_args()

    documents = load_documents(args.docs)
    dataset = load_dataset(args.dataset)

    # The retrieval configuration the threshold will be deployed with. A
    # threshold tuned on one retriever does not transfer to another, because
    # the score scales differ -- so the curve is always measured on the
    # configuration it will be used with.
    config = AblationConfig(label="refusal-calibration", chunker="structure_aware", bm25=True)
    system = build_system(config, documents)

    observations: list[RefusalObservation] = []
    for question in dataset:
        retrieved = system.retriever.retrieve(question.question, top_k=args.top_k)
        confidence = confidence_of(retrieved)
        observations.append(
            RefusalObservation(
                question_id=question.id,
                confidence=float(getattr(confidence, args.signal)),
                answerable=question.answerable,
            )
        )

    n_ans = sum(1 for o in observations if o.answerable)
    print(f"signal:  {args.signal}")
    print(f"corpus:  {len(system.spanned_chunks)} chunks")
    print(
        f"dataset: {len(observations)} questions "
        f"({n_ans} answerable, {len(observations) - n_ans} not)\n"
    )

    curve = refusal_curve(observations)
    chosen = choose_operating_point(curve, max_false_refusal=args.max_false_refusal)
    print(format_curve(curve, chosen=chosen))

    print()
    if chosen is None:
        # A real finding, not an error: this signal does not separate the two
        # classes at the requested ceiling.
        print(
            f"NO OPERATING POINT satisfies false refusal <= {args.max_false_refusal:.0%}.\n"
            f"The confidence signal does not separate answerable from unanswerable "
            f"questions well enough at that ceiling. Options, in order of honesty:\n"
            f"  - raise the ceiling and accept more false refusals;\n"
            f"  - try another signal (--signal margin);\n"
            f"  - improve retrieval, since the signal is derived from it;\n"
            f"  - leave refusal to the generator, which sees the passages."
        )
        exit_code = 1
    else:
        print(
            f"OPERATING POINT: threshold={chosen.threshold:.4f} on {args.signal}\n"
            f"  correct refusals: {chosen.correct_refusal_rate:.1%} "
            f"({chosen.correct_refusals}/{chosen.n_unanswerable} unanswerable)\n"
            f"  false refusals:   {chosen.false_refusal_rate:.1%} "
            f"({chosen.false_refusals}/{chosen.n_answerable} answerable)\n"
            f"  criterion: maximise correct refusals subject to "
            f"false refusal <= {args.max_false_refusal:.0%}"
        )
        exit_code = 0

    if args.out:
        payload = {
            "signal": args.signal,
            "max_false_refusal": args.max_false_refusal,
            "system_config": system.config,
            "curve": [p.to_dict() for p in curve],
            "chosen": chosen.to_dict() if chosen else None,
            "observations": [
                {
                    "question_id": o.question_id,
                    "confidence": o.confidence,
                    "answerable": o.answerable,
                }
                for o in observations
            ],
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwritten to {args.out}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
