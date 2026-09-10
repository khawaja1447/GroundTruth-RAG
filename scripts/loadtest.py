#!/usr/bin/env python3
"""Characterise the service under concurrency.

    python scripts/loadtest.py --requests 300 --concurrency 16
    python scripts/loadtest.py --ramp
    python scripts/loadtest.py --url http://localhost:8000/query   # a live server

By default it drives `RagService` in-process, which measures the pipeline
without the HTTP layer's noise. `--url` drives a real server instead, which is
what you want before claiming a p99.

`--ramp` sweeps concurrency and reports the knee -- where p95 starts to
degrade. "Where it breaks" is the part of a load test that is actually
useful, and a single throughput number never answers it.

Stdlib only, so this runs anywhere the package does.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from gtrag.ablation import AblationConfig, build_system  # noqa: E402
from gtrag.index.embed import HashingEmbedder  # noqa: E402
from gtrag.serve.cache import SemanticCache  # noqa: E402
from gtrag.serve.service import QueryRequest, RagService  # noqa: E402
from gtrag.serve.telemetry import percentile  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from run_sweep import load_documents  # noqa: E402

# A realistic mix: repeats the cache should catch, paraphrases only a semantic
# cache catches, distinct questions, and an unanswerable one. A load test on a
# single repeated question measures the cache, not the system.
QUERY_MIX: tuple[str, ...] = (
    "What was Northwind's total net revenue in fiscal 2024?",
    "Northwind fiscal 2024 net revenue figure",
    "What was Northwind's Ground segment revenue in fiscal 2024?",
    "Which Northwind segment grew fastest in fiscal 2024?",
    "What was Cascade Semiconductor's gross margin?",
    "How does Northwind recover fuel cost increases?",
    "What risks does Northwind face from driver availability?",
    "How many employees did Northwind have at the end of fiscal 2024?",
)


@dataclass
class Outcome:
    latency_ms: float
    ok: bool
    cache_hit: bool = False
    error: str = ""


def report(outcomes: list[Outcome], wall_seconds: float, concurrency: int) -> dict:
    latencies = [o.latency_ms for o in outcomes if o.ok]
    errors = [o for o in outcomes if not o.ok]
    hits = sum(1 for o in outcomes if o.cache_hit)

    stats = {
        "concurrency": concurrency,
        "requests": len(outcomes),
        "errors": len(errors),
        "error_rate": len(errors) / len(outcomes) if outcomes else 0.0,
        "cache_hit_rate": hits / len(outcomes) if outcomes else 0.0,
        "throughput_rps": len(outcomes) / wall_seconds if wall_seconds else 0.0,
        "wall_seconds": round(wall_seconds, 3),
    }
    if latencies:
        stats.update(
            {
                "p50_ms": round(percentile(latencies, 50), 2),
                "p95_ms": round(percentile(latencies, 95), 2),
                "p99_ms": round(percentile(latencies, 99), 2),
                "max_ms": round(max(latencies), 2),
                "mean_ms": round(statistics.mean(latencies), 2),
            }
        )
    return stats


def run_in_process(
    service: RagService, n: int, concurrency: int, use_cache: bool
) -> tuple[list[Outcome], float]:
    def one(i: int) -> Outcome:
        question = QUERY_MIX[i % len(QUERY_MIX)]
        start = time.perf_counter()
        try:
            result = service.query(QueryRequest(question=question, use_cache=use_cache))
            elapsed = (time.perf_counter() - start) * 1000.0
            return Outcome(elapsed, result.quality != "unavailable", result.cache_hit)
        except Exception as exc:  # noqa: BLE001
            return Outcome((time.perf_counter() - start) * 1000.0, False, error=str(exc))

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        outcomes = list(pool.map(one, range(n)))
    return outcomes, time.perf_counter() - started


def run_over_http(
    url: str, n: int, concurrency: int, use_cache: bool
) -> tuple[list[Outcome], float]:
    def one(i: int) -> Outcome:
        payload = json.dumps(
            {"question": QUERY_MIX[i % len(QUERY_MIX)], "use_cache": use_cache}
        ).encode()
        request = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}
        )
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = json.loads(response.read())
            elapsed = (time.perf_counter() - start) * 1000.0
            return Outcome(elapsed, True, bool(body.get("cache_hit")))
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return Outcome((time.perf_counter() - start) * 1000.0, False, error=str(exc))

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        outcomes = list(pool.map(one, range(n)))
    return outcomes, time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--url", default=None, help="drive a live server instead of in-process")
    parser.add_argument("--docs", default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--ramp", action="store_true", help="sweep concurrency to find the knee")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    service: RagService | None = None
    if args.url is None:
        documents = load_documents(args.docs)
        system = build_system(
            AblationConfig(label="loadtest", chunker="structure_aware", bm25=True), documents
        )
        service = RagService(
            system, cache=SemanticCache(HashingEmbedder(), corpus_version="loadtest")
        )
        print(f"target:  in-process ({len(system.spanned_chunks)} chunks)")
    else:
        print(f"target:  {args.url}")
    print(
        f"mix:     {len(QUERY_MIX)} distinct questions, cache {'off' if args.no_cache else 'on'}\n"
    )

    levels = [1, 2, 4, 8, 16, 32] if args.ramp else [args.concurrency]
    rows: list[dict] = []

    header = (
        f"{'conc':>5} {'reqs':>6} {'p50 ms':>9} {'p95 ms':>9} {'p99 ms':>9} "
        f"{'rps':>9} {'errors':>7} {'cache':>7}"
    )
    print(header)
    print("-" * len(header))

    for concurrency in levels:
        if service is not None:
            if service.cache:
                service.cache.clear()
            outcomes, wall = run_in_process(service, args.requests, concurrency, not args.no_cache)
        else:
            outcomes, wall = run_over_http(args.url, args.requests, concurrency, not args.no_cache)
        stats = report(outcomes, wall, concurrency)
        rows.append(stats)
        print(
            f"{concurrency:>5} {stats['requests']:>6} "
            f"{stats.get('p50_ms', 0):>9.2f} {stats.get('p95_ms', 0):>9.2f} "
            f"{stats.get('p99_ms', 0):>9.2f} {stats['throughput_rps']:>9.1f} "
            f"{stats['errors']:>7} {stats['cache_hit_rate']:>6.0%}"
        )

    if args.ramp and len(rows) > 1:
        # The knee: where p95 first degrades materially against the
        # single-threaded baseline. Throughput alone hides this -- it keeps
        # climbing while individual requests get slower.
        baseline = rows[0].get("p95_ms", 0.0) or 1.0
        knee = next((r for r in rows[1:] if r.get("p95_ms", 0.0) > baseline * 2.0), None)
        print()
        if knee:
            print(
                f"knee: p95 exceeds 2x the single-threaded baseline ({baseline:.2f}ms) "
                f"at concurrency {knee['concurrency']} ({knee['p95_ms']:.2f}ms)"
            )
        else:
            print(
                f"no knee up to concurrency {levels[-1]}: p95 stayed under 2x the "
                f"{baseline:.2f}ms baseline"
            )

        # A p95 threshold crossing on one run is not a capacity limit. On the
        # in-process pipeline these timings are sub-millisecond and the
        # reported knee has been observed to land at 8, 16 and 32 across
        # consecutive runs of the same commit -- i.e. it was noise, and
        # docs/serving.md 1 records a page that once published one of those
        # as a result. Say so here rather than letting the line above be
        # quoted on its own.
        p50s = [r.get("p50_ms", 0.0) for r in rows]
        if p50s and max(p50s) < 2.0 * (min(p50s) or 1.0):
            print(
                "p50 is flat across the ramp, so nothing is queueing: treat the knee "
                "as a single sample, not a capacity limit. Re-run before quoting it."
            )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
