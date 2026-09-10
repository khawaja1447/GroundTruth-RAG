"""Spans, metrics, and per-request cost accounting.

Stdlib-only by design, with OpenTelemetry as an optional export target. The
tracer here is deliberately small: what a RAG service actually needs is
per-stage latency, token counts, cost, and the ability to answer "why was
*this* request slow" from a trace id -- and that is a few hundred lines, not
a dependency.

Two things this exists to make possible:

  * **Attribution.** A p95 that says "1.2s" is not actionable. A p95 broken
    down by stage says the reranker is the problem.
  * **Cost per query, measured.** Not estimated from a price list at the end
    of the month, but accumulated per request from reported token usage.
"""

from __future__ import annotations

import contextlib
import threading
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Span", "Trace", "Tracer", "MetricsRegistry", "percentile", "new_trace_id"]


def new_trace_id() -> str:
    return uuid.uuid4().hex[:16]


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile over a sorted copy."""
    if not values:
        raise ValueError("percentile of empty sequence")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * (q / 100.0)
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return ordered[low] * (1 - frac) + ordered[high] * frac


@dataclass
class Span:
    name: str
    start: float
    end: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def duration_ms(self) -> float:
        return ((self.end or time.perf_counter()) - self.start) * 1000.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "duration_ms": round(self.duration_ms, 3),
            "attributes": self.attributes,
            "error": self.error,
        }


@dataclass
class Trace:
    """One request's spans, keyed by a trace id the client also receives.

    The id is returned in the response and stamped on every log line, so a
    user reporting "this answer was wrong" hands you a string that pulls up
    exactly which chunks were retrieved and how long each stage took.
    """

    trace_id: str
    spans: list[Span] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return sum(s.duration_ms for s in self.spans)

    def stage_timings(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for span in self.spans:
            out[span.name] = out.get(span.name, 0.0) + span.duration_ms
        out["total"] = sum(out.values())
        return {k: round(v, 3) for k, v in out.items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "duration_ms": round(self.duration_ms, 3),
            "spans": [s.to_dict() for s in self.spans],
            "attributes": self.attributes,
        }


class Tracer:
    """Creates traces and records spans.

    `otel_exporter` is any object with `export(trace_dict)`. Keeping the
    integration to that one method means OpenTelemetry is an optional
    adapter, not a hard dependency of the request path.
    """

    def __init__(self, otel_exporter: Any = None) -> None:
        self.otel_exporter = otel_exporter
        self._local = threading.local()

    @property
    def current(self) -> Trace | None:
        return getattr(self._local, "trace", None)

    @contextmanager
    def trace(self, trace_id: str | None = None, **attributes: Any) -> Iterator[Trace]:
        trace = Trace(trace_id=trace_id or new_trace_id(), attributes=dict(attributes))
        previous = self.current
        self._local.trace = trace
        try:
            yield trace
        finally:
            self._local.trace = previous
            if self.otel_exporter is not None:
                # Telemetry must never break the request it is measuring.
                with contextlib.suppress(Exception):
                    self.otel_exporter.export(trace.to_dict())

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Span]:
        span = Span(name=name, start=time.perf_counter(), attributes=dict(attributes))
        trace = self.current
        if trace is not None:
            trace.spans.append(span)
        try:
            yield span
        except Exception as exc:
            span.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            span.end = time.perf_counter()


class MetricsRegistry:
    """In-process counters, gauges and latency histograms.

    Bounded ring buffers rather than unbounded lists: a long-running server
    accumulating every latency it has ever seen is a memory leak that only
    shows up in production.
    """

    def __init__(self, window: int = 2000) -> None:
        self.window = window
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = {}
        self._latencies: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=window))
        self._lock = threading.Lock()

    def increment(self, name: str, value: float = 1.0) -> None:
        with self._lock:
            self._counters[name] += value

    def gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = value

    def observe(self, name: str, value_ms: float) -> None:
        with self._lock:
            self._latencies[name].append(value_ms)

    def record_trace(self, trace: Trace) -> None:
        for name, duration in trace.stage_timings().items():
            self.observe(f"latency.{name}", duration)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            counters = dict(self._counters)
            gauges = dict(self._gauges)
            latencies = {name: list(values) for name, values in self._latencies.items()}

        histograms: dict[str, dict[str, float]] = {}
        for name, values in latencies.items():
            if not values:
                continue
            histograms[name] = {
                "count": len(values),
                "p50": round(percentile(values, 50), 3),
                "p95": round(percentile(values, 95), 3),
                "p99": round(percentile(values, 99), 3),
                "max": round(max(values), 3),
            }

        requests = counters.get("requests.total", 0.0)
        return {
            "counters": counters,
            "gauges": gauges,
            "latency_ms": histograms,
            "derived": {
                "error_rate": (counters.get("requests.error", 0.0) / requests)
                if requests
                else None,
                "refusal_rate": (
                    (counters.get("requests.refused", 0.0) / requests) if requests else None
                ),
                "cache_hit_rate": (
                    counters.get("cache.hits", 0.0)
                    / (counters.get("cache.hits", 0.0) + counters.get("cache.misses", 0.0))
                    if (counters.get("cache.hits", 0.0) + counters.get("cache.misses", 0.0))
                    else None
                ),
                "cost_per_query_usd": (
                    (counters.get("cost.usd.total", 0.0) / requests) if requests else None
                ),
            },
        }

    def render_prometheus(self) -> str:
        """Prometheus text exposition, so this scrapes without an adapter."""
        snapshot = self.snapshot()
        lines: list[str] = []
        for name, value in sorted(snapshot["counters"].items()):
            metric = "gtrag_" + name.replace(".", "_")
            lines.append(f"# TYPE {metric} counter")
            lines.append(f"{metric} {value}")
        for name, value in sorted(snapshot["gauges"].items()):
            metric = "gtrag_" + name.replace(".", "_")
            lines.append(f"# TYPE {metric} gauge")
            lines.append(f"{metric} {value}")
        for name, stats in sorted(snapshot["latency_ms"].items()):
            metric = "gtrag_" + name.replace(".", "_")
            lines.append(f"# TYPE {metric} summary")
            for quantile in ("p50", "p95", "p99"):
                lines.append(f'{metric}{{quantile="{quantile[1:]}"}} {stats[quantile]}')
            lines.append(f"{metric}_count {stats['count']}")
        return "\n".join(lines) + "\n"
