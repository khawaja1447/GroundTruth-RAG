"""The service: a RAG system wrapped in cache, telemetry and degradation.

Deliberately framework-free. Everything the HTTP layer needs is here, so the
FastAPI adapter is a thin translation of request to `QueryRequest` and
`QueryResult` to JSON -- and the whole request path is testable without
starting a server.

The ordering is load-bearing:

    trace -> cache lookup -> [degradable pipeline] -> cache store -> metrics

Cache before the pipeline, because a hit should cost a vector comparison and
nothing else. Metrics last, so a cache hit is still counted as a request.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..types import SystemResponse
from .cache import SemanticCache
from .degradation import CircuitBreaker, Degradable, DegradationReport
from .telemetry import MetricsRegistry, Tracer, new_trace_id

__all__ = ["QueryRequest", "QueryResult", "RagService"]


@dataclass(frozen=True, slots=True)
class QueryRequest:
    question: str
    history: tuple[tuple[str, str], ...] = ()
    top_k: int | None = None
    trace_id: str = ""
    use_cache: bool = True
    # Roles the caller holds. Phase 6 enforces these at retrieval; carried
    # here so the cache can be partitioned by them rather than leaking one
    # user's answer to another.
    roles: tuple[str, ...] = ()

    def cache_key(self) -> str:
        """Questions with different history or permissions are different questions."""
        parts = [self.question]
        if self.history:
            parts.append("|".join(f"{q}->{a}" for q, a in self.history))
        if self.roles:
            parts.append("roles:" + ",".join(sorted(self.roles)))
        return "\x00".join(parts)


@dataclass(frozen=True, slots=True)
class QueryResult:
    """What the service returns: the answer plus everything needed to audit it."""

    trace_id: str
    answer: str
    refused: bool
    citations: tuple[dict[str, Any], ...]
    passages: tuple[dict[str, Any], ...]
    timings: dict[str, float]
    usage: dict[str, float]
    quality: str = "full"
    degraded_components: dict[str, str] = field(default_factory=dict)
    cache_hit: bool = False
    error: str | None = None
    trace: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "answer": self.answer,
            "refused": self.refused,
            "citations": list(self.citations),
            "passages": list(self.passages),
            "timings_ms": self.timings,
            "usage": self.usage,
            "quality": self.quality,
            "degraded_components": self.degraded_components,
            "cache_hit": self.cache_hit,
            "error": self.error,
        }


def _passage_dict(chunk: Any) -> dict[str, Any]:
    return {
        "rank": chunk.rank,
        "chunk_id": chunk.chunk_id,
        "score": chunk.score,
        "text": chunk.text,
        "company": chunk.metadata.get("company"),
        "fiscal_year": chunk.metadata.get("fiscal_year"),
        "section": chunk.metadata.get("section"),
        "source_url": chunk.metadata.get("source_url"),
    }


class RagService:
    """Production wrapper around a `RagSystem`."""

    def __init__(
        self,
        system: Any,
        *,
        cache: SemanticCache | None = None,
        tracer: Tracer | None = None,
        metrics: MetricsRegistry | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.system = system
        self.cache = cache
        self.tracer = tracer or Tracer()
        self.metrics = metrics or MetricsRegistry()
        self.timeout_seconds = timeout_seconds
        self.started_at = time.time()

        # The pipeline is one required stage. A failure inside it degrades to
        # an error rather than a fabricated answer: "degraded" must never mean
        # "answered without retrieving".
        self.pipeline = Degradable(
            name="pipeline",
            breaker=CircuitBreaker(name="pipeline", failure_threshold=5, recovery_seconds=30.0),
            required=True,
        )

    # -- health -----------------------------------------------------------

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.started_at

    def health(self) -> dict[str, Any]:
        """Liveness plus the state of every breaker.

        Reports `degraded` rather than unhealthy when a breaker is open: the
        service is still serving, and a load balancer should keep sending it
        traffic while a human looks at the component.
        """
        breaker = self.pipeline.breaker.to_dict()
        healthy = breaker["state"] == "closed"
        return {
            "status": "ok" if healthy else "degraded",
            "uptime_seconds": round(self.uptime_seconds, 1),
            "corpus_chunks": len(getattr(self.system, "spanned_chunks", []) or []),
            "breakers": [breaker],
            "cache": self.cache.stats.to_dict() if self.cache else None,
        }

    def config(self) -> dict[str, Any]:
        config = dict(getattr(self.system, "config", {}) or {})
        if self.cache:
            config.update(self.cache.config)
        return config

    # -- the request path -------------------------------------------------

    def query(self, request: QueryRequest) -> QueryResult:
        trace_id = request.trace_id or new_trace_id()
        report = DegradationReport()
        self.metrics.increment("requests.total")

        with self.tracer.trace(trace_id, question=request.question) as trace:
            if self.cache is not None and request.use_cache:
                with self.tracer.span("cache_lookup"):
                    cached = self.cache.get(request.cache_key())
                if cached is not None:
                    self.metrics.increment("cache.hits")
                    self.metrics.record_trace(trace)
                    return self._to_result(trace_id, cached, report, trace, cache_hit=True)
                self.metrics.increment("cache.misses")

            try:
                with self.tracer.span("pipeline"):
                    response = self.pipeline.run(
                        lambda: self.system.answer(
                            request.question,
                            history=list(request.history) if request.history else None,
                        ),
                        report,
                    )
            except RuntimeError as exc:
                self.metrics.increment("requests.error")
                self.metrics.record_trace(trace)
                return QueryResult(
                    trace_id=trace_id,
                    answer="",
                    refused=False,
                    citations=(),
                    passages=(),
                    timings=trace.stage_timings(),
                    usage={},
                    quality="unavailable",
                    degraded_components=report.degraded,
                    error=str(exc),
                    trace=trace.to_dict(),
                )

            assert response is not None  # required=True guarantees this
            if self.cache is not None and request.use_cache:
                self.cache.put(request.cache_key(), response)

            if response.refused:
                self.metrics.increment("requests.refused")
            if response.error:
                self.metrics.increment("requests.error")
            self.metrics.increment("cost.usd.total", response.usage.cost_usd)
            self.metrics.increment("tokens.input", response.usage.input_tokens)
            self.metrics.increment("tokens.output", response.usage.output_tokens)
            self.metrics.record_trace(trace)
            if self.cache is not None:
                self.metrics.gauge("cache.entries", len(self.cache))

            return self._to_result(trace_id, response, report, trace)

    def stream(self, request: QueryRequest) -> Iterator[dict[str, Any]]:
        """Server-sent events for one query.

        Passages are emitted **before** the answer. The user sees what the
        system is reading while the model is still writing, which is both
        better perceived latency and better epistemics -- the sources arrive
        with the claim rather than after it.
        """
        result = self.query(request)

        yield {"event": "trace", "data": {"trace_id": result.trace_id, "quality": result.quality}}
        yield {"event": "passages", "data": {"passages": list(result.passages)}}

        if result.error:
            yield {"event": "error", "data": {"error": result.error}}
        elif result.refused:
            yield {"event": "refusal", "data": {"reason": "insufficient evidence in the corpus"}}
        else:
            # Chunked so a client renders progressively. A real streaming
            # generator would push these as the model produces them; the
            # boundary is the same either way.
            for piece in _chunk_text(result.answer):
                yield {"event": "token", "data": {"text": piece}}

        yield {
            "event": "done",
            "data": {
                "citations": list(result.citations),
                "timings_ms": result.timings,
                "usage": result.usage,
                "cache_hit": result.cache_hit,
            },
        }

    # -- internals --------------------------------------------------------

    def _to_result(
        self,
        trace_id: str,
        response: SystemResponse,
        report: DegradationReport,
        trace: Any,
        *,
        cache_hit: bool = False,
    ) -> QueryResult:
        return QueryResult(
            trace_id=trace_id,
            answer=response.answer,
            refused=response.refused,
            citations=tuple(
                {"claim_index": c.claim_index, "chunk_ids": list(c.chunk_ids), "text": c.text}
                for c in response.citations
            ),
            passages=tuple(
                _passage_dict(c) for c in sorted(response.retrieved, key=lambda c: c.rank)
            ),
            timings=trace.stage_timings(),
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "cost_usd": round(response.usage.cost_usd, 6),
            },
            quality=report.quality_flag,
            degraded_components=dict(report.degraded),
            cache_hit=cache_hit,
            error=response.error,
            trace=trace.to_dict(),
        )


def _chunk_text(text: str, size: int = 48) -> Sequence[str]:
    if not text:
        return ()
    return [text[i : i + size] for i in range(0, len(text), size)]
