"""Phase 5: caching, telemetry, degradation, and the HTTP surface.

The tests that carry weight here are the invalidation ones (a cache that
cannot go stale correctly is a correctness bug with a latency benefit) and
the degradation ones (a silently degraded service looks healthy on every
dashboard while serving worse answers).
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from gtrag.ablation import AblationConfig, build_system
from gtrag.index.embed import HashingEmbedder
from gtrag.ingest.parse import parse_filing
from gtrag.serve.cache import SemanticCache
from gtrag.serve.degradation import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    Degradable,
    DegradationReport,
)
from gtrag.serve.service import QueryRequest, RagService
from gtrag.serve.telemetry import MetricsRegistry, Tracer, percentile
from gtrag.types import RetrievedChunk, SystemResponse, Usage

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def documents():
    return [
        parse_filing(
            (FIXTURES / name).read_text(encoding="utf-8"),
            metadata={"company": company, "fiscal_year": year},
        )
        for name, company, year in (
            ("filing_sample.html", "Northwind Logistics, Inc.", 2024),
            ("filing_prior_year.html", "Northwind Logistics, Inc.", 2023),
            ("filing_peer.html", "Cascade Semiconductor Corp.", 2024),
        )
    ]


@pytest.fixture(scope="module")
def system(documents):
    return build_system(
        AblationConfig(label="serving", chunker="structure_aware", bm25=True), documents
    )


def response(answer: str = "An answer.", *, chunk_ids=("c1",), cost=0.001) -> SystemResponse:
    return SystemResponse(
        answer=answer,
        retrieved=tuple(
            RetrievedChunk(chunk_id=cid, rank=i, text="ctx", score=1.0)
            for i, cid in enumerate(chunk_ids, start=1)
        ),
        usage=Usage(input_tokens=100, output_tokens=20, cost_usd=cost),
        timings={"total": 250.0},
    )


# --------------------------------------------------------------------------
# Semantic cache
# --------------------------------------------------------------------------


class TestSemanticCache:
    @pytest.fixture
    def cache(self):
        return SemanticCache(HashingEmbedder(dimension=256), threshold=0.95, corpus_version="v1")

    def test_miss_then_hit(self, cache):
        assert cache.get("what was revenue?") is None
        cache.put("what was revenue?", response())
        hit = cache.get("what was revenue?")
        assert hit is not None
        assert hit.metadata["cache"]["hit"] is True

    def test_unrelated_question_misses(self, cache):
        cache.put("what was Northwind net revenue in fiscal 2024?", response())
        assert cache.get("where is the wafer fabrication facility located?") is None

    def test_hit_reports_zero_latency(self, cache):
        """A cache hit must not report the original generation latency.

        Carrying it through would make p95 meaningless -- the fast path would
        inflate the very metric it exists to improve.
        """
        cache.put("q", response())
        assert cache.get("q").timings["total"] == 0.0

    def test_stats_track_savings(self, cache):
        cache.put("q", response(cost=0.004))
        cache.get("q")
        assert cache.stats.hits == 1
        assert cache.stats.cost_saved_usd == pytest.approx(0.004)
        assert cache.stats.latency_saved_ms == pytest.approx(250.0)

    def test_errored_responses_are_not_cached(self, cache):
        """Otherwise the next request gets the same failure for free, for an hour."""
        failed = SystemResponse(answer="", error="upstream timeout")
        cache.put("q", failed)
        assert len(cache) == 0

    def test_chunk_invalidation(self, cache):
        cache.put("q1", response(chunk_ids=("a", "b")))
        cache.put("q2", response(chunk_ids=("c",)))
        assert cache.invalidate_chunks(["a"]) == 1
        assert len(cache) == 1

    def test_corpus_version_bump_invalidates_everything(self, cache):
        cache.put("q1", response())
        cache.put("q2", response(answer="another"))
        assert cache.set_corpus_version("v2") == 2
        assert len(cache) == 0

    def test_stale_version_entries_are_not_served(self, cache):
        cache.put("q", response())
        cache.corpus_version = "v2"  # simulate a re-ingest between put and get
        assert cache.get("q") is None
        assert cache.stats.stale_invalidations == 1

    def test_ttl_expiry(self):
        cache = SemanticCache(HashingEmbedder(dimension=128), ttl_seconds=0.01)
        cache.put("q", response())
        time.sleep(0.02)
        assert cache.get("q") is None

    def test_eviction_at_capacity(self):
        cache = SemanticCache(HashingEmbedder(dimension=128), max_entries=2)
        for i in range(4):
            cache.put(f"question number {i}", response(answer=str(i)))
        assert len(cache) == 2
        assert cache.stats.evictions == 2

    def test_eviction_keeps_the_popular_entry(self):
        cache = SemanticCache(HashingEmbedder(dimension=256), max_entries=2, threshold=0.99)
        cache.put("alpha bravo charlie delta", response(answer="popular"))
        cache.get("alpha bravo charlie delta")  # one hit
        cache.put("echo foxtrot golf hotel", response(answer="unpopular"))
        cache.put("india juliet kilo lima", response(answer="new"))
        assert cache.get("alpha bravo charlie delta") is not None

    def test_disabled_cache_is_a_no_op(self):
        cache = SemanticCache(HashingEmbedder(dimension=64), enabled=False)
        cache.put("q", response())
        assert cache.get("q") is None
        assert len(cache) == 0

    def test_rejects_bad_config(self):
        with pytest.raises(ValueError, match="threshold"):
            SemanticCache(HashingEmbedder(), threshold=0.0)
        with pytest.raises(ValueError, match="max_entries"):
            SemanticCache(HashingEmbedder(), max_entries=0)


# --------------------------------------------------------------------------
# Telemetry
# --------------------------------------------------------------------------


class TestTelemetry:
    def test_spans_are_recorded_on_the_trace(self):
        tracer = Tracer()
        with tracer.trace("abc") as trace:
            with tracer.span("retrieval"):
                pass
            with tracer.span("generation"):
                pass
        assert [s.name for s in trace.spans] == ["retrieval", "generation"]

    def test_stage_timings_include_a_total(self):
        tracer = Tracer()
        with tracer.trace() as trace, tracer.span("a"):
            pass
        timings = trace.stage_timings()
        assert "total" in timings and timings["total"] >= timings["a"]

    def test_span_records_an_exception_and_reraises(self):
        tracer = Tracer()
        with tracer.trace() as trace, pytest.raises(ValueError), tracer.span("boom"):
            raise ValueError("kaboom")
        assert "kaboom" in trace.spans[0].error

    def test_exporter_failure_never_breaks_the_request(self):
        class BrokenExporter:
            def export(self, _):
                raise RuntimeError("collector down")

        tracer = Tracer(otel_exporter=BrokenExporter())
        with tracer.trace() as trace, tracer.span("a"):  # must not raise
            pass
        assert trace.spans

    def test_percentiles(self):
        values = list(range(1, 101))
        assert percentile(values, 50) == pytest.approx(50.5)
        assert percentile(values, 99) == pytest.approx(99.01, abs=0.1)

    def test_metrics_snapshot(self):
        metrics = MetricsRegistry()
        metrics.increment("requests.total", 4)
        metrics.increment("requests.error", 1)
        metrics.observe("latency.total", 100.0)
        metrics.observe("latency.total", 200.0)
        snapshot = metrics.snapshot()
        assert snapshot["counters"]["requests.total"] == 4
        assert snapshot["derived"]["error_rate"] == pytest.approx(0.25)
        assert snapshot["latency_ms"]["latency.total"]["count"] == 2

    def test_latency_window_is_bounded(self):
        """An unbounded latency list is a memory leak that only shows up in
        production."""
        metrics = MetricsRegistry(window=10)
        for i in range(100):
            metrics.observe("latency.total", float(i))
        assert metrics.snapshot()["latency_ms"]["latency.total"]["count"] == 10

    def test_prometheus_exposition(self):
        metrics = MetricsRegistry()
        metrics.increment("requests.total", 3)
        metrics.observe("latency.total", 50.0)
        text = metrics.render_prometheus()
        assert "gtrag_requests_total 3" in text
        assert 'gtrag_latency_total{quantile="50"}' in text

    def test_derived_rates_are_none_without_traffic(self):
        assert MetricsRegistry().snapshot()["derived"]["error_rate"] is None


# --------------------------------------------------------------------------
# Degradation
# --------------------------------------------------------------------------


class TestCircuitBreaker:
    def test_opens_after_consecutive_failures(self):
        breaker = CircuitBreaker(name="x", failure_threshold=3)
        for _ in range(3):
            with pytest.raises(RuntimeError):
                breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("down")))
        assert breaker.state is CircuitState.OPEN
        assert breaker.trips == 1

    def test_open_circuit_short_circuits(self):
        breaker = CircuitBreaker(name="x", failure_threshold=1)
        with pytest.raises(RuntimeError):
            breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("down")))
        with pytest.raises(CircuitOpenError):
            breaker.call(lambda: "never runs")

    def test_intermittent_failures_do_not_open_it(self):
        """A component failing one request in ten is degraded but usable.

        Opening on that would take a working dependency offline.
        """
        breaker = CircuitBreaker(name="x", failure_threshold=3)
        for _ in range(5):
            with pytest.raises(RuntimeError):
                breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("blip")))
            breaker.call(lambda: "ok")
        assert breaker.state is CircuitState.CLOSED

    def test_half_open_after_recovery_then_closes(self):
        breaker = CircuitBreaker(name="x", failure_threshold=1, recovery_seconds=0.01)
        with pytest.raises(RuntimeError):
            breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("down")))
        time.sleep(0.02)
        assert breaker.state is CircuitState.HALF_OPEN
        breaker.call(lambda: "recovered")
        assert breaker.state is CircuitState.CLOSED

    def test_failed_probe_reopens_immediately(self):
        breaker = CircuitBreaker(name="x", failure_threshold=5, recovery_seconds=0.01)
        for _ in range(5):
            with pytest.raises(RuntimeError):
                breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("down")))
        time.sleep(0.02)
        assert breaker.state is CircuitState.HALF_OPEN
        with pytest.raises(RuntimeError):
            breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("still down")))
        # Still down: must not wait for the threshold again.
        assert breaker.state is CircuitState.OPEN

    def test_rejects_bad_config(self):
        with pytest.raises(ValueError):
            CircuitBreaker(name="x", failure_threshold=0)
        with pytest.raises(ValueError):
            CircuitBreaker(name="x", recovery_seconds=0)


class TestDegradable:
    def test_optional_component_falls_back_and_is_flagged(self):
        report = DegradationReport()
        stage = Degradable(
            name="reranker",
            breaker=CircuitBreaker(name="reranker"),
            fallback=lambda: "unranked",
        )
        result = stage.run(lambda: (_ for _ in ()).throw(RuntimeError("model down")), report)
        assert result == "unranked"
        assert report.is_degraded
        assert report.quality_flag == "degraded"
        assert "reranker" in report.degraded

    def test_optional_component_without_fallback_returns_none(self):
        report = DegradationReport()
        stage = Degradable(name="verifier", breaker=CircuitBreaker(name="verifier"))
        assert stage.run(lambda: (_ for _ in ()).throw(RuntimeError("x")), report) is None

    def test_required_component_raises(self):
        """'Degraded' must never mean 'answered without retrieving'."""
        report = DegradationReport()
        stage = Degradable(
            name="retriever", breaker=CircuitBreaker(name="retriever"), required=True
        )
        with pytest.raises(RuntimeError, match="required component"):
            stage.run(lambda: (_ for _ in ()).throw(RuntimeError("index gone")), report)

    def test_success_leaves_the_report_clean(self):
        report = DegradationReport()
        stage = Degradable(name="ok", breaker=CircuitBreaker(name="ok"))
        assert stage.run(lambda: "fine", report) == "fine"
        assert not report.is_degraded
        assert report.quality_flag == "full"


# --------------------------------------------------------------------------
# Service
# --------------------------------------------------------------------------


class TestRagService:
    def test_answers(self, system):
        service = RagService(system)
        result = service.query(QueryRequest(question="What was net revenue in fiscal 2024?"))
        assert result.trace_id
        assert result.passages
        assert result.quality == "full"

    def test_trace_id_is_honoured(self, system):
        service = RagService(system)
        result = service.query(QueryRequest(question="revenue", trace_id="abc123"))
        assert result.trace_id == "abc123"

    def test_timings_are_broken_down_by_stage(self, system):
        service = RagService(system)
        timings = service.query(QueryRequest(question="revenue")).timings
        assert "pipeline" in timings and "total" in timings

    def test_cache_hit_on_repeat(self, system):
        service = RagService(
            system, cache=SemanticCache(HashingEmbedder(dimension=256), corpus_version="v1")
        )
        service.query(QueryRequest(question="What was net revenue in fiscal 2024?"))
        second = service.query(QueryRequest(question="What was net revenue in fiscal 2024?"))
        assert second.cache_hit

    def test_cache_is_partitioned_by_roles(self, system):
        """Two callers with different permissions must not share an answer."""
        service = RagService(
            system, cache=SemanticCache(HashingEmbedder(dimension=256), corpus_version="v1")
        )
        service.query(QueryRequest(question="revenue", roles=("analyst",)))
        other = service.query(QueryRequest(question="revenue", roles=("intern",)))
        assert not other.cache_hit

    def test_cache_can_be_bypassed_per_request(self, system):
        service = RagService(
            system, cache=SemanticCache(HashingEmbedder(dimension=256), corpus_version="v1")
        )
        service.query(QueryRequest(question="revenue"))
        assert not service.query(QueryRequest(question="revenue", use_cache=False)).cache_hit

    def test_pipeline_failure_returns_unavailable_not_a_fabricated_answer(self):
        class BrokenSystem:
            name = "broken"
            config: dict = {}

            def answer(self, question, *, history=None):
                raise RuntimeError("index unavailable")

        service = RagService(BrokenSystem())
        result = service.query(QueryRequest(question="anything"))
        assert result.quality == "unavailable"
        assert result.answer == ""
        assert "index unavailable" in result.error

    def test_metrics_accumulate(self, system):
        service = RagService(system)
        for _ in range(3):
            service.query(QueryRequest(question="net revenue fiscal 2024"))
        snapshot = service.metrics.snapshot()
        assert snapshot["counters"]["requests.total"] == 3
        assert snapshot["latency_ms"]["latency.total"]["count"] == 3

    def test_health_reports_breaker_state(self, system):
        health = RagService(system).health()
        assert health["status"] == "ok"
        assert health["breakers"][0]["state"] == "closed"

    def test_health_degrades_when_the_breaker_opens(self):
        class BrokenSystem:
            name = "broken"
            config: dict = {}

            def answer(self, question, *, history=None):
                raise RuntimeError("down")

        service = RagService(BrokenSystem())
        for _ in range(5):
            service.query(QueryRequest(question="q"))
        assert service.health()["status"] == "degraded"

    def test_stream_emits_passages_before_the_answer(self, system):
        service = RagService(system)
        events = list(service.stream(QueryRequest(question="What was net revenue?")))
        names = [e["event"] for e in events]
        assert names[0] == "trace"
        assert names[1] == "passages"
        assert names[-1] == "done"
        assert names.index("passages") < max(
            [i for i, n in enumerate(names) if n in ("token", "refusal")], default=2
        )

    def test_stream_ends_with_citations_and_usage(self, system):
        service = RagService(system)
        done = list(service.stream(QueryRequest(question="net revenue")))[-1]
        assert "citations" in done["data"] and "timings_ms" in done["data"]


# --------------------------------------------------------------------------
# HTTP surface
# --------------------------------------------------------------------------


fastapi = pytest.importorskip("fastapi", reason="API tests need the [serve] extra")
from fastapi.testclient import TestClient  # noqa: E402

from gtrag.serve.app import create_app  # noqa: E402


@pytest.fixture(scope="module")
def client(system):
    service = RagService(
        system, cache=SemanticCache(HashingEmbedder(dimension=256), corpus_version="v1")
    )
    return TestClient(create_app(service))


class TestApi:
    def test_query(self, client):
        r = client.post("/query", json={"question": "What was net revenue in fiscal 2024?"})
        assert r.status_code == 200
        body = r.json()
        assert body["trace_id"] and body["passages"] and "timings_ms" in body

    def test_query_with_history(self, client):
        r = client.post(
            "/query",
            json={
                "question": "And what drove that increase?",
                "history": [{"question": "What was net revenue?", "answer": "$4,218 million."}],
            },
        )
        assert r.status_code == 200

    def test_inbound_trace_id_is_honoured(self, client):
        r = client.post(
            "/query", json={"question": "revenue"}, headers={"x-trace-id": "from-gateway"}
        )
        assert r.json()["trace_id"] == "from-gateway"

    def test_validation_rejects_empty_question(self, client):
        assert client.post("/query", json={"question": ""}).status_code == 422

    def test_validation_rejects_absurd_top_k(self, client):
        assert client.post("/query", json={"question": "x", "top_k": 5000}).status_code == 422

    def test_stream_is_server_sent_events(self, client):
        r = client.post("/query/stream", json={"question": "What was net revenue?"})
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        assert "event: passages" in r.text
        assert "event: done" in r.text

    def test_healthz(self, client):
        assert client.get("/healthz").json()["status"] == "ok"

    def test_readyz_reports_the_corpus(self, client):
        body = client.get("/readyz").json()
        assert body["ready"] and body["corpus_chunks"] > 0

    def test_metrics_is_prometheus_text(self, client):
        client.post("/query", json={"question": "revenue"})
        r = client.get("/metrics")
        assert r.status_code == 200
        assert "gtrag_requests_total" in r.text

    def test_stats_and_config(self, client):
        assert "metrics" in client.get("/stats").json()
        assert client.get("/config").json()["retriever"] == "hybrid"

    def test_pipeline_failure_is_503_not_500(self, system):
        class BrokenSystem:
            name = "broken"
            config: dict = {}
            spanned_chunks: list = []

            def answer(self, question, *, history=None):
                raise RuntimeError("down")

        broken = TestClient(create_app(RagService(BrokenSystem())))
        # 503: the request was valid and the service expects to recover, so
        # the client should retry rather than treat it as a bug.
        assert broken.post("/query", json={"question": "x"}).status_code == 503
