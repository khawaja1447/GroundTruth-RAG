"""FastAPI adapter.

Deliberately thin. Every decision that matters -- caching, degradation,
tracing, cost accounting -- lives in `RagService`, which is framework-free and
unit-tested without starting a server. This file translates HTTP to that and
back.

Endpoints:

    POST /query          answer a question
    POST /query/stream   the same, as server-sent events
    GET  /healthz        liveness + breaker state
    GET  /readyz         readiness (is the corpus actually loaded?)
    GET  /metrics        Prometheus text exposition
    GET  /stats          human-readable metrics snapshot
    GET  /config         the resolved configuration this process is running
"""

# NOTE: deliberately NO `from __future__ import annotations` in this module.
#
# PEP 563 turns annotations into strings, and FastAPI resolves them with
# `get_type_hints` against the *module* globals. The request models below are
# defined inside `create_app` so that pydantic stays a lazy import, which means
# they never appear in module globals -- FastAPI cannot resolve them, silently
# falls back to treating the body as query parameters, and every POST returns
# 422 with "field required: query.body". Without the future import the
# signature carries the real class objects and no resolution is needed.

import json
import os
from typing import Any

from ..ablation import AblationConfig, build_system
from ..index.embed import HashingEmbedder
from .cache import SemanticCache
from .service import QueryRequest, RagService
from .telemetry import MetricsRegistry, Tracer

__all__ = ["create_app", "build_service_from_env"]


def build_service_from_env() -> RagService:
    """Assemble the service a server process should run.

    Configuration comes from the environment so a container is configured the
    way containers are configured, and the resolved values are exposed at
    `/config` so what is running is inspectable rather than assumed.
    """
    import glob
    from pathlib import Path

    from ..ingest.document import Document
    from ..ingest.parse import parse_filing

    docs_dir = os.environ.get("GTRAG_DOCS", "corpus/documents")
    files = sorted(glob.glob(os.path.join(docs_dir, "*.json")))

    if files:
        documents = [
            Document.from_dict(json.loads(Path(f).read_text(encoding="utf-8"))) for f in files
        ]
        corpus_version = f"{len(files)}-docs"
    else:
        # Fall back to the fixture corpus so the container starts and is
        # explorable before `make ingest` has run. `/readyz` reports which.
        root = Path(__file__).resolve().parents[3]
        fixtures = root / "tests" / "fixtures"
        documents = [
            parse_filing(
                (fixtures / name).read_text(encoding="utf-8"),
                metadata={"company": company, "fiscal_year": year, "form_type": "10-K"},
            )
            for name, company, year in (
                ("filing_sample.html", "Northwind Logistics, Inc.", 2024),
                ("filing_prior_year.html", "Northwind Logistics, Inc.", 2023),
                ("filing_peer.html", "Cascade Semiconductor Corp.", 2024),
            )
            if (fixtures / name).exists()
        ]
        corpus_version = "fixtures"

    config = AblationConfig(
        label=os.environ.get("GTRAG_LABEL", "serving"),
        chunker=os.environ.get("GTRAG_CHUNKER", "structure_aware"),
        bm25=os.environ.get("GTRAG_BM25", "1") not in ("0", "false", ""),
        rewriter=os.environ.get("GTRAG_REWRITER", "heuristic"),
        verifier=os.environ.get("GTRAG_VERIFIER", "lexical"),
        top_k=int(os.environ.get("GTRAG_TOP_K", "5")),
    )
    system = build_system(config, documents)

    cache = SemanticCache(
        embedder=HashingEmbedder(),
        threshold=float(os.environ.get("GTRAG_CACHE_THRESHOLD", "0.95")),
        max_entries=int(os.environ.get("GTRAG_CACHE_ENTRIES", "1000")),
        ttl_seconds=float(os.environ.get("GTRAG_CACHE_TTL", "3600")),
        corpus_version=corpus_version,
        enabled=os.environ.get("GTRAG_CACHE", "1") not in ("0", "false", ""),
    )

    service = RagService(system, cache=cache, tracer=Tracer(), metrics=MetricsRegistry())
    service.corpus_source = "ingested" if files else "fixtures"  # type: ignore[attr-defined]
    return service


def create_app(service: RagService | None = None) -> Any:
    """Build the ASGI app. Imports FastAPI lazily so the package stays usable
    without it installed -- the eval harness must never need a web framework."""
    try:
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import PlainTextResponse, StreamingResponse
        from pydantic import BaseModel, Field
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "The API server needs FastAPI:\n"
            "    pip install -e '.[serve]'\n"
            "The eval harness and the CLI do not."
        ) from exc

    service = service or build_service_from_env()
    app = FastAPI(
        title="groundtruth-rag",
        description="Retrieval over financial filings, with citations and refusal.",
        version="0.5.0",
    )

    class Turn(BaseModel):
        question: str
        answer: str

    class QueryBody(BaseModel):
        question: str = Field(min_length=1, max_length=2000)
        history: list[Turn] = Field(default_factory=list, max_length=20)
        top_k: int | None = Field(default=None, ge=1, le=50)
        use_cache: bool = True
        roles: list[str] = Field(default_factory=list, max_length=20)

    def to_request(body: QueryBody, request: Request) -> QueryRequest:
        return QueryRequest(
            question=body.question,
            history=tuple((t.question, t.answer) for t in body.history),
            top_k=body.top_k,
            # Honour an inbound trace id so a request can be followed across
            # a gateway, but never trust its length.
            trace_id=(request.headers.get("x-trace-id") or "")[:64],
            use_cache=body.use_cache,
            roles=tuple(body.roles),
        )

    @app.post("/query")
    def query(body: QueryBody, request: Request) -> dict[str, Any]:
        result = service.query(to_request(body, request))
        if result.quality == "unavailable":
            # 503, not 500: the request was valid and the service expects to
            # recover, so clients should retry rather than treat it as a bug.
            raise HTTPException(status_code=503, detail=result.to_dict())
        return result.to_dict()

    @app.post("/query/stream")
    def query_stream(body: QueryBody, request: Request) -> Any:
        def events() -> Any:
            for event in service.stream(to_request(body, request)):
                yield f"event: {event['event']}\ndata: {json.dumps(event['data'])}\n\n"

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return service.health()

    @app.get("/readyz")
    def readyz() -> dict[str, Any]:
        chunks = len(getattr(service.system, "spanned_chunks", []) or [])
        source = getattr(service, "corpus_source", "unknown")
        if not chunks:
            raise HTTPException(status_code=503, detail={"ready": False, "reason": "empty corpus"})
        return {"ready": True, "corpus_chunks": chunks, "corpus_source": source}

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> str:
        return service.metrics.render_prometheus()

    @app.get("/stats")
    def stats() -> dict[str, Any]:
        return {
            "metrics": service.metrics.snapshot(),
            "cache": service.cache.stats.to_dict() if service.cache else None,
            "health": service.health(),
        }

    @app.get("/config")
    def config() -> dict[str, Any]:
        return service.config()

    return app
