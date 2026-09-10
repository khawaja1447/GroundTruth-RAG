# Phase 5 — the serving layer

Making "production" in the README load-bearing.

    trace -> cache lookup -> [degradable pipeline] -> cache store -> metrics

---

## 1. Measured behaviour

`make loadtest` drives the pipeline in-process across a concurrency ramp:

```
 conc   reqs    p50 ms    p95 ms    p99 ms       rps  errors   cache
    1    200      0.28      0.37      1.44    2806.3       0    96%
    2    200      0.29      0.51      1.38    2741.6       0    96%
    4    200      0.29      0.53      1.56    2668.9       0    96%
    8    200      0.30      0.58      2.21    2431.6       0    96%
   16    200      0.30      0.71      1.58    2337.7       0    96%
   32    200      0.30      0.87      1.69    2315.4       0    96%

knee: p95 exceeds 2x the single-threaded baseline (0.37ms) at concurrency 32
```

**Throughput falls as concurrency rises** — 2806 rps at one thread down to
2315 at thirty-two. That is the GIL: the pipeline is pure-Python CPU work, so
extra threads contend rather than help. The operational consequence is in the
compose file: one CPU per replica, and scale with replicas rather than
`--workers`.

These numbers are the *retrieval and assembly* path only. The extractive
generator makes no model call, so a real deployment's p95 is dominated by the
LLM and will be three orders of magnitude larger. Reporting sub-millisecond
latency as though it were end-to-end service latency would be dishonest; what
this measures is that the parts *we* wrote are not the bottleneck.

---

## 2. Semantic caching

Exact-match caching is nearly useless here: "what was Northwind's FY24
revenue?" and "Northwind fiscal 2024 net revenue?" want the same answer and
share almost no characters.

The hard part is not the lookup, it is **invalidation**. A cached answer is
derived from specific chunks, and when the corpus is re-ingested there is
nothing in the *question* to tell you the answer went stale. Every entry
therefore records:

- the **chunk ids** it was derived from — `invalidate_chunks()` drops every
  answer that depended on a chunk that changed;
- the **corpus version** it was built against — a version bump invalidates
  everything older, and a stale-version entry is never served even if the TTL
  has not expired.

Three further decisions worth stating:

- **Threshold 0.95, high on purpose.** A loose threshold returns a
  confidently wrong answer to a question nobody asked. A miss costs a
  regeneration; a false positive costs correctness.
- **Errored responses are never cached.** Otherwise the next caller gets the
  same failure for free, for an hour.
- **A hit reports zero latency, not the original generation time.** Carrying
  it through would make p95 meaningless — the fast path would inflate the
  very metric it exists to improve.
- **The cache is partitioned by caller roles.** Two callers with different
  permissions must not share an answer; Phase 6 depends on this.

---

## 3. Degradation

The default behaviour of a RAG pipeline under partial failure is a 500. That
is the wrong answer almost every time: if the reranker is down, hybrid
retrieval still returns useful passages.

**`CircuitBreaker`** is a standard three-state breaker. Two details:

- It opens on *consecutive* failures, not a failure rate. A component failing
  one request in ten is degraded but usable, and opening on that takes a
  working dependency offline.
- A failed half-open probe re-opens immediately rather than waiting for the
  threshold again. The component is still down; there is nothing to learn
  from four more timeouts.

**`Degradable`** wraps a component with a fallback and records that the
fallback was used. The flag matters as much as the fallback: a silently
degraded service looks healthy on every dashboard while serving worse
answers. Responses carry `quality: "degraded"` and name the components.

**A required stage has no fallback.** The pipeline itself is `required=True`,
so a retrieval failure returns `503 unavailable` rather than an answer.
"Degraded" must never mean "answered without retrieving" — that is not a
degraded answer, it is a fabricated one.

---

## 4. Observability

`/metrics` renders Prometheus text exposition directly, so it scrapes without
an exporter sidecar. `/stats` is the same data in a shape a human reads.

Latency is recorded **per stage**, not just end to end. A p95 that says
"1.2s" is not actionable; one that attributes it to the reranker is. Every
request carries a `trace_id` returned in the response and honoured from an
inbound `X-Trace-Id`, so a user reporting a wrong answer hands you a string
that pulls up exactly which chunks were retrieved and how long each stage
took.

Latency histograms use bounded ring buffers. An unbounded list of every
latency the process has ever seen is a memory leak that only shows up in
production.

Cost is **accumulated per request** from reported token usage, not estimated
from a price list at month end.

---

## 5. HTTP surface

| Endpoint | Purpose |
|---|---|
| `POST /query` | Answer a question |
| `POST /query/stream` | The same, as server-sent events |
| `GET /healthz` | Liveness + breaker state |
| `GET /readyz` | Readiness — is the corpus actually loaded? |
| `GET /metrics` | Prometheus text exposition |
| `GET /stats` | Human-readable snapshot |
| `GET /config` | The resolved configuration this process is running |

**Streaming emits passages before the answer.** The user sees what the system
is reading while the model is still writing — better perceived latency, and
better epistemics: the sources arrive with the claim rather than after it.

**Liveness and readiness are separate.** The process can be up while the
corpus is still loading, and routing traffic to it then produces empty
answers. The container healthcheck probes `/readyz`.

**A pipeline failure is 503, not 500.** The request was valid and the service
expects to recover, so clients should retry rather than treat it as a bug.

---

## 6. A bug this phase surfaced

`from __future__ import annotations` in the FastAPI module turned every
annotation into a string. FastAPI resolves annotations with `get_type_hints`
against **module globals**, and the request models are defined inside
`create_app` so pydantic stays a lazy import — so they are not resolvable.
FastAPI fell back to treating the request body as *query parameters*, and
every POST returned `422: field required: query.body`.

It is a silent, total failure of every write endpoint, and nothing about the
error message points at the future import. The module now carries a comment
explaining why it must not have one.

---

## 7. Front matter — a fix with a null result

Investigating a demo query surfaced a real corpus defect: the cover page and
table of contents were being indexed. The TOC names every section heading, so
it matches lexically against a question about any of them — a universal false
positive that retrieves for everything and answers nothing.

`Document.front_matter` now identifies the region before the first Item
section, and every chunker excludes it by default (`exclude_front_matter`,
configurable so the decision stays measurable).

**Measured: no change.** nDCG@10, recall@10 and MRR were identical to four
decimal places with and without it, on this eval set.

```
with front matter        chunks=15   nDCG@10=0.7170  recall@10=1.0000  MRR=0.6308
front matter excluded    chunks=12   nDCG@10=0.7170  recall@10=1.0000  MRR=0.6308
```

Front matter *does* reach the top 5 on 6 of 17 questions, so the pollution is
real. It does not move the metrics because with only 15 chunks, dropping
three simply shifts other non-gold chunks into the vacated slots — the gold
chunks' ranks are unchanged. On a corpus of thousands of chunks, where a
universal false positive competes against genuinely relevant passages, the
effect would not be neutral.

The change is kept: it is principled (front matter contains no answers by
construction) and it removes 20% of the corpus for free. But it is **not**
claimed as a quality improvement, because on the evidence available it is not
one.

---

## 8. Containers

Multi-stage build; the runtime image carries the venv and the package but not
the toolchain or the test suite. Runs as a non-root user — the service reads
a corpus and answers questions, and has no reason to hold root in a
network-exposed container.

The corpus is a **read-only bind mount**, not an image layer. It is large,
regenerable with `make ingest`, changes on a different cadence from the code,
and the service must never be able to mutate it.

---

## 9. What is measured and what is not

**Measured:** cache hit/miss, chunk- and version-based invalidation, TTL
expiry, eviction preferring popular entries, role partitioning, refusal to
cache errors; breaker state transitions including the intermittent-failure
and failed-probe cases; degradation flagging and the required-stage
distinction; trace/span recording, exporter-failure isolation, bounded
histograms, Prometheus rendering; and the full HTTP surface including
validation, streaming order, and the 503 path.

**Not measured:** behaviour under a real LLM's latency and failure modes, and
the container itself — Docker is not available in this environment, so the
Dockerfile and compose file are written but unbuilt. The load-test numbers
are in-process; `--url` drives a live server and is the honest way to claim a
p99.
