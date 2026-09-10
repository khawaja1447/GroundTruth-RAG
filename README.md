# groundtruth-rag

A retrieval system over corporate financial filings where **every
architectural decision is justified by a measured delta** — and the evaluation
harness that produces those numbers is the actual product.

> **Status: complete, Phases 1–7.** Corpus pipeline, evaluation harness, the
> retrieval ablation program, grounded generation, serving and hardening are
> built, tested and running — **505 tests, no network, no API key, $0.00**.
> Every figure below is re-derived from source by `make reproduce`, which
> fails if one moves.
> Docs: [`corpus.md`](docs/corpus.md),
> [`eval-methodology.md`](docs/eval-methodology.md),
> [`ablation.md`](docs/ablation.md), [`generation.md`](docs/generation.md),
> [`serving.md`](docs/serving.md), [`security.md`](docs/security.md).

Most RAG projects are forty lines: load PDF, split at 500 characters, embed,
top-k, stuff into a prompt. They have no numbers, so there is nothing to
discuss. This one inverts the order — the harness was built *before* the
improvements it exists to measure, so every later change has a control to be
measured against.

---

## Quickstart

No API key, no network, no corpus needed:

```bash
git clone <repo> && cd groundtruth-rag
make install
make test           # 505 tests, stdlib only
make validate       # dataset structure + corpus join
make eval-fast      # full deterministic eval
make reproduce      # re-derive every figure in this README; fails if one moved
```

To build the real corpus (the only step that needs the network):

```bash
export GTRAG_SEC_USER_AGENT="groundtruth-rag research you@example.com"
make ingest COMPANIES=10 YEARS=3
make index
make query Q="What was Apple's total net revenue in fiscal 2024?"
```

`make eval-fast` on the fixture corpus produces (abridged — the full output
also lists the three judged metrics as `unscored`, plus latency, cost, and the
empty slices):

```
  ndcg@10                        0.8296  [0.6466, 0.9702]  (n=12, 4 n/a)
  recall@10                      0.8750  [0.7083, 1.0000]  (n=12, 4 n/a)
  answer_recall@10               0.9167  [0.7500, 1.0000]  (n=12, 4 n/a)
  mrr                            0.8333  [0.6250, 1.0000]  (n=12, 4 n/a)
  correct_refusal                0.0000  [0.0000, 0.0000]  (n=4, 12 n/a)
  false_refusal                  0.0833  [0.0000, 0.2500]  (n=12, 4 n/a)
  answered_unanswerable          1.0000  [1.0000, 1.0000]  (n=4, 12 n/a)
  citation_fabrication_rate      0.0000  [0.0000, 0.0000]  (n=15, 1 n/a)

slice                   n           ndcg@10         recall@10
-------------------------------------------------------------
single_hop              4            0.8803            0.8750
numeric_table           3            0.5436            0.6667
multi_hop               2            0.9599            1.0000
comparative_temporal    3            0.9612            1.0000
```

Two things that output is already telling you about the stub retriever, which
is the point of the exercise:

- `answered_unanswerable = 1.00` — it answers **every** question it should have
  refused. That is the hallucination path, quantified.
- `numeric_table` is the weakest slice at 0.54 nDCG — figures inside tables are
  the hardest thing to retrieve on this corpus. That is what motivated
  structure-aware chunking in Phase 3; it is *not* a claim that structure-aware
  chunking fixed it. On this smoke corpus it scored below the fixed-size
  baseline, inconclusively. See [`ablation.md`](docs/ablation.md).

---

## What is actually built

```
src/gtrag/
├── ingest/                  # P1  corpus pipeline
│   ├── document.py          #     document model + Span (the anchoring primitive)
│   ├── edgar.py             #     SEC client: UA enforcement, 10 req/s token bucket
│   └── parse.py             #     HTML -> text, sections (TOC trap), tables
├── chunking/
│   ├── base.py              # P1  Chunker protocol + fixed-token baseline, span-tracked
│   └── strategies.py        # P3  recursive, structure-aware, sentence-window,
│                            #     parent-document, semantic
├── index/
│   ├── embed.py             # P1  Embedder protocol: hashing + sentence-transformers
│   └── store.py             # P1  exhaustive cosine index, embedder-mismatch guard
├── retrieve/
│   ├── retrievers.py        # P3  dense, BM25, RRF hybrid, reranking, metadata filter
│   └── rewrite.py           # P4  multi-turn query rewriting (heuristic + LLM)
├── generate/
│   ├── context.py           # P4  dedup -> budget -> lost-in-the-middle order
│   ├── refusal.py           # P4  confidence signals, refusal curve, operating point
│   ├── verify.py            # P4  claim-level support, numbers checked separately
│   └── generator.py         # P1  extractive (offline) + Anthropic, both refusable
├── serve/
│   ├── cache.py             # P5  semantic cache: chunk + corpus-version invalidation
│   ├── degradation.py       # P5  circuit breaker, fallbacks, required stages
│   ├── telemetry.py         # P5  per-stage tracing, Prometheus, bounded histograms
│   ├── service.py           # P5  the request pipeline
│   └── app.py               # P5  FastAPI surface, streaming, health, 503 path
├── security/
│   ├── access.py            # P6  role ACLs, deny-by-default, pre-filter predicate
│   └── injection.py         # P6  7-payload corpus, detection, sanitisation
├── ablation.py              # P3  the sweep configs and the ladders
├── grounded.py              # P4  the assembled system
├── baseline.py              # P1  the control it is measured against
└── cli.py                   #     ingest / index / query / inspect

evals/                       # P2  the measurement layer — the actual product
├── spans.py                 #     span -> per-chunking relevance resolution
├── types.py                 #     labeled-question model + the invariants that keep it honest
├── dataset.py               #     loading, corpus join checks, composition reporting
├── metrics/
│   ├── retrieval.py         #     recall, precision, MRR, graded nDCG — hand-implemented
│   ├── generation.py        #     refusal 2x2, citation validity, claim splitting
│   └── stats.py             #     bootstrap CIs, paired bootstrap, statistical power
├── judges/
│   ├── base.py              #     judge protocol, scales, rubric versioning
│   ├── llm_judge.py         #     Anthropic-backed, structured output, cached
│   └── rubrics/             #     4 versioned rubrics with worked examples
├── calibration.py           #     weighted Cohen's kappa + the human-label round trip
├── cache.py                 #     content-addressed SQLite response cache
├── runner.py                #     config hashing, execution, scoring, result files
├── report.py                #     run summaries, slice tables, paired comparisons
├── gate.py                  #     the CI regression gate
└── cli.py                   #     8 commands

scripts/
├── reproduce.py             # P7  re-derive every published figure; --check fails on drift
├── run_sweep.py             # P3  the ablation program
├── refusal_curve.py         # P4  the tradeoff curve and operating-point selection
├── security_report.py       # P6  leak test + injection report, non-zero on a leak
├── loadtest.py              # P5  concurrency ramp (machine-dependent; see below)
└── build_ablation_table.py  # P3  the table, generated rather than typed
```

### Design decisions worth defending

**The core harness has zero dependencies.** Retrieval metrics, refusal scoring,
citation validation, bootstrap intervals and Cohen's kappa are all stdlib, so
the test suite and the CI gate run with nothing installed and no API key. Only
the LLM judge needs a model. That split is what makes quality a testable
property on every push, including from forks.

**Metrics are hand-implemented, not imported.** Importing RAGAS turns "what
does that metric compute?" into a question you cannot answer. nDCG here is 12
lines and tested against hand-computed values.

**Undefined is not zero.** A metric that does not apply to a question returns
`None`. Unanswerable questions have no gold chunks, so recall is undefined for
them — scoring them `0.0` would make the headline number a function of dataset
composition rather than retrieval quality. Aggregates report `n` and
`n_undefined` separately.

**Judged and deterministic metrics are separated by design.** Anything checkable
by code is checked by code — which keeps the expensive half honest and the
cheap half always available.

**A delta whose confidence interval includes zero is inconclusive.** The paired
bootstrap compares two configurations on the same questions, cancelling
per-question difficulty. The ablation table marks unresolved deltas `(ns)`.
This is the guard against shipping noise.

---

## Span anchoring — why the chunking ablation is possible

The most consequential design decision in the project, and the one that is
easy to get wrong in a way you do not discover until it is too late.

Gold evidence is labeled by **document character span**, not by chunk id.
Phase 3's first ablation dimension is the chunking — and if labels pointed at
chunk ids, re-chunking would invalidate every one of them. You would have to
re-label the whole eval set per strategy, which nobody does, so in practice
the most valuable ablation never gets run.

```
document text   ......[=== gold span ===]...........
chunking A      [ chunk 1 ][ chunk 2 ][ chunk 3 ]      -> chunk 2 relevant
chunking B      [   chunk 1   ][   chunk 2   ]         -> chunk 1 relevant
```

One human labeling; both chunkings graded from it automatically. Relevance is
graded on how much of **the span** a chunk covers, not how much of the chunk is
gold — a 512-token chunk containing a one-sentence answer is a retrieval
success, and grading the other way would punish exactly the large-chunk
strategies Phase 3 needs to evaluate fairly.

## Parsing filings

The table-of-contents trap: "Item 1A. Risk Factors" appears in the TOC before
it appears in the body, so a first-match parser anchors every section to the
TOC. Measured on the test fixture, that produces sections of **21, 26 and 50
characters** instead of 777, 758 and 996 — a corpus that looks fine and
retrieves nothing. The parser takes the last match per item above a length
floor, and a test guards it.

SEC compliance is enforced, not assumed: a User-Agent with a contact email is
required rather than defaulted, and the 10 req/s limit uses a token bucket
shared across threads — a per-request `sleep` does not bound concurrent
throughput, and there is a threaded test proving the difference.

---

## The ablation program, and the result that matters

Running the full ladder on the smoke corpus, **every delta came back
inconclusive**:

```
baseline: fixed + dense      0.8196
+ structure-aware chunking   0.7527   -0.0669 [-0.2088, +0.0568]  inconclusive
+ bm25 hybrid (RRF)          0.8297   +0.0769 [+0.0000, +0.1941]  inconclusive
+ reranking                  0.7859   -0.0438 [-0.1420, +0.0568]  inconclusive
+ metadata pre-filtering     0.8663   +0.0804 [-0.0888, +0.2295]  inconclusive
```

That is the harness working. With n=13 the interval on any paired difference
is about ±0.15, so an 8-point move is indistinguishable from noise. Claiming
"+7.7 points from hybrid retrieval" off this data is precisely the unearned
result the project exists to prevent.

Two rungs go *down* and both stay in the table. The ladder is a fixed
sequence of one-component steps, not a greedy search — dropping a losing rung
would renumber every rung above it and destroy the attribution. And an
interval spanning [-0.21, +0.06] supports "worse" no better than "better", so
removing it would mean acting on exactly the noise the report refuses to act
on in the other direction.

An all-inconclusive ablation is ambiguous between "the components do nothing"
and "the set is too small to tell" — so the harness answers that too:

```
ndcg@10:   n=13, sd=0.2743 -> can resolve ~0.149; to detect 0.020: need ~723
recall@10: n=13, sd=0.2774 -> can resolve ~0.151; to detect 0.020: need ~739
```

The CI gate's 2-point nDCG threshold needs roughly **720 labeled questions**,
against the 220 the plan had budgeted. That is a concrete next action,
produced before any component decision was made on bad evidence.

## Six chunking strategies, one labeling

Because gold evidence is span-anchored, all six are graded from the same human
labeling — `fixed`, `recursive`, `structure_aware`, `sentence_window`,
`parent_document`, `semantic`. A test asserts the evidence resolves under every
one of them *with disjoint chunk-id sets*, which is the property that makes the
comparison possible at all.

Two rules of `structure_aware` are asserted rather than hoped for: a table is
never split (row labels in one chunk and figures in another answer nothing),
and a chunk never crosses a section boundary.

Retrieval composes as an expression, so each layer is independently
measurable:

```python
RerankingRetriever(
    FilteredRetriever(
        HybridRetriever([DenseRetriever(index), BM25Retriever(chunks)]),
        extractor=MetadataExtractor.from_chunks(chunks),
    ),
    reranker=LexicalReranker(),
)
```

Fusion is rank-based, not score-based: BM25 scores and cosine similarities are
on incomparable scales, and normalising them is a hidden hyperparameter.

---

## Grounded generation, and a conclusion the project got wrong

```
configuration                      answered_unans   false_refusal   fabricated
p4 base: structure-aware + bm25           100.0%            0.0%         0.0%
+ dedup                                   100.0%            0.0%         0.0%
+ lost-in-the-middle order                100.0%            0.0%         0.0%
+ query rewriting                         100.0%            0.0%         0.0%
+ claim verification                      100.0%            0.0%         0.0%
+ refusal (top_score)                      25.0%            7.7%         0.0%
```

**Fabricated citations are zero on every row** — Phase 4's hard gate, asserted
by a test rather than claimed by a report. **Without a refusal policy the
system answers 100% of unanswerable questions**, which is the hallucination
path quantified. **Turning refusal on catches three of the four for the price
of wrongly declining one answerable question in thirteen.**

Getting there is the most instructive thing in the repository, because the
first answer was wrong.

### What Phase 4 measured, and published

```
signal         best J   @ correct   @ false     (a coin flip scores 0.0)
top_score      +0.019         25%       23%
mean_score     +0.154        100%       85%
margin         +0.308        100%       69%
```

No operating point existed at any sane ceiling, so the conclusion drawn was
architectural: *retrieval confidence cannot decide this — the retriever
returns its best five chunks whether or not any of them answer the question,
so the decision belongs to the generator, which sees the passage text.*

### What the same measurement says now

```
signal         best J   @ correct   @ false
top_score      +0.673         75%        8%
mean_score     +0.442         75%       31%
margin         +0.346         50%       15%

OPERATING POINT: threshold=0.0324 on top_score
  criterion: maximise correct refusals subject to false refusal <= 10%
```

The ranking inverted — `top_score` went from worst to best — and the earlier
conclusion did not survive.

**What changed was not the threshold.** Phase 5 excluded the cover page and
table of contents from the corpus, and reported at the time that it had *no
measurable effect*: nDCG, recall and MRR were identical to four decimal
places. That was true. It was also not the whole story. The table of contents
names every section heading, so it matched **unanswerable** questions about as
well as real ones — invisible in retrieval quality, and fatal to the very gap
refusal calibration depends on.

A component was being blocked by a defect in a different component, and the
blocked component's metrics were the only place it showed.

**This was caught by tooling, not by insight.** `scripts/reproduce.py` holds
every published figure in a `PUBLISHED` table, re-derives all eighteen from
source in eleven seconds, and exits non-zero when one moves. On the run that
introduced it, ten of the eleven figures the table then held came back
`MOVED` — which is how both this and the retrieval bug below surfaced in the
same minute. Without it, this README would still carry the superseded
conclusion, stated just as confidently. The full account is in
[`generation.md` §2](docs/generation.md).

The same run caught a plain bug: the system was reporting its *assembled
context* as the retrieval result. `lost_in_the_middle_order` deliberately
moves rank 2 to the end of the context, so every rank-sensitive metric —
nDCG, MRR — had been measuring the presentation layout rather than the
retriever.

Still true, and still enforced: at a **5% ceiling** no signal produces an
operating point and `make refusal-curve MAX_FALSE_REFUSAL=0.05` exits
non-zero saying so. And an early version of the selector returned a point with
0% correct *and* 0% false refusals and called it the answer — it satisfies any
ceiling by never refusing. A criterion satisfiable by doing nothing is not a
criterion, so degenerate points are rejected by default.

---

## Serving: making "production" load-bearing

```
trace -> cache lookup -> [degradable pipeline] -> cache store -> metrics
```

`make loadtest` runs a concurrency ramp. One representative run:

```
 conc   reqs    p50 ms    p95 ms       rps  errors   cache
    1    200      0.29      0.48    2608.4       0    96%
   32    200      0.30      1.35    2411.6       0    96%
```

**These are the only numbers in this repository not under the reproduce
contract, and they do not reproduce.** They are wall-clock timings on shared
hardware. Across four consecutive runs the single-threaded p95 ranged 0.37–0.50
ms and the reported knee landed at concurrency 8, 16 and 32 — so the knee is
noise, not a measurement, and quoting one is how a load test becomes theatre.

What is stable across every run, and is the actual finding:

- **p50 is flat at ~0.28 ms at every concurrency**, and
- **throughput at 32 threads is always below the peak** (which sits at 1–2
  threads), never above it.

Threads buy nothing here: the pipeline is pure-Python CPU work, so they
contend on the GIL rather than overlapping. The operational consequence is in
the compose file — one CPU per replica, scale with replicas, not `--workers`.

And these cover *retrieval and assembly* only. The extractive generator makes
no model call, so a real deployment's p95 is dominated by the LLM and will be
three orders of magnitude larger. Reporting sub-millisecond latency as
end-to-end service latency would be dishonest; what this measures is that the
parts *we* wrote are not the bottleneck.

Three decisions the rest of the layer turns on:

- **The hard part of caching is invalidation, not lookup.** Nothing in a
  *question* tells you the answer went stale. Every entry records the chunk
  ids it was derived from and the corpus version it was built against, so
  re-ingesting a document drops exactly the answers that depended on it. The
  similarity threshold is 0.95, deliberately high: a miss costs a
  regeneration, a false positive costs correctness. Entries are partitioned by
  caller role, so two principals never share an answer.
- **Degrade, but never silently.** A circuit breaker opens on *consecutive*
  failures rather than a failure rate — a component failing one request in ten
  is degraded but usable, and opening on that takes a working dependency
  offline. Responses carry `quality: "degraded"` and name the components,
  because a silently degraded service looks healthy on every dashboard while
  serving worse answers. The retrieval stage is `required=True` and has no
  fallback: "degraded" must never mean "answered without retrieving", which is
  not a degraded answer but a fabricated one.
- **Latency is recorded per stage.** A p95 that says "1.2s" is not actionable;
  one that attributes it to the reranker is. Every response carries a
  `trace_id` honoured from an inbound header, so a user reporting a bad answer
  hands you a string that pulls up which chunks were retrieved and how long
  each stage took.

A bug worth recording: `from __future__ import annotations` in the FastAPI
module made every POST return `422: field required`. FastAPI resolves
annotations against *module globals*, and the request models are defined
inside `create_app` so pydantic stays a lazy import — so it fell back to
treating the body as query parameters. A silent, total failure of every write
endpoint, with nothing in the error pointing at the cause. Details in
[`serving.md`](docs/serving.md).

---

## Hardening: the two questions enterprise buyers ask first

**Permission-aware retrieval.** Restricted content reaches no unauthorised
principal — asserted in passages, in the answer, *and in the cache*:

```
principal             may see  in passages  in answer
finance analyst           yes          yes        yes   PASS
intern (no role)           no           no         no   PASS
anonymous                  no           no         no   PASS
```

Filtering is a retrieval **pre-filter, never a post-filter**. Post-filtering
is wrong twice: it returns fewer than k (the chunks the user *may* read never
got to compete for the slots), and it is not a security boundary at all —
anything reaching the scorer has already been read, so it still lands in the
cache, the logs and the reranker. A test reaches into the cache and asserts
the restricted figure is not there.

Untagged documents are **denied by default**, and a missing principal fails
closed. The permissive default leaks silently the first time a document is
ingested before the tagging pipeline exists.

**Indirect prompt injection.** Seven payloads, three separable measurements —
because collapsing them produces a misleading number:

```
detection rate         86%   (the miss is asserted by a test, not hidden)
behavioural success     0%   (NOT a security claim -- see below)
structural             6 invisible characters stripped
```

The 0% is **immunity by construction, not by defence**: the extractive
generator cannot follow instructions at all. Reporting it as a win would be a
false reassurance, and the number is only meaningful against a real model.

An earlier version of this measurement counted "the canary appears in the
retrieved passages" and reported 100% before and after — trivially true, since
the payload *is* the poisoned chunk. Presence in the context and change in
behaviour are different questions.

The sanitizer deliberately **does not delete** text that matches a pattern: an
attacker who learns the pattern could append it to a paragraph they want
suppressed and have the defence remove it for them. Only what is
unconditionally safe — invisible characters, forged delimiters — is stripped.

---

## Judge calibration

The differentiating piece. An uncalibrated LLM judge produces numbers that look
like measurements and are not.

```bash
make eval                 # judged run (needs ANTHROPIC_API_KEY)
make calibrate-export     # 100 stratified samples, human_scores blank
# ... label them by hand, without reading judge_scores first ...
make calibrate-report     # weighted Cohen's kappa, gate is >= 0.60
```

Weighted kappa (quadratic) for ordinal scales, because confusing "correct" with
"partially correct" is a smaller error than confusing it with "incorrect".
Raw agreement is not enough: on a set where 85% of answers are correct, a judge
that always says "correct" scores 85% agreement and has measured nothing.

Two edge cases return honest answers rather than misleading zeros: a constant
judge scores exactly κ = 0 (no information beyond chance — fails the gate),
while *both* raters constant gives κ **undefined** (no disagreement to measure —
re-sample, don't rewrite the rubric).

---

## The regression gate

```bash
make eval-fast && make baseline    # set the reference
# ... make a change ...
make eval-fast && make gate        # exit 1 on regression
```

Absolute thresholds: 2 points of nDCG, 3 of groundedness, **zero** tolerance
for fabricated citations. A metric unscored in either run is reported as
*skipped*, never as passing — silently passing on an unscored metric is how a
broken judge gets through CI green.

---

## Reproducibility

```bash
make reproduce      # 18 figures, ~11s, no model call, $0.00
```

- **Every number this repository publishes is under contract.**
  `scripts/reproduce.py` re-derives all of them from source and `--check`
  exits non-zero when one moves. Changing a published figure is therefore a
  visible edit to that file, in the same commit as the code change that caused
  it. A number that can drift silently is not a result — and this is the tool
  that caught the refusal conclusion above.
- Run ids are hashes of the system config, judge config, metric config **and
  the dataset's content** — editing the eval set changes the hash, so a run
  cannot be mistaken for a re-run of a different set.
- Judge responses are cached on `sha256(model, prompt, schema, rubric_version)`.
  Rubric version is a content hash, so an edited rubric cannot reuse scores
  from the old wording.
- Determinism is asserted in the test suite, not assumed.
- The ablation table is generated by `scripts/build_ablation_table.py`, never
  edited by hand.

---

## Commands

`make help` lists these from the Makefile itself.

| Command | Does |
|---|---|
| `make install` | Install the package and dev dependencies |
| `make install-judge` | Also install the Anthropic SDK (judged metrics, real generation) |
| `make install-embed` | Also install sentence-transformers (real semantic embeddings) |
| `make lint` | Ruff check + format check across `src`, `evals`, `tests`, `scripts` |
| `make ingest` | Fetch + parse filings from EDGAR (needs `GTRAG_SEC_USER_AGENT`) |
| `make index` | Chunk + embed the document store |
| `make query Q="…"` | Ask the baseline a question |
| `make test` | Test suite — no key, no network |
| `make validate` | Dataset structure + corpus join, `--strict` fails on unverified |
| `make stats` | Slice composition against targets |
| `make sweep` | Run the ablation ladder with deltas and power |
| `make sweep-chunking` | Chunking sweep alone |
| `make sweep-generation` | Phase 4 generation ladder |
| `make refusal-curve` | Measure the refusal tradeoff, pick an operating point |
| `make serve` | Run the API locally |
| `make loadtest` | Concurrency ramp (timings are machine-dependent, not pinned) |
| `make docker-build` | Build the runtime image |
| `make docker-up` | API + Prometheus |
| `make security` | Injection + access-control report (fails on leak) |
| `make eval-fast` | Deterministic metrics only |
| `make eval` | Full judged run |
| `make baseline` | Promote latest run to the CI reference |
| `make gate` | Check latest run against the baseline |
| `make compare BASE=… CAND=…` | Paired bootstrap comparison |
| `make calibrate-export` | Sample a judged run for hand-labeling |
| `make calibrate-report` | Judge/human agreement |
| `make ablation` | Regenerate the ablation table |
| `make reproduce` | Re-derive every published figure; fails if one moved |
| `make clean` | Remove caches and generated artifacts (keeps results and baselines) |

---

## What this does not claim

The most useful thing a measurement project can publish is the list of things
it did not measure, because that is the list an unscrupulous version would
quietly leave out.

| Claim you might expect | Actual status |
|---|---|
| These retrieval numbers are results | **No.** Three fixture filings, 17 questions. `make ingest && make sweep` reruns identical code on the real corpus; until then every figure is a smoke test |
| A component was shown to help | **No.** Every delta in the ladder is inconclusive, and the harness says how large the set must be (~720) before any of them could be |
| Semantic chunking and neural reranking were evaluated | **No.** `SentenceTransformerEmbedder` and `CrossEncoderReranker` are written against documented interfaces and have never executed — no model weights here. The offline stand-ins report `neural: false` and `split_embedder_semantic: false` so a run made that way is identifiable, not quietly comparable |
| The sentence-window collapse is a finding | **No.** It is an artifact of a bag-of-words embedder against 77 one-sentence candidates. The caveat travels with the number |
| Groundedness improved | **Not established.** It is a judged metric and no judge has run here. The machinery, cache and calibration gate are built; the number needs an API key |
| The judge is calibrated | **Not yet.** κ ≥ 0.60 is the gate and the round trip is implemented and tested; the human labels do not exist |
| Prompt injection is handled | **No.** 0% behavioural success is immunity *by construction* — the extractive generator cannot follow instructions at all. That number is meaningless until a real model runs. What is real: 86% detection with the miss asserted by a test, and unconditional structural neutralisation |
| The EDGAR client works | **Unverified.** `sec.gov` is blocked at this environment's gateway, so `HttpFetcher` has never made a live request. Logic is covered against recorded fixtures; expect to adjust `ITEM_PATTERNS` on first contact with real filers |
| The container runs | **Unbuilt.** Docker is unavailable here. The Dockerfile and compose file are written, not exercised |
| The load-test p99 is a service latency | **No.** In-process, retrieval and assembly only, and the *only* figures here outside the reproduce contract — they are wall-clock timings and they do not repeat. The reported knee moved between 8, 16 and 32 across four runs, so there isn't one. `--url` drives a live server and is the honest way to claim a p99 |

Every one of these is stated the same way in the phase docs, next to the
number it qualifies, rather than collected here to be forgotten.

---

## A note on the fixture data

`src/gtrag/fixtures/` contains a small synthetic corpus — **invented companies,
invented figures.** It was written so the harness had something to run against
before EDGAR ingestion existed, and it stays because the tests need fixed,
hand-checkable inputs that do not change when a filer reformats their HTML.
None of it is real financial data. It is shaped like the
real thing in the ways that matter for retrieval: two similar companies with
overlapping vocabulary, two fiscal years of near-identical boilerplate, figures
that live in tables, and a footnote that qualifies the number above it.

## Licence

MIT
