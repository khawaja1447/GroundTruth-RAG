# groundtruth-rag

A retrieval system over corporate financial filings where **every
architectural decision is justified by a measured delta** — and the evaluation
harness that produces those numbers is the actual product.

> **Status: Phases 1–2 complete.** The corpus pipeline and the evaluation
> harness are built, tested and running — 257 tests, no network, no API key.
> Phases 3–7 (the retrieval ablation program, serving, hardening) are next.
> See [`docs/eval-methodology.md`](docs/eval-methodology.md) for the argument
> behind the numbers and [`docs/corpus.md`](docs/corpus.md) for the ingestion
> design.

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
make test           # 257 tests, stdlib only
make validate       # dataset structure + corpus join
make eval-fast      # full deterministic eval
```

To build the real corpus (the only step that needs the network):

```bash
export GTRAG_SEC_USER_AGENT="groundtruth-rag research you@example.com"
make ingest COMPANIES=10 YEARS=3
make index
make query Q="What was Apple's total net revenue in fiscal 2024?"
```

`make eval-fast` on the fixture corpus produces:

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
  the hardest thing to retrieve on this corpus, which is what motivates
  structure-aware chunking in Phase 3.

---

## What is actually built

```
src/gtrag/                # Phase 1 — the corpus pipeline and baseline
├── ingest/
│   ├── document.py       # document model + Span (the anchoring primitive)
│   ├── edgar.py          # SEC client: UA enforcement, 10 req/s token bucket
│   └── parse.py          # HTML -> text, sections (TOC trap), tables
├── chunking/base.py      # Chunker protocol + fixed-token baseline, span-tracked
├── index/
│   ├── embed.py          # Embedder protocol: hashing (offline) + sentence-transformers
│   └── store.py          # exhaustive cosine index, embedder-mismatch guard
├── generate/generator.py # extractive (offline) + Anthropic, both refusable
├── baseline.py           # the Phase 1 control system
└── cli.py                # ingest / index / query / inspect

evals/                    # Phase 2 — the measurement layer
├── spans.py              # span -> per-chunking relevance resolution
├── types.py          # labeled-question model + the invariants that keep it honest
├── dataset.py        # loading, corpus join checks, composition reporting
├── metrics/
│   ├── retrieval.py  # recall, precision, MRR, graded nDCG — hand-implemented
│   ├── generation.py # refusal 2x2, citation validity, claim splitting
│   └── stats.py      # bootstrap CIs, paired bootstrap
├── judges/
│   ├── base.py       # judge protocol, scales, rubric versioning
│   ├── llm_judge.py  # Anthropic-backed, structured output, cached
│   └── rubrics/      # 4 versioned rubrics with worked examples
├── calibration.py    # weighted Cohen's kappa + the human-label round trip
├── cache.py          # content-addressed SQLite response cache
├── runner.py         # config hashing, execution, scoring, result files
├── report.py         # run summaries, slice tables, paired comparisons
├── gate.py           # the CI regression gate
└── cli.py            # 8 commands
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

| Command | Does |
|---|---|
| `make ingest` | Fetch + parse filings from EDGAR (needs `GTRAG_SEC_USER_AGENT`) |
| `make index` | Chunk + embed the document store |
| `make query Q="…"` | Ask the baseline a question |
| `make test` | Test suite — no key, no network |
| `make validate` | Dataset structure + corpus join, `--strict` fails on unverified |
| `make stats` | Slice composition against targets |
| `make eval-fast` | Deterministic metrics only |
| `make eval` | Full judged run |
| `make baseline` | Promote latest run to the CI reference |
| `make gate` | Check latest run against the baseline |
| `make compare BASE=… CAND=…` | Paired bootstrap comparison |
| `make calibrate-export` | Sample a judged run for hand-labeling |
| `make calibrate-report` | Judge/human agreement |
| `make ablation` | Regenerate the ablation table |

---

## A note on the fixture data

`src/gtrag/fixtures/` contains a small synthetic corpus — **invented companies,
invented figures.** It exists so the harness has something to run against
before Phase 1 delivers real EDGAR ingestion, and so the tests have fixed,
hand-checkable inputs. None of it is real financial data. It is shaped like the
real thing in the ways that matter for retrieval: two similar companies with
overlapping vocabulary, two fiscal years of near-identical boilerplate, figures
that live in tables, and a footnote that qualifies the number above it.

## Licence

MIT
