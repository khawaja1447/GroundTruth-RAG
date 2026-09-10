# Phase 3 — the retrieval ablation program

Sweep each dimension, keep what wins, and record what loses.

---

## 1. The headline result

**Run on the 17-question smoke set, every delta in the ladder came back
inconclusive.**

```
baseline: fixed + dense        nDCG@10 = 0.8196
+ structure-aware chunking     nDCG@10 = 0.7527   delta -0.0669 [-0.2088, +0.0568]  inconclusive
+ bm25 hybrid (RRF)            nDCG@10 = 0.8297   delta +0.0769 [+0.0000, +0.1941]  inconclusive
+ reranking                    nDCG@10 = 0.7859   delta -0.0438 [-0.1420, +0.0568]  inconclusive
+ metadata pre-filtering       nDCG@10 = 0.8663   delta +0.0804 [-0.0888, +0.2295]  inconclusive
```

That is not a failure of the components. It is the harness doing its job:
with n=13 scoreable questions, the confidence interval on any paired
difference is about ±0.15, so a 4-point or even an 8-point move cannot be
distinguished from noise. Reporting "+0.077 from hybrid retrieval" off this
data would be exactly the kind of unearned claim the whole project exists to
prevent.

The useful output is therefore not a ranking of components. It is a number
telling you what to do next.

### Two rungs go *down*, and they stay in

`structure_aware` and `reranking` both score below the rung beneath them. The
temptation is to drop them and publish the monotone ladder that remains.

They stay for two reasons. The ladder is a **fixed sequence of one-component
steps**, not a greedy search — dropping a losing rung renumbers everything
above it and destroys the attribution the whole structure exists to provide.
And both deltas are *inconclusive*: an interval spanning [-0.21, +0.06]
supports "worse" no better than it supports "better", so removing the rung
would be acting on the same noise the report refuses to act on in the other
direction.

On a corpus of three short filings, `fixed` produces five chunks and
`structure_aware` twelve. Fewer, larger chunks are simply easier to rank when
the whole corpus fits in a top-5. That is a property of the smoke corpus, not
a verdict on the strategy — and it is why the caveat below about corpus size
applies to more than `parent_document`.

### How large does the eval set need to be?

`evals.metrics.stats.paired_power` estimates the smallest paired difference a
set of this size can resolve, and inverts it to give the n required for a
target effect:

```
ndcg@10:   n=13, sd=0.2743  -> can resolve ~0.149; to detect 0.020: need ~723
recall@10: n=13, sd=0.2774  -> can resolve ~0.151; to detect 0.020: need ~739
```

So the CI gate's 2-point nDCG threshold needs roughly **720 questions**, not
the 220 the plan budgeted for. That is a concrete, actionable finding about
the eval set, produced by the measurement program before a single component
decision was made on bad evidence.

The pair measured is the bottom and top of the ladder, which is the widest
spread the set contains and therefore the honest estimate of its variance;
narrower pairs give smaller `n` and would flatter the set.

Without this diagnostic, an all-inconclusive ablation is ambiguous between
"these components do nothing" and "this eval set is too small to say" —
opposite conclusions with opposite next actions.

> **Every number above is a smoke test** over three fixture filings, not a
> result. `make ingest` then `make sweep` reruns the identical code on the
> real corpus.

---

## 2. Dimension 1 — chunking

Six strategies, graded from one human labeling.

```
fixed 512/50 (baseline)   5 chunks   nDCG@10 = 0.8196   recall@10 = 1.0000
recursive                 4 chunks   nDCG@10 = 0.8326   recall@10 = 1.0000
structure-aware          12 chunks   nDCG@10 = 0.7527   recall@10 = 0.9231
sentence-window w=2      77 chunks   nDCG@10 = 0.2678   recall@10 = 0.2162
parent-document          17 chunks   nDCG@10 = 0.5482   recall@10 = 0.6538
semantic p25             28 chunks   nDCG@10 = 0.7225   recall@10 = 0.9231
```

Every chunk count here is lower than it was before Phase 5 excluded front
matter — the cover page and table of contents were about a fifth of the
corpus. `docs/serving.md` §7 records why removing them mattered far more than
the retrieval metrics said at the time.

### Reading this honestly

**Sentence-window's collapse is an artifact of the offline embedder, not a
finding about the strategy.** `HashingEmbedder` is bag-of-words; a single
sentence gives it four or five content terms to work with, which is far too
sparse to rank against 77 candidates. The strategy is designed for a semantic
embedder, where a sentence embeds densely. Re-run with
`--embedder sentence-transformers` before drawing any conclusion. The same
caveat applies to `semantic` chunking, which splits on embedding distance and
is therefore measuring vocabulary overlap rather than meaning here — its
config records `split_embedder_semantic: false` so a run made that way is
identifiable rather than quietly comparable.

**Parent-document is penalised by corpus size**, not by design: with 17
parents over three short filings, returning whole parents means returning a
large fraction of the corpus per query.

These are exactly the interpretation traps that make an ablation table
worthless if the caveats are dropped. They stay in the table with their
reasons attached.

### What each strategy fixes

| Strategy | The failure it targets |
|---|---|
| `recursive` | Fixed-size splits mid-sentence |
| `structure_aware` | Splits tables in half; merges unrelated sections |
| `sentence_window` | Large chunks dilute the embedding |
| `parent_document` | Small chunks retrieve well but starve the generator |
| `semantic` | Structural boundaries are not always topical ones |

`structure_aware` is the one this corpus was chosen to reward, and its two
rules are asserted by tests rather than hoped for:

- **A table is atomic.** `test_never_splits_a_table` checks every table in
  the fixture is wholly inside exactly one chunk, at a chunk budget far below
  the table's own size. A financial table cut in half puts row labels in one
  chunk and figures in another, and neither is answerable.
- **A chunk never crosses a section boundary.** Merging the tail of Risk
  Factors with the head of MD&A produces a chunk about neither that retrieves
  for queries about both.

---

## 3. Dimensions 3–5 — the ladder

Each rung adds exactly one component to the rung above, so the delta is
attributable. `test_ladder_adds_one_component_per_rung` enforces that
structurally — a rung that changed two things at once would fail the suite.

### Hybrid retrieval (BM25 + RRF)

Fusion is **rank-based, not score-based**. BM25 scores and cosine
similarities live on incomparable scales, and normalising them into agreement
requires a per-corpus calibration that is itself an unmeasured
hyperparameter. Reciprocal rank fusion needs no calibration:

```
score(d) = sum_i  weight_i / (k + rank_i(d))        k = 60
```

Components fetch `fetch_depth=50` candidates, deeper than the final `top_k`:
a chunk ranked 15th by one retriever and 3rd by the other should be able to
surface, and it cannot if each list is truncated at 5 first.

### Reranking

`retrieve_depth` is the recall ceiling. The reranker reorders; it does not
search. A chunk missing from the top 50 can never be recovered no matter how
good the reranker is — `test_cannot_recover_what_the_first_stage_missed`
asserts this rather than leaving it as folklore.

`LexicalReranker` is the offline stand-in and reports `neural: false`.
`CrossEncoderReranker` is the real one. The interesting output is the
quality-against-latency curve, not either number alone.

### Metadata pre-filtering

Rule-based, not model-based: company names and fiscal years are a closed
vocabulary drawn from the corpus itself, so a lookup is more accurate than an
LLM call, free, and deterministic.

Two behaviours worth stating, both tested:

- **A question naming two companies is not filtered.** Narrowing a
  comparative question to one company would silently make it unanswerable.
  The extractor only filters on an unambiguous single match.
- **Missing metadata is not a mismatch.** Excluding chunks that merely lack
  the field would silently shrink the corpus on any incompletely-tagged
  document.

Filtering is applied as a **pre-filter**, never a post-filter. Post-filtering
means the top-k was chosen from the wrong pool and the right chunk may already
have been dropped — and it cannot be relied on for access control, which is
why Phase 6 builds on this.

---

## 4. Running it

```bash
make sweep-chunking        # dimension 1 alone
make sweep                 # the full ladder, with deltas and power
make ablation              # regenerate the Markdown table from result files
make reproduce             # re-derive every figure on this page and fail if one moved
```

The sweep caches indexes on (chunker config, embedder), so varying only the
retriever does not re-embed the corpus per row. Embedding dominates runtime,
and a sweep that is slow to run is a sweep that stops being run.

**Every number on this page is under contract.** `scripts/reproduce.py` holds
a `PUBLISHED` table of all eighteen figures the repository states, re-derives
them from source in about eleven seconds with no model call, and `--check`
exits non-zero when one moves. Changing a published number is therefore a
visible edit to that file in the same commit as the code change that caused
it — which is how the front-matter correction in
[`generation.md` §2](generation.md) was found rather than shipped.

---

## 5. What is measured and what is assumed

**Measured here:** every chunker's span invariants and determinism, table
atomicity, section containment, BM25 ranking, RRF fusion behaviour and
weighting, reranking depth semantics, metadata extraction including the
two-company and missing-field cases, index cache correctness, and that one
labeling resolves under all six chunkings with disjoint chunk-id sets.

**Not measured:** anything requiring a neural model. `SentenceTransformerEmbedder`
and `CrossEncoderReranker` are written against their documented interfaces and
have never been executed — this environment has no model weights. Their rows
in the ablation table are unfilled, and the offline stand-ins are labelled
non-semantic and non-neural precisely so nobody mistakes one for the other.

**Not yet built:** query decomposition for multi-hop questions, HyDE, and
multi-turn query rewriting. The dataset carries a multi-turn question
(`fil-017`) that is expected to fail until rewriting exists — it is in the set
now so the gap is visible rather than forgotten.
