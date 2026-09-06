# Evaluation methodology

This document is the argument for why the numbers in this repository can be
trusted. It is written to be read by someone deciding whether to believe the
ablation table.

---

## 1. Why evaluation came first

The harness was built before the retrieval improvements it exists to measure.
That ordering is deliberate. Without a measured control, every later change is
an assertion rather than a result, and the usual outcome is a pipeline full of
components that were added because a blog post recommended them.

The cost of this ordering is real: nothing looked impressive for the first
stretch of the project. The benefit is that every row of the ablation table
carries a measured delta and a confidence interval, and three components were
rejected on evidence.

---

## 2. The labeled set

### Composition

Questions are stratified across seven slices, each stressing a different
failure mode. Per-slice reporting is the default everywhere in the harness,
because an aggregate that improves while the `numeric_table` slice collapses is
a regression wearing a disguise.

| Slice | Stresses |
|---|---|
| `single_hop` | Baseline retrieval quality |
| `numeric_table` | Table-aware chunking, exact-value fidelity |
| `multi_hop` | Recall@k, query decomposition |
| `comparative_temporal` | Metadata filtering, fiscal-year disambiguation |
| `unanswerable` | Refusal behaviour, hallucination rate |
| `ambiguous` | Clarification vs. confident wrong answer |
| `adversarial` | Injection resistance via retrieved content (Phase 6) |

Run `make stats` for the current counts against target.

### Graded relevance

Gold chunks carry three levels, not two:

| Level | Meaning |
|---|---|
| `2` | Contains the answer |
| `1` | Supporting context, insufficient alone |
| `0` | Irrelevant (implicit — unlabeled chunks) |

Binary labels would make nDCG meaningless, and they would hide the failure
this corpus produces most often: retrieving the surrounding discussion while
missing the table cell with the figure in it. `answer_recall@k` measures
exactly that, restricted to relevance-2 chunks.

### Answerability is derived, never declared

A question is answerable exactly when it has gold chunks. Three cases fall out
of that one rule:

- `unanswerable` — nothing in the corpus answers it; refusing is correct.
- `ambiguous` — underspecified as asked (no company named when two are in the
  corpus), so asking for clarification is correct and a confident answer is
  the failure.
- everything else — gold chunks exist and an answer is expected.

Refusal scoring and the "is this metric defined" check both key off this, so
it cannot be set independently of the labels. The dataset loader rejects any
record where the type and the labels disagree.

### Verification

Candidates were generated semi-automatically — sample chunks, have a model
propose questions answerable from them — and then **every one was verified by
hand**. Records carry `provenance` and `verified_by`, and the runner refuses to
execute on an unverified dataset unless `--allow-unverified` is passed
explicitly, which marks the run as exploratory.

> **Acceptance rate: to be filled in when the full set is built.** Report the
> fraction of generated candidates kept. It is evidence of rigour, and its
> absence is conspicuous to anyone who has built one of these.

An eval set a model wrote and grades unsupervised is circular. The verification
flag exists so that circularity is impossible to reach by accident.

---

## 3. Metrics

### The undefined convention

**A metric that does not apply to a question returns `None`, never `0.0`.**

This is the single most important convention in the harness. Unanswerable
questions have no gold chunks, so recall is undefined for them. Scoring them
`0.0` would drag the mean down by however many unanswerable questions the set
happens to contain, making the headline number a function of dataset
composition rather than retrieval quality — and it would then move whenever
the eval set grew.

Aggregates report `n` and `n_undefined` separately so the denominator of every
number is visible.

### Retrieval

| Metric | Definition |
|---|---|
| `recall@k` | Fraction of relevance≥1 chunks in the top k |
| `answer_recall@k` | Same, restricted to relevance-2 chunks |
| `precision@k` | Relevant fraction of what was returned (denominator is the returned count capped at k, not k) |
| `mrr` | Reciprocal rank of the first relevant chunk |
| `ndcg@k` | Graded, exponential gain: `(2^rel − 1) / log2(rank + 1)` |

nDCG uses exponential rather than linear gain, so a relevance-2 chunk is worth
3× a relevance-1 chunk rather than 2× — the right shape when one of the two
actually answers the question. The ideal ranking is the gold relevances sorted
descending and truncated at k.

### Generation — deterministic

These need no model, no key, and no budget, which is why they run in CI on
every push:

- `correct_refusal` / `false_refusal` / `answered_unanswerable` — the full 2×2
  of refusal behaviour. Reporting only the first is the common mistake: a
  system that refuses everything scores 100% on the unanswerable slice while
  being useless.
- `citation_fabrication_rate` — cited chunk ids checked against what was
  actually retrieved. A citation pointing at a chunk that was never retrieved
  cannot support anything, and it is invisible to a human spot-check because
  it looks perfectly plausible. **The CI gate for this is zero.**

### Generation — judged

- `context_sufficiency` (binary) — could a careful reader answer from these
  passages alone? Judged **independently of what the system answered**, which
  is what separates a retrieval failure from a generation failure and makes
  debugging tractable.
- `answer_correctness` (0/1/2) — against the reference answer.
- `groundedness` — the answer is decomposed into atomic claims and each is
  checked for entailment against the retrieved context. Reported as the
  fraction of supported claims.

---

## 4. Judge calibration

**An uncalibrated LLM judge is a random number generator with good manners.**

### Protocol

1. Run the eval with the judge enabled.
2. `make calibrate-export` — samples 100 items, stratified by question type
   and seeded, into a file with `human_scores` blank.
3. Label them by hand **without reading `judge_scores` first**.
4. `make calibrate-report` — computes agreement.

### Why Cohen's kappa

Raw agreement is inflated by the base rate. On a set where 85% of answers are
correct, a judge that always says "correct" scores 85% agreement and has
measured nothing. Kappa corrects for agreement expected by chance.

**Weighted kappa** (quadratic) for the ordinal scales: confusing "correct" with
"partially correct" is a smaller error than confusing it with "incorrect", and
unweighted kappa treats those identically. Binary scales are reported
unweighted, since with two categories every weighting scheme is the same and
labelling it "quadratic" would misrepresent what was computed.

### The gate

**κ ≥ 0.60** on `answer_correctness` and `context_sufficiency`
(Landis & Koch: "substantial" begins at 0.61; below 0.60 is "moderate" at
best). Below the gate, the judge is not measuring what the rubric intends. The
fix is to tighten the rubric's boundary definitions and add worked examples at
the points of disagreement, then re-run and re-label — not to accept the number.

Two edge cases are handled explicitly rather than returning a misleading zero:

- **A constant judge scores κ = 0 exactly.** Observed and expected disagreement
  coincide. That is correct and informative: the judge carries no information
  beyond chance, and it fails the gate.
- **Both raters constant** on a wider scale gives κ *undefined*, not zero. There
  was no disagreement to measure; the sample needs harder items. Reporting 0.0
  here would send you rewriting a perfectly good rubric.

> **Measured κ: to be filled in after the first calibration run.** Report it per
> metric, with the confusion matrix, in the eval report and in the README body.

---

## 5. Reproducibility

- **Content-addressed cache.** Every model call is keyed on
  `sha256(model, prompt, schema, rubric_version)` in SQLite. Re-running after a
  retrieval-only change reuses every judgment, so the measured delta is
  attributable to the change rather than to judge sampling noise.
- **Rubric version in the key.** Derived from the rubric's content hash, so an
  edited rubric cannot silently reuse scores from the old wording.
- **Run ids are config hashes.** Over the system config, judge config, metric
  config, *and the dataset's content*. Editing the eval set changes the hash,
  so a run cannot be mistaken for a re-run of a different set.
- **Seeded bootstrap.** The seed is written into every result file.
- **Determinism is tested.** `test_deterministic_across_runs` asserts that the
  same config produces the same numbers at different worker counts.

---

## 6. Statistical treatment

A single mean over a few hundred questions is a noisy estimate, and RAG
ablations routinely produce differences smaller than that noise.

- **Bootstrap confidence intervals** (10,000 resamples, percentile method) on
  every headline metric.
- **Paired bootstrap** for comparing two configurations. Both are evaluated on
  the same questions, so resampling questions and differencing *within* each
  resample cancels per-question difficulty — the dominant source of variance.
  An unpaired comparison of the same data needs a much larger gap to reach the
  same conclusion.
- **A delta whose CI includes zero is reported as inconclusive**, and marked
  `(ns)` in the ablation table. It has not been shown to do anything on this
  eval set, however inviting the point estimate looks. This is the guard
  against shipping noise, and it is the reason some plausible-sounding
  components were rejected.

Intervals are computed for headline metrics only. The full sweep is ~30 metrics
across 7 slices, and resampling all of them costs minutes to produce intervals
nobody reads on `precision@1`.

---

## 7. The regression gate

CI runs the harness on every push and fails the build on:

| Check | Threshold |
|---|---|
| `ndcg@10` drop | > 0.02 |
| `recall@10` drop | > 0.02 |
| `groundedness` drop | > 0.03 |
| `answer_correctness` drop | > 0.03 |
| `false_refusal` rise | > 0.05 |
| `citation_fabrication_rate` | **> 0** |
| judge errors | any |

Thresholds are absolute rather than statistical because CI needs a fast, stable
yes/no; the paired bootstrap is the tool for deciding whether a change is real.

Two properties worth noting:

- A metric unscored in either run is reported as **skipped**, never as passing.
  Silently passing on an unscored metric is how a broken judge gets through CI
  green.
- A run with any judge error cannot be recorded as a passing baseline.

---

## 8. Known limitations

- **The judge and the system under test can share a model family.** Correlated
  blind spots are possible. The calibration study bounds this against human
  judgement but does not eliminate it.
- **Claim decomposition is heuristic.** The model-based splitter falls back to
  a sentence splitter on failure, and an over-split compound sentence inflates
  the claim count that groundedness averages over.
- **The eval set is single-annotator.** Inter-annotator agreement among
  multiple humans is not measured, so "human" here means one person's
  judgement applied consistently.
- **Confidence intervals assume questions are independent.** Questions drawn
  from the same filing are not fully independent, so intervals are likely
  slightly optimistic.
- **`ambiguous` questions are scored only for refusal behaviour.** Whether the
  clarifying question asked was a *good* one is not measured.
