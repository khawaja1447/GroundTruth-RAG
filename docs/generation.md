# Phase 4 — grounded generation

Phase 3 stopped at "the right chunks were retrieved". This covers everything
between that and an answer a reader can check.

    query -> rewrite -> retrieve -> assemble -> refuse? -> generate -> verify

---

## 1. The headline result

```
configuration                      answered_unans   false_refusal   fabricated
p3 winner (no generation stages)          100.0%            0.0%         0.0%
+ dedup                                   100.0%            0.0%         0.0%
+ lost-in-the-middle order                100.0%            0.0%         0.0%
+ query rewriting                         100.0%            0.0%         0.0%
+ claim verification                      100.0%            0.0%         0.0%
+ refusal (margin)                          0.0%           69.2%         0.0%
```

Three things to read out of it.

**Fabricated citations are zero on every row.** That is Phase 4's hard gate,
and it holds across the whole eval set. A test asserts it rather than a
report claiming it.

**Without a refusal policy the system answers 100% of the unanswerable
questions.** That is the hallucination path, quantified. It is not a
regression introduced here — it is what the pipeline has always done, now
measured.

**Turning refusal on fixes that and costs 69% false refusals.** Catching all
four unanswerable questions means declining nine of the thirteen answerable
ones. That is not a usable system, and the number says so plainly.

---

## 2. Why refusal is hard here

The refusal threshold was chosen from a measured curve, per the exit gate.
The curve says the signal is not good enough.

```
signal         best J   @ correct   @ false    minimum usable ceiling
top_score      +0.019         25%       23%                       23%
mean_score     +0.154        100%       85%                       38%
margin         +0.308        100%       69%                       23%
```

A perfect separator scores J = +1.0; a coin flip scores 0.0. `top_score` at
+0.019 is a coin flip. `margin` is the best of the three and still needs a
69% false-refusal rate to catch every unanswerable question.

**No operating point exists at a 5% false-refusal ceiling on any signal.**
`scripts/refusal_curve.py` reports that and exits non-zero rather than
returning a threshold that happens to satisfy the constraint by refusing
nothing.

### The degenerate-point trap

The first version of `choose_operating_point` returned a threshold with 0%
correct refusals and 0% false refusals, and called it the answer. It
satisfies any ceiling — by never refusing. A criterion that can be met by
doing nothing is not a criterion, so `RefusalPoint.degenerate` now identifies
those points and the selector rejects them by default.

### The conclusion

**Retrieval-derived confidence is the wrong place to make this decision.** The
retriever returns its best five chunks whether or not any of them answer the
question; the scores reflect similarity within the corpus, not sufficiency
for the question. The generator sees the passage text and can tell the answer
is not in it — which is why `AnthropicGenerator` returns a structured
`refused` flag and the rubric instructs it that refusing is correct when the
information is absent.

That is an architectural finding produced by measurement, and it is the
opposite of what the plan assumed when it listed "tune the threshold".

---

## 3. Context assembly

Three stages, in this order, each for a stated reason.

### Deduplication

Filings repeat boilerplate almost verbatim across fiscal years and across
peers. Two near-identical chunks spend the budget twice and skew the model
toward whatever got repeated. This corpus makes it acute: the smoke corpus
alone contains two Northwind filings whose Item 1 sections differ in three
figures.

Detection uses **4-gram shingles, not a bag of words**. Two filings from
different years share nearly all vocabulary and differ in exactly the figures
that matter, so an order-insensitive comparison calls them duplicates.
`test_word_order_matters` pins this.

### Ordering

Models attend unevenly across a long context; the middle is where information
goes to die. `lost_in_the_middle_order` deals the ranked list alternately to
the front and back, so rank 1 opens the context, rank 2 closes it, and the
weakest chunk lands dead centre:

```
[1, 2, 3, 4, 5]  ->  [1, 3, 5, 4, 2]
```

Whether it helps is an empirical question on your own set, which is why it is
a toggle rather than a hardcoded reorder.

### Budget

A hard ceiling with a stated drop policy: least relevant first, and never
drop the last chunk. Ranks are **renumbered to presentation order** after
assembly, because the citation indices the model returns are positions in
what it was shown — any other numbering makes every citation off by an
unpredictable amount.

Every dropped chunk is accounted for on the returned `AssembledContext`, so a
result file can explain why a chunk the retriever found never reached the
model.

---

## 4. Verification

`LexicalVerifier` runs offline; `JudgeVerifier` reuses the Phase 2
`claim_support` rubric, so verification is scored by the same calibrated
standard that measures groundedness rather than a second, uncalibrated notion
of support that would quietly disagree with it.

**Numbers are checked strictly and separately from words.** A claim asserting
"$9,999 million" when the context says "$4,218 million" shares most of its
vocabulary and none of its meaning, so a single blended similarity score
would call it supported. Any numeric token absent from the context fails the
claim outright.

A failed judge call falls back to the lexical check and says so in the
reason — it never becomes a silent "supported".

Unsupported claims are **annotated, not suppressed**. The claim may well be
true; a system that deletes everything it cannot verify is less useful than
one that says which parts it could not.

---

## 5. Multi-turn rewriting

"And what drove that increase?" contains no company, no period and no
subject. The retriever sees four stopwords and a noun.

`HeuristicRewriter` detects context-dependence and splices in the entities
and fiscal year from recent turns, producing "Northwind fiscal 2024: And what
drove that increase?". Crude, offline, deterministic, and enough to make the
query retrievable. `LLMRewriter` handles what rules cannot and falls back to
the heuristic on any failure.

`is_dependent` is deliberately over-inclusive: rewriting a query that did not
need it is usually harmless, whereas missing one means retrieving on
stopwords.

---

## 6. A bug this phase surfaced

`ExtractiveGenerator` had `min_score: float = 0.05` — an absolute threshold
on the top retrieval score.

Cosine similarities sit near 0.3. Reciprocal-rank-fusion scores sit near
1/(60 + rank) ≈ 0.016–0.033. So the default behaved sensibly under dense
retrieval and **refused every question under any hybrid configuration**.

The Phase 3 retrieval metrics (nDCG, recall, MRR) are unaffected — they are
computed from what the retriever returned, before generation. But the
generation-side metrics on the three hybrid rows of that ladder were wrong:
`false_refusal` would have read 100%. They now read 0%.

This is the same trap documented in `RefusalPolicy` — score scales do not
transfer across retrievers — reached from the other direction. The default is
now 0.0, and refusal is `RefusalPolicy`'s job, which calibrates against the
retriever it will actually be deployed with.

---

## 7. Exit gate

| Gate | Status |
|---|---|
| Zero fabricated citations across the eval set | **Met.** 0.0% on every ladder row, asserted by a test |
| Refusal operating point chosen from a measured curve | **Met**, and the finding is that no acceptable point exists — documented above with the criterion and the numbers |
| Multi-turn rewriting passes a conversational slice | **Partially met.** The heuristic rewriter makes `fil-017` retrievable and is tested; the slice is one question |
| Groundedness improved at equal retrieval quality | **Not established.** See below |

The last one is honest: groundedness is a judged metric, and no judge has run
in this environment. The lexical verifier measures a proxy, and the ladder
shows the verification stage does not move the refusal or citation metrics —
which is expected, since it annotates rather than changes what was retrieved.
Establishing the groundedness delta needs `make eval` with an API key.

---

## 8. What is measured and what is not

**Measured:** shingle-based duplicate detection including the word-order
case, the reorder's exact placement, budget dropping and rank renumbering,
full accounting of dropped chunks, confidence signals, the refusal curve and
its degenerate-point rejection, lexical verification including the fabricated
figure case and judge-failure fallback, dependence detection and heuristic
rewriting, refusal short-circuiting generation, and that with every Phase 4
stage disabled the system behaves exactly as the Phase 3 system did.

**Not measured:** anything needing a model. `LLMRewriter`, `JudgeVerifier`
with a live judge, and `AnthropicGenerator`'s structured refusal path are
written against documented interfaces and have not been executed here.

**Not built:** the post-verification cost/benefit recommendation the plan
asks for needs the judged run to produce a quality delta to weigh against the
extra pass. The machinery is in place; the number is not.
