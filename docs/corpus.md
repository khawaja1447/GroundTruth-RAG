# Phase 1 — corpus and baseline

The control. Everything Phase 3 measures is measured against the number this
produces.

---

## 1. Why the baseline is deliberately bad

Fixed 512-token chunks, one dense retriever, top-k=5, everything stuffed into
a prompt. No reranking, no hybrid retrieval, no metadata filtering, no query
rewriting.

The temptation is to start with the sophisticated version. Without a measured
control, every later improvement is an assertion, and assertions are exactly
what make other people's RAG repos unconvincing.

It is not throwaway code. It satisfies the same `RagSystem` protocol as every
later configuration, it is fully config-driven, and it stays in the repo as
the `baseline` configuration forever.

---

## 2. Span anchoring — the decision that makes Phase 3 possible

This is the most consequential design choice in the project, and it is easy to
get wrong in a way that is not discovered until Phase 3 is underway.

**The problem.** Phase 2 labels gold evidence by `chunk_id`. Phase 3's first
ablation dimension *is the chunking*. Re-chunk the corpus and every `chunk_id`
changes, so every label points at something that no longer exists. You would
have to re-label the entire eval set once per chunking strategy — which nobody
does, so in practice people quietly stop comparing chunking strategies and the
most valuable ablation never happens.

**The fix.** Anchor evidence to a character span in the *document*, which
chunking cannot move. Each chunk records the span it covers. Relevance is
derived by overlap at eval time.

```
document text   ......[=== gold span ===]...........
chunking A      [ chunk 1 ][ chunk 2 ][ chunk 3 ]      -> chunk 2 relevant
chunking B      [   chunk 1   ][   chunk 2   ]         -> chunk 1 relevant
```

Both labelings come from the same span. Neither needed a human. One labeling
effort survives every chunking strategy you will ever try.

### The grading rule

A chunk's relevance is decided by how much of **the gold span** it covers, not
how much of the chunk is gold:

| Coverage of the span | Relevance |
|---|---|
| ≥ 90% | 2 — contains the answer |
| ≥ 25% | 1 — supporting context |
| below | 0 — unlabeled |

Asymmetric on purpose. A 512-token chunk containing a one-sentence answer is a
retrieval success even though 95% of its text is unrelated. Grading on the
chunk's own composition would report 0.05 and punish exactly the large-chunk
strategies Phase 3 needs to evaluate fairly.

The partial band exists because a chunk boundary can cut an answer in half.
Neither half alone answers the question, and calling both irrelevant would make
fixed-size chunking look better than it is by hiding its worst failure mode.

### Diagnosing a chunking

`evals.spans.coverage_report` reports which questions lost their evidence under
a given chunking. Run it whenever the chunker changes: a question that loses
its answer-bearing chunk is a property of the chunking, not a retrieval
result, and it must not be mistaken for a worse retriever.

---

## 3. Parsing filings

### The table-of-contents trap

"Item 1A. Risk Factors" appears at least twice in every 10-K: once in the table
of contents, once at the real section. A first-match parser anchors every
section to the TOC.

Measured on the test fixture, that mistake produces sections of **21, 26 and 50
characters** against correct lengths of **777, 758, 996 and 549**. The symptom
is not a crash — it is a corpus that looks fine and retrieves nothing.

The parser takes the **last** match per item that is followed by enough text to
be a real section. Last because the TOC always precedes the body; the length
floor because a filing that genuinely omits an item (Item 1B is often "None.")
must not swallow the rest of the document.
`tests/test_ingest.py::test_does_not_anchor_to_the_table_of_contents` guards it.

### Tables

Extracted structurally before any text flattening, then linearized in place
with aligned columns. Both forms are kept: the linearized text is embedded and
shown to the model, the structure is what lets Phase 3's structure-aware
chunker refuse to split a table down the middle.

Column alignment is not cosmetic — it keeps a row's cells adjacent in the token
stream, so a figure stays retrievable together with its row label.

### Offsets

Whitespace normalization happens exactly once, at ingestion, before any offset
is recorded. Every span indexes into the normalized text, so a second pass
would shift every offset and silently corrupt every label. `normalize_whitespace`
is tested for idempotence for that reason.

---

## 4. SEC compliance

Two requirements that are not optional. SEC blocks on both.

- **User-Agent must identify you**, with a contact email. A default
  `python-urllib/3.11` gets a 403 and repeat offenders are IP-banned.
  `GTRAG_SEC_USER_AGENT` is required rather than defaulted, so the failure is a
  clear message at startup instead of a confusing 403 later.
- **Rate limit is 10 requests/second**, enforced with a token bucket shared
  across threads. A per-request `sleep` does not bound concurrent throughput —
  eight threads each sleeping still issue eight requests at once. Only
  serialising the *grant* of permission works, and there is a test that spawns
  threads to prove it.

Responses are cached to disk, because re-parsing the same filings while
developing the section parser is normal and re-downloading each time is both
slow and rude.

---

## 5. Offline by default

Every stage has a real implementation and an offline one behind the same
protocol:

| Stage | Offline default | Real |
|---|---|---|
| Fetching | `RecordedFetcher` (fixtures) | `HttpFetcher` |
| Embedding | `HashingEmbedder` | `SentenceTransformerEmbedder` |
| Generation | `ExtractiveGenerator` | `AnthropicGenerator` |

This keeps the zero-dependency discipline from Phase 2: the full pipeline is
constructible, runnable and testable with no API key, no model download and no
network, which is what makes CI deterministic.

**`HashingEmbedder` is not semantic.** It is hashed bag-of-words with
sub-linear term weighting — a lexical retriever wearing a vector interface. Its
`config` reports `"semantic": False`, and there is a test asserting that, so no
number produced with it can be mistaken for a semantic-retrieval result.

---

## 6. Index

Exhaustive cosine search. At Phase 1 corpus size an exact scan is milliseconds
and always correct, whereas an ANN index adds a recall parameter that would
confound every retrieval measurement in Phase 3 — you would not know whether a
change moved retrieval quality or just the approximation. Qdrant arrives in
Phase 3, when metadata pre-filtering is genuinely needed and the
recall/latency tradeoff is itself worth measuring.

Loading an index built with a different embedder is refused. Vectors from
different models are not comparable, and searching anyway returns plausible
nonsense — far worse than an error, because nothing about the output looks
wrong.

---

## 7. Running it

```bash
export GTRAG_SEC_USER_AGENT="groundtruth-rag research you@example.com"

make ingest COMPANIES=10 YEARS=3     # fetch + parse (the only networked step)
make index CHUNK_TOKENS=512          # chunk + embed
make query Q="What was Apple's total net revenue in fiscal 2024?"
```

`ingest` and `index` are separate commands on purpose: Phase 3 re-chunks and
re-embeds the same corpus dozens of times, and none of that should re-download
anything.

For real semantic retrieval:

```bash
make install-embed
make index EMBEDDER=sentence-transformers
```

---

## 8. What is verified, and what is not

Honest status, because it affects how much to trust what follows.

**Verified here:** parsing (including the TOC trap, against a fixture filing),
table extraction, span arithmetic, span→relevance resolution under two
different chunkings, chunk-id stability, the index and its embedder guard, the
EDGAR client's listing and URL construction against a recorded submissions
response, the rate limiter under threads, and the assembled baseline answering
end to end. 257 tests, no network, no API key.

**Not verified here:** a live EDGAR fetch. The environment this was built in
blocks `sec.gov` and `data.sec.gov` at the network gateway, so `HttpFetcher`
has never made a real request. Its logic is covered by the recorded fetcher and
the User-Agent/rate-limit tests, but the first live `make ingest` is the real
test of it — expect to adjust the section patterns in `ITEM_PATTERNS` once you
see how a few real filers format their headings. Real 10-K HTML is messier than
any fixture.

**Also not verified:** `SentenceTransformerEmbedder` and `AnthropicGenerator`
have not been executed, only written against their documented interfaces.
