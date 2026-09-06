.DEFAULT_GOAL := help
PY      ?= python3
DATASET ?= evals/datasets/qa_seed.jsonl
SYSTEM  ?= gtrag.fixtures.system:FixtureRagSystem
LABEL   ?= baseline
COMPANIES    ?= 10
YEARS        ?= 3
CHUNK_TOKENS ?= 512
EMBEDDER     ?= hashing
Q            ?= What was total net revenue in the most recent fiscal year?
RUN     ?= $(shell ls -t evals/results/*.json 2>/dev/null | head -1)

.PHONY: help install install-judge install-embed test lint ingest index query \
        sweep sweep-chunking validate stats eval eval-fast baseline gate compare calibrate-export \
        calibrate-report ablation clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package and dev dependencies
	$(PY) -m pip install -e '.[dev]'

install-judge:  ## Also install the Anthropic SDK (judged metrics + real generation)
	$(PY) -m pip install -e '.[judge,dev]'

install-embed:  ## Also install sentence-transformers (real semantic embeddings)
	$(PY) -m pip install -e '.[judge,embed,dev]'

ingest:  ## Fetch and parse filings from EDGAR (needs GTRAG_SEC_USER_AGENT)
	$(PY) -m gtrag.cli ingest --companies $(COMPANIES) --years $(YEARS)

index:  ## Chunk and embed the document store into a vector index
	$(PY) -m gtrag.cli index --chunk-tokens $(CHUNK_TOKENS) --embedder $(EMBEDDER)

query:  ## Ask the baseline a question: make query Q="what was revenue?"
	$(PY) -m gtrag.cli query "$(Q)" --embedder $(EMBEDDER)

test:  ## Run the test suite (no API key, no network)
	$(PY) -m pytest

lint:  ## Lint and format-check
	$(PY) -m ruff check src evals tests
	$(PY) -m ruff format --check src evals tests

sweep:  ## Run the full ablation ladder (deltas + statistical power)
	$(PY) scripts/run_sweep.py --sweep both $(if $(DOCS),--docs $(DOCS),)

sweep-chunking:  ## Run the chunking sweep alone (dimension 1)
	$(PY) scripts/run_sweep.py --sweep chunking $(if $(DOCS),--docs $(DOCS),)

validate:  ## Validate the dataset's structure and its join to the corpus
	$(PY) -m evals.cli validate --dataset $(DATASET) --corpus --strict

stats:  ## Show eval-set composition against slice targets
	$(PY) -m evals.cli stats --dataset $(DATASET)

eval:  ## Full eval, judged (needs ANTHROPIC_API_KEY)
	$(PY) -m evals.cli run --dataset $(DATASET) --system $(SYSTEM) --label "$(LABEL)"

eval-fast:  ## Deterministic metrics only -- no judge, no API key, no cost
	$(PY) -m evals.cli run --dataset $(DATASET) --system $(SYSTEM) --label "$(LABEL)" --no-judge

baseline:  ## Promote the most recent run to the stored CI baseline
	$(PY) -m evals.cli baseline $(RUN)

gate:  ## Check the most recent run against the stored baseline
	$(PY) -m evals.cli gate $(RUN)

compare:  ## Paired comparison: make compare BASE=<file> CAND=<file>
	$(PY) -m evals.cli compare $(BASE) $(CAND)

calibrate-export:  ## Sample a judged run for hand-labeling
	$(PY) -m evals.cli calibrate-export $(RUN) --dataset $(DATASET) -n 100

calibrate-report:  ## Judge/human agreement (Cohen's kappa) -- gate is kappa >= 0.60
	$(PY) -m evals.cli calibrate-report

ablation:  ## Regenerate the ablation table from every result file
	$(PY) scripts/build_ablation_table.py

clean:  ## Remove caches and generated artifacts (keeps results and baselines)
	rm -rf .pytest_cache **/__pycache__ evals/.cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
