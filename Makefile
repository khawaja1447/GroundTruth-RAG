.DEFAULT_GOAL := help
PY      ?= python3
DATASET ?= evals/datasets/qa_seed.jsonl
SYSTEM  ?= gtrag.fixtures.system:FixtureRagSystem
LABEL   ?= baseline
RUN     ?= $(shell ls -t evals/results/*.json 2>/dev/null | head -1)

.PHONY: help install test lint validate stats eval eval-fast baseline gate \
        compare calibrate-export calibrate-report ablation clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package and dev dependencies
	$(PY) -m pip install -e '.[dev]'

install-judge:  ## Also install the Anthropic SDK (needed only for judged metrics)
	$(PY) -m pip install -e '.[judge,dev]'

test:  ## Run the test suite (no API key, no network)
	$(PY) -m pytest

lint:  ## Lint and format-check
	$(PY) -m ruff check src evals tests
	$(PY) -m ruff format --check src evals tests

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
