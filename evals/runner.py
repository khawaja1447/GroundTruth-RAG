"""The eval runner: config hashing, execution, scoring, and result files.

Three properties this file exists to guarantee:

  * **Reproducible.** A run is identified by a hash of the dataset, the
    system config, the judge config, and the metric config. Two runs with the
    same hash must produce the same numbers, so the hash covers everything
    that can change a result and nothing that cannot.
  * **Attributable.** Every per-question score is written out, not just the
    aggregate. The ablation table is generated from these files, and the
    failure taxonomy is built by clustering them.
  * **Honest about gaps.** A question the judge could not score is recorded
    as unscored, never as zero.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import platform
import sys
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gtrag.types import RagSystem, SystemResponse

from .cache import ResponseCache
from .dataset import composition_report
from .judges.base import SCALES, Judge, Judgment, NullJudge
from .metrics import generation as gen
from .metrics import retrieval as ret
from .metrics.stats import Aggregate, aggregate
from .types import Dataset, EvalQuestion, QuestionType

__all__ = [
    "RunConfig",
    "QuestionResult",
    "RunResult",
    "run_eval",
    "load_system",
    "config_hash",
]

DEFAULT_K_VALUES = (1, 5, 10, 20)
PRIMARY_K = 10
RESULT_SCHEMA_VERSION = 2

# The metrics that lead every report, in reading order: retrieval quality
# first (what Phase 3 moves), then generation quality, then the honesty
# checks, then cost.
HEADLINE_METRICS: tuple[str, ...] = (
    "ndcg@10",
    "recall@10",
    "answer_recall@10",
    "mrr",
    "context_sufficiency",
    "answer_correctness",
    "groundedness",
    "correct_refusal",
    "false_refusal",
    "answered_unanswerable",
    "citation_fabrication_rate",
    "latency_ms",
    "cost_usd",
)

# Bootstrap intervals are computed only for the headline metrics. The full
# sweep (every k in every slice) is ~30 metrics x 7 slices, and resampling all
# of them costs minutes on a real eval set to produce intervals nobody reads
# on precision@1. Everything else still gets a mean and an n.
CI_METRICS: frozenset[str] = frozenset(HEADLINE_METRICS)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Everything that can change a number, and nothing that cannot.

    Note what is absent: timestamps, hostnames, output paths, worker counts.
    Those go in the result file's provenance block but never into the hash,
    or two identical runs on different machines would look like different
    configurations.
    """

    dataset_path: str
    system_name: str
    system_config: dict[str, Any]
    judge_model: str
    judge_enabled: bool
    k_values: tuple[int, ...] = DEFAULT_K_VALUES
    primary_k: int = PRIMARY_K
    bootstrap_resamples: int = 10_000
    bootstrap_seed: int = 20260906
    label: str = ""

    def hashable(self) -> dict[str, Any]:
        return {
            "schema": RESULT_SCHEMA_VERSION,
            "dataset": self.dataset_path,
            "system": {"name": self.system_name, "config": self.system_config},
            "judge": {"model": self.judge_model, "enabled": self.judge_enabled},
            "k_values": list(self.k_values),
            "primary_k": self.primary_k,
            "bootstrap": {
                "resamples": self.bootstrap_resamples,
                "seed": self.bootstrap_seed,
            },
        }


def config_hash(config: RunConfig, dataset: Dataset) -> str:
    """Hash of the config plus the dataset's content.

    Hashing the dataset *content*, not just its path, is what stops a silent
    comparison against a set that has since been edited. Add ten questions
    and the hash changes, so the run gets its own file instead of being
    mistaken for a re-run of the old one.
    """
    payload = config.hashable()
    payload["dataset_digest"] = hashlib.sha256(dataset.to_jsonl().encode("utf-8")).hexdigest()
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def load_system(spec: str, **kwargs: Any) -> RagSystem:
    """Load a system under test from a `module.path:attribute` spec.

    The attribute may be a class or a factory; either is called with kwargs.
    Keeping this dynamic is what lets one harness measure the Phase 1
    baseline, every Phase 3 ablation, and a stub, without the harness ever
    importing a retriever.
    """
    if ":" not in spec:
        raise ValueError(
            f"system spec must look like 'module.path:attribute', got {spec!r}\n"
            f"e.g. 'gtrag.fixtures.system:FixtureRagSystem'"
        )
    module_path, attr = spec.split(":", 1)
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(f"could not import {module_path!r} from system spec {spec!r}") from exc
    try:
        factory = getattr(module, attr)
    except AttributeError as exc:
        raise AttributeError(f"{module_path!r} has no attribute {attr!r}") from exc
    return factory(**kwargs)


# --------------------------------------------------------------------------
# Per-question result
# --------------------------------------------------------------------------


@dataclass
class QuestionResult:
    """All scores for one question, plus the trace needed to debug it."""

    question_id: str
    question_type: str
    difficulty: str
    answerable: bool

    # Carried into the result file so per-question rows are readable on their
    # own -- calibration export and failure clustering both work off these
    # files without needing to re-join to the dataset.
    question: str = ""
    gold_answer: str | None = None

    answer: str = ""
    refused: bool = False
    retrieved_ids: list[str] = field(default_factory=list)
    error: str | None = None

    # Deterministic metrics -- always present, no judge required.
    retrieval: dict[str, float | None] = field(default_factory=dict)
    refusal: dict[str, bool | None] = field(default_factory=dict)
    citations: dict[str, Any] = field(default_factory=dict)

    # Judged metrics -- None when unscored.
    answer_correctness: float | None = None
    context_sufficiency: float | None = None
    groundedness: float | None = None
    judge_errors: list[str] = field(default_factory=list)
    judgments: list[dict[str, Any]] = field(default_factory=list)

    # Operational
    timings: dict[str, float] = field(default_factory=dict)
    usage: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def score_retrieval(
    question: EvalQuestion,
    response: SystemResponse,
    k_values: Sequence[int],
    primary_k: int,
    relevance: dict[str, int] | None = None,
) -> dict[str, float | None]:
    """Deterministic retrieval metrics for one question.

    `relevance` is the resolved `chunk_id -> relevance` map. It is passed in
    rather than read off the question because a span-labeled question has no
    fixed answer until you know the chunking -- which is exactly what makes
    the Phase 3 chunking ablation possible.
    """
    retrieved = response.retrieved_ids
    if relevance is None:
        relevance = {g.chunk_id: g.relevance for g in question.gold_chunks}
    gold = {cid for cid, rel in relevance.items() if rel >= 1}
    answer_ids = {cid for cid, rel in relevance.items() if rel == 2}

    scores: dict[str, float | None] = {}
    for k in k_values:
        scores[f"recall@{k}"] = ret.recall_at_k(retrieved, gold, k)
        scores[f"precision@{k}"] = ret.precision_at_k(retrieved, gold, k)
        scores[f"hit_rate@{k}"] = ret.hit_rate_at_k(retrieved, gold, k)
        scores[f"ndcg@{k}"] = ret.ndcg_at_k(retrieved, relevance, k)
        scores[f"answer_recall@{k}"] = ret.answer_bearing_recall_at_k(retrieved, answer_ids, k)
    scores["mrr"] = ret.mrr(retrieved, gold)
    scores["primary_ndcg"] = scores.get(f"ndcg@{primary_k}")
    scores["primary_recall"] = scores.get(f"recall@{primary_k}")
    return scores


def _judge_question(
    judge: Judge,
    question: EvalQuestion,
    response: SystemResponse,
    result: QuestionResult,
) -> None:
    """Run judged metrics, recording failures instead of swallowing them."""
    context = [c.text for c in sorted(response.retrieved, key=lambda c: c.rank) if c.text]
    judgments: list[Judgment] = []

    # Context sufficiency: retrieval-side, so it only applies where a gold
    # answer exists to be sufficient *for*.
    if question.answerable and question.gold_answer:
        j = judge.score(
            "context_sufficiency",
            question=question.question,
            answer=response.answer,
            context=context,
            gold_answer=question.gold_answer,
        )
        judgments.append(j)
        if j.ok:
            result.context_sufficiency = SCALES["context_sufficiency"].normalise(j.score)

    # Correctness: only meaningful against a reference answer. On the
    # unanswerable slice, refusal scoring already covers the right behaviour.
    if question.answerable and question.gold_answer:
        j = judge.score(
            "answer_correctness",
            question=question.question,
            answer=response.answer,
            context=context,
            gold_answer=question.gold_answer,
        )
        judgments.append(j)
        if j.ok:
            result.answer_correctness = SCALES["answer_correctness"].normalise(j.score)

    # Groundedness: applies to any substantive answer, including a wrong one
    # and including answers on the unanswerable slice -- an ungrounded answer
    # to an unanswerable question is exactly what we most want to catch.
    if response.answer.strip() and not response.refused:
        claims = judge.decompose_claims(response.answer)
        supported: list[bool] = []
        for claim in claims:
            j = judge.score(
                "claim_support",
                question=question.question,
                answer=response.answer,
                context=context,
                extra={"claim": claim},
            )
            judgments.append(j)
            if j.ok:
                supported.append(j.score == 2)
        if supported:
            result.groundedness = gen.groundedness_from_claims(supported)

    result.judgments = [j.to_dict() for j in judgments]
    result.judge_errors = [j.error for j in judgments if j.error]


def evaluate_question(
    question: EvalQuestion,
    system: RagSystem,
    judge: Judge,
    *,
    k_values: Sequence[int],
    primary_k: int,
    relevance: dict[str, int] | None = None,
) -> QuestionResult:
    """Run one question end to end and score it."""
    result = QuestionResult(
        question_id=question.id,
        question_type=question.question_type.value,
        difficulty=question.difficulty.value,
        answerable=question.answerable,
        question=question.question,
        gold_answer=question.gold_answer,
    )

    started = time.perf_counter()
    try:
        response = system.answer(
            question.question, history=list(question.history) if question.history else None
        )
    except Exception as exc:  # noqa: BLE001 - one bad question must not kill the run
        result.error = f"{type(exc).__name__}: {exc}"
        result.timings = {"total": (time.perf_counter() - started) * 1000.0}
        return result

    if response.error:
        result.error = response.error

    result.answer = response.answer
    result.refused = response.refused
    result.retrieved_ids = response.retrieved_ids
    result.timings = dict(response.timings) or {"total": (time.perf_counter() - started) * 1000.0}
    result.usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "cached_input_tokens": response.usage.cached_input_tokens,
        "cost_usd": response.usage.cost_usd,
    }

    result.retrieval = score_retrieval(question, response, k_values, primary_k, relevance)
    result.refusal = gen.score_refusal(question, response.refused).to_dict()
    result.citations = gen.citation_validity(response).to_dict()

    if not isinstance(judge, NullJudge):
        _judge_question(judge, question, response, result)

    return result


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------


@dataclass
class RunResult:
    run_id: str
    config: dict[str, Any]
    provenance: dict[str, Any]
    dataset: dict[str, Any]
    aggregates: dict[str, dict[str, Any]]
    slices: dict[str, dict[str, dict[str, Any]]]
    per_question: list[dict[str, Any]]
    cache: dict[str, Any]
    warnings: list[str]

    def metric(self, name: str) -> float | None:
        entry = self.aggregates.get(name)
        return entry["mean"] if entry else None

    def per_question_scores(self, metric: str) -> dict[str, float | None]:
        """Scores keyed by question id, for paired comparison between runs."""
        out: dict[str, float | None] = {}
        for row in self.per_question:
            out[row["question_id"]] = _extract_metric(row, metric)
        return out

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, directory: str | Path = "evals/results") -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{self.run_id}.json"
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False, sort_keys=False),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> RunResult:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


def _extract_metric(row: dict[str, Any], metric: str) -> float | None:
    """Pull one metric out of a per-question row, whichever group it lives in."""
    if metric in row.get("retrieval", {}):
        return row["retrieval"][metric]
    if metric in ("answer_correctness", "context_sufficiency", "groundedness"):
        return row.get(metric)
    if metric in row.get("refusal", {}):
        val = row["refusal"][metric]
        return None if val is None else float(val)
    if metric == "citation_fabrication_rate":
        return row.get("citations", {}).get("fabrication_rate")
    if metric == "latency_ms":
        return row.get("timings", {}).get("total")
    if metric == "cost_usd":
        return row.get("usage", {}).get("cost_usd")
    return None


def _metric_names(results: Sequence[QuestionResult]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for r in results:
        for k in r.retrieval:
            if k not in seen and not k.startswith("primary_"):
                seen.add(k)
                names.append(k)
    for k in ("answer_correctness", "context_sufficiency", "groundedness"):
        names.append(k)
    for k in ("correct_refusal", "false_refusal", "answered_unanswerable"):
        names.append(k)
    names.extend(["citation_fabrication_rate", "latency_ms", "cost_usd"])
    return names


def _aggregate_all(
    rows: Sequence[dict[str, Any]], metrics: Sequence[str], config: RunConfig
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for metric in metrics:
        values = [_extract_metric(r, metric) for r in rows]
        if metric in CI_METRICS:
            agg: Aggregate = aggregate(
                metric,
                values,
                resamples=config.bootstrap_resamples,
                seed=config.bootstrap_seed,
            )
        else:
            defined = [v for v in values if v is not None]
            agg = Aggregate(
                metric=metric,
                mean=(sum(defined) / len(defined)) if defined else None,
                ci_low=None,
                ci_high=None,
                n=len(defined),
                n_undefined=len(values) - len(defined),
            )
        out[metric] = agg.to_dict()
    return out


def run_eval(
    dataset: Dataset,
    system: RagSystem,
    *,
    judge: Judge | None = None,
    config: RunConfig | None = None,
    workers: int = 4,
    cache: ResponseCache | None = None,
    allow_unverified: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> RunResult:
    """Evaluate `system` on `dataset` and return a complete result.

    `allow_unverified` is False by default and refuses to run on a dataset
    containing machine-generated questions no human has checked. That is a
    deliberate obstacle: the whole credibility of the published number rests
    on the labels having been verified, and making it easy to skip would
    defeat the purpose of tracking verification at all.
    """
    judge = judge or NullJudge()
    cache = cache or ResponseCache(enabled=False)
    warnings: list[str] = []

    unverified = dataset.unverified()
    if unverified and not allow_unverified:
        raise ValueError(
            f"{len(unverified)} of {len(dataset)} questions are unverified "
            f"(e.g. {', '.join(q.id for q in unverified[:5])}).\n"
            f"Hand-verify them and set `verified_by`, or pass --allow-unverified "
            f"for an exploratory run whose numbers must not be published."
        )
    if unverified:
        warnings.append(
            f"{len(unverified)}/{len(dataset)} questions are unverified; "
            f"these numbers are exploratory and must not be published"
        )

    config = config or RunConfig(
        dataset_path=str(dataset.path or "<in-memory>"),
        system_name=getattr(system, "name", type(system).__name__),
        system_config=dict(getattr(system, "config", {}) or {}),
        judge_model=getattr(judge, "model", "null"),
        judge_enabled=not isinstance(judge, NullJudge),
    )
    run_id = config_hash(config, dataset)

    questions = list(dataset)

    # Resolve span-anchored labels against this system's actual chunking. A
    # system that exposes `spanned_chunks` gets its span labels resolved;
    # anything else falls back to chunk-id labels.
    chunks = getattr(system, "spanned_chunks", None)
    relevance_maps: dict[str, dict[str, int]] = {q.id: q.relevance_map(chunks) for q in questions}

    lost = [
        q.id for q in questions if q.gold_spans and chunks is not None and not relevance_maps[q.id]
    ]
    if lost:
        # Not a retrieval result: this chunking split the evidence so that no
        # chunk carries enough of it. Surfacing it as a warning keeps it from
        # being misread as a worse retriever.
        warnings.append(
            f"{len(lost)} question(s) lost their gold evidence under this chunking "
            f"(e.g. {', '.join(lost[:3])}); their retrieval metrics are undefined"
        )

    results: list[QuestionResult] = [None] * len(questions)  # type: ignore[list-item]
    started = time.time()

    def work(index_and_question: tuple[int, EvalQuestion]) -> None:
        idx, q = index_and_question
        results[idx] = evaluate_question(
            q,
            system,
            judge,
            k_values=config.k_values,
            primary_k=config.primary_k,
            relevance=relevance_maps[q.id],
        )
        if progress is not None:
            progress(sum(1 for r in results if r is not None), len(questions))

    if workers <= 1:
        for pair in enumerate(questions):
            work(pair)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(work, enumerate(questions)))

    rows = [r.to_dict() for r in results]
    metrics = _metric_names(results)
    aggregates = _aggregate_all(rows, metrics, config)

    slices: dict[str, dict[str, dict[str, Any]]] = {}
    for qtype in QuestionType:
        subset = [r for r in rows if r["question_type"] == qtype.value]
        if subset:
            slices[qtype.value] = _aggregate_all(subset, metrics, config)

    judge_error_count = sum(len(r.judge_errors) for r in results)
    if judge_error_count:
        warnings.append(
            f"{judge_error_count} judge call(s) failed; affected metrics are "
            f"unscored rather than zero -- see per-question judge_errors"
        )
    run_errors = [r for r in results if r.error]
    if run_errors:
        warnings.append(f"{len(run_errors)} question(s) errored during system.answer()")
    if isinstance(judge, NullJudge):
        warnings.append(
            "no judge configured: correctness, context sufficiency and groundedness "
            "are unscored (deterministic metrics are unaffected)"
        )

    comp = composition_report(dataset)
    return RunResult(
        run_id=run_id,
        config=config.hashable() | {"label": config.label},
        provenance={
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "duration_s": round(time.time() - started, 2),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "workers": workers,
            "harness_schema": RESULT_SCHEMA_VERSION,
        },
        dataset=comp.to_dict() | {"path": str(dataset.path or "<in-memory>")},
        aggregates=aggregates,
        slices=slices,
        per_question=rows,
        cache=cache.stats.to_dict(),
        warnings=warnings,
    )
