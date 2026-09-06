"""Core domain types shared by the retrieval pipeline and the eval harness.

Deliberately stdlib-only. The eval harness must be runnable in CI with no
third-party packages installed, so nothing here may import `anthropic`,
`pydantic`, or any vector-store client.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

# --------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Chunk:
    """An indexed unit of the corpus.

    `chunk_id` is the join key between the corpus, a system's retrieval output,
    and the gold labels in the eval set. It must be stable across re-ingestion
    of unchanged content, otherwise every label in the dataset silently rots.
    """

    chunk_id: str
    text: str
    doc_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.chunk_id:
            raise ValueError("chunk_id must be non-empty")


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A chunk returned by a retriever, with its score and 1-indexed rank."""

    chunk_id: str
    rank: int
    score: float = 0.0
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError(f"rank is 1-indexed, got {self.rank}")


# --------------------------------------------------------------------------
# System responses
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Citation:
    """Attribution of one span of the answer to supporting chunks.

    `claim_index` refers to the position of the claim (usually a sentence) in
    the answer. `chunk_ids` are the chunks the system says support it -- which
    is not the same as the chunks that actually do. Phase 4 measures the gap.
    """

    claim_index: int
    chunk_ids: tuple[str, ...]
    text: str = ""


@dataclass(frozen=True, slots=True)
class Usage:
    """Token and cost accounting for a single system call."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cost_usd: float = 0.0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
        )


@dataclass
class StageTimings:
    """Wall-clock milliseconds per pipeline stage.

    Recorded per query so p50/p95 can be reported per stage rather than only
    end to end -- without this you cannot tell whether a latency regression
    came from the reranker or the generator.
    """

    stages: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start) * 1000.0
            self.stages[stage] = self.stages.get(stage, 0.0) + elapsed

    @property
    def total_ms(self) -> float:
        return sum(self.stages.values())

    def to_dict(self) -> dict[str, float]:
        out = {k: round(v, 3) for k, v in self.stages.items()}
        out["total"] = round(self.total_ms, 3)
        return out


@dataclass(frozen=True, slots=True)
class SystemResponse:
    """What the system under test returns for one question.

    A refusal is a first-class outcome, not an error: on the `unanswerable`
    slice, refusing IS the correct answer. `answer` may be empty when
    `refused` is True.
    """

    answer: str
    retrieved: tuple[RetrievedChunk, ...] = ()
    citations: tuple[Citation, ...] = ()
    refused: bool = False
    usage: Usage = field(default_factory=Usage)
    timings: dict[str, float] = field(default_factory=dict)
    error: str | None = None
    # Per-stage trace (Phase 4). Carried into the result file so a failure
    # can be attributed to a stage -- rewriting, assembly, refusal,
    # verification -- without re-running the question by hand.
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def retrieved_ids(self) -> list[str]:
        """Chunk ids in rank order."""
        return [c.chunk_id for c in sorted(self.retrieved, key=lambda c: c.rank)]

    @property
    def cited_ids(self) -> set[str]:
        return {cid for c in self.citations for cid in c.chunk_ids}


# --------------------------------------------------------------------------
# System under test
# --------------------------------------------------------------------------


@runtime_checkable
class RagSystem(Protocol):
    """The interface the eval harness measures.

    Anything satisfying this can be evaluated -- the Phase 1 baseline, each
    Phase 3 ablation, or a stub. The harness never imports a concrete system;
    it is loaded by dotted path at runtime (see `evals.runner.load_system`),
    which is what keeps the harness reusable across every later phase.
    """

    name: str

    @property
    def config(self) -> dict[str, Any]:
        """Fully-resolved configuration. Hashed into the run id, so it must
        contain every knob that can change results and nothing that cannot
        (no timestamps, no absolute paths, no run ids)."""
        ...

    def answer(
        self, question: str, *, history: list[tuple[str, str]] | None = None
    ) -> SystemResponse: ...
