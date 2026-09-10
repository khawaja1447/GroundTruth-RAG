"""What the service does when a dependency fails.

The default behaviour of a RAG pipeline under partial failure is a 500. That
is the wrong answer almost every time: if the reranker is down, hybrid
retrieval still returns useful passages; if the generator is down, the
retrieved passages are themselves valuable. Returning *something* with an
honest quality flag beats returning nothing.

Two mechanisms:

  * `CircuitBreaker` -- stop calling a component that is failing. Without
    one, every request pays the full timeout of a dead dependency, and a
    single failed component turns into a service-wide latency collapse.
  * `Degradable` -- wrap a component with a fallback and record that the
    fallback was used, so the response can say so and the metrics can show
    how long the service has been degraded.

The flag matters as much as the fallback. A silently degraded service looks
healthy on every dashboard while quietly serving worse answers.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Generic, TypeVar

__all__ = [
    "CircuitState",
    "CircuitBreaker",
    "CircuitOpenError",
    "Degradable",
    "DegradationReport",
]

T = TypeVar("T")


class CircuitState(StrEnum):
    CLOSED = "closed"  # healthy, calls pass through
    OPEN = "open"  # failing, calls short-circuit
    HALF_OPEN = "half_open"  # probing whether it recovered


class CircuitOpenError(RuntimeError):
    """Raised when a call is short-circuited by an open breaker."""


@dataclass
class CircuitBreaker:
    """Standard three-state breaker.

    `failure_threshold` consecutive failures opens the circuit. After
    `recovery_seconds` one probe request is allowed through (half-open); if it
    succeeds the circuit closes, if it fails the timer restarts.

    Consecutive rather than windowed failures: a component that fails one
    request in ten is degraded but usable, and opening on that would take a
    working dependency offline.
    """

    name: str
    failure_threshold: int = 5
    recovery_seconds: float = 30.0
    _state: CircuitState = CircuitState.CLOSED
    _consecutive_failures: int = 0
    _opened_at: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    trips: int = 0

    def __post_init__(self) -> None:
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if self.recovery_seconds <= 0:
            raise ValueError("recovery_seconds must be positive")

    @property
    def state(self) -> CircuitState:
        with self._lock:
            return self._resolve_state()

    def _resolve_state(self) -> CircuitState:
        if (
            self._state is CircuitState.OPEN
            and time.monotonic() - self._opened_at >= self.recovery_seconds
        ):
            self._state = CircuitState.HALF_OPEN
        return self._state

    def allows(self) -> bool:
        with self._lock:
            return self._resolve_state() is not CircuitState.OPEN

    def record_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0
            self._state = CircuitState.CLOSED

    def record_failure(self) -> None:
        with self._lock:
            self._consecutive_failures += 1
            was = self._resolve_state()
            if (
                self._consecutive_failures >= self.failure_threshold
                or was is CircuitState.HALF_OPEN
            ):
                # A failed probe re-opens immediately: the component is still
                # down and there is no point waiting for the threshold again.
                if self._state is not CircuitState.OPEN:
                    self.trips += 1
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()

    def call(self, fn: Callable[[], T]) -> T:
        if not self.allows():
            raise CircuitOpenError(
                f"circuit for {self.name!r} is open after {self._consecutive_failures} "
                f"consecutive failures; retrying in "
                f"{max(0.0, self.recovery_seconds - (time.monotonic() - self._opened_at)):.1f}s"
            )
        try:
            result = fn()
        except Exception:
            self.record_failure()
            raise
        self.record_success()
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "state": str(self.state),
            "consecutive_failures": self._consecutive_failures,
            "trips": self.trips,
        }


@dataclass
class DegradationReport:
    """Which components are currently degraded, and why."""

    degraded: dict[str, str] = field(default_factory=dict)

    @property
    def is_degraded(self) -> bool:
        return bool(self.degraded)

    @property
    def quality_flag(self) -> str:
        if not self.degraded:
            return "full"
        return "degraded"

    def record(self, component: str, reason: str) -> None:
        self.degraded[component] = reason

    def to_dict(self) -> dict[str, Any]:
        return {"quality": self.quality_flag, "degraded_components": dict(self.degraded)}


@dataclass
class Degradable(Generic[T]):
    """A component plus what to do when it fails.

    `fallback` returning None means "this stage is skipped", which is the
    right answer for a reranker (serve the un-reranked order) and the wrong
    one for a retriever (there is nothing to serve). The distinction is the
    caller's to make, which is why the fallback is supplied rather than
    inferred.
    """

    name: str
    breaker: CircuitBreaker
    fallback: Callable[[], T] | None = None
    required: bool = False

    def run(self, fn: Callable[[], T], report: DegradationReport) -> T | None:
        try:
            return self.breaker.call(fn)
        except CircuitOpenError as exc:
            reason = str(exc)
        except Exception as exc:  # noqa: BLE001 - degradation is the point
            reason = f"{type(exc).__name__}: {exc}"

        report.record(self.name, reason)
        if self.required:
            # Some stages have no meaningful fallback. Failing loudly is
            # correct there -- a "degraded" answer with no retrieval is not a
            # degraded answer, it is a fabricated one.
            raise RuntimeError(f"required component {self.name!r} failed: {reason}")
        return self.fallback() if self.fallback is not None else None
