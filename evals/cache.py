"""Content-addressed response cache.

This is what makes `make eval` cheap and deterministic. Every model call is
keyed on a hash of everything that can change its output, so:

  * re-running an eval after a *retrieval-only* change re-uses every judge
    response, and the measured delta is attributable to the change rather
    than to sampling noise in the judge;
  * CI can run the full judged suite on a cache hit for the cost of some
    SQLite reads;
  * a published number can be regenerated months later.

The key deliberately includes the rubric version. Editing a rubric changes
what the judge measures, and silently reusing scores from the old wording is
the subtlest way to corrupt a comparison.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["ResponseCache", "CacheStats", "cache_key"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    key         TEXT PRIMARY KEY,
    namespace   TEXT NOT NULL,
    payload     TEXT NOT NULL,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_responses_namespace ON responses(namespace);
"""


def cache_key(namespace: str, payload: dict[str, Any]) -> str:
    """Stable hash of a request.

    `sort_keys=True` is load-bearing: dict ordering varies between runs for
    anything built from a set or a comprehension over unordered input, and an
    unsorted dump would produce a different key for an identical request --
    a cache that never hits and a comparison that never reproduces.
    """
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(f"{namespace}\x00{blob}".encode()).hexdigest()
    return f"{namespace}:{digest}"


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    writes: int = 0

    @property
    def lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float | None:
        return self.hits / self.lookups if self.lookups else None

    def to_dict(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "hit_rate": self.hit_rate,
        }


class ResponseCache:
    """Thread-safe SQLite cache for model responses.

    The runner calls judges from a thread pool, so writes are serialised
    behind a lock and the connection is opened with `check_same_thread=False`.
    """

    def __init__(self, path: str | Path | None = None, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self.stats = CacheStats()
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None

        if not enabled:
            self.path = None
            return

        self.path = Path(path) if path else Path("evals/.cache/responses.sqlite")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        # WAL keeps concurrent readers from blocking the writer; the runner is
        # read-heavy across threads.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def get(self, key: str) -> dict[str, Any] | None:
        if not self.enabled or self._conn is None:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM responses WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                self.stats.misses += 1
                return None
            self.stats.hits += 1
        return json.loads(row[0])

    def put(self, key: str, namespace: str, value: dict[str, Any]) -> None:
        if not self.enabled or self._conn is None:
            return
        blob = json.dumps(value, ensure_ascii=False)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO responses (key, namespace, payload, created_at) "
                "VALUES (?, ?, ?, ?)",
                (key, namespace, blob, time.time()),
            )
            self._conn.commit()
            self.stats.writes += 1

    def purge(self, namespace: str | None = None) -> int:
        """Delete cached responses; returns the number removed.

        Use after editing a rubric if you did not bump its version -- though
        bumping the version is the better habit, since it keeps old scores
        reproducible instead of destroying them.
        """
        if not self.enabled or self._conn is None:
            return 0
        with self._lock:
            if namespace is None:
                cur = self._conn.execute("DELETE FROM responses")
            else:
                cur = self._conn.execute("DELETE FROM responses WHERE namespace = ?", (namespace,))
            self._conn.commit()
            return cur.rowcount

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def __enter__(self) -> ResponseCache:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
