"""Retrieval metrics, implemented directly rather than imported.

Two reasons this is hand-written instead of pulled from a framework: you have
to be able to say exactly what each number computes, and the graded/binary
distinction below is the kind of thing libraries make an opaque default.

Conventions used throughout:
  * ranks are 1-indexed; `retrieved` is in rank order
  * gold relevance is graded 0/1/2; "relevant" for binary metrics means >= 1
  * a metric that is undefined for a question returns None, never 0.0

That last rule matters more than it looks. Unanswerable questions have no gold
chunks, so recall is undefined for them -- scoring them as 0.0 would drag the
mean down by however many unanswerable questions the set happens to contain,
and make the headline number a function of dataset composition rather than
retrieval quality.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import log2

__all__ = [
    "recall_at_k",
    "precision_at_k",
    "hit_rate_at_k",
    "mrr",
    "dcg",
    "ndcg_at_k",
    "answer_bearing_recall_at_k",
]


def _truncate(retrieved: Sequence[str], k: int) -> list[str]:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    return list(retrieved[:k])


def recall_at_k(retrieved: Sequence[str], gold_ids: set[str], k: int) -> float | None:
    """Fraction of relevant chunks (relevance >= 1) appearing in the top k.

    Returns None when the question has no relevant chunks, i.e. the metric is
    undefined rather than zero.
    """
    if not gold_ids:
        return None
    top = set(_truncate(retrieved, k))
    return len(top & gold_ids) / len(gold_ids)


def answer_bearing_recall_at_k(
    retrieved: Sequence[str], answer_ids: set[str], k: int
) -> float | None:
    """Recall restricted to relevance-2 chunks.

    The stricter companion to `recall_at_k`: retrieving the supporting context
    but missing the chunk that actually contains the figure still produces a
    wrong answer, and this is the metric that shows it.
    """
    if not answer_ids:
        return None
    top = set(_truncate(retrieved, k))
    return len(top & answer_ids) / len(answer_ids)


def precision_at_k(retrieved: Sequence[str], gold_ids: set[str], k: int) -> float | None:
    """Fraction of the returned top-k that is relevant.

    Denominator is the number actually returned, capped at k -- not k itself.
    A system returning 3 chunks when k=10 is measured on the 3 it returned;
    dividing by 10 would report a precision ceiling of 0.3 for a system that
    got all three right.
    """
    if not gold_ids:
        return None
    top = _truncate(retrieved, k)
    if not top:
        return 0.0
    return len(set(top) & gold_ids) / len(top)


def hit_rate_at_k(retrieved: Sequence[str], gold_ids: set[str], k: int) -> float | None:
    """1.0 if any relevant chunk is in the top k, else 0.0."""
    if not gold_ids:
        return None
    return 1.0 if set(_truncate(retrieved, k)) & gold_ids else 0.0


def mrr(retrieved: Sequence[str], gold_ids: set[str], k: int | None = None) -> float | None:
    """Reciprocal rank of the first relevant chunk; 0.0 if none is retrieved.

    Note the asymmetry with the None convention: "no relevant chunk was
    retrieved" is a real score of zero, whereas "this question has no relevant
    chunks" is undefined.
    """
    if not gold_ids:
        return None
    pool = list(retrieved) if k is None else _truncate(retrieved, k)
    for rank, chunk_id in enumerate(pool, start=1):
        if chunk_id in gold_ids:
            return 1.0 / rank
    return 0.0


def dcg(relevances: Sequence[int]) -> float:
    """Discounted cumulative gain with exponential gain.

        DCG = sum_i (2^rel_i - 1) / log2(i + 1)      [i is 1-indexed]

    Exponential gain rather than linear, so a relevance-2 chunk (contains the
    answer) is worth 3x a relevance-1 chunk (supporting context) rather than
    2x -- the right shape when one of the two actually answers the question.
    """
    return sum((2**rel - 1) / log2(i + 1) for i, rel in enumerate(relevances, start=1))


def ndcg_at_k(retrieved: Sequence[str], relevance_map: dict[str, int], k: int) -> float | None:
    """Normalised DCG at k against graded relevance.

    The ideal ranking is the gold relevances sorted descending, truncated to k.
    Retrieved chunks absent from `relevance_map` score 0.

    Returns None when there is no relevant chunk at all (IDCG would be 0 and
    the ratio undefined).
    """
    graded = {cid: rel for cid, rel in relevance_map.items() if rel > 0}
    if not graded:
        return None

    actual = dcg([graded.get(cid, 0) for cid in _truncate(retrieved, k)])
    ideal = dcg(sorted(graded.values(), reverse=True)[:k])
    if ideal == 0:
        return None
    return actual / ideal
