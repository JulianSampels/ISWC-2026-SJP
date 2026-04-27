"""Entity-centric metrics for relation-tail instance completion."""

from __future__ import annotations

import math
from typing import Sequence, Set, Tuple


Pair = Tuple[int, int]


def entity_hit_at_k(ranked: Sequence[Pair], gold: Set[Pair], k: int) -> float:
    """Return 1 if any gold pair appears in the top-k ranked pairs."""
    if not gold or k <= 0:
        return 0.0
    return 1.0 if any(pair in gold for pair in ranked[: int(k)]) else 0.0


def entity_recall_at_k(ranked: Sequence[Pair], gold: Set[Pair], k: int) -> float:
    """Fraction of this head's gold pairs recovered in top-k."""
    if not gold or k <= 0:
        return 0.0
    return len(set(ranked[: int(k)]) & gold) / float(len(gold))


def budget_to_first_hit(ranked: Sequence[Pair], gold: Set[Pair]) -> int | None:
    """One-based rank of the first hit, or None when no hit is present."""
    if not gold:
        return None
    for index, pair in enumerate(ranked, start=1):
        if pair in gold:
            return index
    return None


def mean_reciprocal_rank(ranked: Sequence[Pair], gold: Set[Pair]) -> float:
    """Mean reciprocal rank over all gold pairs for one head.

    Missing gold pairs contribute zero, which keeps the metric aligned with the
    candidate set recall ceiling.
    """
    if not gold:
        return 0.0
    rank_by_pair = {}
    for index, pair in enumerate(ranked, start=1):
        rank_by_pair.setdefault(pair, index)
    return sum(1.0 / rank_by_pair[pair] for pair in gold if pair in rank_by_pair) / float(len(gold))


def average_precision(ranked: Sequence[Pair], gold: Set[Pair]) -> float:
    """Average precision for one head over the full ranked list."""
    if not gold:
        return 0.0
    seen: Set[Pair] = set()
    hits = 0
    precision_sum = 0.0
    for index, pair in enumerate(ranked, start=1):
        if pair in seen:
            continue
        seen.add(pair)
        if pair in gold:
            hits += 1
            precision_sum += hits / float(index)
    return precision_sum / float(len(gold))


def ndcg_at_k(ranked: Sequence[Pair], gold: Set[Pair], k: int) -> float:
    """Binary relevance nDCG@k for one head."""
    if not gold or k <= 0:
        return 0.0
    cutoff = int(k)
    dcg = 0.0
    seen: Set[Pair] = set()
    for index, pair in enumerate(ranked[:cutoff], start=1):
        if pair in seen:
            continue
        seen.add(pair)
        if pair in gold:
            dcg += 1.0 / math.log2(index + 1)

    ideal_len = min(len(gold), cutoff)
    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_len + 1))
    return dcg / idcg if idcg > 0.0 else 0.0


def ndcg(ranked: Sequence[Pair], gold: Set[Pair]) -> float:
    """Binary relevance nDCG over the full ranked list."""
    return ndcg_at_k(ranked, gold, len(ranked))
