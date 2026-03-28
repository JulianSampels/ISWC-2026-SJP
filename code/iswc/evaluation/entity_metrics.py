"""
Entity-Level Evaluation Metrics (RQ3)
======================================
Novel metrics proposed for entity-centric fact suggestion evaluation.

Conventional tuple-level metrics (MRR, Hits@K) evaluate ranking quality of
individual isolated facts. For entity-centric fact suggestion, the practical
unit of interaction is an anchor entity with a ranked suggestion list.

We define three complementary entity-level metrics:

  EntityHit@K(h)       = 1 if ≥ 1 gold fact appears in top-K suggestions for h
  EntityRecall@K(h)    = |TopK(h) ∩ G(h)| / |G(h)|
  Budget-to-First-Hit(h) = min{k : TopK(h) ∩ G(h) ≠ ∅}

These capture practical utility for curation workflows:
  - EntityHit@K: "Does the user see at least one correct fact?"
  - EntityRecall@K: "What fraction of the entity's missing facts are surfaced?"
  - B2FH: "How many candidates must the user inspect before finding the first correct one?"

Reference: TODO.org §RQ3 and paper_outline §6.3.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MetricResults:
    """Container for evaluation results."""
    # Tuple-level metrics
    mrr:        float = 0.0
    hits_at_1:  float = 0.0
    hits_at_3:  float = 0.0
    hits_at_10: float = 0.0
    recall_at_k: Dict[int, float] = field(default_factory=dict)

    # Entity-level metrics
    entity_hit_at_k:      Dict[int, float] = field(default_factory=dict)
    entity_recall_at_k:   Dict[int, float] = field(default_factory=dict)
    budget_to_first_hit:  float = 0.0
    budget_to_first_hit_coverage: float = 0.0  # fraction of entities where ≥1 hit exists

    # Candidate quality metrics
    avg_candidate_size: float = 0.0
    candidate_coverage: float = 0.0


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def entity_hit_at_k(
    top_k_facts: List[Tuple[int, int]],   # predicted top-K (r, t) pairs
    gold_facts:  Set[Tuple[int, int]],     # ground-truth (r, t) pairs for this entity
    k: int,
) -> float:
    """
    EntityHit@K for a single anchor entity.

    Returns 1.0 if any of the top-K predictions is a gold fact, else 0.0.
    """
    if not gold_facts:
        return 0.0
    top = set(top_k_facts[:k])
    return 1.0 if top & gold_facts else 0.0


def entity_recall_at_k(
    top_k_facts: List[Tuple[int, int]],
    gold_facts:  Set[Tuple[int, int]],
    k: int,
) -> float:
    """
    EntityRecall@K for a single anchor entity.

    Returns |TopK(h) ∩ G(h)| / |G(h)|.
    """
    if not gold_facts:
        return 0.0
    top = set(top_k_facts[:k])
    return len(top & gold_facts) / len(gold_facts)


def budget_to_first_hit(
    ranked_facts: List[Tuple[int, int]],  # all ranked (r, t) predictions
    gold_facts:   Set[Tuple[int, int]],
) -> Optional[int]:
    """
    Budget-to-First-Hit for a single anchor entity.

    Returns the 1-indexed rank of the first correct prediction.
    Returns None if no correct fact is found in the entire ranked list.
    """
    for k, (r, t) in enumerate(ranked_facts, start=1):
        if (r, t) in gold_facts:
            return k
    return None


def mean_reciprocal_rank(
    ranked_facts: List[Tuple[int, int]],
    gold_facts:   Set[Tuple[int, int]],
) -> float:
    """
    Tuple-level MRR: for each gold fact, find its rank and compute reciprocal.
    Average over all gold facts.
    """
    if not gold_facts:
        return 0.0
    rr_sum = 0.0
    fact_rank: Dict[Tuple[int, int], int] = {f: i + 1 for i, f in enumerate(ranked_facts)}
    for gf in gold_facts:
        rank = fact_rank.get(gf)
        if rank is not None:
            rr_sum += 1.0 / rank
    return rr_sum / len(gold_facts)


def hits_at_k_tuple(
    ranked_facts: List[Tuple[int, int]],
    gold_facts:   Set[Tuple[int, int]],
    k: int,
) -> float:
    """
    Tuple-level Hits@K: fraction of gold facts that appear in the top-K predictions.
    """
    if not gold_facts:
        return 0.0
    top_k_set = set(ranked_facts[:k])
    return len(top_k_set & gold_facts) / len(gold_facts)


def recall_at_k_tuple(
    ranked_facts: List[Tuple[int, int]],
    gold_facts:   Set[Tuple[int, int]],
    k: int,
) -> float:
    """
    Tuple-level Recall@K: same as hits_at_k_tuple — fraction of gold facts
    recovered within the top-K predictions.
    """
    return hits_at_k_tuple(ranked_facts, gold_facts, k)


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_entity_centric(
    predictions: Dict[int, List[Tuple[int, int]]],
    gold_triples: Tensor,
    k_values:    List[int] = (1, 3, 5, 10, 20, 50, 100),
    test_heads:  Optional[List[int]] = None,
) -> MetricResults:
    """
    Evaluate a system's fact suggestions against ground-truth test triples
    using both tuple-level and entity-level metrics.

    Args:
        predictions: Dict[head_id -> list of (r, t) pairs in ranked order].
        gold_triples: (N, 3) tensor of test ground-truth (h, r, t) triples.
        k_values:    List of K values to compute metrics at.
        test_heads:  If given, restrict evaluation to these entities.

    Returns:
        MetricResults with all computed metrics.
    """
    gold_np = gold_triples.numpy() if isinstance(gold_triples, Tensor) else np.array(gold_triples)

    # Build gold facts per head
    gold_per_head: Dict[int, Set[Tuple[int, int]]] = defaultdict(set)
    for h, r, t in gold_np:
        gold_per_head[int(h)].add((int(r), int(t)))

    if test_heads is None:
        test_heads = list(gold_per_head.keys())

    # ---- Accumulators ----
    mrr_vals:    List[float] = []
    h1_vals:     List[float] = []
    h3_vals:     List[float] = []
    h10_vals:    List[float] = []

    e_hit:       Dict[int, List[float]] = {k: [] for k in k_values}
    e_recall:    Dict[int, List[float]] = {k: [] for k in k_values}
    r_at_k:      Dict[int, List[float]] = {k: [] for k in k_values}

    b2fh_vals:   List[float] = []
    cand_sizes:  List[int]   = []
    covered:     int         = 0
    total_gold:  int         = 0

    for h in test_heads:
        gold = gold_per_head.get(h, set())
        if not gold:
            continue

        ranked = predictions.get(h, [])  # list of (r, t) in ranked order
        ranked_set = set(ranked)

        # --- Candidate quality ---
        cand_sizes.append(len(ranked))
        covered    += len(gold & ranked_set)
        total_gold += len(gold)

        # --- Tuple-level metrics ---
        mrr_vals.append(mean_reciprocal_rank(ranked, gold))
        h1_vals.append(hits_at_k_tuple(ranked, gold, 1))
        h3_vals.append(hits_at_k_tuple(ranked, gold, 3))
        h10_vals.append(hits_at_k_tuple(ranked, gold, 10))

        # --- Entity-level metrics at each K ---
        for k in k_values:
            e_hit[k].append(entity_hit_at_k(ranked, gold, k))
            e_recall[k].append(entity_recall_at_k(ranked, gold, k))
            r_at_k[k].append(recall_at_k_tuple(ranked, gold, k))

        # --- Budget-to-First-Hit ---
        b2fh = budget_to_first_hit(ranked, gold)
        if b2fh is not None:
            b2fh_vals.append(float(b2fh))

    n = len(test_heads) if test_heads else 1

    def _mean(lst: List[float]) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    results = MetricResults(
        mrr        = _mean(mrr_vals),
        hits_at_1  = _mean(h1_vals),
        hits_at_3  = _mean(h3_vals),
        hits_at_10 = _mean(h10_vals),
        recall_at_k = {k: _mean(r_at_k[k]) for k in k_values},
        entity_hit_at_k    = {k: _mean(e_hit[k]) for k in k_values},
        entity_recall_at_k = {k: _mean(e_recall[k]) for k in k_values},
        budget_to_first_hit          = _mean(b2fh_vals),
        budget_to_first_hit_coverage = len(b2fh_vals) / len(test_heads) if test_heads else 0.0,
        avg_candidate_size = _mean([float(s) for s in cand_sizes]),
        candidate_coverage = covered / total_gold if total_gold > 0 else 0.0,
    )

    logger.info(
        f"Evaluated {len(test_heads)} entities. "
        f"MRR={results.mrr:.4f}, Hits@10={results.hits_at_10:.4f}, "
        f"EntityHit@10={results.entity_hit_at_k.get(10, 0):.4f}, "
        f"EntityRecall@10={results.entity_recall_at_k.get(10, 0):.4f}, "
        f"B2FH={results.budget_to_first_hit:.1f}, "
        f"Coverage={results.candidate_coverage:.4f}."
    )
    return results


# ---------------------------------------------------------------------------
# Fixed-budget comparison table
# ---------------------------------------------------------------------------

def evaluate_at_fixed_budgets(
    predictions:  Dict[int, List[Tuple[int, int]]],
    gold_triples: Tensor,
    budgets:      List[int] = (50, 100, 200, 500),
    test_heads:   Optional[List[int]] = None,
) -> Dict[int, MetricResults]:
    """
    Evaluate a method at multiple fixed candidate budgets.

    For each budget b, truncate each entity's prediction list to top-b
    and compute all metrics. This enables fair fixed-budget comparisons
    between methods (RQ2 fixed-budget analysis in §6.7).

    Args:
        predictions:  Full ranked predictions (will be truncated per budget).
        gold_triples: Ground-truth test triples.
        budgets:      List of candidate budgets to evaluate at.
        test_heads:   Entities to evaluate on.

    Returns:
        Dict[budget -> MetricResults]
    """
    results: Dict[int, MetricResults] = {}

    for b in budgets:
        truncated = {
            h: preds[:b] for h, preds in predictions.items()
        }
        results[b] = evaluate_entity_centric(
            truncated, gold_triples,
            k_values=[1, 5, 10, b],
            test_heads=test_heads,
        )
        logger.info(f"Budget {b}: Coverage={results[b].candidate_coverage:.4f}, MRR={results[b].mrr:.4f}")

    return results


# ---------------------------------------------------------------------------
# Formatting utilities
# ---------------------------------------------------------------------------

def format_results_table(
    method_results: Dict[str, MetricResults],
    k: int = 10,
) -> str:
    """
    Format a comparison table for multiple methods.

    Args:
        method_results: Dict[method_name -> MetricResults]
        k: K value to use for @K metrics.

    Returns:
        Markdown-formatted table string.
    """
    header = (
        f"| Method | Cand.Size | Coverage | MRR | Hits@{k} | "
        f"EntityHit@{k} | EntityRecall@{k} | B2FH |\n"
        f"|--------|-----------|----------|-----|---------|"
        f"------------|----------------|------|\n"
    )
    rows = []
    for name, res in method_results.items():
        row = (
            f"| {name:<30} | "
            f"{res.avg_candidate_size:>9.1f} | "
            f"{res.candidate_coverage:>8.4f} | "
            f"{res.mrr:>5.4f} | "
            f"{res.hits_at_10:>7.4f} | "
            f"{res.entity_hit_at_k.get(k, 0):>12.4f} | "
            f"{res.entity_recall_at_k.get(k, 0):>14.4f} | "
            f"{res.budget_to_first_hit:>4.1f} |"
        )
        rows.append(row)
    return header + "\n".join(rows)


def format_fixed_budget_table(
    method_budget_results: Dict[str, Dict[int, MetricResults]],
    budgets: List[int] = (50, 100, 200, 500),
) -> str:
    """
    Format a fixed-budget comparison table.

    Produces a table showing MRR and EntityHit@K at each budget.
    """
    lines = [f"| Method | " + " | ".join(f"B={b} MRR | B={b} EHit@{b}" for b in budgets) + " |"]
    lines.append("|" + "---|" * (1 + 2 * len(budgets)))

    for method, budget_res in method_budget_results.items():
        cols = [f"| {method:<30}"]
        for b in budgets:
            res = budget_res.get(b)
            if res:
                cols.append(f" {res.mrr:.4f} | {res.entity_hit_at_k.get(b, 0):.4f}")
            else:
                cols.append(" --- | ---")
        lines.append(" | ".join(cols) + " |")

    return "\n".join(lines)
