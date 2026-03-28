"""
Relation-First Adaptation Baseline (RQ1)
=========================================
This module implements the Relation-First adapted baseline for RQ1:

  "Can conventional relation-conditioned KGC be naturally adapted to solve
   entity-centric fact suggestion?"

Algorithm:
  Step 1: For each anchor entity h, rank all relations by P(r | h).
          Two options:
            A. Frequency-based prior (simple): rank by training frequency of r in h's neighbourhood.
            B. Learned relation predictor:     use the Phase-1 relation scores from SJP.
  Step 2: Select top-k_r candidate relations.
  Step 3: For each selected relation r, run standard tail prediction P(t | h, r).
  Step 4: Keep top-k_t tails per relation.
  Step 5: Construct fact candidates {(h, r, t) | r ∈ R_{k_r}(h), t ∈ T_{k_t}(h,r)}.
  Step 6: Score each candidate: s(h,r,t) = P(r|h) · P(t|h,r), then rank.

Reference: TODO.org, "Relation-First Adaptation" section.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Option A: Frequency-based relation prior
# ---------------------------------------------------------------------------

def build_relation_frequency_prior(
    train_triples: Tensor,
    num_relations: int,
) -> Dict[int, Dict[int, float]]:
    """
    Compute P(r | h) from training-set co-occurrence frequencies.

    For each head entity h, compute:
        count(r | h) = number of training triples (h, r, *) for each relation r
        P(r | h) ∝ count(r | h)

    Returns:
        entity_rel_prob: Dict[entity_id -> Dict[relation_id -> probability]]
    """
    triples_np = train_triples.numpy() if isinstance(train_triples, Tensor) else np.array(train_triples)

    entity_rel_count: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for h, r, t in triples_np:
        entity_rel_count[int(h)][int(r)] += 1

    entity_rel_prob: Dict[int, Dict[int, float]] = {}
    for h, rel_counts in entity_rel_count.items():
        total = sum(rel_counts.values())
        entity_rel_prob[h] = {r: c / total for r, c in rel_counts.items()}

    return entity_rel_prob


def build_tail_co_occurrence_index(
    train_triples: Tensor,
) -> Dict[Tuple[int, int], Dict[int, float]]:
    """
    Build P(t | h, r) from training co-occurrence.

    Returns:
        tail_prob: Dict[(head_id, rel_id) -> Dict[tail_id -> probability]]
    """
    triples_np = train_triples.numpy() if isinstance(train_triples, Tensor) else np.array(train_triples)
    hr_tail_count: Dict[Tuple[int, int], Dict[int, int]] = defaultdict(lambda: defaultdict(int))

    for h, r, t in triples_np:
        hr_tail_count[(int(h), int(r))][int(t)] += 1

    tail_prob: Dict[Tuple[int, int], Dict[int, float]] = {}
    for (h, r), counts in hr_tail_count.items():
        total = sum(counts.values())
        tail_prob[(h, r)] = {t: c / total for t, c in counts.items()}

    return tail_prob


def build_global_tail_prob_for_relation(
    train_triples: Tensor,
) -> Dict[int, Dict[int, float]]:
    """
    Build P(t | r) globally (marginalised over h).
    Used as fallback for entities with no (h, r) training evidence.

    Returns:
        rel_tail_prob: Dict[relation_id -> Dict[tail_id -> probability]]
    """
    triples_np = train_triples.numpy() if isinstance(train_triples, Tensor) else np.array(train_triples)
    r_tail_count: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

    for h, r, t in triples_np:
        r_tail_count[int(r)][int(t)] += 1

    rel_tail_prob: Dict[int, Dict[int, float]] = {}
    for r, counts in r_tail_count.items():
        total = sum(counts.values())
        rel_tail_prob[r] = {t: c / total for t, c in counts.items()}

    return rel_tail_prob


# ---------------------------------------------------------------------------
# Relation-First Baseline
# ---------------------------------------------------------------------------

class RelationFirstBaseline:
    """
    Relation-First adaptation of conventional KGC for entity-centric fact suggestion.

    Uses the frequency-based relation prior (Option A) and the training co-occurrence
    index for tail prediction. For entities with no training evidence, falls back to
    the global tail distribution for that relation.
    """

    def __init__(
        self,
        train_triples: Tensor,
        num_relations: int,
        k_r: int = 10,
        k_t: int = 50,
        alpha: float = 0.5,
    ):
        """
        Args:
            train_triples: (N, 3) training triple tensor.
            num_relations: Total number of relation types.
            k_r:   Number of top relations to consider per head entity.
            k_t:   Number of top tails to consider per (head, relation) pair.
            alpha: Weight for relation score in final scoring:
                   s(h,r,t) = P(r|h)^alpha * P(t|h,r)^(1-alpha)
        """
        self.k_r   = k_r
        self.k_t   = k_t
        self.alpha = alpha

        logger.info("Building relation frequency prior…")
        self.rel_prior   = build_relation_frequency_prior(train_triples, num_relations)
        logger.info("Building tail co-occurrence index…")
        self.tail_prob   = build_tail_co_occurrence_index(train_triples)
        self.global_tail = build_global_tail_prob_for_relation(train_triples)

        # Precompute global relation frequencies as fallback
        triples_np = train_triples.numpy() if isinstance(train_triples, Tensor) else np.array(train_triples)
        r_count = defaultdict(int)
        for h, r, t in triples_np:
            r_count[int(r)] += 1
        total = sum(r_count.values())
        self.global_rel_prior = {r: c / total for r, c in r_count.items()}

    def generate_candidates(
        self,
        head: int,
        max_candidates: Optional[int] = None,
    ) -> List[Tuple[int, int, int, float]]:
        """
        Generate (head, relation, tail, score) candidates for head entity h.

        Returns:
            Sorted list of (head, r, t, score) in descending score order.
        """
        # Step 1-2: Select top-k_r relations for h
        rel_scores = self.rel_prior.get(head, self.global_rel_prior)
        top_rels = sorted(rel_scores.items(), key=lambda x: -x[1])[:self.k_r]

        candidates: List[Tuple[int, int, int, float]] = []

        for r, p_r in top_rels:
            # Step 3-4: Get top-k_t tails for (h, r)
            tail_dist = self.tail_prob.get((head, r), self.global_tail.get(r, {}))
            top_tails = sorted(tail_dist.items(), key=lambda x: -x[1])[:self.k_t]

            for t, p_t in top_tails:
                # Step 6: Combined score
                score = (p_r ** self.alpha) * (p_t ** (1.0 - self.alpha))
                candidates.append((head, r, t, score))

        # Rank by combined score
        candidates.sort(key=lambda x: -x[3])

        if max_candidates is not None:
            candidates = candidates[:max_candidates]

        return candidates

    def generate_candidates_batch(
        self,
        heads: List[int],
        max_candidates: Optional[int] = None,
    ) -> Dict[int, List[Tuple[int, int, int, float]]]:
        """Generate candidates for a batch of head entities."""
        return {h: self.generate_candidates(h, max_candidates) for h in heads}

    def set_learned_relation_scores(
        self,
        entity_to_rel_scores: Dict[int, Dict[int, float]],
    ) -> None:
        """
        Override frequency-based relation prior with learned scores from
        Phase-1 of the Split-Join-Predict framework.

        This enables Option B (learned relation predictor) for the baseline.

        Args:
            entity_to_rel_scores: Dict[entity_id -> Dict[relation_id -> score]]
        """
        self.rel_prior = entity_to_rel_scores
        logger.info(
            f"Relation-First baseline: switched to learned relation scores "
            f"for {len(entity_to_rel_scores)} entities."
        )
