"""
Independent Combination Baseline (RQ1)
=======================================
This module implements the Independent Combination baseline for RQ1.

This is the most important baseline for RQ1: it tests whether the structured
Split-Join-Predict candidate generation is doing more than simply combining
two independently predicted lists.

Algorithm:
  Step 1: For h, predict top-k_r relation scores P(r | h) independently.
  Step 2: For h, predict top-k_t tail entity scores P(t | h) independently.
  Step 3: Select top-k_r relations and top-k_t tails.
  Step 4: Form all pairwise combinations: C(h) = R_{k_r}(h) × T_{k_t}(h).
  Step 5: Score each (h, r, t) via log-additive fusion:
          s(h,r,t) = α log P(r|h) + β log P(t|h)
  Step 6: Rank all candidates by s(h,r,t).

Formally: P(r,t | h) ≈ P(r | h) · P(t | h)  [independence assumption]

Reference: TODO.org, "Independent Combination Baseline" section.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class IndependentCombinationBaseline:
    """
    Independent Combination baseline for entity-centric fact suggestion.

    Assumes relation and tail predictions are independent given h:
        P(r, t | h) ≈ P(r | h) · P(t | h)

    This is the cleanest baseline for testing whether the SJP framework's
    structured joint scoring (Eq. 5.4) provides value over simple factorisation.

    If structured SJP significantly outperforms this baseline, it confirms
    that the joint scoring and global normalisation are essential.
    """

    def __init__(
        self,
        train_triples: Tensor,
        k_r: int = 10,
        k_t: int = 100,
        alpha: float = 0.5,
        beta: float = 0.5,
    ):
        """
        Args:
            train_triples: (N, 3) training triple tensor.
            k_r:   Number of top relations to select per entity.
            k_t:   Number of top tail entities to select per entity.
            alpha: Log-weight for relation score in fusion.
            beta:  Log-weight for tail entity score in fusion.
                   Note: alpha + beta need not sum to 1.
        """
        self.k_r   = k_r
        self.k_t   = k_t
        self.alpha = alpha
        self.beta  = beta

        triples_np = train_triples.numpy() if isinstance(train_triples, Tensor) else np.array(train_triples)

        # Build P(r | h) from training frequency
        hr_count:  Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        ht_count:  Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        r_global:  Dict[int, int]            = defaultdict(int)
        t_global:  Dict[int, int]            = defaultdict(int)

        for h, r, t in triples_np:
            h, r, t = int(h), int(r), int(t)
            hr_count[h][r] += 1
            ht_count[h][t] += 1
            r_global[r]    += 1
            t_global[t]    += 1

        # Normalise to probabilities
        self.rel_prob: Dict[int, Dict[int, float]] = {}
        for h, counts in hr_count.items():
            total = sum(counts.values())
            self.rel_prob[h] = {r: c / total for r, c in counts.items()}

        self.tail_prob: Dict[int, Dict[int, float]] = {}
        for h, counts in ht_count.items():
            total = sum(counts.values())
            self.tail_prob[h] = {t: c / total for t, c in counts.items()}

        # Global fallback distributions
        r_total = sum(r_global.values())
        t_total = sum(t_global.values())
        self.global_rel_prob  = {r: c / r_total for r, c in r_global.items()}
        self.global_tail_prob = {t: c / t_total for t, c in t_global.items()}

        logger.info(
            f"IndependentCombination: {len(self.rel_prob)} entities with relation priors, "
            f"{len(self.tail_prob)} entities with tail priors."
        )

    def generate_candidates(
        self,
        head: int,
        max_candidates: Optional[int] = None,
    ) -> List[Tuple[int, int, int, float]]:
        """
        Generate (head, relation, tail, score) candidates for entity h.

        Score: s(h,r,t) = α·log P(r|h) + β·log P(t|h)

        Returns:
            Sorted list of (head, r, t, score) in descending score order.
        """
        # Step 1-3: Get top-k_r relations and top-k_t tails
        rel_dist  = self.rel_prob.get(head,  self.global_rel_prob)
        tail_dist = self.tail_prob.get(head, self.global_tail_prob)

        top_rels  = sorted(rel_dist.items(),  key=lambda x: -x[1])[:self.k_r]
        top_tails = sorted(tail_dist.items(), key=lambda x: -x[1])[:self.k_t]

        eps = 1e-12  # Numerical stability

        candidates: List[Tuple[int, int, int, float]] = []

        # Step 4-5: Cartesian product + log-additive scoring
        for r, p_r in top_rels:
            log_r = math.log(p_r + eps)
            for t, p_t in top_tails:
                log_t = math.log(p_t + eps)
                score = self.alpha * log_r + self.beta * log_t
                candidates.append((head, r, t, score))

        # Step 6: Rank
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

    def set_learned_scores(
        self,
        rel_scores:  Dict[int, Dict[int, float]],
        tail_scores: Dict[int, Dict[int, float]],
    ) -> None:
        """
        Replace frequency-based distributions with learned scores from
        Phase-1 of the Split-Join-Predict framework.

        This enables a fair comparison: both SJP Phase-2 and this baseline
        start from the same Phase-1 predictions, but SJP uses joint scoring
        while this baseline combines them independently.

        Args:
            rel_scores:  Dict[entity_id -> Dict[relation_id -> score]]
            tail_scores: Dict[entity_id -> Dict[entity_id -> score]]
        """
        self.rel_prob  = rel_scores
        self.tail_prob = tail_scores
        logger.info(
            f"IndependentCombination: updated to learned scores for "
            f"{len(rel_scores)} entities (relations) and "
            f"{len(tail_scores)} entities (tails)."
        )


# ---------------------------------------------------------------------------
# Tail-First Adaptation Baseline
# ---------------------------------------------------------------------------

class TailFirstBaseline:
    """
    Tail-First adaptation baseline (symmetric to Relation-First).

    Algorithm:
      Step 1: Predict top-k_t likely tail entities for anchor h.
      Step 2: For each selected tail t, predict plausible relations P(r | h, t).
      Step 3: Construct candidates (h, r, t).
      Step 4: Score s(h,r,t) = P(t|h) · P(r|h,t).

    Reference: TODO.org, "Tail-First Adaptation" section.
    """

    def __init__(
        self,
        train_triples: Tensor,
        k_t: int = 50,
        k_r: int = 5,
        alpha: float = 0.5,
    ):
        """
        Args:
            k_t:   Number of top tail entities per head entity.
            k_r:   Number of top relations per (head, tail) pair.
            alpha: Weight for tail score in combined scoring.
        """
        self.k_t   = k_t
        self.k_r   = k_r
        self.alpha = alpha

        triples_np = train_triples.numpy() if isinstance(train_triples, Tensor) else np.array(train_triples)

        # P(t | h) from training frequency
        ht_count: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        # P(r | h, t): relation frequency for known (h, t) pairs
        htr_count: Dict[Tuple[int, int], Dict[int, int]] = defaultdict(lambda: defaultdict(int))

        for h, r, t in triples_np:
            h, r, t = int(h), int(r), int(t)
            ht_count[h][t]         += 1
            htr_count[(h, t)][r]   += 1

        self.tail_prob: Dict[int, Dict[int, float]] = {}
        for h, counts in ht_count.items():
            total = sum(counts.values())
            self.tail_prob[h] = {t: c / total for t, c in counts.items()}

        self.rel_given_ht: Dict[Tuple[int, int], Dict[int, float]] = {}
        for (h, t), counts in htr_count.items():
            total = sum(counts.values())
            self.rel_given_ht[(h, t)] = {r: c / total for r, c in counts.items()}

        # Global fallbacks
        t_global = defaultdict(int)
        r_global = defaultdict(int)
        for h, r, t in triples_np:
            t_global[int(t)] += 1
            r_global[int(r)] += 1
        t_total = sum(t_global.values())
        r_total = sum(r_global.values())
        self.global_tail_prob = {t: c / t_total for t, c in t_global.items()}
        self.global_rel_prob  = {r: c / r_total for r, c in r_global.items()}

    def generate_candidates(
        self,
        head: int,
        max_candidates: Optional[int] = None,
    ) -> List[Tuple[int, int, int, float]]:
        """
        Generate (head, relation, tail, score) candidates.
        Score: s(h,r,t) = P(t|h)^alpha · P(r|h,t)^(1-alpha)
        """
        tail_dist = self.tail_prob.get(head, self.global_tail_prob)
        top_tails = sorted(tail_dist.items(), key=lambda x: -x[1])[:self.k_t]

        candidates: List[Tuple[int, int, int, float]] = []

        for t, p_t in top_tails:
            rel_dist = self.rel_given_ht.get((head, t), self.global_rel_prob)
            top_rels = sorted(rel_dist.items(), key=lambda x: -x[1])[:self.k_r]

            for r, p_r in top_rels:
                score = (p_t ** self.alpha) * (p_r ** (1.0 - self.alpha))
                candidates.append((head, r, t, score))

        candidates.sort(key=lambda x: -x[3])

        if max_candidates is not None:
            candidates = candidates[:max_candidates]

        return candidates

    def generate_candidates_batch(
        self,
        heads: List[int],
        max_candidates: Optional[int] = None,
    ) -> Dict[int, List[Tuple[int, int, int, float]]]:
        return {h: self.generate_candidates(h, max_candidates) for h in heads}
