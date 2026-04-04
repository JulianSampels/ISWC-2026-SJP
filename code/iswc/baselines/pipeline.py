"""
Instance Completion Pipeline
==============================

Combines any relation predictor with any KGC tail-prediction model to
produce (head, relation, tail, score) candidate triples for the instance
completion task.

Two-stage pipeline
------------------
Stage 1 — Relation prediction (k_r relations per entity):
    Controlled by `k_r`.  Any BaseRelationPredictor works here
    (BPRRelationPredictor, RecoinRelationPredictor, …).

Stage 2 — Tail prediction for each (h, r_i, ?) (k_t tails per query):
    Controlled by `k_t`.  Any trained BaseKGCModel works here
    (TransEModel, TransHModel, TransRModel, DistMultModel, RotatEModel, …).

Scoring
-------
Raw scores from different KGC models are not comparable across models,
so within each (h, r) bucket the KGC tail scores are softmax-normalised
to probabilities before combination with the relation score:

    score(h, r, t) = alpha * rel_score(h, r) + (1 - alpha) * kgc_tail_prob(h, r, t)

`alpha = 0.0` means ignore relation score and rank purely by KGC;
`alpha = 1.0` means the relation score dominates.

Usage
-----
::

    from iswc.baselines import (
        BPRRelationPredictor, RecoinRelationPredictor,
        build_kgc_model, KGCTrainer,
        InstanceCompletionPipeline,
    )

    # 1. Train KGC model
    kgc = build_kgc_model("rotate", num_entities, num_relations, embed_dim=128)
    KGCTrainer(kgc, train_triples, num_entities, device=device).train(200)

    # 2. Train relation predictor
    rel = BPRRelationPredictor(train_triples, num_entities, num_relations)
    rel.train(100)

    # 3. Build pipeline and generate candidates
    pipe = InstanceCompletionPipeline(rel, kgc, k_r=10, k_t=50, device=device)
    candidates = pipe.generate_candidates_batch(test_heads, max_candidates=500)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .bpr_recoin import BaseRelationPredictor
from .kgc_tails import PyKEENWrapper

logger = logging.getLogger(__name__)


class InstanceCompletionPipeline:
    """
    Two-stage candidate generator for entity-centric fact suggestion.

    Parameters
    ----------
    relation_predictor:
        Any BaseRelationPredictor (BPR, Recoin, …).
    kgc_model:
        A trained PyKEENWrapper (from build_kgc_model or PyKEENTrainer.train()).
    k_r:
        Number of candidate relations to retrieve per head entity (Stage 1).
    k_t:
        Number of candidate tails to retrieve per (head, relation) query (Stage 2).
    alpha:
        Mixing weight for relation vs tail score in final ranking (see module doc).
    device:
        Device used for KGC inference.  Defaults to the model's current device.
    """

    def __init__(
        self,
        relation_predictor: BaseRelationPredictor,
        kgc_model: PyKEENWrapper,
        k_r: int = 10,
        k_t: int = 50,
        alpha: float = 0.5,
        device: Optional[torch.device] = None,
    ):
        self.relation_predictor = relation_predictor
        self.kgc_model          = kgc_model
        self.k_r                = k_r
        self.k_t                = k_t
        self.alpha              = alpha
        self.device             = device or next(kgc_model.model.parameters()).device

        kgc_model.model.eval()   # PyKEENWrapper holds the nn.Module in .model

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def generate_candidates(
        self,
        head: int,
        max_candidates: Optional[int] = None,
    ) -> List[Tuple[int, int, int, float]]:
        """
        Generate ranked (head, relation, tail, score) candidates for one entity.

        Stage 1: retrieve top-k_r (relation, rel_score) pairs.
        Stage 2: for each (head, relation, ?), retrieve top-k_t tails from KGC.
        Combine: softmax-normalise KGC scores per bucket, then blend with rel_score.

        Returns:
            Sorted list of (head, relation, tail, combined_score), descending.
        """
        # --- Stage 1: relation prediction ---
        top_rels = self.relation_predictor.predict_relations(head, top_k=self.k_r)
        # top_rels: List[(rel_id, rel_score)]  — rel_scores in [0,1]

        all_candidates: List[Tuple[int, int, int, float]] = []

        for rel_id, rel_score in top_rels:
            # --- Stage 2: tail prediction via KGC ---
            # Returns List[(tail_id, raw_kgc_score)], descending by raw score
            tail_hits = self.kgc_model.predict_tails(
                head=head,
                relation=rel_id,
                top_k=self.k_t,
                device=self.device,
            )

            if not tail_hits:
                continue

            # Softmax-normalise KGC scores within this (head, relation) bucket
            raw = np.array([s for _, s in tail_hits], dtype=np.float64)
            raw -= raw.max()                    # numerical stability
            exp  = np.exp(raw)
            tail_probs = exp / exp.sum()        # sums to 1 within the bucket

            for (tail_id, _), tail_prob in zip(tail_hits, tail_probs):
                combined = self.alpha * rel_score + (1.0 - self.alpha) * float(tail_prob)
                all_candidates.append((head, rel_id, tail_id, combined))

        all_candidates.sort(key=lambda x: -x[3])
        if max_candidates is not None:
            all_candidates = all_candidates[:max_candidates]

        return all_candidates

    def generate_candidates_batch(
        self,
        heads: List[int],
        max_candidates: Optional[int] = None,
    ) -> Dict[int, List[Tuple[int, int, int, float]]]:
        """
        Generate candidates for a list of head entities.

        Returns:
            Dict mapping each head entity id to its sorted candidate list.
        """
        results: Dict[int, List[Tuple[int, int, int, float]]] = {}
        for i, h in enumerate(heads):
            results[h] = self.generate_candidates(h, max_candidates)
            if (i + 1) % 100 == 0:
                logger.debug(f"  Pipeline: processed {i + 1}/{len(heads)} entities.")
        return results
