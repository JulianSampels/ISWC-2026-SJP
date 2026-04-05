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
import multiprocessing as mp
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .bpr_recoin import BaseRelationPredictor
from .kgc_tails import PyKEENWrapper

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level worker state for Stage-1 multiprocessing
# ---------------------------------------------------------------------------
# These globals are set once per worker process via Pool(initializer=…) so
# the predictor object is pickled/transferred only at worker startup, not
# for every individual task.

_worker_predictor: Optional[BaseRelationPredictor] = None
_worker_k_r: int = 0


def _init_stage1_worker(predictor: BaseRelationPredictor, k_r: int) -> None:
    """Pool initializer: store the predictor in each worker process."""
    global _worker_predictor, _worker_k_r
    _worker_predictor = predictor
    _worker_k_r = k_r


def _stage1_worker(h: int) -> Tuple[int, List[Tuple[int, float]]]:
    """Worker task: predict relations for one head entity."""
    return h, _worker_predictor.predict_relations(h, top_k=_worker_k_r)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

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

    def _run_stage1(
        self,
        heads: List[int],
        num_workers: int,
    ) -> Dict[int, List[Tuple[int, float]]]:
        """
        Run Stage 1 (relation prediction) for all heads.

        With num_workers > 0, a multiprocessing Pool distributes the work
        across CPU cores.  The predictor is transferred to each worker once
        via the Pool initializer, so per-task overhead is just one int (the
        head ID).

        Note: num_workers > 0 requires the relation_predictor to be CPU-based.
        Forking a process that holds a live CUDA context is unsafe; if BPR is
        on GPU, set num_workers=0 (the default).
        """
        if num_workers > 0:
            ctx = mp.get_context("fork")      # fork: fast, zero-copy on Linux
            with ctx.Pool(
                processes=num_workers,
                initializer=_init_stage1_worker,
                initargs=(self.relation_predictor, self.k_r),
            ) as pool:
                pairs = pool.map(_stage1_worker, heads)
            return dict(pairs)

        return {
            h: self.relation_predictor.predict_relations(h, top_k=self.k_r)
            for h in heads
        }

    def generate_candidates_batch(
        self,
        heads: List[int],
        max_candidates: Optional[int] = None,
        chunk_size: int = 512,
        num_workers: int = 0,
    ) -> Dict[int, List[Tuple[int, int, int, float]]]:
        """
        Batched candidate generation with parallel Stage 1 and chunked GPU Stage 2.

        Stage 1 — relation prediction (CPU):
            Optionally parallelised across `num_workers` processes via
            multiprocessing.Pool.  Each worker receives the predictor once at
            startup (via the Pool initializer) and then only receives head IDs,
            keeping inter-process communication minimal.

        Stage 2 — KGC tail scoring (GPU):
            All (h, r) pairs from the entire batch are packed into a single
            (N, 2) tensor and fed to model.score_t() in chunks of `chunk_size`
            rows.  TopK, softmax-normalisation, and score combination are done
            on-GPU before a single .cpu() transfer.

        Parameters
        ----------
        heads : List[int]
            Head entity IDs to generate candidates for.
        max_candidates : int, optional
            Maximum candidates kept per head after ranking.
        chunk_size : int
            (h, r) pairs per score_t call.  Peak GPU mem ≈ chunk × entities × 4 B.
            Default 512 uses ~30 MB for FB15k-237 (14 k entities).
        num_workers : int
            CPU processes for Stage 1.  0 = sequential (safe with GPU predictors).
            Set > 0 only when the relation_predictor is CPU-based.

        Returns
        -------
        Dict mapping each head entity id to its sorted candidate list.
        """
        if not heads:
            return {}

        # ── Stage 1: relation prediction ─────────────────────────────────────
        stage1: Dict[int, List[Tuple[int, float]]] = self._run_stage1(heads, num_workers)

        # Pack all (h, r) pairs in input order and record per-head spans.
        hr_rows: List[Tuple[int, int]] = []
        rel_scores_list: List[float]   = []
        head_spans: List[Tuple[int, int]] = []    # parallel to heads

        for h in heads:
            top_rels = stage1.get(h, [])
            start = len(hr_rows)
            for rel_id, rel_score in top_rels:
                hr_rows.append((h, rel_id))
                rel_scores_list.append(float(rel_score))
            head_spans.append((start, len(hr_rows)))

        if not hr_rows:
            return {h: [] for h in heads}

        # ── Stage 2: chunked GPU scoring ─────────────────────────────────────
        N   = len(hr_rows)
        k_t = min(self.k_t, self.kgc_model.num_entities)
        dev = self.device

        hr_tensor    = torch.tensor(hr_rows,         dtype=torch.long,    device=dev)  # (N, 2)
        rel_scores_t = torch.tensor(rel_scores_list, dtype=torch.float32, device=dev)  # (N,)

        top_tail_idx   = torch.empty(N, k_t, dtype=torch.long,    device=dev)
        top_tail_probs = torch.empty(N, k_t, dtype=torch.float32, device=dev)

        model = self.kgc_model.model
        with torch.no_grad():
            for start in range(0, N, chunk_size):
                end  = min(start + chunk_size, N)
                raw  = model.score_t(hr_tensor[start:end])         # (C, num_entities)
                vals, idx = torch.topk(raw, k_t, dim=1)            # (C, k_t)
                top_tail_idx[start:end]   = idx
                top_tail_probs[start:end] = F.softmax(vals, dim=1) # per-bucket normalise

        # ── Combine rel_score with tail probs (vectorised on GPU) ─────────────
        combined = (
            self.alpha * rel_scores_t.unsqueeze(1) +
            (1.0 - self.alpha) * top_tail_probs
        )                                                            # (N, k_t)

        # Single transfer to CPU for the grouping step
        top_tail_idx_cpu = top_tail_idx.cpu()
        combined_cpu     = combined.cpu()
        hr_rels_cpu      = hr_tensor[:, 1].cpu()

        # ── Group by head, sort, truncate ─────────────────────────────────────
        results: Dict[int, List[Tuple[int, int, int, float]]] = {}
        for i, h in enumerate(heads):
            start, end = head_spans[i]
            if start == end:
                results[h] = []
                continue

            rels   = hr_rels_cpu[start:end].tolist()               # (n_rels,)
            t_idx  = top_tail_idx_cpu[start:end]                   # (n_rels, k_t)
            scores = combined_cpu[start:end]                       # (n_rels, k_t)

            cands: List[Tuple[int, int, int, float]] = [
                (h, rel_id, int(tail_id), float(score))
                for rel_id, row_idx, row_scores in zip(rels, t_idx.tolist(), scores.tolist())
                for tail_id, score in zip(row_idx, row_scores)
            ]

            cands.sort(key=lambda x: -x[3])
            if max_candidates is not None:
                cands = cands[:max_candidates]
            results[h] = cands

            if (i + 1) % 100 == 0:
                logger.debug(f"  Pipeline: grouped {i + 1}/{len(heads)} entities.")

        return results
