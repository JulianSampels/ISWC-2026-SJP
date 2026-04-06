"""
MVF/GFRT — Full Pipeline: Filter + Rank
=========================================
Reproduced from:
  Li, Zhang, Yu.
  "A Multi-View Filter for Relation-Free Knowledge Graph Completion."
  Big Data Research, 2023. https://doi.org/10.1016/j.bdr.2023.100397

End-to-end usage:
  1. Build the two graphs with mvf_graphs.py.
  2. Train GFRTModel with this trainer.
  3. Use GFRTFilter.generate_candidates() to generate and score (r, t) pairs
     for a given head entity h.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple  # noqa: F401

import torch
import torch.optim as optim
from torch import Tensor

from .gfrt_graphs import GFRTGraph, build_head_relation_graph, build_tail_relation_graph
from .gfrt_model import GFRTModel, find_aligned_entities

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MVF Trainer
# ---------------------------------------------------------------------------

class GFRTTrainer:
    """
    Trainer for the GFRT model.

    Training strategy (MVF paper, Section 3.4):
      - Optimise L_H_intra, L_T_intra, and L_Cross alternately.
      - Two learning rates: η_1 for intra-view, η_2 for inter-view.
      - L2-normalise all embeddings after each gradient step (§3.3).
    """

    def __init__(
        self,
        model: GFRTModel,
        graph_H: GFRTGraph,
        graph_T: GFRTGraph,
        train_triples: Tensor,
        device: torch.device = torch.device("cpu"),
        lr_intra: float = 0.01,
        lr_inter: float = 0.001,
    ):
        self.model        = model.to(device)
        self.graph_H      = graph_H
        self.graph_T      = graph_T
        self.device       = device
        self.train_triples = train_triples

        self.aligned_entities = find_aligned_entities(train_triples).to(device)

        # Separate optimisers as in GFRT
        self.opt_intra = optim.Adam(
            list(model.gnn_head.parameters()) + list(model.gnn_tail.parameters()),
            lr=lr_intra,
        )
        self.opt_inter = optim.Adam(
            list(model.gnn_head.parameters()) + list(model.gnn_tail.parameters()),
            lr=lr_inter,
        )

        # Precompute training pair lists for quick sampling
        self._train_np = train_triples.numpy()

    def train_epoch(self, batch_size: int = 256) -> Dict[str, float]:
        """
        Run one training epoch. Returns dict of losses.
        """
        self.model.train()

        # Forward pass through both GNNs
        h_emb, rH_emb, t_emb, rT_emb = self.model(
            self.graph_H, self.graph_T, self.device
        )

        # Sample a batch of training triples
        idx     = torch.randperm(len(self._train_np))[:batch_size]
        batch   = self.train_triples[idx].to(self.device)
        pos_h   = batch[:, 0]
        pos_r   = batch[:, 1]
        pos_t   = batch[:, 2]

        # Intra-view losses (negatives are relation-corrupted inside intra_loss)
        loss_H = self.model.intra_loss(
            pos_h, pos_r, pos_t,
            h_emb, rH_emb, t_emb, rT_emb,
            is_head_graph=True,
        )
        loss_T = self.model.intra_loss(
            pos_h, pos_r, pos_t,
            h_emb, rH_emb, t_emb, rT_emb,
            is_head_graph=False,
        )

        # Inter-view alignment loss
        loss_C = self.model.inter_view_loss(self.aligned_entities, h_emb, t_emb)

        # Step intra
        self.opt_intra.zero_grad()
        (loss_H + loss_T).backward(retain_graph=True)
        self.opt_intra.step()

        # Step inter
        self.opt_inter.zero_grad()
        loss_C.backward()
        self.opt_inter.step()

        # L2-normalise all embeddings after each gradient step (MVF paper §3.3)
        self.model.normalize_embeddings()

        return {
            "loss_H":    loss_H.item(),
            "loss_T":    loss_T.item(),
            "loss_cross": loss_C.item(),
        }

    @torch.no_grad()
    def get_embeddings(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return current embeddings for inference."""
        self.model.eval()
        return self.model(self.graph_H, self.graph_T, self.device)


# ---------------------------------------------------------------------------
# MVF Filter: candidate generation + scoring
# ---------------------------------------------------------------------------

class GFRTFilter:
    """
    Given trained GFRT embeddings, generate and rank (r, t) candidates for a head entity h.

    Candidate generation (Section 4.3 in MVF paper, "without type information"):
      1. Select top-m relations for h (from head-rel score f(h, r_H)).
      2. Select top-n tail entities for each h (from tail-rel score f(h, t_T) — not
         directly, so we use the entity-side embedding similarity in G_T).
      3. Score all (r, t) pairs from {top-m relations} × {top-n tail candidates}
         using the combined score S_t = f(h, r_H) + f(t, r_T) + 1.
      4. Select top-x pairs as final candidates.

    This corresponds to the "without type information" protocol in the MVF paper.
    """

    def __init__(
        self,
        h_emb:   Tensor,    # (num_entities, D)
        rH_emb:  Tensor,    # (num_relations, D)
        t_emb:   Tensor,    # (num_entities, D)
        rT_emb:  Tensor,    # (num_relations, D)
        model:   GFRTModel,
        train_triples: Tensor,
        top_m_relations: int = 20,
        top_n_tails:     int = 100,
    ):
        self.h_emb   = h_emb
        self.rH_emb  = rH_emb
        self.t_emb   = t_emb
        self.rT_emb  = rT_emb
        self.model   = model
        self.top_m   = top_m_relations
        self.top_n   = top_n_tails

        # Build a relation-side tail index from training triples
        # (r -> set of observed tails) for faster candidate selection
        self._r_to_tails: Dict[int, List[int]] = defaultdict(list)
        triples_np = train_triples.numpy() if isinstance(train_triples, Tensor) else train_triples
        for h, r, t in triples_np:
            self._r_to_tails[int(r)].append(int(t))

    @torch.no_grad()
    def generate_candidates(
        self,
        head: int,
        candidate_budget: Optional[int] = None,
    ) -> List[Tuple[int, int, int, float]]:
        """
        Generate (head, relation, tail, score) candidates for a head entity.

        Args:
            head:             Head entity id.
            candidate_budget: Maximum candidates to return (top-x).

        Returns:
            Sorted list of (head, relation, tail, score) tuples.
        """
        num_relations = self.rH_emb.size(0)
        num_entities  = self.h_emb.size(0)
        device        = self.h_emb.device

        h_e = self.h_emb[head]  # (D,)

        # Step 1: Top-m relations by f(h, r_H) = h · r_H
        rel_scores = (h_e.unsqueeze(0) * self.rH_emb).sum(dim=-1)  # (R,)
        top_m_rels = rel_scores.topk(min(self.top_m, num_relations)).indices.tolist()

        # Step 2: Top-n tail entities using mean rT embedding of top-m relations as query.
        # f(t, r_T) = t · r_T; query = mean(rT[top_m_rels]) aggregates relation context.
        rT_query    = self.rT_emb[torch.tensor(top_m_rels, device=self.rT_emb.device)].mean(0)  # (D,)
        tail_scores = (self.t_emb * rT_query.unsqueeze(0)).sum(dim=-1)   # (E,)
        top_n_tails = tail_scores.topk(min(self.top_n, num_entities)).indices.tolist()

        # Step 3: Score all (r, t) combinations
        candidates: List[Tuple[int, int, int, float]] = []

        for r in top_m_rels:
            r_tensor = torch.tensor([r] * len(top_n_tails), dtype=torch.long, device=device)
            t_tensor = torch.tensor(top_n_tails, dtype=torch.long, device=device)
            h_tensor = torch.full_like(r_tensor, head)

            scores = self.model.score_candidates(
                h_tensor, r_tensor, t_tensor,
                self.h_emb, self.rH_emb, self.t_emb, self.rT_emb,
            )

            for t, s in zip(top_n_tails, scores.tolist()):
                candidates.append((head, r, t, s))

        # Step 4: Sort and return top-x
        candidates.sort(key=lambda x: -x[3])
        if candidate_budget is not None:
            candidates = candidates[:candidate_budget]
        return candidates

    def generate_candidates_batch(
        self,
        heads: List[int],
        candidate_budget: Optional[int] = None,
    ) -> Dict[int, List[Tuple[int, int, int, float]]]:
        """Generate candidates for a batch of head entities."""
        return {
            h: self.generate_candidates(h, candidate_budget)
            for h in heads
        }


# ---------------------------------------------------------------------------
# Convenience: build full MVF pipeline from scratch
# ---------------------------------------------------------------------------

def build_gfrt_pipeline(
    train_triples: Tensor,
    num_entities: int,
    num_relations: int,
    embed_dim: int = 100,
    num_layers: int = 2,
    top_k1: int = 100,
    top_k2: int = 30,
    margin: float = 1.0,
    device: torch.device = torch.device("cpu"),
) -> Tuple[GFRTModel, GFRTGraph, GFRTGraph]:
    """
    Convenience function: build both graphs and the GFRT model.

    Returns:
        (model, graph_H, graph_T) ready for training.
    """
    graph_H = build_head_relation_graph(
        train_triples, num_entities, num_relations, top_k1=top_k1, top_k2=top_k2
    )
    graph_T = build_tail_relation_graph(
        train_triples, num_entities, num_relations, top_k1=top_k1, top_k2=top_k2
    )
    model = GFRTModel(
        num_entities=num_entities,
        num_relations=num_relations,
        embed_dim=embed_dim,
        num_layers=num_layers,
        margin_intra=margin,
    ).to(device)

    return model, graph_H, graph_T
