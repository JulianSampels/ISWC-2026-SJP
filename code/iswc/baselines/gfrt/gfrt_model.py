"""
MVF/GFRT — Intra-View GNN and Inter-View Alignment Model
=========================================================
Reproduced from:
  Li, Zhang, Yu.
  "A Multi-View Filter for Relation-Free Knowledge Graph Completion."
  Big Data Research, 2023. https://doi.org/10.1016/j.bdr.2023.100397

Architecture overview (GFRT):
  1. Intra-view module:
     - Head-rel GNN  (GNN_H): learns h_emb and r_emb^H from G_H.
     - Tail-rel GNN  (GNN_T): learns t_emb and r_emb^T from G_T.
  2. Inter-view alignment:
     - For entities that appear in both G_H and G_T (the "aligned" set E_A),
       minimise ||e_a^H - e_a^T||_2.
  3. Scoring:
     - Head-rel score: f(h, r_H) = h_i · (r_H)_i
     - Tail-rel score: f(t, r_T) = t_i · (r_T)_i
     - Candidate score for (h, r, t): S = f(h, r_H) + f(t, r_T) + 1
       (the +1 avoids negative scores, following the MVF paper implementation).

Note: This reproduction uses a standard message-passing GNN with attention
aggregation as a proxy for the attention-based GNN described in the paper.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .gfrt_graphs import GFRTGraph

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Attention-based GNN layer (intra-view)
# ---------------------------------------------------------------------------

class AttentionGNNLayer(nn.Module):
    """
    One message-passing layer with attention coefficients ζ_ij or η_ij.

    For head entity h_i, the attention coefficient to neighbour relation r_j is:
        ζ_ij = w_0^T σ(W[r_j; h_i] + b) + b_0     (MVF Eq. 10 / 18)
    Normalised via softmax over N_d(h_i).

    This layer handles both entity→relation and entity→entity message passing
    within a single unified node space (entities 0..E-1, relations E..E+R-1).
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        # Attention parameters
        self.W_attn    = nn.Linear(2 * embed_dim, embed_dim, bias=True)
        self.w0        = nn.Linear(embed_dim, 1, bias=True)
        # Self-transformation
        self.W_self    = nn.Linear(embed_dim, embed_dim, bias=True)
        # Neighbour aggregation
        self.W_neigh   = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(
        self,
        node_emb: Tensor,           # (N_total, D)
        er_src: Tensor,             # entity-relation src (entity ids)
        er_dst: Tensor,             # entity-relation dst (relation node ids)
        ee_src: Optional[Tensor],   # entity-entity src
        ee_dst: Optional[Tensor],   # entity-entity dst
        ee_weight: Optional[Tensor],# entity-entity weights
    ) -> Tensor:
        """
        Single GNN update step.

        Returns:
            Updated node_emb of shape (N_total, D).
        """
        N = node_emb.size(0)
        agg = torch.zeros_like(node_emb)

        # --- Entity-relation attention aggregation ---
        if er_src.numel() > 0:
            h_emb = node_emb[er_src]   # (E_edges, D) head entity embeddings
            r_emb = node_emb[er_dst]   # (E_edges, D) relation embeddings

            # Attention: ζ_ij = softmax(w0 σ(W[r_j; h_i] + b))
            pair = torch.cat([r_emb, h_emb], dim=-1)   # (E_edges, 2D)
            e_ij = self.w0(torch.tanh(self.W_attn(pair))).squeeze(-1)  # (E_edges,)

            # Softmax per source node
            attn = _segment_softmax(e_ij, er_src, N)   # (E_edges,)

            # Aggregate: agg[h] += Σ_j attn_ij * W_neigh(r_j)
            r_msg = self.W_neigh(r_emb)  # (E_edges, D)
            agg.index_add_(0, er_src, attn.unsqueeze(-1) * r_msg)

        # --- Entity-entity aggregation (mean, weighted by similarity) ---
        if ee_src is not None and ee_src.numel() > 0:
            src_emb = node_emb[ee_src]  # (EE, D)
            if ee_weight is not None:
                w = ee_weight.unsqueeze(-1).to(node_emb.device)
                agg.index_add_(0, ee_src, w * self.W_neigh(src_emb))
            else:
                agg.index_add_(0, ee_src, self.W_neigh(src_emb))

        # Combine self + aggregated
        out = torch.tanh(self.W_self(node_emb) + agg)
        return out


# ---------------------------------------------------------------------------
# Intra-view GNN (stacks multiple attention layers)
# ---------------------------------------------------------------------------

class IntraViewGNN(nn.Module):
    """
    Multi-layer attention GNN for one graph view (head-rel or tail-rel).
    Produces entity embeddings v^e and relation embeddings v^r.
    """

    def __init__(self, total_nodes: int, embed_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.node_emb = nn.Embedding(total_nodes, embed_dim)
        nn.init.xavier_uniform_(self.node_emb.weight)

        self.layers = nn.ModuleList([
            AttentionGNNLayer(embed_dim) for _ in range(num_layers)
        ])

    def forward(
        self,
        graph: GFRTGraph,
        device: torch.device = torch.device("cpu"),
    ) -> Tensor:
        """
        Run the full GNN forward pass.

        Returns:
            node_emb: (total_nodes, D) final node embeddings.
        """
        er_src = graph.er_src.to(device)
        er_dst = graph.er_dst.to(device)
        ee_src = graph.ee_src.to(device) if graph.ee_src.numel() > 0 else None
        ee_dst = graph.ee_dst.to(device) if graph.ee_dst.numel() > 0 else None
        ee_w   = graph.ee_weight.to(device) if graph.ee_weight.numel() > 0 else None

        x = self.node_emb.weight.clone()
        for layer in self.layers:
            x = layer(x, er_src, er_dst, ee_src, ee_dst, ee_w)
        return x  # (N_total, D)


# ---------------------------------------------------------------------------
# Full GFRT model
# ---------------------------------------------------------------------------

class GFRTModel(nn.Module):
    """
    Full GFRT model: two intra-view GNNs + inter-view alignment.

    Usage:
      1. Call forward() to get entity and relation embeddings from both views.
      2. score_candidates() to score (h, r, t) candidates.
      3. Training: minimise intra_loss + cross_loss (inter-view alignment).
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embed_dim: int = 64,
        num_layers: int = 2,
        margin_intra: float = 1.0,
    ):
        super().__init__()
        total_nodes = num_entities + num_relations
        self.num_entities  = num_entities
        self.num_relations = num_relations
        self.embed_dim     = embed_dim
        self.margin_intra  = margin_intra

        self.gnn_head = IntraViewGNN(total_nodes, embed_dim, num_layers)
        self.gnn_tail = IntraViewGNN(total_nodes, embed_dim, num_layers)

    def forward(
        self,
        graph_H: GFRTGraph,
        graph_T: GFRTGraph,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Run both GNNs and return all embeddings.

        Returns:
            h_emb:   (num_entities,  D) — entity embeddings from G_H
            rH_emb:  (num_relations, D) — relation embeddings from G_H
            t_emb:   (num_entities,  D) — entity embeddings from G_T
            rT_emb:  (num_relations, D) — relation embeddings from G_T
        """
        E = self.num_entities
        xH = self.gnn_head(graph_H, device)   # (E+R, D)
        xT = self.gnn_tail(graph_T, device)   # (E+R, D)

        h_emb  = xH[:E]    # entity side of head-rel graph
        rH_emb = xH[E:]    # relation side of head-rel graph
        t_emb  = xT[:E]    # entity side of tail-rel graph
        rT_emb = xT[E:]    # relation side of tail-rel graph

        return h_emb, rH_emb, t_emb, rT_emb

    def score_candidates(
        self,
        heads: Tensor,       # (B,) head entity ids
        relations: Tensor,   # (B,) relation ids
        tails: Tensor,       # (B,) tail entity ids
        h_emb: Tensor,       # (num_entities, D)
        rH_emb: Tensor,      # (num_relations, D)
        t_emb: Tensor,       # (num_entities, D)
        rT_emb: Tensor,      # (num_relations, D)
    ) -> Tensor:
        """
        Compute candidate scores.

        Score(h, r, t) = f(h, r_H) + f(t, r_T) + 1
        where f(h, r_H) = h_i · (r_H)_i  (MVF paper Eq. 15 and 23).

        Returns:
            scores: (B,) scalar scores.
        """
        h_e  = h_emb[heads]          # (B, D)
        rH_e = rH_emb[relations]     # (B, D)
        t_e  = t_emb[tails]          # (B, D)
        rT_e = rT_emb[relations]     # (B, D)

        score_head = (h_e * rH_e).sum(dim=-1)    # f(h, r_H)
        score_tail = (t_e * rT_e).sum(dim=-1)    # f(t, r_T)
        return score_head + score_tail + 1.0

    def intra_loss(
        self,
        pos_h: Tensor, pos_r: Tensor, pos_t: Tensor,
        neg_h: Tensor, neg_r: Tensor, neg_t: Tensor,
        h_emb: Tensor, rH_emb: Tensor,
        t_emb: Tensor, rT_emb: Tensor,
        is_head_graph: bool = True,
    ) -> Tensor:
        """
        Intra-view hinge loss on (entity, relation) pairs.

        For head-rel graph: pairs are (h, r).
        For tail-rel graph: pairs are (t, r).

        L_intra = (1/|G|) Σ [γ + f(neg) - f(pos)]_+

        MVF paper Eq. (16) and (24).
        """
        if is_head_graph:
            e_emb   = h_emb
            r_emb   = rH_emb
            pos_e   = pos_h
            neg_e   = neg_h
        else:
            e_emb   = t_emb
            r_emb   = rT_emb
            pos_e   = pos_t
            neg_e   = neg_t

        pos_e_emb = e_emb[pos_e]
        neg_e_emb = e_emb[neg_e]
        r_e       = r_emb[pos_r]

        pos_score = (pos_e_emb * r_e).sum(dim=-1)
        neg_score = (neg_e_emb * r_e).sum(dim=-1)

        loss = F.relu(self.margin_intra + neg_score - pos_score)
        return loss.mean()

    def inter_view_loss(
        self,
        aligned_entities: Tensor,  # (A,) entity ids that appear in both views
        h_emb: Tensor,
        t_emb: Tensor,
    ) -> Tensor:
        """
        Inter-view alignment loss: force aligned entities' embeddings to be close.

        L_cross = (1/|E_A|) Σ_{e ∈ E_A} ||e_a^H - e_a^T||_2

        MVF paper Eq. (25).
        """
        e_H = h_emb[aligned_entities]
        e_T = t_emb[aligned_entities]
        return (e_H - e_T).norm(p=2, dim=-1).mean()


# ---------------------------------------------------------------------------
# Aligned entity detection
# ---------------------------------------------------------------------------

def find_aligned_entities(
    train_triples: torch.Tensor,
) -> torch.Tensor:
    """
    Find entities that appear as BOTH head AND tail in training triples.
    These are the "aligned" entities E_A that appear in both G_H and G_T.
    """
    triples_np = train_triples.numpy() if isinstance(train_triples, torch.Tensor) else train_triples
    heads = set(int(h) for h, r, t in triples_np)
    tails = set(int(t) for h, r, t in triples_np)
    aligned = sorted(heads & tails)
    return torch.tensor(aligned, dtype=torch.long)


# ---------------------------------------------------------------------------
# Segment softmax helper
# ---------------------------------------------------------------------------

def _segment_softmax(values: Tensor, segment_ids: Tensor, num_segments: int) -> Tensor:
    """
    Compute per-segment softmax. Used to normalise attention coefficients
    within each entity's neighbourhood.

    Args:
        values:      (E,) raw attention logits.
        segment_ids: (E,) segment membership (entity node id).
        num_segments: total number of segments (nodes).

    Returns:
        normalised: (E,) attention weights summing to 1 within each segment.
    """
    # Numerical stability: subtract segment max
    seg_max = torch.zeros(num_segments, device=values.device).scatter_reduce_(
        0, segment_ids, values, reduce="amax", include_self=True
    )
    shifted = values - seg_max[segment_ids]
    exp_v   = torch.exp(shifted)
    seg_sum = torch.zeros(num_segments, device=values.device).scatter_add_(
        0, segment_ids, exp_v
    )
    return exp_v / (seg_sum[segment_ids] + 1e-9)
