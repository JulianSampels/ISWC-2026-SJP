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

GNN layer architecture (MVF paper Eq. 11/19):
  out_i = tanh(W1 · x_i) + tanh(W2 · attn_agg_i) + tanh(W3 · mean_agg_i)

  where:
    - attn_agg_i = Σ_j ζ_ij · x_j   (attention over entity-relation neighbours)
    - mean_agg_i = (1/|N|) Σ_j x_j  (unweighted mean over same-type neighbours)
    - ζ_ij = softmax_j(w0^T σ(W_attn [x_j; x_i] + b))

Intra-view loss (MVF paper Eq. 16/24):
  Negative samples are formed by CORRUPTING THE RELATION, not the entity.
  L_intra = mean [γ + f(e, r'_neg) - f(e, r_pos)]_+

L2 normalisation (MVF paper §3.3):
  All embeddings are normalised to unit norm after each gradient step.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

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
    One message-passing layer handling all three edge types.

    Entity nodes aggregate from:
      - Relation neighbours via attention (er edges, W2 path)
      - Entity neighbours via weighted mean (ee edges, W3 path)

    Relation nodes aggregate from:
      - Entity neighbours via mean (reverse er edges, W3 path)
      - Relation neighbours via mean (rr edges, W3 path)

    Output:
      out_i = tanh(W1·x_i) + tanh(W2·attn_agg_i) + tanh(W3·mean_agg_i)
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        # Attention parameters (entity ← relation)
        self.W_attn = nn.Linear(2 * embed_dim, embed_dim, bias=True)
        self.w0     = nn.Linear(embed_dim, 1, bias=True)
        # Three separate transformation matrices (MVF Eq. 11/19)
        self.W1 = nn.Linear(embed_dim, embed_dim, bias=True)   # self
        self.W2 = nn.Linear(embed_dim, embed_dim, bias=True)   # attention agg
        self.W3 = nn.Linear(embed_dim, embed_dim, bias=True)   # mean agg

    def forward(
        self,
        node_emb:  Tensor,              # (N_total, D)
        er_src:    Tensor,              # entity ids (head of er edges)
        er_dst:    Tensor,              # relation node ids (tail of er edges, offset by E)
        ee_src:    Optional[Tensor],    # entity-entity src
        ee_dst:    Optional[Tensor],    # entity-entity dst (neighbours to aggregate from)
        ee_weight: Optional[Tensor],    # entity-entity Jaccard weights
        rr_src:    Optional[Tensor],    # relation-relation src (offset by E)
        rr_dst:    Optional[Tensor],    # relation-relation dst (offset by E)
    ) -> Tensor:
        N = node_emb.size(0)
        attn_agg = torch.zeros_like(node_emb)
        mean_agg = torch.zeros_like(node_emb)
        mean_cnt = torch.zeros(N, device=node_emb.device)

        # --- Entities aggregate from relation neighbours (attention, W2) ---
        if er_src.numel() > 0:
            h_emb = node_emb[er_src]   # entity embeddings
            r_emb = node_emb[er_dst]   # relation embeddings

            pair = torch.cat([r_emb, h_emb], dim=-1)   # (E_edges, 2D)
            e_ij = self.w0(torch.tanh(self.W_attn(pair))).squeeze(-1)  # (E_edges,)
            attn = _segment_softmax(e_ij, er_src, N)   # (E_edges,)

            # Aggregate relation messages → entity nodes
            attn_agg.index_add_(0, er_src, attn.unsqueeze(-1) * r_emb)

        # --- Relations aggregate from entity neighbours (reverse er, mean, W3) ---
        if er_src.numel() > 0:
            e_msgs = node_emb[er_src]   # entity embeddings as messages
            mean_agg.index_add_(0, er_dst, e_msgs)
            mean_cnt.index_add_(0, er_dst, torch.ones(er_dst.size(0), device=node_emb.device))

        # --- Entity-entity mean aggregation (weighted by Jaccard, W3) ---
        if ee_src is not None and ee_src.numel() > 0:
            # ee_dst are the NEIGHBOURS to aggregate from (not self)
            neigh_emb = node_emb[ee_dst]
            if ee_weight is not None:
                w = ee_weight.unsqueeze(-1).to(node_emb.device)
                mean_agg.index_add_(0, ee_src, w * neigh_emb)
                mean_cnt.index_add_(0, ee_src, ee_weight.to(node_emb.device))
            else:
                mean_agg.index_add_(0, ee_src, neigh_emb)
                mean_cnt.index_add_(0, ee_src, torch.ones(ee_src.size(0), device=node_emb.device))

        # --- Relation-relation mean aggregation (W3) ---
        if rr_src is not None and rr_src.numel() > 0:
            rr_neigh = node_emb[rr_dst]
            mean_agg.index_add_(0, rr_src, rr_neigh)
            mean_cnt.index_add_(0, rr_src, torch.ones(rr_src.size(0), device=node_emb.device))

        # Normalise mean aggregation by count
        safe_cnt = mean_cnt.clamp(min=1).unsqueeze(-1)
        mean_agg = mean_agg / safe_cnt

        # Output: tanh(W1·self) + tanh(W2·attn) + tanh(W3·mean)
        return (
            torch.tanh(self.W1(node_emb))
            + torch.tanh(self.W2(attn_agg))
            + torch.tanh(self.W3(mean_agg))
        )


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
        self.node_emb  = nn.Embedding(total_nodes, embed_dim)
        nn.init.xavier_uniform_(self.node_emb.weight)

        self.layers = nn.ModuleList([
            AttentionGNNLayer(embed_dim) for _ in range(num_layers)
        ])

    def forward(
        self,
        graph: GFRTGraph,
        device: torch.device = torch.device("cpu"),
    ) -> Tensor:
        """Returns node embeddings (total_nodes, D)."""
        er_src = graph.er_src.to(device)
        er_dst = graph.er_dst.to(device)
        ee_src = graph.ee_src.to(device) if graph.ee_src.numel() > 0 else None
        ee_dst = graph.ee_dst.to(device) if graph.ee_dst.numel() > 0 else None
        ee_w   = graph.ee_weight.to(device) if graph.ee_weight.numel() > 0 else None
        rr_src = graph.rr_src.to(device) if graph.rr_src.numel() > 0 else None
        rr_dst = graph.rr_dst.to(device) if graph.rr_dst.numel() > 0 else None

        x = self.node_emb.weight.clone()
        for layer in self.layers:
            x = layer(x, er_src, er_dst, ee_src, ee_dst, ee_w, rr_src, rr_dst)
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
      3. Training: minimise intra_loss + inter_view_loss; call normalize_embeddings() after each step.
    """

    def __init__(
        self,
        num_entities:  int,
        num_relations: int,
        embed_dim:     int = 64,
        num_layers:    int = 2,
        margin_intra:  float = 1.0,
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
            h_emb:  (num_entities,  D) — entity embeddings from G_H
            rH_emb: (num_relations, D) — relation embeddings from G_H
            t_emb:  (num_entities,  D) — entity embeddings from G_T
            rT_emb: (num_relations, D) — relation embeddings from G_T
        """
        E = self.num_entities
        xH = self.gnn_head(graph_H, device)   # (E+R, D)
        xT = self.gnn_tail(graph_T, device)   # (E+R, D)
        return xH[:E], xH[E:], xT[:E], xT[E:]

    @torch.no_grad()
    def normalize_embeddings(self) -> None:
        """
        L2-normalise all node embeddings (entities + relations) in both GNNs.
        Called after each optimizer step per MVF paper §3.3.
        """
        for gnn in (self.gnn_head, self.gnn_tail):
            w = gnn.node_emb.weight
            gnn.node_emb.weight.copy_(F.normalize(w, p=2, dim=-1))

    def score_candidates(
        self,
        heads:     Tensor,   # (B,) head entity ids
        relations: Tensor,   # (B,) relation ids
        tails:     Tensor,   # (B,) tail entity ids
        h_emb:     Tensor,   # (num_entities, D)
        rH_emb:    Tensor,   # (num_relations, D)
        t_emb:     Tensor,   # (num_entities, D)
        rT_emb:    Tensor,   # (num_relations, D)
    ) -> Tensor:
        """
        S(h, r, t) = f(h, r_H) + f(t, r_T) + 1
        where f(x, y) = x · y  (MVF paper Eq. 15 and 23).

        Returns:
            scores: (B,) scalar scores.
        """
        h_e  = h_emb[heads]
        rH_e = rH_emb[relations]
        t_e  = t_emb[tails]
        rT_e = rT_emb[relations]
        return (h_e * rH_e).sum(dim=-1) + (t_e * rT_e).sum(dim=-1) + 1.0

    def intra_loss(
        self,
        pos_h:        Tensor,   # (B,) positive head entity ids
        pos_r:        Tensor,   # (B,) positive relation ids
        pos_t:        Tensor,   # (B,) positive tail entity ids
        h_emb:        Tensor,
        rH_emb:       Tensor,
        t_emb:        Tensor,
        rT_emb:       Tensor,
        is_head_graph: bool = True,
    ) -> Tensor:
        """
        Intra-view hinge loss.

        Negatives are formed by CORRUPTING THE RELATION (not the entity),
        as described in MVF paper Eq. (16) and (24).

        For head-rel graph:
            L_H = mean [γ + f(h, r'_H) - f(h, r_H)]_+
        For tail-rel graph:
            L_T = mean [γ + f(t, r'_T) - f(t, r_T)]_+
        """
        if is_head_graph:
            e_emb = h_emb
            r_emb = rH_emb
            pos_e = pos_h
        else:
            e_emb = t_emb
            r_emb = rT_emb
            pos_e = pos_t

        # Corrupt the relation
        neg_r = torch.randint(0, self.num_relations, pos_r.shape, device=pos_r.device)

        pos_e_emb = e_emb[pos_e]
        pos_r_emb = r_emb[pos_r]
        neg_r_emb = r_emb[neg_r]

        pos_score = (pos_e_emb * pos_r_emb).sum(dim=-1)
        neg_score = (pos_e_emb * neg_r_emb).sum(dim=-1)

        return F.relu(self.margin_intra + neg_score - pos_score).mean()

    def inter_view_loss(
        self,
        aligned_entities: Tensor,  # (A,) entity ids appearing in both views
        h_emb: Tensor,
        t_emb: Tensor,
    ) -> Tensor:
        """
        Inter-view alignment loss (MVF paper Eq. 25).

        L_cross = (1/|E_A|) Σ_{e ∈ E_A} ||e_a^H - e_a^T||_2
        """
        e_H = h_emb[aligned_entities]
        e_T = t_emb[aligned_entities]
        return (e_H - e_T).norm(p=2, dim=-1).mean()


# ---------------------------------------------------------------------------
# Aligned entity detection
# ---------------------------------------------------------------------------

def find_aligned_entities(train_triples: torch.Tensor) -> torch.Tensor:
    """
    Find entities that appear as BOTH head AND tail in training triples.
    These are the "aligned" entities E_A in both G_H and G_T.
    """
    triples_np = train_triples.numpy() if isinstance(train_triples, torch.Tensor) else train_triples
    heads   = set(int(h) for h, r, t in triples_np)
    tails   = set(int(t) for h, r, t in triples_np)
    aligned = sorted(heads & tails)
    return torch.tensor(aligned, dtype=torch.long)


# ---------------------------------------------------------------------------
# Segment softmax helper
# ---------------------------------------------------------------------------

def _segment_softmax(values: Tensor, segment_ids: Tensor, num_segments: int) -> Tensor:
    """
    Compute per-segment softmax for attention normalisation.

    Args:
        values:       (E,) raw attention logits.
        segment_ids:  (E,) segment membership (entity node id).
        num_segments: total number of segments (nodes).

    Returns:
        (E,) attention weights summing to 1 within each segment.
    """
    seg_max = torch.zeros(num_segments, device=values.device).scatter_reduce_(
        0, segment_ids, values, reduce="amax", include_self=True
    )
    shifted = values - seg_max[segment_ids]
    exp_v   = torch.exp(shifted)
    seg_sum = torch.zeros(num_segments, device=values.device).scatter_add_(
        0, segment_ids, exp_v
    )
    return exp_v / (seg_sum[segment_ids] + 1e-9)
