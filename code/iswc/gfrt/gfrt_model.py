"""
MVF/GFRT — Intra-View GNN and Inter-View Alignment Model
=========================================================
Reproduced from:
  Li, Zhang, Yu.
  "A Multi-View Filter for Relation-Free Knowledge Graph Completion."
  Big Data Research, 2023. https://doi.org/10.1016/j.bdr.2023.100397

GNN layer architecture (MVF paper Eq. 9-11 / 17-19):

  Entity node h_i:
    out = tanh(W1·h_i)                        [self]
        + tanh(W2 · Σ_j ζ_ij r_j)            [attention over relation neighbours]
        + tanh(W3 · (1/|S|) Σ_k h_k)         [unweighted mean over similar entities]

  Relation node r_i:
    out = tanh(W1·r_i)                        [self]
        + tanh(W2 · Σ_j η_ij h_j)            [attention over entity neighbours, Eq. 18]
        + tanh(W3 · (1/|S|) Σ_k r_k)         [unweighted mean over similar relations]

  Attention coefficients use SEPARATE parameters for entity→relation (ζ) and
  relation→entity (η) directions (Eq. 10 vs Eq. 18).

Intra-view loss (MVF paper Eq. 16 / 24):
  Head-rel view: negatives corrupt the RELATION only.
  Tail-rel view: negatives corrupt the RELATION OR the TAIL ENTITY (50/50).

Scoring (MVF paper):
  S(h, r, t) = f(h, r_H) + f(t, r_T)    [no +1 for GFRT itself]

L2 normalisation (MVF paper §3.3):
  All embeddings normalised to unit norm after each gradient step.
"""

from __future__ import annotations

import logging
import os
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

    Entity nodes (0..E-1):
      - Attention over relation neighbours via er edges  → attn_agg (W2)
      - Unweighted mean over similar entities via ee edges → mean_agg (W3)

    Relation nodes (E..E+R-1):
      - Attention over entity neighbours via reverse er edges → attn_agg (W2)
        (separate attention parameters η from entity attention ζ, per Eq. 10/18)
      - Unweighted mean over similar relations via rr edges → mean_agg (W3)

    Output: tanh(W1·x) + tanh(W2·attn_agg) + tanh(W3·mean_agg)
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

        # Entity-side attention: ζ_ij (entity h_i ← relation r_j, Eq. 10)
        self.W_attn_e = nn.Linear(2 * embed_dim, embed_dim, bias=True)
        self.w0       = nn.Linear(embed_dim, 1, bias=True)

        # Relation-side attention: η_ij (relation r_i ← entity h_j, Eq. 18)
        self.W_attn_r = nn.Linear(2 * embed_dim, embed_dim, bias=True)
        self.w1       = nn.Linear(embed_dim, 1, bias=True)

        # Three transformation matrices (shared across entity/relation node types)
        self.W1 = nn.Linear(embed_dim, embed_dim, bias=True)   # self
        self.W2 = nn.Linear(embed_dim, embed_dim, bias=True)   # attention agg
        self.W3 = nn.Linear(embed_dim, embed_dim, bias=True)   # mean agg

    def forward(
        self,
        node_emb:  Tensor,              # (N_total, D)
        er_src:    Tensor,              # entity ids (head of er edges)
        er_dst:    Tensor,              # relation node ids (tail of er edges, offset by E)
        ee_src:    Optional[Tensor],    # entity-entity src
        ee_dst:    Optional[Tensor],    # entity-entity dst (neighbours)
        rr_src:    Optional[Tensor],    # relation-relation src (offset by E)
        rr_dst:    Optional[Tensor],    # relation-relation dst (offset by E)
    ) -> Tensor:
        N = node_emb.size(0)
        attn_agg = torch.zeros_like(node_emb)
        mean_agg = torch.zeros_like(node_emb)
        mean_cnt = torch.zeros(N, device=node_emb.device)

        if er_src.numel() > 0:
            h_emb = node_emb[er_src]   # entity embeddings
            r_emb = node_emb[er_dst]   # relation embeddings

            # Entities aggregate from relation neighbours via attention (ζ, Eq. 10)
            pair_e = torch.cat([r_emb, h_emb], dim=-1)          # [r_j; h_i]
            e_ij   = self.w0(torch.tanh(self.W_attn_e(pair_e))).squeeze(-1)
            attn_e = _segment_softmax(e_ij, er_src, N)
            attn_agg.index_add_(0, er_src, attn_e.unsqueeze(-1) * r_emb)

            # Relations aggregate from entity neighbours via attention (η, Eq. 18)
            pair_r = torch.cat([h_emb, r_emb], dim=-1)          # [h_j; r_i]
            e_ij_r = self.w1(torch.tanh(self.W_attn_r(pair_r))).squeeze(-1)
            attn_r = _segment_softmax(e_ij_r, er_dst, N)
            attn_agg.index_add_(0, er_dst, attn_r.unsqueeze(-1) * h_emb)

        # Entity-entity UNWEIGHTED mean (Eq. 11: 1/|S(h_i)| Σ h_k)
        # Similarity scores select neighbours; they do NOT weight the aggregation.
        if ee_src is not None and ee_src.numel() > 0:
            neigh_emb = node_emb[ee_dst]
            mean_agg.index_add_(0, ee_src, neigh_emb)
            mean_cnt.index_add_(0, ee_src, torch.ones(ee_src.size(0), device=node_emb.device))

        # Relation-relation unweighted mean (Eq. 19: 1/|S(r_i)| Σ r_k)
        if rr_src is not None and rr_src.numel() > 0:
            rr_neigh = node_emb[rr_dst]
            mean_agg.index_add_(0, rr_src, rr_neigh)
            mean_cnt.index_add_(0, rr_src, torch.ones(rr_src.size(0), device=node_emb.device))

        # Normalise mean by count
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

    def __init__(self, total_nodes: int, embed_dim: int = 100, num_layers: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.node_emb  = nn.Embedding(total_nodes, embed_dim)
        nn.init.normal_(self.node_emb.weight, mean=0.0, std=1.0)

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
        rr_src = graph.rr_src.to(device) if graph.rr_src.numel() > 0 else None
        rr_dst = graph.rr_dst.to(device) if graph.rr_dst.numel() > 0 else None

        x = self.node_emb.weight.clone()
        for layer in self.layers:
            x = layer(x, er_src, er_dst, ee_src, ee_dst, rr_src, rr_dst)
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
        embed_dim:     int   = 100,
        num_layers:    int   = 2,
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
        E  = self.num_entities
        xH = self.gnn_head(graph_H, device)
        xT = self.gnn_tail(graph_T, device)
        return xH[:E], xH[E:], xT[:E], xT[E:]

    @torch.no_grad()
    def normalize_embeddings(self) -> None:
        """
        L2-normalise all node embeddings in both GNNs (MVF paper §3.3).
        Call after each optimizer step.
        """
        for gnn in (self.gnn_head, self.gnn_tail):
            w = gnn.node_emb.weight
            gnn.node_emb.weight.copy_(F.normalize(w, p=2, dim=-1))

    def score_candidates(
        self,
        heads:     Tensor,   # (B,) head entity ids
        relations: Tensor,   # (B,) relation ids
        tails:     Tensor,   # (B,) tail entity ids
        h_emb:     Tensor,
        rH_emb:    Tensor,
        t_emb:     Tensor,
        rT_emb:    Tensor,
    ) -> Tensor:
        """
        S(h, r, t) = f(h, r_H) + f(t, r_T)    (MVF paper Eq. 15 / 23)
        where f(x, y) = x · y.

        Note: no +1 offset for GFRT itself (that appears only for RETA-filter
        candidates in the paper's combined scoring formula).

        Returns: (B,) scalar scores.
        """
        h_e  = h_emb[heads]
        rH_e = rH_emb[relations]
        t_e  = t_emb[tails]
        rT_e = rT_emb[relations]
        return (h_e * rH_e).sum(dim=-1) + (t_e * rT_e).sum(dim=-1)

    def intra_loss(
        self,
        pos_h:         Tensor,   # (B,) positive head entity ids
        pos_r:         Tensor,   # (B,) positive relation ids
        pos_t:         Tensor,   # (B,) positive tail entity ids
        h_emb:         Tensor,
        rH_emb:        Tensor,
        t_emb:         Tensor,
        rT_emb:        Tensor,
        is_head_graph: bool = True,
    ) -> Tensor:
        """
        Intra-view hinge loss (MVF paper Eq. 16 / 24).

        Head-rel view (Eq. 16):
            Negative corrupts RELATION only.
            L_H = mean [γ + f(h, r'_H) - f(h, r_H)]_+

        Tail-rel view (Eq. 24):
            Negative corrupts RELATION OR TAIL ENTITY (50/50).
            L_T = mean [γ + f(neg_t, neg_r_T) - f(t, r_T)]_+
        """
        if is_head_graph:
            # Corrupt relation only
            neg_r = torch.randint(0, self.num_relations, pos_r.shape, device=pos_r.device)

            pos_e_emb = h_emb[pos_h]
            pos_score = (pos_e_emb * rH_emb[pos_r]).sum(dim=-1)
            neg_score = (pos_e_emb * rH_emb[neg_r]).sum(dim=-1)
        else:
            # Corrupt relation OR tail entity (50/50)
            B = pos_t.size(0)
            corrupt_tail = torch.rand(B, device=pos_r.device) < 0.5

            neg_r = torch.randint(0, self.num_relations, (B,), device=pos_r.device)
            neg_t = torch.randint(0, self.num_entities,  (B,), device=pos_r.device)

            # When corrupt_tail: use (pos_r, neg_t); else: use (neg_r, pos_t)
            eff_r = torch.where(corrupt_tail, pos_r, neg_r)
            eff_t = torch.where(corrupt_tail, neg_t, pos_t)

            pos_score = (t_emb[pos_t] * rT_emb[pos_r]).sum(dim=-1)
            neg_score = (t_emb[eff_t] * rT_emb[eff_r]).sum(dim=-1)

        return F.relu(self.margin_intra + neg_score - pos_score).mean()

    def inter_view_loss(
        self,
        aligned_entities: Tensor,   # (A,) entity ids appearing in both views
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

    def save(self, path: str) -> None:
        """Serialize the model to disk using torch.save()."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        checkpoint = {
            "model_state": self.state_dict(),
            "num_entities": self.num_entities,
            "num_relations": self.num_relations,
            "embed_dim": self.embed_dim,
            "num_layers": len(self.gnn_head.layers),
            "margin_intra": self.margin_intra,
        }
        torch.save(checkpoint, path)
        logger.info(f"GFRTModel saved → {path}")

    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> "GFRTModel":
        """Load a model saved by save()."""
        device = device or torch.device("cpu")
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            num_entities=checkpoint["num_entities"],
            num_relations=checkpoint["num_relations"],
            embed_dim=checkpoint["embed_dim"],
            num_layers=checkpoint["num_layers"],
            margin_intra=checkpoint.get("margin_intra", 1.0),
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        logger.info(f"GFRTModel loaded ← {path}")
        return model


# ---------------------------------------------------------------------------
# Aligned entity detection
# ---------------------------------------------------------------------------

def find_aligned_entities(train_triples: torch.Tensor) -> torch.Tensor:
    """
    Find entities that appear as BOTH head AND tail in training triples.
    These are the aligned entities E_A shared between G_H and G_T.
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
    Per-segment softmax for attention normalisation.

    Args:
        values:       (E,) raw attention logits.
        segment_ids:  (E,) segment membership (node id).
        num_segments: total number of nodes.

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
