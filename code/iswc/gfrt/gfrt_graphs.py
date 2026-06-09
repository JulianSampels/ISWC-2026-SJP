"""
MVF/GFRT — Graph Construction Module
=====================================
Reproduced from:
  Li, Zhang, Yu.
  "A Multi-View Filter for Relation-Free Knowledge Graph Completion."
  Big Data Research, 2023. https://doi.org/10.1016/j.bdr.2023.100397

This module constructs the two heterogeneous graphs used by GFRT:
  - Head-relation graph G_H: models correlations between HEAD entities and relations.
  - Tail-relation graph G_T: models correlations between TAIL entities and relations.

Both graphs contain three edge types:
  1. Entity-entity edges (based on number of shared relations)
  2. Relation-relation edges (based on number of shared entities)
  3. Entity-relation edges (participation in a (h, r) or (r, t) pair)

Similarity cutoffs top-k1 and top-k2 are used to limit edge density
(default: k1=100 similar entities, k2=30 similar relations per node).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GFRTGraph:
    """
    Heterogeneous graph (either head-relation or tail-relation).

    Edge indices are stored as COO format: (src, dst) with edge_type label.
    All node ids share a unified namespace:
      - Entity nodes: 0 … num_entities-1
      - Relation nodes: num_entities … num_entities + num_relations - 1
    """
    num_entities: int
    num_relations: int

    # Entity-entity edge indices (by shared relations)
    ee_src: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    ee_dst: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    ee_weight: torch.Tensor = field(default_factory=lambda: torch.empty(0))

    # Relation-relation edge indices (by shared entities)
    rr_src: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    rr_dst: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    rr_weight: torch.Tensor = field(default_factory=lambda: torch.empty(0))

    # Entity-relation edge indices
    er_src: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    er_dst: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))

    @property
    def total_nodes(self) -> int:
        return self.num_entities + self.num_relations

    def relation_node_id(self, r: int) -> int:
        """Map a relation id to its node id in the unified namespace."""
        return self.num_entities + r


# ---------------------------------------------------------------------------
# Head-relation graph construction
# ---------------------------------------------------------------------------

def build_head_relation_graph(
    train_triples: torch.Tensor,
    num_entities: int,
    num_relations: int,
    top_k1: int = 100,
    top_k2: int = 30,
) -> GFRTGraph:
    """
    Build the head-relation graph G_H.

    Nodes: head entities ∪ relations.
    Edges:
      - (e_i, e_j) if they share >= 1 head-side relation (Jaccard-style sim_H Eq.1-2)
      - (r_i, r_j) if they share >= 1 head entity        (Jaccard-style sim_H Eq.3-4)
      - (e_i, r_j) for every training triple (e_i, r_j, *)

    Reference: MVF paper, Section 3.1 (Construction of head-rel graph).
    """
    triples_np = train_triples.numpy() if isinstance(train_triples, torch.Tensor) else np.array(train_triples)

    # h -> set of relations it participates in as HEAD
    h_to_rels: Dict[int, Set[int]] = defaultdict(set)
    # r -> set of head entities it connects FROM
    r_to_heads: Dict[int, Set[int]] = defaultdict(set)

    for h, r, t in triples_np:
        h_to_rels[int(h)].add(int(r))
        r_to_heads[int(r)].add(int(h))

    graph = GFRTGraph(num_entities=num_entities, num_relations=num_relations)

    # ------ Entity-entity edges ------
    logger.info("Building head-rel entity-entity edges…")
    ee_src, ee_dst, ee_w = _build_entity_entity_edges(
        entity_to_relations=h_to_rels,
        top_k=top_k1,
    )
    graph.ee_src    = torch.tensor(ee_src, dtype=torch.long)
    graph.ee_dst    = torch.tensor(ee_dst, dtype=torch.long)
    graph.ee_weight = torch.tensor(ee_w,   dtype=torch.float)

    # ------ Relation-relation edges ------
    logger.info("Building head-rel relation-relation edges…")
    rr_src, rr_dst, rr_w = _build_relation_relation_edges(
        relation_to_entities=r_to_heads,
        top_k=top_k2,
        offset=num_entities,
    )
    graph.rr_src    = torch.tensor(rr_src, dtype=torch.long)
    graph.rr_dst    = torch.tensor(rr_dst, dtype=torch.long)
    graph.rr_weight = torch.tensor(rr_w,   dtype=torch.float)

    # ------ Entity-relation edges ------
    # G_H contains one edge per observed (head, relation) pair. Repeating the
    # same pair once per triple would change attention weights away from the
    # graph described in the paper.
    er_edges = sorted((int(h), num_entities + int(r)) for h, r, _ in triples_np)
    er_edges = list(dict.fromkeys(er_edges))
    er_src_list = [h for h, _ in er_edges]
    er_dst_list = [r_node for _, r_node in er_edges]
    graph.er_src = torch.tensor(er_src_list, dtype=torch.long)
    graph.er_dst = torch.tensor(er_dst_list, dtype=torch.long)

    logger.info(
        f"Head-rel graph: {num_entities} entities, {num_relations} relations, "
        f"{len(ee_src)} EE edges, {len(rr_src)} RR edges, {len(er_src_list)} ER edges."
    )
    return graph


# ---------------------------------------------------------------------------
# Tail-relation graph construction
# ---------------------------------------------------------------------------

def build_tail_relation_graph(
    train_triples: torch.Tensor,
    num_entities: int,
    num_relations: int,
    top_k1: int = 100,
    top_k2: int = 30,
) -> GFRTGraph:
    """
    Build the tail-relation graph G_T.

    Symmetric to G_H, but from the perspective of TAIL entities.
    Reference: MVF paper, Section 3.1 (Construction of tail-rel graph).
    """
    triples_np = train_triples.numpy() if isinstance(train_triples, torch.Tensor) else np.array(train_triples)

    # t -> set of relations it participates in as TAIL
    t_to_rels: Dict[int, Set[int]] = defaultdict(set)
    # r -> set of tail entities it connects TO
    r_to_tails: Dict[int, Set[int]] = defaultdict(set)

    for h, r, t in triples_np:
        t_to_rels[int(t)].add(int(r))
        r_to_tails[int(r)].add(int(t))

    graph = GFRTGraph(num_entities=num_entities, num_relations=num_relations)

    # ------ Entity-entity edges ------
    logger.info("Building tail-rel entity-entity edges…")
    ee_src, ee_dst, ee_w = _build_entity_entity_edges(
        entity_to_relations=t_to_rels,
        top_k=top_k1,
    )
    graph.ee_src    = torch.tensor(ee_src, dtype=torch.long)
    graph.ee_dst    = torch.tensor(ee_dst, dtype=torch.long)
    graph.ee_weight = torch.tensor(ee_w,   dtype=torch.float)

    # ------ Relation-relation edges ------
    logger.info("Building tail-rel relation-relation edges…")
    rr_src, rr_dst, rr_w = _build_relation_relation_edges(
        relation_to_entities=r_to_tails,
        top_k=top_k2,
        offset=num_entities,
    )
    graph.rr_src    = torch.tensor(rr_src, dtype=torch.long)
    graph.rr_dst    = torch.tensor(rr_dst, dtype=torch.long)
    graph.rr_weight = torch.tensor(rr_w,   dtype=torch.float)

    # ------ Entity-relation edges ------
    # G_T contains one edge per observed (tail, relation) pair.
    er_edges = sorted((int(t), num_entities + int(r)) for _, r, t in triples_np)
    er_edges = list(dict.fromkeys(er_edges))
    er_src_list = [t for t, _ in er_edges]
    er_dst_list = [r_node for _, r_node in er_edges]
    graph.er_src = torch.tensor(er_src_list, dtype=torch.long)
    graph.er_dst = torch.tensor(er_dst_list, dtype=torch.long)

    logger.info(
        f"Tail-rel graph: {num_entities} entities, {num_relations} relations, "
        f"{len(ee_src)} EE edges, {len(rr_src)} RR edges, {len(er_src_list)} ER edges."
    )
    return graph

def _build_entity_entity_edges(
    entity_to_relations: Dict[int, Set[int]],
    top_k: int,
    chunk_size: int = 200,
) -> Tuple[List[int], List[int], List[float]]:
    
    logger.info("Parsing dictionary mappings into sparse matrix layout...")
    entity_ids = list(entity_to_relations.keys())
    if not entity_ids:
        return [], [], []

    num_entities = max(entity_ids) + 1
    
    max_rel = 0
    for rels in entity_to_relations.values():
        if rels:
            max_rel = max(max_rel, max(rels))
    num_relations = max_rel + 1

    rows, cols = [], []
    for e, rels in entity_to_relations.items():
        for r in rels:
            rows.append(e)
            cols.append(r)

    A = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(num_entities, num_relations)
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Moving matrix and metadata to device: {device}...")
    
    A_torch = torch.from_numpy(A.toarray()).to(device)
    entity_sizes = A_torch.sum(dim=1, keepdim=True)
    A_T = A_torch.T

    src_list, dst_list, w_list = [], [], []

    logger.info(f"Computing top-{top_k} asymmetric overlaps...")
    
    for start_idx in tqdm(range(0, num_entities, chunk_size), desc="Entity-entity edges"):
        end_idx = min(start_idx + chunk_size, num_entities)
        current_chunk_size = end_idx - start_idx
        
        intersections = torch.mm(A_torch[start_idx:end_idx], A_T)
        
        # Safety guard: Prevent division by zero for entities with no relations
        chunk_sizes = entity_sizes[start_idx:end_idx]
        scores = torch.where(chunk_sizes > 0, intersections / chunk_sizes, 0.0)
        
        # Exclude self-loops
        global_rows = torch.arange(start_idx, end_idx, device=device)
        scores[torch.arange(current_chunk_size, device=device), global_rows] = 0.0
        scores[scores <= 0] = 0.0
        
        actual_k = min(top_k, num_entities)
        topk_values, topk_indices = torch.topk(scores, k=actual_k, dim=1, largest=True)
        
        # Vectorised generation of source indices
        src_indices = torch.arange(start_idx, end_idx, device=device).unsqueeze(1).expand(-1, actual_k)
        
        # Filter out trailing zeros (where a row had fewer than top_k matches)
        valid_mask = topk_values > 0
        
        src_chunk = src_indices[valid_mask].cpu().numpy()
        dst_chunk = topk_indices[valid_mask].cpu().numpy()
        w_chunk = topk_values[valid_mask].cpu().numpy()
        
        # Append to the final lists in mass bulk
        src_list.extend(src_chunk.tolist())
        dst_list.extend(dst_chunk.astype(int).tolist())
        w_list.extend(w_chunk.astype(float).tolist())

    return src_list, dst_list, w_list


def _build_relation_relation_edges(
    relation_to_entities: Dict[int, Set[int]],
    top_k: int,
    offset: int,
) -> Tuple[List[int], List[int], List[float]]:
    """
    Build relation-relation edges by selecting top-k similar relations per relation.
    Similarity = Jaccard-like (number of shared entities normalised by |r1_entities|).
    Returns (src_ids, dst_ids, weights) where ids include the node offset.
    """
    relations = list(relation_to_entities.keys())
    src_list, dst_list, w_list = [], [], []

    for r1 in relations:
        ents1 = relation_to_entities[r1]
        if not ents1:
            continue
        sims: List[Tuple[float, int]] = []
        for r2 in relations:
            if r1 == r2:
                continue
            ents2 = relation_to_entities[r2]
            s = len(ents1 & ents2) / len(ents1)
            if s > 0:
                sims.append((s, r2))
        sims.sort(key=lambda x: -x[0])
        for sim, r2 in sims[:top_k]:
            src_list.append(offset + r1)
            dst_list.append(offset + r2)
            w_list.append(sim)

    return src_list, dst_list, w_list
