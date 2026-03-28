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
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from scipy.sparse import coo_matrix

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
        sim_fn=_sim_entity_head,
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
    er_src_list, er_dst_list = [], []
    for h, r, _ in triples_np:
        h, r = int(h), int(r)
        er_src_list.append(h)
        er_dst_list.append(num_entities + r)
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
        sim_fn=_sim_entity_tail,
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
    er_src_list, er_dst_list = [], []
    for _, r, t in triples_np:
        r, t = int(r), int(t)
        er_src_list.append(t)
        er_dst_list.append(num_entities + r)
    graph.er_src = torch.tensor(er_src_list, dtype=torch.long)
    graph.er_dst = torch.tensor(er_dst_list, dtype=torch.long)

    logger.info(
        f"Tail-rel graph: {num_entities} entities, {num_relations} relations, "
        f"{len(ee_src)} EE edges, {len(rr_src)} RR edges, {len(er_src_list)} ER edges."
    )
    return graph


# ---------------------------------------------------------------------------
# Shared similarity utilities
# ---------------------------------------------------------------------------

def _sim_entity_head(e1_rels: Set[int], e2_rels: Set[int]) -> float:
    """
    Head-entity similarity: fraction of e1's relations that e2 also has outgoing.
    MVF paper Eq. (1):
        sim_H(e1)(e2) = |{r ∈ R_H : (e1,r)∈G_H} ∩ {r ∈ R_H : (e2,r)∈G_H}|
                       / |{r ∈ R_H : (e1,r) ∈ G_H}|
    """
    if not e1_rels:
        return 0.0
    return len(e1_rels & e2_rels) / len(e1_rels)


def _sim_entity_tail(e1_rels: Set[int], e2_rels: Set[int]) -> float:
    """Tail-entity similarity (symmetric definition from the tail-rel graph)."""
    if not e1_rels:
        return 0.0
    return len(e1_rels & e2_rels) / len(e1_rels)


def _build_entity_entity_edges(
    entity_to_relations: Dict[int, Set[int]],
    top_k: int,
    sim_fn,
) -> Tuple[List[int], List[int], List[float]]:
    """
    Build entity-entity edges by selecting top-k similar entities per entity.
    Returns (src_ids, dst_ids, weights).
    """
    entities = list(entity_to_relations.keys())
    src_list, dst_list, w_list = [], [], []

    for e1 in entities:
        rels1 = entity_to_relations[e1]
        sims: List[Tuple[float, int]] = []
        for e2 in entities:
            if e1 == e2:
                continue
            s = sim_fn(rels1, entity_to_relations[e2])
            if s > 0:
                sims.append((s, e2))
        # Keep top-k
        sims.sort(key=lambda x: -x[0])
        for sim, e2 in sims[:top_k]:
            src_list.append(e1)
            dst_list.append(e2)
            w_list.append(sim)

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
