"""
Relation Predictors: BPR and Recoin
=====================================

Both classes implement BaseRelationPredictor and expose:

    predict_relations(head, top_k) -> List[(rel_id, score)]

They are consumed by InstanceCompletionPipeline (pipeline.py) which
appends a KGC model for tail prediction.

BPR — Bayesian Personalised Ranking [Rendle et al., 2012]
  Learns entity / relation embeddings via pairwise ranking:
      score(h, r) = E[h] · R[r]
  Training: for each observed (h, r+) sample r- ∉ observed(h) and
  maximise log σ(score(h,r+) − score(h,r-)).

Recoin — Relative Completeness in Wikidata [Balaraman et al., 2018]
  Frequency-based collaborative filtering: for h, find all entities
  that share ≥1 entity type (Boolean similarity), then rank relations
  by their occurrence rate among those similar entities:
      score(h, r) = |{e ∈ Similar(h) : ∃t (e,r,t) ∈ KG}| / |Similar(h)|
"""

from __future__ import annotations

import logging
import os
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

# ---------------------------------------------------------------------------
# Default cache directory — mirrors the layout used by kgc_tails.py
# Layout: <project_root>/iswc_data/cache/models/
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_DIR: Path = _PROJECT_ROOT / "iswc_data" / "cache" / "models"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseRelationPredictor(ABC):
    """
    Common interface for all relation predictors.

    Subclasses implement predict_relations() and the three cache hooks
    (_cache_filename, _build_checkpoint, _restore_checkpoint); all
    persistence logic (save / load / _cache_path) lives here.
    """

    # Declared here so type-checkers know all subclasses have them.
    _dataset_name: Optional[str]
    _cache_dir: Path

    # ── Public API ───────────────────────────────────────────────────────────

    @abstractmethod
    def predict_relations(
        self, head: int, top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        Return relations ranked for entity `head`.

        Returns:
            List of (relation_id, score) sorted by score descending.
            Scores are normalised to a probability-like range [0, 1].
        """

    def predict_relations_batch(
        self,
        heads: List[int],
        top_k: Optional[int] = None,
    ) -> Dict[int, List[Tuple[int, float]]]:
        return {h: self.predict_relations(h, top_k) for h in heads}

    # ── Cache hooks (implemented by subclasses) ──────────────────────────────

    @abstractmethod
    def _cache_filename(self) -> str:
        """Bare filename (no directory) for this predictor's cache file."""

    @abstractmethod
    def _build_checkpoint(self) -> dict:
        """Return everything needed to restore this predictor as a dict."""

    @abstractmethod
    def _restore_checkpoint(
        self, checkpoint: dict, device: Optional[torch.device] = None
    ) -> None:
        """Restore predictor state from a checkpoint dict."""

    # ── Persistence (shared implementation) ──────────────────────────────────

    def _init_cache(
        self,
        dataset_name: Optional[str],
        cache_dir: Optional[Path],
    ) -> None:
        """Call from subclass __init__ to initialise caching fields."""
        self._dataset_name = dataset_name
        self._cache_dir    = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR

    def _cache_path(self) -> Optional[Path]:
        """Return the full cache file path, or None if caching is disabled."""
        if not self._dataset_name:
            return None
        return self._cache_dir / self._cache_filename()

    def save(self, path: str) -> None:
        """Serialise this predictor to disk."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(self._build_checkpoint(), path)
        logger.info(f"{type(self).__name__} saved → {path}")

    @classmethod
    def load(
        cls,
        path: str,
        device: Optional[torch.device] = None,
    ) -> "BaseRelationPredictor":
        """Load a predictor saved by :meth:`save`."""
        device     = device or torch.device("cpu")
        checkpoint = torch.load(path, map_location=device)
        obj        = object.__new__(cls)
        obj._dataset_name = None
        obj._cache_dir    = DEFAULT_CACHE_DIR
        obj._restore_checkpoint(checkpoint, device)
        logger.info(f"{cls.__name__} loaded ← {path}")
        return obj


# ---------------------------------------------------------------------------
# BPR model + trainer
# ---------------------------------------------------------------------------

class _BPRModel(nn.Module):
    """Dot-product scoring model: score(h, r) = E[h] · R[r]."""

    def __init__(self, num_entities: int, num_relations: int, embed_dim: int):
        super().__init__()
        self.entity_emb   = nn.Embedding(num_entities,  embed_dim)
        self.relation_emb = nn.Embedding(num_relations, embed_dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def score(self, heads: Tensor, relations: Tensor) -> Tensor:
        """(B,) dot-product scores."""
        return (self.entity_emb(heads) * self.relation_emb(relations)).sum(dim=-1)

    def bpr_loss(self, heads: Tensor, pos_rels: Tensor, neg_rels: Tensor) -> Tensor:
        return -torch.log(
            torch.sigmoid(self.score(heads, pos_rels) - self.score(heads, neg_rels)) + 1e-10
        ).mean()

    @torch.no_grad()
    def all_scores(self, head: int) -> np.ndarray:
        """Softmax-normalised scores over all relations for one entity."""
        h   = self.entity_emb.weight[head]                    # (d,)
        raw = (self.relation_emb.weight * h).sum(dim=-1)      # (R,)
        raw = raw - raw.max()
        exp = torch.exp(raw)
        return (exp / exp.sum()).cpu().numpy()


class BPRRelationPredictor(BaseRelationPredictor):
    """
    BPR-based relation predictor.

    Usage::

        pred = BPRRelationPredictor(train_triples, num_entities, num_relations,
                                    dataset_name="fb15k237")
        pred.train(num_epochs=100)
        rels = pred.predict_relations(head_id, top_k=10)
    """

    def __init__(
        self,
        train_triples: Tensor,
        num_entities: int,
        num_relations: int,
        embed_dim: int = 64,
        lr: float = 1e-3,
        l2: float = 1e-4,
        device: Optional[torch.device] = None,
        dataset_name: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ):
        self._init_cache(dataset_name, cache_dir)
        self.device     = device or torch.device("cpu")
        self._embed_dim = embed_dim
        self._lr        = lr
        self._l2        = l2

        self.model = _BPRModel(num_entities, num_relations, embed_dim).to(self.device)
        self._opt  = optim.Adam(self.model.parameters(), lr=lr, weight_decay=l2)

        triples_np = train_triples.numpy() if isinstance(train_triples, Tensor) else np.array(train_triples)

        self._num_relations = num_relations
        self._entity_pos_rels: Dict[int, Set[int]] = defaultdict(set)
        self._hr_pairs: List[Tuple[int, int]] = []

        for h, r, _t in triples_np:
            self._entity_pos_rels[int(h)].add(int(r))
        for h, rels in self._entity_pos_rels.items():
            for r in rels:
                self._hr_pairs.append((h, r))

        logger.info(
            f"BPRRelationPredictor: {len(self._hr_pairs)} (head, relation) pairs "
            f"from {len(self._entity_pos_rels)} entities."
        )

    # ── Cache hooks ──────────────────────────────────────────────────────────

    def _cache_filename(self) -> str:
        return f"bpr_{self._dataset_name}_d{self._embed_dim}.pt"

    def _build_checkpoint(self) -> dict:
        return {
            "model_state":     self.model.state_dict(),
            "num_entities":    self.model.entity_emb.num_embeddings,
            "num_relations":   self._num_relations,
            "embed_dim":       self._embed_dim,
            "entity_pos_rels": {k: list(v) for k, v in self._entity_pos_rels.items()},
            "hr_pairs":        self._hr_pairs,
        }

    def _restore_checkpoint(
        self, checkpoint: dict, device: Optional[torch.device] = None
    ) -> None:
        device = device or torch.device("cpu")
        self.device         = device
        self._num_relations = checkpoint["num_relations"]
        self._embed_dim     = checkpoint["embed_dim"]
        self._lr            = None
        self._l2            = None
        self._entity_pos_rels = {
            int(k): set(v) for k, v in checkpoint["entity_pos_rels"].items()
        }
        self._hr_pairs = checkpoint["hr_pairs"]

        self.model = _BPRModel(
            checkpoint["num_entities"],
            checkpoint["num_relations"],
            checkpoint["embed_dim"],
        ).to(device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self._opt = None  # optimizer not restored; re-create if fine-tuning needed

    # ── Training ─────────────────────────────────────────────────────────────

    def _sample_neg(self, head: int) -> int:
        pos = self._entity_pos_rels[head]
        while True:
            r = random.randint(0, self._num_relations - 1)
            if r not in pos:
                return r

    def train(
        self,
        num_epochs: int = 100,
        batch_size: int = 512,
        verbose: bool = True,
    ) -> List[float]:
        """Train BPR and return per-epoch losses.

        If a cached checkpoint exists for this dataset/embed_dim, the
        trained weights are loaded from disk and training is skipped.
        """
        cache_path = self._cache_path()
        if cache_path is not None and cache_path.exists():
            logger.info(f"Loading cached BPR model from {cache_path}")
            checkpoint = torch.load(str(cache_path), map_location=self.device)
            self._restore_checkpoint(checkpoint, self.device)
            return []

        losses = []
        for epoch in range(1, num_epochs + 1):
            self.model.train()
            random.shuffle(self._hr_pairs)
            ep_loss, n = 0.0, 0
            for start in range(0, len(self._hr_pairs), batch_size):
                batch    = self._hr_pairs[start : start + batch_size]
                heads    = torch.tensor([b[0] for b in batch], dtype=torch.long, device=self.device)
                pos_rels = torch.tensor([b[1] for b in batch], dtype=torch.long, device=self.device)
                neg_rels = torch.tensor(
                    [self._sample_neg(b[0]) for b in batch],
                    dtype=torch.long, device=self.device,
                )
                self._opt.zero_grad()
                loss = self.model.bpr_loss(heads, pos_rels, neg_rels)
                loss.backward()
                self._opt.step()
                ep_loss += loss.item(); n += 1
            avg = ep_loss / max(n, 1)
            losses.append(avg)
            if verbose and epoch % 10 == 0:
                logger.info(f"  BPR epoch {epoch:3d}/{num_epochs} — loss {avg:.4f}")
        self.model.eval()

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.save(str(cache_path))

        return losses

    def predict_relations(
        self, head: int, top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        probs  = self.model.all_scores(head)                  # (R,)
        ranked = sorted(enumerate(probs), key=lambda x: -x[1])
        return ranked[:top_k] if top_k is not None else ranked


# ---------------------------------------------------------------------------
# Recoin
# ---------------------------------------------------------------------------

class RecoinRelationPredictor(BaseRelationPredictor):
    """
    Recoin-style relation predictor (Balaraman et al., 2018).

    For entity h, finds all entities sharing ≥1 type (Boolean similarity),
    then ranks relations by their occurrence rate among those neighbours:

        score(h, r) = |{e ∈ Similar(h) : ∃t (e,r,t) ∈ KG}| / |Similar(h)|

    Falls back to global relation frequency when h has no similar entities.

    Usage::

        pred = RecoinRelationPredictor(train_triples, entity_types,
                                       dataset_name="fb15k237")
        rels = pred.predict_relations(head_id, top_k=10)
    """

    def __init__(
        self,
        train_triples: Tensor,
        entity_types: Dict[int, List[int]],
        dataset_name: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ):
        """
        Args:
            train_triples: (N, 3) LongTensor of (head, relation, tail).
            entity_types:  {entity_id: [type_id, ...]}
            dataset_name:  Used to key the cache file. Required for caching.
            cache_dir:     Override the default cache directory.
        """
        self._init_cache(dataset_name, cache_dir)

        cache_path = self._cache_path()
        if cache_path is not None and cache_path.exists():
            logger.info(f"Loading cached Recoin model from {cache_path}")
            checkpoint = torch.load(str(cache_path), map_location="cpu")
            self._restore_checkpoint(checkpoint)
            return

        triples_np = train_triples.numpy() if isinstance(train_triples, Tensor) else np.array(train_triples)

        # e → set of relations it participates in (as head)
        self._entity_rels: Dict[int, Set[int]] = defaultdict(set)
        r_count: Dict[int, int] = defaultdict(int)
        for h, r, _t in triples_np:
            self._entity_rels[int(h)].add(int(r))
            r_count[int(r)] += 1

        # type → set of entities
        self._type_to_ents: Dict[int, Set[int]] = defaultdict(set)
        self._ent_types: Dict[int, Set[int]] = {}
        for ent, types in entity_types.items():
            s = set(types)
            self._ent_types[ent] = s
            for t in s:
                self._type_to_ents[t].add(ent)

        total = sum(r_count.values())
        self._global_rel_prob = {r: c / total for r, c in r_count.items()}

        logger.info(
            f"RecoinRelationPredictor: {len(self._entity_rels)} entities with relations, "
            f"{len(self._type_to_ents)} distinct types."
        )

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.save(str(cache_path))

    # ── Cache hooks ──────────────────────────────────────────────────────────

    def _cache_filename(self) -> str:
        return f"recoin_{self._dataset_name}.pt"

    def _build_checkpoint(self) -> dict:
        return {
            "entity_rels":     {k: list(v) for k, v in self._entity_rels.items()},
            "type_to_ents":    {k: list(v) for k, v in self._type_to_ents.items()},
            "ent_types":       {k: list(v) for k, v in self._ent_types.items()},
            "global_rel_prob": self._global_rel_prob,
        }

    def _restore_checkpoint(
        self, checkpoint: dict, device: Optional[torch.device] = None
    ) -> None:
        self._entity_rels     = {int(k): set(v) for k, v in checkpoint["entity_rels"].items()}
        self._type_to_ents    = {int(k): set(v) for k, v in checkpoint["type_to_ents"].items()}
        self._ent_types       = {int(k): set(v) for k, v in checkpoint["ent_types"].items()}
        self._global_rel_prob = checkpoint["global_rel_prob"]

    # ── Inference ────────────────────────────────────────────────────────────

    def _similar_entities(self, head: int) -> Set[int]:
        """Entities sharing ≥1 type with head (excluding head itself)."""
        h_types = self._ent_types.get(head, set())
        similar: Set[int] = set()
        for tp in h_types:
            similar |= self._type_to_ents.get(tp, set())
        similar.discard(head)
        return similar

    def predict_relations(
        self, head: int, top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        similar = self._similar_entities(head)
        if not similar:
            ranked = sorted(self._global_rel_prob.items(), key=lambda x: -x[1])
            return ranked[:top_k] if top_k is not None else ranked

        rel_cnt: Dict[int, int] = defaultdict(int)
        for e in similar:
            for r in self._entity_rels.get(e, set()):
                rel_cnt[r] += 1

        n      = len(similar)
        ranked = sorted(((r, cnt / n) for r, cnt in rel_cnt.items()), key=lambda x: -x[1])
        return ranked[:top_k] if top_k is not None else ranked
