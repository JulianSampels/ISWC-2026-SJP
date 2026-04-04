"""
KGC Tail Prediction via PyKEEN
================================

Wraps PyKEEN (https://pykeen.readthedocs.io, tested on v1.11) to provide
tail prediction for queries of the form (head, relation, ?).

Integration with your data pipeline
-------------------------------------
The primary entry point in this codebase is via KgLoader
(SJP_code/PathE/pathe/kgloader.py), which already builds a proper PyKEEN
TriplesFactory with correct entity/relation ID mappings.  Pass it directly:

    kg = KgLoader("fb15k237")
    wrapper = PyKEENTrainer.from_kg_loader("transe", kg).train(200)

If you do not have a KgLoader (e.g. for unit tests or custom datasets),
you can also construct from raw integer triples:

    wrapper = PyKEENTrainer.from_integer_triples(
        "transe", train_triples, num_entities, num_relations
    ).train(200)

The two-stage pipeline then is:

    Stage 1 — relation prediction  (BPR or Recoin)
        top_rels = rel_predictor.predict_relations(head, top_k=k_r)

    Stage 2 — tail prediction  (any PyKEEN model)
        for rel, rel_score in top_rels:
            tails = wrapper.predict_tails(head, rel, top_k=k_t)

    This is orchestrated by InstanceCompletionPipeline (pipeline.py).

Why NOT rebuild the TriplesFactory from scratch?
    KgLoader.triple_factory already contains a fully consistent entity_to_id /
    relation_to_id mapping across train / val / test.  Re-creating it from the
    raw integer tensor with artificial str(i) labels would duplicate work and,
    for datasets where PyKEEN's IDs are not the identity mapping, produce the
    wrong model.

Available models (40+, selected):
    Translational:  transe, transh, transr, transd, se
    Bilinear:       distmult, complex, rescal, simple, tucker, cp, hole
    Neural:         conve, convkb, ermlp, ermlpe, ntn, proje
    Geometric:      rotate, mure, quate, boxe, toruse, kg2e, pairre
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Default cache directory — anchored to the project root so it works
# regardless of the current working directory.
# Layout: <project_root>/iswc_data/cache/models/
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_DIR: Path = _PROJECT_ROOT / "iswc_data" / "cache" / "models"

if TYPE_CHECKING:
    # Avoid hard import at module load time — pykeen is heavy
    from pykeen.triples import TriplesFactory as _TriplesFactory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional PyKEEN import
# ---------------------------------------------------------------------------

try:
    from pykeen.models import model_resolver
    from pykeen.triples import TriplesFactory
    from pykeen.training import SLCWATrainingLoop
    _PYKEEN_AVAILABLE = True
    PYKEEN_MODELS: Set[str] = set(model_resolver.options)
except ImportError:  # pragma: no cover
    _PYKEEN_AVAILABLE = False
    PYKEEN_MODELS: Set[str] = set()

MODEL_REGISTRY: Set[str] = PYKEEN_MODELS   # alias used by __init__.py


def _require_pykeen() -> None:
    if not _PYKEEN_AVAILABLE:
        raise ImportError(
            "PyKEEN is required for KGC tail prediction. "
            "Install with: pip install pykeen"
        )


# ---------------------------------------------------------------------------
# TriplesFactory helpers
# ---------------------------------------------------------------------------

def _factory_from_integer_triples(
    train_triples: Tensor,
    num_entities: int,
    num_relations: int,
) -> "TriplesFactory":
    """
    Build a PyKEEN TriplesFactory from a raw integer-encoded triple tensor.

    Entity / relation integer IDs are preserved by using str(id) labels with
    an explicit identity mapping str(i) → i, so entity 42 in our codebase
    maps to index 42 inside PyKEEN.

    Use this ONLY when you do not already have a TriplesFactory (e.g. unit
    tests or custom datasets).  When using KgLoader, pass
    kg_loader.triple_factory.training directly to PyKEENTrainer.
    """
    _require_pykeen()

    arr = (
        train_triples.numpy() if isinstance(train_triples, Tensor) else np.asarray(train_triples)
    ).astype(int)

    entity_to_id   = {str(i): i for i in range(num_entities)}
    relation_to_id = {str(i): i for i in range(num_relations)}
    labeled        = np.array([[str(h), str(r), str(t)] for h, r, t in arr])

    return TriplesFactory.from_labeled_triples(
        labeled,
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
    )


# ---------------------------------------------------------------------------
# Inference wrapper
# ---------------------------------------------------------------------------

class PyKEENWrapper:
    """
    Thin adapter around a trained PyKEEN model.

    Exposes predict_tails() so it plugs directly into
    InstanceCompletionPipeline without any changes to the pipeline code.

    Training happens ONCE (in PyKEENTrainer.train()), then the wrapper is
    saved to disk.  Subsequent pipeline runs load the saved wrapper instead
    of re-training, which matters for large KGs (FB15K-237 training can
    take minutes to hours depending on the model and hardware).

    Typical workflow
    ----------------
    ::

        # ── Step 1: train once ───────────────────────────────────────────
        wrapper = PyKEENTrainer.from_kg_loader("transe", kg).train(200)
        wrapper.save("checkpoints/transe_fb15k237.pt")

        # ── Step 2: load on subsequent runs ──────────────────────────────
        wrapper = PyKEENWrapper.load("checkpoints/transe_fb15k237.pt")
        tails   = wrapper.predict_tails(head=42, relation=3, top_k=50)

    Attributes
    ----------
    model : pykeen.models.base.Model
    num_entities : int
    num_relations : int
    """

    def __init__(
        self,
        model: Any,
        num_entities: int,
        num_relations: int,
        embed_dim: Optional[int] = None,
    ):
        _require_pykeen()
        self.model         = model
        self.num_entities  = num_entities
        self.num_relations = num_relations
        self._embed_dim    = embed_dim   # original value from trainer; None for loaded models
        self.model.eval()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Save the trained wrapper to disk.

        Serialises the PyKEEN model's state dict plus the metadata needed
        to reconstruct the wrapper on load (model class name, num_entities,
        num_relations).  Using state dict (not pickle of the whole object)
        makes the checkpoint robust to code changes.

        Parameters
        ----------
        path : str
            File path for the checkpoint, e.g.
            ``"checkpoints/transe_fb15k237.pt"``.

        Example
        -------
        ::

            wrapper.save("checkpoints/transe_fb15k237.pt")
        """
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        checkpoint = {
            "model_class":    self.model.__class__.__name__,
            "model_state":    self.model.state_dict(),
            "num_entities":   self.num_entities,
            "num_relations":  self.num_relations,
            # Store the original embed_dim from training — NOT inferred from weights.
            # Inferring is wrong for models like RotatE that store 2×embed_dim internally.
            "embedding_dim":  self._embed_dim,
        }
        torch.save(checkpoint, path)
        logger.info(f"PyKEENWrapper saved → {path}")

    @classmethod
    def load(
        cls,
        path: str,
        device: Optional[torch.device] = None,
        triples_factory: Optional[Any] = None,
    ) -> "PyKEENWrapper":
        """
        Load a saved wrapper from disk.

        Parameters
        ----------
        path : str
            Path to a checkpoint saved by :meth:`save`.
        device : torch.device, optional
            Device to load the model onto.  Defaults to CPU.
        triples_factory : TriplesFactory, optional
            Required by some PyKEEN models (e.g. ConvE) whose constructor
            needs shape information from the factory.  For standard
            embedding models (TransE, RotatE, DistMult, …) this can be
            omitted — a minimal dummy factory is created automatically.

        Returns
        -------
        PyKEENWrapper

        Example
        -------
        ::

            wrapper = PyKEENWrapper.load("checkpoints/transe_fb15k237.pt")
        """
        _require_pykeen()
        device = device or torch.device("cpu")

        checkpoint     = torch.load(path, map_location=device)
        model_cls_name = checkpoint["model_class"]
        num_entities   = checkpoint["num_entities"]
        num_relations  = checkpoint["num_relations"]
        embedding_dim  = checkpoint.get("embedding_dim")  # None for old checkpoints

        # Resolve the PyKEEN model class by name
        model_cls = model_resolver.lookup(model_cls_name)

        # Build a minimal dummy TriplesFactory so the model constructor
        # can infer its embedding shapes.  The actual triples do not matter
        # here; only num_entities and num_relations are used.
        if triples_factory is None:
            dummy_triples = np.array(
                [[str(i % num_entities), str(i % num_relations), str((i + 1) % num_entities)]
                 for i in range(max(num_relations, 2))],
                dtype=str,
            )
            entity_to_id   = {str(i): i for i in range(num_entities)}
            relation_to_id = {str(i): i for i in range(num_relations)}
            triples_factory = TriplesFactory.from_labeled_triples(
                dummy_triples,
                entity_to_id=entity_to_id,
                relation_to_id=relation_to_id,
            )

        # Reconstruct with the same embedding_dim used during training
        model_kwargs = {}
        if embedding_dim is not None:
            model_kwargs["embedding_dim"] = embedding_dim

        model = model_cls(triples_factory=triples_factory, **model_kwargs).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        logger.info(
            f"PyKEENWrapper loaded ← {path} "
            f"({model_cls_name}, {num_entities} entities, {num_relations} relations)."
        )
        return cls(model, num_entities, num_relations)

    @torch.no_grad()
    def predict_tails(
        self,
        head: int,
        relation: int,
        top_k: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> List[Tuple[int, float]]:
        """
        Rank all entities as candidate tails for query (head, relation, ?).

        PyKEEN's score_t() scores ALL entities in one vectorised call, so
        no manual chunking is required.

        Parameters
        ----------
        head, relation : int
            Integer IDs matching the mapping inside the TriplesFactory used
            during training (i.e. the IDs from KgLoader.train_triples).
        top_k : int, optional
            Return only the top-k tails.  None means return all entities.
        device : torch.device, optional
            Device for inference.  Defaults to the model's current device.

        Returns
        -------
        List of (tail_id, score) sorted by score descending.
        """
        self.model.eval()
        dev = device or next(self.model.parameters()).device
        self.model = self.model.to(dev)

        # score_t expects shape (batch, 2): [head_id, relation_id]
        hr_batch = torch.tensor([[head, relation]], dtype=torch.long, device=dev)
        scores   = self.model.score_t(hr_batch).squeeze(0)  # (num_entities,)

        k = min(top_k, self.num_entities) if top_k is not None else self.num_entities
        top_vals, top_idx = torch.topk(scores, k)
        return list(zip(top_idx.cpu().tolist(), top_vals.cpu().tolist()))

    @torch.no_grad()
    def score_triples(self, triples: Tensor) -> Tensor:
        """
        Score a batch of explicit (h, r, t) triples.

        Parameters
        ----------
        triples : (N, 3) LongTensor

        Returns
        -------
        (N,) float tensor of scores.
        """
        self.model.eval()
        dev = next(self.model.parameters()).device
        return self.model.score_hrt(triples.to(dev))


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class PyKEENTrainer:
    """
    Trains any PyKEEN model and returns a PyKEENWrapper for tail prediction.

    Two construction paths
    ----------------------
    **Preferred — when using KgLoader:**

        trainer = PyKEENTrainer.from_kg_loader("transe", kg_loader,
                                               embed_dim=128, lr=1e-3)
        wrapper = trainer.train(num_epochs=200)

    This reuses the TriplesFactory that KgLoader already built, which
    guarantees consistent entity/relation IDs across the whole pipeline.

    **Fallback — from raw integer triples:**

        trainer = PyKEENTrainer.from_integer_triples(
            "transe", train_triples, num_entities, num_relations,
            embed_dim=128, lr=1e-3,
        )
        wrapper = trainer.train(num_epochs=200)

    Parameters (shared)
    --------------------
    model_name : str
        PyKEEN model name.  See ``sorted(PYKEEN_MODELS)`` for the full list,
        e.g. ``"transe"``, ``"rotate"``, ``"distmult"``, ``"complex"``, …
    embed_dim : int
        Embedding dimension passed to the model as ``embedding_dim``.
        Models with asymmetric spaces (TransR, Tucker) accept extra dims
        via ``model_kwargs``, e.g. ``model_kwargs={"relation_dim": 50}``.
    lr : float
        Adam learning rate.
    device : torch.device, optional
        Training device.  Defaults to CUDA if available, else CPU.
    model_kwargs : dict, optional
        Additional keyword arguments forwarded to the PyKEEN model
        constructor verbatim.
    negative_sampler : str
        PyKEEN negative sampler:
          ``"basic"``     — uniform random entity replacement (default)
          ``"bernoulli"`` — relation-frequency-aware (TransH paper trick)
    """

    def __init__(
        self,
        model_name: str,
        triples_factory: "TriplesFactory",
        num_entities: int,
        num_relations: int,
        embed_dim: int = 128,
        lr: float = 1e-3,
        device: Optional[torch.device] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        negative_sampler: str = "basic",
        dataset_name: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ):
        _require_pykeen()

        key = model_name.lower()
        if key not in PYKEEN_MODELS:
            raise ValueError(
                f"Unknown PyKEEN model '{model_name}'. "
                f"Available: sorted(PYKEEN_MODELS)."
            )

        self._model_name       = key
        self._tf               = triples_factory
        self._num_entities     = num_entities
        self._num_relations    = num_relations
        self._embed_dim        = embed_dim
        self._lr               = lr
        self._device           = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model_kwargs     = model_kwargs or {}
        self._negative_sampler = negative_sampler
        self._dataset_name     = dataset_name
        self._cache_dir        = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cache_path(self) -> Optional[Path]:
        """
        Return the full cache file path, or None if caching is disabled.

        Caching is disabled when dataset_name is not set (no stable name to
        key on).  Format: <cache_dir>/<model>_<dataset>_d<embed_dim>.pt
        """
        if not self._dataset_name:
            return None
        fname = f"{self._model_name}_{self._dataset_name}_d{self._embed_dim}.pt"
        return self._cache_dir / fname

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_kg_loader(
        cls,
        model_name: str,
        kg_loader: Any,
        embed_dim: int = 128,
        lr: float = 1e-3,
        device: Optional[torch.device] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        negative_sampler: str = "basic",
        dataset_name: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ) -> "PyKEENTrainer":
        """
        Build a trainer directly from a KgLoader instance.

        Reuses the TriplesFactory that KgLoader already constructed so that
        entity/relation IDs are guaranteed to be consistent with the integer
        tensors in kg_loader.train_triples / val_triples / test_triples.

        Parameters
        ----------
        kg_loader : KgLoader
            A loaded KgLoader (SJP_code/PathE/pathe/kgloader.py).
        model_name : str
            PyKEEN model name (e.g. ``"transe"``, ``"rotate"``).

        Example
        -------
        ::

            from PathE.pathe.kgloader import KgLoader
            from iswc.baselines import PyKEENTrainer, InstanceCompletionPipeline

            kg      = KgLoader("fb15k237")
            trainer = PyKEENTrainer.from_kg_loader("rotate", kg, embed_dim=128)
            wrapper = trainer.train(num_epochs=200)

            # wrapper.predict_tails() now uses the same IDs as kg.train_triples
        """
        # KgLoader.triple_factory is a PyKEEN EagerDataset with a .training attr
        training_factory = kg_loader.triple_factory.training
        num_entities     = kg_loader.num_nodes_total
        num_relations    = training_factory.num_relations

        # Auto-infer dataset name from KgLoader if not provided explicitly
        inferred_dataset = dataset_name or getattr(kg_loader, "dataset", None)

        logger.info(
            f"PyKEENTrainer.from_kg_loader: dataset={inferred_dataset or '?'}, "
            f"entities={num_entities}, relations={num_relations}, "
            f"train_triples={training_factory.num_triples}."
        )

        return cls(
            model_name=model_name,
            triples_factory=training_factory,
            num_entities=num_entities,
            num_relations=num_relations,
            embed_dim=embed_dim,
            lr=lr,
            device=device,
            model_kwargs=model_kwargs,
            negative_sampler=negative_sampler,
            dataset_name=inferred_dataset,
            cache_dir=cache_dir,
        )

    @classmethod
    def from_integer_triples(
        cls,
        model_name: str,
        train_triples: Tensor,
        num_entities: int,
        num_relations: int,
        embed_dim: int = 128,
        lr: float = 1e-3,
        device: Optional[torch.device] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        negative_sampler: str = "basic",
        dataset_name: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ) -> "PyKEENTrainer":
        """
        Build a trainer from a raw integer-encoded triple tensor.

        Use this when you do NOT have a KgLoader (e.g. unit tests, custom
        datasets).  Entity IDs must be contiguous integers in [0, num_entities).

        Parameters
        ----------
        train_triples : (N, 3) LongTensor of (head, relation, tail).
        num_entities, num_relations : int
            Total counts in the full KG (not just the training split).
        dataset_name : str, optional
            Used to construct the cache filename.  Required for caching.
        cache_dir : Path, optional
            Override the default cache directory.
        """
        logger.info(
            f"PyKEENTrainer.from_integer_triples: building TriplesFactory "
            f"from {len(train_triples)} triples "
            f"({num_entities} entities, {num_relations} relations)."
        )
        tf = _factory_from_integer_triples(train_triples, num_entities, num_relations)
        return cls(
            model_name=model_name,
            triples_factory=tf,
            num_entities=num_entities,
            num_relations=num_relations,
            embed_dim=embed_dim,
            lr=lr,
            device=device,
            model_kwargs=model_kwargs,
            negative_sampler=negative_sampler,
            dataset_name=dataset_name,
            cache_dir=cache_dir,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        num_epochs: int = 200,
        batch_size: int = 1024,
        verbose: bool = True,
    ) -> PyKEENWrapper:
        """
        Train the model and return a PyKEENWrapper ready for tail prediction.

        The SLCWA (Stochastic Local Closed World Assumption) training loop is
        used: for each positive triple, one or more negatives are sampled by
        corrupting the tail entity.  This is the standard training regime for
        TransE, RotatE, DistMult, etc.

        Parameters
        ----------
        num_epochs : int
        batch_size : int
            Number of positive triples per mini-batch.
        verbose : bool
            Show a tqdm progress bar during training.

        Returns
        -------
        PyKEENWrapper
            The trained model, ready to plug into InstanceCompletionPipeline.
        """
        _require_pykeen()

        # ── Cache check ───────────────────────────────────────────────────────
        cache_path = self._cache_path()
        if cache_path is not None and cache_path.exists():
            logger.info(f"Loading cached model from {cache_path}")
            return PyKEENWrapper.load(str(cache_path), device=self._device)

        logger.info(
            f"Training '{self._model_name}' — "
            f"epochs={num_epochs}, batch={batch_size}, "
            f"embed_dim={self._embed_dim}, lr={self._lr}, "
            f"device={self._device}."
        )

        model_cls = model_resolver.lookup(self._model_name)
        model = model_cls(
            triples_factory=self._tf,
            embedding_dim=self._embed_dim,
            **self._model_kwargs,
        ).to(self._device)

        loop = SLCWATrainingLoop(
            model=model,
            triples_factory=self._tf,
            optimizer="adam",
            optimizer_kwargs={"lr": self._lr},
            negative_sampler=self._negative_sampler,
        )

        losses = loop.train(
            triples_factory=self._tf,
            num_epochs=num_epochs,
            batch_size=batch_size,
            use_tqdm=verbose,
        )

        if losses:
            logger.info(f"Training complete — final loss: {losses[-1]:.4f}.")

        model.eval()
        wrapper = PyKEENWrapper(model, self._num_entities, self._num_relations, embed_dim=self._embed_dim)

        # ── Cache save ────────────────────────────────────────────────────────
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            wrapper.save(str(cache_path))
            logger.info(f"Model cached → {cache_path}")

        return wrapper


# ---------------------------------------------------------------------------
# Convenience one-call factory
# ---------------------------------------------------------------------------

def build_kgc_model(
    name: str,
    train_triples: Tensor,
    num_entities: int,
    num_relations: int,
    embed_dim: int = 128,
    num_epochs: int = 200,
    batch_size: int = 1024,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    negative_sampler: str = "basic",
    verbose: bool = True,
    dataset_name: Optional[str] = None,
    cache_dir: Optional[Path] = None,
) -> PyKEENWrapper:
    """
    Build and train a KGC model from raw integer triples in a single call.

    This is the shorthand for the ``from_integer_triples`` path.  When using
    KgLoader, prefer ``PyKEENTrainer.from_kg_loader(...)`` instead.

    Example
    -------
    ::

        wrapper = build_kgc_model("rotate", train_triples, E, R,
                                  embed_dim=128, num_epochs=300)
        tails = wrapper.predict_tails(head=0, relation=2, top_k=50)
    """
    return PyKEENTrainer.from_integer_triples(
        model_name=name,
        train_triples=train_triples,
        num_entities=num_entities,
        num_relations=num_relations,
        embed_dim=embed_dim,
        lr=lr,
        device=device,
        model_kwargs=model_kwargs or {},
        negative_sampler=negative_sampler,
        dataset_name=dataset_name,
        cache_dir=cache_dir,
    ).train(num_epochs=num_epochs, batch_size=batch_size, verbose=verbose)
