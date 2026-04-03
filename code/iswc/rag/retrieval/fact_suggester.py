"""
SJP fact-suggestion retriever.

This module bridges the SJP instance-completion model with the RAG pipeline.
The trained model's `suggest_facts_for_entity` function is called to retrieve
scored (h, r, t) triples for a given head entity.

Integration path
----------------
The placeholder below returns an empty list.  To activate the real model:

  1. Load a trained PathEModelWrapperUniqueHeads checkpoint.
  2. Load the corresponding UniqueHeadEntityMultiPathDataset for the target split.
  3. Instantiate a BaseCandidateGenerator (e.g. CandidateGeneratorGlobalWithTail).
  4. Construct SJPFactSuggester and pass it to SJPRAGPipeline.

Example::

    from pathe.pathe_trainer import suggest_facts_for_entity
    from pathe.wrappers import PathEModelWrapperUniqueHeads
    from pathe.pathdata import UniqueHeadEntityMultiPathDataset
    from pathe.candidates import CandidateGeneratorGlobalWithTail

    model = PathEModelWrapperUniqueHeads.load_from_checkpoint(ckpt_path)
    model.eval()
    dataset = ...   # load UniqueHeadEntityMultiPathDataset for the target split
    cand_gen = CandidateGeneratorGlobalWithTail(alpha=0.5, beta=0.3, temperature=1.0, ...)
    relation_maps = dataset.relation_maps

    retriever = SJPFactSuggester(
        model=model,
        dataset=dataset,
        candidate_generator=cand_gen,
        relation_maps=relation_maps,
        entity_id_map=entity_str_to_int,   # maps Freebase MID → int vocab ID
        id_to_entity=int_to_entity_str,    # maps int vocab ID → label string
        id_to_relation=int_to_relation_str,
    )
"""
import logging
from typing import Callable, Dict, List, Optional, Tuple

from .base import BaseRetriever, Triple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type aliases matching suggest_facts_for_entity's return type
# ---------------------------------------------------------------------------
# (head_id: int, relation_id: int, tail_id: int, score: float)
_SJPResult = Tuple[int, int, int, float]


class SJPFactSuggester(BaseRetriever):
    """Retriever backed by the SJP instance-completion model.

    The retriever wraps `suggest_facts_for_entity` (pathe_trainer.py) and
    handles the vocabulary translation between string entity IDs (Freebase /
    Wikidata) and the integer IDs used internally by the SJP model.

    When *model* is None the retriever operates in **placeholder mode**:
    it logs a warning and returns an empty list.  This lets the rest of the
    pipeline run end-to-end before the real model is integrated.

    Args:
        model:               Trained PathEModelWrapperUniqueHeads, or None.
        dataset:             UniqueHeadEntityMultiPathDataset for the split.
        candidate_generator: BaseCandidateGenerator instance.
        relation_maps:       RelationMaps for the split.
        entity_id_map:       Dict mapping string entity ID → SJP integer vocab ID.
        id_to_entity:        Dict or callable mapping SJP integer → entity label.
        id_to_relation:      Dict or callable mapping SJP integer → relation label.
        device:              Torch device string (default: auto-detect from model).
    """

    def __init__(
        self,
        model=None,
        dataset=None,
        candidate_generator=None,
        relation_maps=None,
        entity_id_map: Optional[Dict[str, int]] = None,
        id_to_entity: Optional[Dict[int, str]] = None,
        id_to_relation: Optional[Dict[int, str]] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.candidate_generator = candidate_generator
        self.relation_maps = relation_maps
        self.entity_id_map: Dict[str, int] = entity_id_map or {}
        self.id_to_entity: Dict[int, str] = id_to_entity or {}
        self.id_to_relation: Dict[int, str] = id_to_relation or {}
        self.device = device

        self._placeholder_mode = model is None
        if self._placeholder_mode:
            logger.warning(
                "SJPFactSuggester running in placeholder mode (model=None). "
                "All retrieve() calls will return []. "
                "See docstring for integration instructions."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, entity_id: str, top_k: int = 10) -> List[Triple]:
        """Return top-k SJP-scored triples for *entity_id*.

        Args:
            entity_id: String entity ID (e.g. Freebase MID "m.06w2sn5").
            top_k:     Maximum number of triples to return.

        Returns:
            List of Triple objects sorted by SJP score descending.
            Returns [] in placeholder mode or if entity_id is not in vocab.
        """
        if self._placeholder_mode:
            return self._placeholder_retrieve(entity_id, top_k)
        return self._model_retrieve(entity_id, top_k)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _placeholder_retrieve(self, entity_id: str, top_k: int) -> List[Triple]:
        """Return empty list and log a debug message."""
        logger.debug("Placeholder retrieve for entity '%s' (top_k=%d)", entity_id, top_k)
        return []

    def _model_retrieve(self, entity_id: str, top_k: int) -> List[Triple]:
        """Call suggest_facts_for_entity and translate back to string triples."""
        # Late import to avoid requiring SJP dependencies when in placeholder mode
        try:
            from pathe.pathe_trainer import suggest_facts_for_entity  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Could not import suggest_facts_for_entity from pathe. "
                "Make sure the SJP_code/PathE package is on your PYTHONPATH."
            ) from exc

        int_entity_id = self.entity_id_map.get(entity_id)
        if int_entity_id is None:
            logger.warning("Entity '%s' not found in SJP vocabulary; returning [].", entity_id)
            return []

        try:
            raw: List[_SJPResult] = suggest_facts_for_entity(
                entity_id=int_entity_id,
                model=self.model,
                dataset=self.dataset,
                candidate_generator=self.candidate_generator,
                relation_maps=self.relation_maps,
                top_k=top_k,
                device=self.device,
            )
        except ValueError as exc:
            logger.warning("suggest_facts_for_entity raised ValueError: %s", exc)
            return []

        triples = []
        for h_int, r_int, t_int, score in raw:
            triples.append(Triple(
                head=self.id_to_entity.get(h_int, str(h_int)),
                relation=self.id_to_relation.get(r_int, str(r_int)),
                tail=self.id_to_entity.get(t_int, str(t_int)),
                score=score,
            ))
        return triples  # already sorted by score desc from suggest_facts_for_entity
