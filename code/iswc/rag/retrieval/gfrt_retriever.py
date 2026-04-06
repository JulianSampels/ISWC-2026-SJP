"""
GFRT fact-retrieval adapter for the RAG pipeline.

Given a topic entity (string ID, e.g. Freebase MID), the retriever:
  1. Maps the string ID to the GFRT integer vocabulary.
  2. Calls GFRTFilter.generate_candidates_batch to get ranked (r, t) candidates.
  3. Translates integer IDs back to string labels and returns Triple objects.

Factory function
----------------
Use ``build_gfrt_retriever()`` to load an existing GFRT checkpoint or train
a new model from scratch, then wrap it as a GFRTFactRetriever::

    retriever = build_gfrt_retriever(
        dataset_name="webqsp",
        model_path="iswc_data/cache/models/gfrt_webqsp.pt",
    )
    triples = retriever.retrieve("m.06w2sn5", budget=50)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

from .base import BaseRetriever, Triple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class GFRTFactRetriever(BaseRetriever):
    """Retriever backed by the GFRT instance-completion model.

    Handles vocabulary translation between string entity IDs (Freebase MIDs,
    Wikidata QIDs, etc.) and the integer IDs used internally by GFRT.

    Args:
        gfrt_filter:   Trained GFRTFilter instance.
        entity_id_map: Dict mapping string entity ID → GFRT integer vocab ID.
        id_to_entity:  Dict mapping GFRT integer ID → entity label/string.
        id_to_relation:Dict mapping GFRT integer ID → relation label/string.
    """

    def __init__(
        self,
        gfrt_filter,
        entity_id_map:  Dict[str, int],
        id_to_entity:   Dict[int, str],
        id_to_relation: Dict[int, str],
    ) -> None:
        self.gfrt_filter   = gfrt_filter
        self.entity_id_map  = entity_id_map
        self.id_to_entity   = id_to_entity
        self.id_to_relation = id_to_relation

    def retrieve(self, entity_id: str, budget: int = 10) -> List[Triple]:
        """Return budget-many GFRT-scored triples for *entity_id*.

        Args:
            entity_id: String entity ID (e.g. Freebase MID "m.06w2sn5").
            budget:    Maximum number of triples to return.

        Returns:
            List of Triple objects sorted by GFRT score descending.
            Returns [] if the entity is not in the GFRT vocabulary.
        """
        int_id = self.entity_id_map.get(entity_id)
        if int_id is None:
            logger.debug("Entity '%s' not in GFRT vocab; returning [].", entity_id)
            return []

        raw = self.gfrt_filter.generate_candidates_batch([int_id], budget)
        candidates = raw.get(int_id, [])

        triples: List[Triple] = []
        for item in candidates[:budget]:
            # item is (h, r, t, score) from GFRTFilter
            h, r, t, score = item[0], item[1], item[2], item[3]
            triples.append(Triple(
                head=self.id_to_entity.get(int(h), str(h)),
                relation=self.id_to_relation.get(int(r), str(r)),
                tail=self.id_to_entity.get(int(t), str(t)),
                score=float(score),
            ))
        return triples  # already sorted by score desc from GFRTFilter


# ---------------------------------------------------------------------------
# Factory: load or train GFRT, return a GFRTFactRetriever
# ---------------------------------------------------------------------------

# def build_gfrt_retrievers(
#     dataset_obj:  Any,
#     model_dir:    str,
#     device:        Optional[str] = None,
#     # GFRT graph-construction hyperparameters (paper Table 2 defaults)
#     embed_dim:     int   = 100,
#     num_layers:    int   = 2,
#     top_k1:        Optional[int] = None,   # None → auto (100 or 20 for UMLS)
#     top_k2:        Optional[int] = None,   # None → auto (30 or 10 for UMLS)
#     margin:        float = 1.0,
#     # Candidate-generation hyperparameters
#     k_r:           int   = 20,    # top-k relations per head
#     k_t:           int   = 100,   # top-k tails per (head, relation)
#     # Training hyperparameters (only used when training from scratch)
#     epochs:        int   = 100,
#     batch_size:    int   = 256,
#     lr_intra:      float = 0.01,
#     lr_inter:      float = 0.001,
#     log_every:     int   = 10,
# ) -> GFRTFactRetriever:
#     """Load an existing GFRT checkpoint or train from scratch.

#     If *model_path* exists the saved weights are loaded.  Otherwise the model
#     is trained on the KG specified by *dataset_name* (via KgLoader) and
#     saved to *model_path*.

#     Args:
#         dataset_name:  KgLoader-compatible dataset name (e.g. "webqsp", "fb15k237").
#         model_path:    Path to save / load the GFRT model checkpoint.
#         device:        Torch device string; defaults to CUDA if available.
#         embed_dim:     Embedding dimensionality.
#         num_layers:    Number of GNN layers.
#         top_k1:        Entities per entity in graph construction (None = auto).
#         top_k2:        Relations per relation in graph construction (None = auto).
#         margin:        Hinge margin for intra-view loss.
#         k_r:           Top-k relations selected per head during retrieval.
#         k_t:           Top-k tail entities per (head, relation) during retrieval.
#         epochs:        Training epochs (ignored when loading from checkpoint).
#         batch_size:    Training batch size.
#         lr_intra:      Learning rate for intra-view GNN parameters.
#         lr_inter:      Learning rate for inter-view alignment.
#         log_every:     Log training loss every N epochs.

#     Returns:
#         GFRTFactRetriever ready for use in a RAG pipeline.
#     """
#     import torch
#     from iswc.gfrt import GFRTModel, GFRTFilter, GFRTTrainer, build_gfrt_pipeline
#     sid_to_retriever = {}
#     for idx, sample in enumerate(dataset_obj):
#         triple_factory = dataset_obj.to_triple_factory(idx)
#         train_triples = triple_factory.mapped_triples
#         num_entities = triple_factory.num_entities
#         num_relations = triple_factory.num_relations

#         if device is None:
#             device = "cuda" if torch.cuda.is_available() else "cpu"
#         torch_device = torch.device(device)

#         # ── Load KG ───────────────────────────────────────────────────────────────
#         top_k1 = 100
#         top_k2 = 30
#         # ── Build graphs (always needed, even when loading from checkpoint) ───────
#         logger.info("Building GFRT graphs (top_k1=%d, top_k2=%d) …", top_k1, top_k2)
#         model, graph_H, graph_T = build_gfrt_pipeline(
#             train_triples=train_triples,
#             num_entities=num_entities,
#             num_relations=num_relations,
#             embed_dim=embed_dim,
#             num_layers=num_layers,
#             top_k1=top_k1,
#             top_k2=top_k2,
#             margin=margin,
#             device=torch_device,
#         )

#         # ── Load or train ─────────────────────────────────────────────────────────

#         model_path = Path(model_dir) / "rag/models" / f"gfrt_{dataset_obj.name}_{sample.question_id.lower()}.pt"

#         if model_path.exists():
#             logger.info("Loading GFRT model from %s", model_path)
#             model = GFRTModel.load(str(model_path), device=torch_device)
#         else:
#             logger.info("Training GFRT for %d epochs (model will be saved to %s) …", epochs, model_path)
#             trainer = GFRTTrainer(
#                 model=model,
#                 graph_H=graph_H,
#                 graph_T=graph_T,
#                 train_triples=train_triples,
#                 device=torch_device,
#                 lr_intra=lr_intra,
#                 lr_inter=lr_inter,
#             )
#             for epoch in range(1, epochs + 1):
#                 losses = trainer.train_epoch(batch_size=batch_size)
#                 if epoch % log_every == 0 or epoch == 1:
#                     logger.info(
#                         "  Epoch %4d/%d  L_H=%.4f  L_T=%.4f  L_cross=%.4f",
#                         epoch, epochs,
#                         losses["loss_H"], losses["loss_T"], losses["loss_cross"],
#                     )
#             model_path.parent.mkdir(parents=True, exist_ok=True)
#             model.save(str(model_path))
#             logger.info("Model saved to %s", model_path)

#         # ── Build filter for inference ─────────────────────────────────────────────
#         inference_trainer = GFRTTrainer(
#             model=model,
#             graph_H=graph_H,
#             graph_T=graph_T,
#             train_triples=train_triples,
#             device=torch_device,
#         )
#         h_emb, rH_emb, t_emb, rT_emb = inference_trainer.get_embeddings()
#         gfrt_filter = GFRTFilter(
#             h_emb=h_emb,
#             rH_emb=rH_emb,
#             t_emb=t_emb,
#             rT_emb=rT_emb,
#             model=model,
#             train_triples=train_triples,
#             top_m_relations=k_r,
#             top_n_tails=k_t,
#         )

#         # ── Build vocabulary maps ─────────────────────────────────────────────────
#         # triple_factory = kg.triple_factory.training
#         entity_id_map = dict(triple_factory.entity_to_id)                              # str → int
#         id_to_entity = {int(k): str(v) for k, v in triple_factory.entity_id_to_label.items()}
#         id_to_relation = {int(k): str(v) for k, v in triple_factory.relation_id_to_label.items()}

#         logger.info(
#             "GFRTFactRetriever ready: %d entities, %d relations, k_r=%d, k_t=%d",
#             len(entity_id_map), len(id_to_relation), k_r, k_t,
#         )
#         retriever = GFRTFactRetriever(
#             gfrt_filter=gfrt_filter,
#             entity_id_map=entity_id_map,
#             id_to_entity=id_to_entity,
#             id_to_relation=id_to_relation,
#         )
#         sid_to_retriever[sample.question_id] = retriever
#         break
#     import ipdb; ipdb.set_trace()
#     return sid_to_retriever
