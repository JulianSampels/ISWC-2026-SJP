"""
GFRT-enhanced RAG pipeline.

Strategy: given the topic entity h, use the trained GFRT instance-completion
model to retrieve a ranked list of (h, r, t) candidate facts.  The facts are
sorted by GFRT score descending, so the LLM receives the most model-confident
triples first.
"""
import logging
from pathlib import Path
from typing import Dict, Optional
from ..datasets.base import QASample
from ..llm.base import BaseLLM
from ..retrieval.base import BaseRetriever
from ..retrieval.gfrt_retriever import GFRTFactRetriever
from .base import BaseRAGPipeline, PipelineResult

logger = logging.getLogger(__name__)


class GFRTRAGPipeline(BaseRAGPipeline):
    """GFRT instance-completion RAG pipeline.

    Args:
        retriever: A GFRTFactRetriever (or any BaseRetriever that returns
                   scored triples).  Triples must already be sorted by score
                   descending when returned.
        llm:       LLM reader.  Should be the same instance used in
                   NativeRAGPipeline / SJPRAGPipeline for a fair comparison.
    """

    def __init__(self, llm: BaseLLM, model_dir: str) -> None:
        self.model_dir = model_dir
        self.llm = llm

    @property
    def name(self) -> str:
        return "gfrt_rag"

    def run(self, sample: QASample, budget: int = 10) -> PipelineResult:
        """Retrieve GFRT-scored facts, then ask the LLM.

        When multiple topic entities are present, triples are retrieved for
        each entity and merged; duplicates are removed while preserving score
        ordering.

        Args:
            sample: KGQA question with linked topic entities.
            budget:  Number of GFRT-ranked triples to use as context.

        Returns:
            PipelineResult with LLM-predicted answers.
        """
        all_triples = []
        seen = set()
        retriever = build_gfrt_retriever(sample, model_dir=self.model_dir)
        for entity in sample.topic_entities:
            for triple in retriever.retrieve(entity, budget=budget):
                key = (triple.head, triple.relation, triple.tail)
                if key not in seen:
                    seen.add(key)
                    all_triples.append(triple)

        all_triples.sort(key=lambda t: t.score if t.score is not None else 0.0, reverse=True)
        top_triples = all_triples[:budget]

        if not top_triples:
            logger.warning(
                "[%s] No triples retrieved for question '%s' (entities: %s).",
                self.name, sample.question_id, sample.topic_entities,
            )

        context = self._format_triples_as_context(top_triples)
        prompt = self._build_prompt(context, sample.question)
        response = self.llm.generate(prompt)
        predicted = self.llm.extract_answers(response.text)

        return PipelineResult(
            question_id=sample.question_id,
            question=sample.question,
            predicted_answers=predicted,
            gold_answers=sample.answers,
            retrieved_triples=top_triples,
            raw_response=response.text,
            context_text=context,
        )


def to_triple_factory(sample: QASample):
    from pykeen.triples import TriplesFactory
    import numpy as np
    triples = []
    for triple in sample.graph:
        if len(triple) != 3:
            raise Exception("Invalid triples")
            # if '.' not in triple[1]:
            #     if '#' not in triple[1]:
            #         print(triple)
        h, r, t = triple
        triples.append([str(h), str(r), str(t)])

    logging.info(f"triples: {len(triples)}")
    triples = np.asarray(triples, dtype=str)
    tf = TriplesFactory.from_labeled_triples(
        triples=triples,
        create_inverse_triples=False,
    )
    return tf


def build_gfrt_retriever(
    sample: QASample,
    model_dir:    str,
    device:        Optional[str] = None,
    # GFRT graph-construction hyperparameters (paper Table 2 defaults)
    embed_dim:     int   = 100,
    num_layers:    int   = 2,
    top_k1:        Optional[int] = None,   # None → auto (100 or 20 for UMLS)
    top_k2:        Optional[int] = None,   # None → auto (30 or 10 for UMLS)
    margin:        float = 1.0,
    # Candidate-generation hyperparameters
    k_r:           int   = 20,    # top-k relations per head
    k_t:           int   = 100,   # top-k tails per (head, relation)
    # Training hyperparameters (only used when training from scratch)
    epochs:        int   = 100,
    batch_size:    int   = 256,
    lr_intra:      float = 0.01,
    lr_inter:      float = 0.001,
    log_every:     int   = 10,
) -> GFRTFactRetriever:
    """Load an existing GFRT checkpoint or train from scratch.

    If *model_path* exists the saved weights are loaded.  Otherwise the model
    is trained on the KG specified by *dataset_name* (via KgLoader) and
    saved to *model_path*.

    Args:
        dataset_name:  KgLoader-compatible dataset name (e.g. "webqsp", "fb15k237").
        model_dir:     Path to save / load the GFRT model checkpoint.
        device:        Torch device string; defaults to CUDA if available.
        embed_dim:     Embedding dimensionality.
        num_layers:    Number of GNN layers.
        top_k1:        Entities per entity in graph construction (None = auto).
        top_k2:        Relations per relation in graph construction (None = auto).
        margin:        Hinge margin for intra-view loss.
        k_r:           Top-k relations selected per head during retrieval.
        k_t:           Top-k tail entities per (head, relation) during retrieval.
        epochs:        Training epochs (ignored when loading from checkpoint).
        batch_size:    Training batch size.
        lr_intra:      Learning rate for intra-view GNN parameters.
        lr_inter:      Learning rate for inter-view alignment.
        log_every:     Log training loss every N epochs.

    Returns:
        GFRTFactRetriever ready for use in a RAG pipeline.
    """
    import torch
    from iswc.gfrt import GFRTModel, GFRTFilter, GFRTTrainer, build_gfrt_pipeline
    triple_factory = to_triple_factory(sample)
    train_triples = triple_factory.mapped_triples
    num_entities = triple_factory.num_entities
    num_relations = triple_factory.num_relations

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    # ── Load KG ───────────────────────────────────────────────────────────────
    top_k1 = 100
    top_k2 = 30
    # ── Build graphs (always needed, even when loading from checkpoint) ───────
    logger.info("Building GFRT graphs (top_k1=%d, top_k2=%d) …", top_k1, top_k2)
    model, graph_H, graph_T = build_gfrt_pipeline(
        train_triples=train_triples,
        num_entities=num_entities,
        num_relations=num_relations,
        embed_dim=embed_dim,
        num_layers=num_layers,
        top_k1=top_k1,
        top_k2=top_k2,
        margin=margin,
        device=torch_device,
    )

    # ── Load or train ─────────────────────────────────────────────────────────
    model_path = Path(model_dir) / "rag/models" / f"gfrt_r{k_r}_t{k_t}_epochs{epochs}_{sample.question_id.lower()}.pt"
    if model_path.exists():
        logger.info("Loading GFRT model from %s", model_path)
        model = GFRTModel.load(str(model_path), device=torch_device)
    else:
        logger.info("Training GFRT for %d epochs (model will be saved to %s) …", epochs, model_path)
        trainer = GFRTTrainer(
            model=model,
            graph_H=graph_H,
            graph_T=graph_T,
            train_triples=train_triples,
            device=torch_device,
            lr_intra=lr_intra,
            lr_inter=lr_inter,
        )
        for epoch in range(1, epochs + 1):
            losses = trainer.train_epoch(batch_size=batch_size)
            if epoch % log_every == 0 or epoch == 1:
                logger.info(
                    "  Epoch %4d/%d  L_H=%.4f  L_T=%.4f  L_cross=%.4f",
                    epoch, epochs,
                    losses["loss_H"], losses["loss_T"], losses["loss_cross"],
                )
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path))
        logger.info("Model saved to %s", model_path)

    # ── Build filter for inference ─────────────────────────────────────────────
    inference_trainer = GFRTTrainer(
        model=model,
        graph_H=graph_H,
        graph_T=graph_T,
        train_triples=train_triples,
        device=torch_device,
    )
    h_emb, rH_emb, t_emb, rT_emb = inference_trainer.get_embeddings()
    gfrt_filter = GFRTFilter(
        h_emb=h_emb,
        rH_emb=rH_emb,
        t_emb=t_emb,
        rT_emb=rT_emb,
        model=model,
        train_triples=train_triples,
        top_m_relations=k_r,
        top_n_tails=k_t,
    )

    # ── Build vocabulary maps ─────────────────────────────────────────────────
    # triple_factory = kg.triple_factory.training
    entity_id_map = dict(triple_factory.entity_to_id)                              # str → int
    id_to_entity = {int(k): str(v) for k, v in triple_factory.entity_id_to_label.items()}
    id_to_relation = {int(k): str(v) for k, v in triple_factory.relation_id_to_label.items()}

    logger.info(
        "GFRTFactRetriever ready: %d entities, %d relations, k_r=%d, k_t=%d",
        len(entity_id_map), len(id_to_relation), k_r, k_t,
    )
    retriever = GFRTFactRetriever(
        gfrt_filter=gfrt_filter,
        entity_id_map=entity_id_map,
        id_to_entity=id_to_entity,
        id_to_relation=id_to_relation,
    )
    return retriever
