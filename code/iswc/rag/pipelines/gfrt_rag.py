"""
GFRT-enhanced RAG pipeline.

Strategy: given the topic entity h, use the trained GFRT instance-completion
model to retrieve a ranked list of (h, r, t) candidate facts.  The facts are
sorted by GFRT score descending, so the LLM receives the most model-confident
triples first.
"""
import logging
from typing import Dict
from ..datasets.base import QASample
from ..llm.base import BaseLLM
from ..retrieval.base import BaseRetriever
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

    def __init__(self, retrievers: Dict[str, BaseRetriever], llm: BaseLLM) -> None:
        self.retrievers = retrievers
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
        retriever = self.retrievers[sample.question_id]
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
