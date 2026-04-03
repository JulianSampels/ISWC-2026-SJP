"""
Native RAG pipeline — baseline.

Strategy: given the topic entity, retrieve its direct 1-hop neighbours
from Wikidata (no learned scoring), then prompt the LLM with those triples
as structured context.

This is the "vanilla KG-RAG" baseline: same LLM, same number of triples,
but retrieval is done by simple graph traversal rather than the SJP
instance-completion model.

Why this baseline?
  - It isolates the contribution of the SJP scoring / ranking from the
    gain of simply having KG triples in the prompt.
  - If SJP RAG outperforms native RAG, the improvement is attributable
    to better retrieval (higher-quality, more relevant triples), not just
    to the presence of structured context.
"""
import logging

from ..datasets.base import QASample
from ..llm.base import BaseLLM
from ..retrieval.base import BaseRetriever
from .base import BaseRAGPipeline, PipelineResult

logger = logging.getLogger(__name__)


class NativeRAGPipeline(BaseRAGPipeline):
    """1-hop KG-neighbour RAG baseline.

    Args:
        retriever: A BaseRetriever that fetches triples for an entity.
                   Typically NativeKGRetriever (Wikidata 1-hop).
        llm:       LLM reader (e.g. ClaudeLLM).
    """

    def __init__(self, retriever: BaseRetriever, llm: BaseLLM) -> None:
        self.retriever = retriever
        self.llm = llm

    @property
    def name(self) -> str:
        return "native_rag"

    def run(self, sample: QASample, top_k: int = 10) -> PipelineResult:
        """Retrieve 1-hop Wikidata triples, then ask the LLM.

        Args:
            sample: KGQA question with linked topic entities.
            top_k:  Number of triples to retrieve per entity.

        Returns:
            PipelineResult with LLM-predicted answers.
        """
        entity = sample.primary_entity()
        if entity is None:
            logger.warning("[%s] No topic entity for question '%s'", self.name, sample.question_id)

        # Retrieve triples (empty list if entity is None or retrieval fails)
        triples = self.retriever.retrieve(entity, top_k=top_k) if entity else []

        # Build context and prompt
        context = self._format_triples_as_context(triples)
        prompt = self._build_prompt(context, sample.question)

        # Query LLM
        response = self.llm.generate(prompt)
        predicted = self.llm.extract_answers(response.text)

        return PipelineResult(
            question_id=sample.question_id,
            question=sample.question,
            predicted_answers=predicted,
            gold_answers=sample.answers,
            retrieved_triples=triples,
            raw_response=response.text,
            context_text=context,
        )
