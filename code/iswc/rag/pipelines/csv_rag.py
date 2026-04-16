"""
CSV-backed RAG pipeline.

Uses our general CSVRetriever (which reads harmonized Candidates.csv files, e.g. from RETA)
and pipes the standardized predictions directly into the language model.
"""
import logging

from ..datasets.base import QASample
from ..llm.base import BaseLLM
from ..retrieval.base import BaseRetriever
from .base import BaseRAGPipeline, PipelineResult

logger = logging.getLogger(__name__)


class CSVRAGPipeline(BaseRAGPipeline):
    """CSV-backed RAG pipeline for use with any harmonized candidate generator.

    Args:
        retriever: A CSVRetriever giving scored triples mapped to natural language.
        llm:       LLM reader (e.g. ClaudeLLM).
        name:      Name identifier for the pipeline output.
    """

    def __init__(self, retriever: BaseRetriever, llm: BaseLLM, name: str = "csv_rag") -> None:
        self.retriever = retriever
        self.llm = llm
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def run(self, sample: QASample, budget: int = 10) -> PipelineResult:
        """Retrieve pre-computed scored triples, then ask the LLM."""
        # Retrieve scored triples for every topic entity
        all_triples = []
        seen = set()
        for entity in sample.topic_entities:
            for triple in self.retriever.retrieve(entity, budget=budget):
                key = (triple.head, triple.relation, triple.tail)
                if key not in seen:
                    seen.add(key)
                    all_triples.append(triple)

        # Sort merged triples by score descending, clip to budget
        all_triples.sort(key=lambda t: t.score if t.score is not None else 0.0, reverse=True)
        top_triples = all_triples[:budget]

        if not top_triples:
            logger.warning(
                "[%s] No triples retrieved for question '%s' (entities: %s).",
                self.name, sample.question_id, sample.topic_entities,
            )

        # Build context and prompt
        context = self._format_triples_as_context(top_triples)
        prompt = self._build_prompt(context, sample.question)

        # Query LLM
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
