"""
SJP-enhanced RAG pipeline — our proposed approach.

Strategy: given the topic entity h, use the trained SJP instance-completion
model to retrieve a ranked list of (h, r, t) candidate facts.  The facts are
sorted by the SJP scoring formula (log-probability under the learned model),
so the LLM receives the most model-confident triples first.

Motivation: unlike 1-hop neighbourhood retrieval, the SJP model predicts
which relations and tail entities are *most likely to be true* for the given
head — effectively performing instance completion before retrieval.  This
produces a smaller, more focused context, reducing noise for the LLM reader.

Comparison with NativeRAGPipeline
----------------------------------
Both pipelines use the same LLM and the same prompt template.
The only difference is how context triples are selected:

  Native RAG:  all 1-hop neighbours (uniform, unscored)
  SJP RAG:     top-k SJP-scored candidates (ranked by model confidence)

A significant Hits@1 / F1 improvement of SJP RAG over Native RAG therefore
demonstrates the value of the instance-completion scoring, not just the
presence of structured KG context.
"""
import logging

from ..datasets.base import QASample
from ..llm.base import BaseLLM
from ..retrieval.base import BaseRetriever
from .base import BaseRAGPipeline, PipelineResult

logger = logging.getLogger(__name__)


class SJPRAGPipeline(BaseRAGPipeline):
    """SJP instance-completion RAG pipeline.

    Args:
        retriever: A SJPFactSuggester (or any BaseRetriever that returns
                   scored triples).  Triples must already be sorted by score
                   descending when returned.
        llm:       LLM reader (e.g. ClaudeLLM).  Should be the same instance
                   used in NativeRAGPipeline for a fair comparison.
    """

    def __init__(self, retriever: BaseRetriever, llm: BaseLLM) -> None:
        self.retriever = retriever
        self.llm = llm

    @property
    def name(self) -> str:
        return "sjp_rag"

    def run(self, sample: QASample, top_k: int = 10) -> PipelineResult:
        """Retrieve SJP-scored facts, then ask the LLM.

        The SJP model is given the topic entity h and returns candidate triples
        (h, r, t) ranked by its log-probability scoring formula.  The top-k
        triples are passed to the LLM as structured context.

        When multiple topic entities are present (rare in WebQSP, more common
        in CWQ), triples are retrieved for each entity and merged; duplicates
        are removed while preserving score ordering.

        Args:
            sample: KGQA question with linked topic entities.
            top_k:  Number of SJP-ranked triples to use as context.

        Returns:
            PipelineResult with LLM-predicted answers.
        """
        # Retrieve SJP-scored triples for every topic entity
        all_triples = []
        seen = set()
        for entity in sample.topic_entities:
            for triple in self.retriever.retrieve(entity, top_k=top_k):
                key = (triple.head, triple.relation, triple.tail)
                if key not in seen:
                    seen.add(key)
                    all_triples.append(triple)

        # Sort merged triples by score descending, clip to top_k
        all_triples.sort(key=lambda t: t.score if t.score is not None else 0.0, reverse=True)
        top_triples = all_triples[:top_k]

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
