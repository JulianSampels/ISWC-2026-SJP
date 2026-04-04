"""
Native RAG pipeline — embedding-based cosine similarity baseline.

Strategy: given the per-sample KG subgraph (provided by the dataset),
encode the question and each triple as dense vectors and return the
top-k triples by cosine similarity to the question.

This is the correct "naive RAG" baseline:
  - Uses the graph that comes WITH the dataset (rmanluo/RoG-webqsp graph field).
  - No live SPARQL queries; no learned KG-specific scoring.
  - Retrieval is purely semantic similarity via sentence-transformers.

Why this baseline?
  - If SJP RAG outperforms native RAG, the improvement is attributable
    to the SJP learned scoring, not just to the presence of structured context.
"""
import logging

from ..datasets.base import QASample
from ..llm.base import BaseLLM
from ..retrieval.embedding_retriever import EmbeddingRetriever
from .base import BaseRAGPipeline, PipelineResult

logger = logging.getLogger(__name__)


class NativeRAGPipeline(BaseRAGPipeline):
    """Embedding-based KG-RAG baseline.

    Args:
        retriever: An EmbeddingRetriever that scores triples by cosine
                   similarity to the question.
        llm:       LLM reader.
    """

    def __init__(self, retriever: EmbeddingRetriever, llm: BaseLLM) -> None:
        self.retriever = retriever
        self.llm = llm

    @property
    def name(self) -> str:
        return "native_rag"

    def run(self, sample: QASample, top_k: int = 10) -> PipelineResult:
        """Retrieve top-k triples by cosine similarity, then ask the LLM.

        Args:
            sample: KGQA question with a per-sample KG subgraph in sample.graph.
            top_k:  Number of triples to retrieve.

        Returns:
            PipelineResult with LLM-predicted answers.
        """
        if not sample.graph:
            logger.warning("[%s] No graph triples for question '%s'", self.name, sample.question_id)

        triples = self.retriever.retrieve(sample.question, sample.graph, top_k=top_k)

        context = self._format_triples_as_context(triples)
        prompt = self._build_prompt(context, sample.question)

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
