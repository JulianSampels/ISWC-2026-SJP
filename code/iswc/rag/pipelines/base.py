"""
Shared data structures and abstract base class for RAG pipelines.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

from ..datasets.base import QASample
from ..retrieval.base import Triple


@dataclass
class PipelineResult:
    """Everything produced by a pipeline for a single QASample.

    Attributes:
        question_id:       Matches QASample.question_id.
        question:          The natural-language question.
        predicted_answers: Answers extracted from the LLM's response.
        gold_answers:      Gold answers from the dataset.
        retrieved_triples: Triples used as context (empty for no-retrieval baseline).
        raw_response:      Verbatim LLM output (useful for debugging).
        context_text:      The context string that was fed to the LLM.
    """
    question_id: str
    question: str
    predicted_answers: List[str]
    gold_answers: List[str]
    retrieved_triples: List[Triple] = field(default_factory=list)
    raw_response: str = ""
    context_text: str = ""


class BaseRAGPipeline(ABC):
    """Abstract base class for RAG pipelines.

    A pipeline combines a retriever (knowledge source) and an LLM reader.
    Subclasses differ only in how they build the context for the LLM.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in result tables and filenames."""

    @abstractmethod
    def run(self, sample: QASample, top_k: int = 10) -> PipelineResult:
        """Process a single QASample and return predictions.

        Args:
            sample: A question-answer sample from a KGQA dataset.
            top_k:  Number of triples / passages to retrieve.

        Returns:
            PipelineResult with predicted answers and supporting evidence.
        """

    # ------------------------------------------------------------------
    # Shared prompt helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_triples_as_context(triples: List[Triple]) -> str:
        """Linearise a list of triples into a numbered list for the LLM prompt."""
        if not triples:
            return "(no context available)"
        lines = [f"{i+1}. {t.to_text()}" for i, t in enumerate(triples)]
        return "\n".join(lines)

    @staticmethod
    def _build_prompt(context: str, question: str) -> str:
        return (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            f"Answer:"
        )
