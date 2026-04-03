"""
Abstract LLM interface for the RAG evaluation framework.

Both pipelines (native and SJP) use the same LLM; only the context
passed to the LLM differs between them.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class LLMResponse:
    """Output from a single LLM call."""
    text: str               # The generated answer text
    input_tokens: int = 0   # Prompt token count (for cost tracking)
    output_tokens: int = 0  # Completion token count


class BaseLLM(ABC):
    """Abstract language model interface."""

    @abstractmethod
    def generate(self, prompt: str) -> LLMResponse:
        """Generate a response for the given prompt.

        Args:
            prompt: Full prompt string (system context + question).

        Returns:
            LLMResponse with the generated text and token counts.
        """

    def extract_answers(self, response_text: str) -> List[str]:
        """Parse the LLM's response into a list of answer strings.

        Default implementation: split by commas and newlines, strip whitespace.
        Override in subclasses if the LLM follows a specific output format.

        Args:
            response_text: Raw text from the LLM.

        Returns:
            List of candidate answer strings (lowercased, stripped).
        """
        candidates = []
        for part in response_text.replace("\n", ",").split(","):
            cleaned = part.strip().lower()
            if cleaned:
                candidates.append(cleaned)
        return candidates
