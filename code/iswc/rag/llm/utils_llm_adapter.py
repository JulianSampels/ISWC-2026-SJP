"""
utils_llm adapter — wraps utils_llm.generate() as a BaseLLM for the RAG pipelines.

Supported model names are whatever is registered in utils_llm.model_map, e.g.:
  gemini-2.0-flash, deepseek-chat, deepseek-reasoner, gpt-3.5-turbo,
  ollama-deepseek-v3, ollama-qwen2.5-coder-32b, ollama-gptoss-120b, ...

API keys are read from environment variables by utils_llm (DEEPSEEK_API_KEY,
GEMINI_API_KEY, OPENAI_API_KEY, etc.).
"""
import logging
from typing import Optional

from .base import BaseLLM, LLMResponse
from . import utils_llm as _utils_llm

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a factual question-answering assistant. "
    "Answer the question using ONLY the information in the provided context. "
    "Rules:\n"
    "1. If there is one answer, output it as a single short phrase or entity name.\n"
    "2. If there are multiple answers, output each one separated by a comma "
    "(e.g. 'answer one, answer two, answer three').\n"
    "3. Do not output explanations, sentences, or reasoning — answers only.\n"
    "4. Do not add punctuation that is not part of the answer itself.\n"
    "5. If the context does not contain enough information to answer, reply with 'unknown'."
)


class UtilsLLM(BaseLLM):
    """LLM reader backed by utils_llm.generate().

    Args:
        model_name:    Any key from utils_llm.model_map.
        system_prompt: System-level instruction passed to the model.
    """

    def __init__(
        self,
        model_name: str,
        system_prompt: str = _SYSTEM_PROMPT,
    ) -> None:
        if model_name not in _utils_llm.model_map:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Available: {list(_utils_llm.model_map.keys())}"
            )
        self.model_name = model_name
        self.system_prompt = system_prompt

    # ------------------------------------------------------------------
    # BaseLLM interface
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> LLMResponse:
        """Call the model and return a LLMResponse.

        Args:
            prompt: Full human-turn prompt (context + question).

        Returns:
            LLMResponse with generated text (token counts not available).
        """
        ok, text = _utils_llm.generate(
            self.model_name,
            prompt,
            system_prompt=self.system_prompt,
        )
        if not ok:
            logger.error("[%s] generation failed: %s", self.model_name, text)
            return LLMResponse(text="unknown", input_tokens=0, output_tokens=0)
        return LLMResponse(text=text.strip(), input_tokens=0, output_tokens=0)
