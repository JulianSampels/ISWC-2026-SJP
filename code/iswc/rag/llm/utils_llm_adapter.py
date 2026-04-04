"""
utils_llm adapter — wraps utils_llm.generate() as a BaseLLM for the RAG pipelines.

Supported model names are whatever is registered in utils_llm.model_map, e.g.:
  gemini-2.0-flash, deepseek-chat, deepseek-reasoner, gpt-3.5-turbo,
  ollama-deepseek-v3, ollama-qwen2.5-coder-32b, ollama-gptoss-120b, ...

API keys are read from environment variables by utils_llm (DEEPSEEK_API_KEY,
GEMINI_API_KEY, OPENAI_API_KEY, etc.).

LLM response cache
------------------
Pass cache_dir to persist responses across runs. Each unique prompt is stored
as one JSON file:
    iswc_data/cache/rag/llm/<model>/<prompt_hash>.json

The hash covers both the system prompt and the user prompt, so changing the
system prompt or the retrieved context (different budget / method) correctly
produces a cache miss. Re-running the same evaluation is then free.
"""
import hashlib
import json
import logging
import os
import re
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


def _prompt_hash(system_prompt: str, prompt: str) -> str:
    """Return a short hash that uniquely identifies a (system, user) prompt pair."""
    content = f"[SYSTEM]\n{system_prompt}\n[USER]\n{prompt}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


class UtilsLLM(BaseLLM):
    """LLM reader backed by utils_llm.generate().

    Args:
        model_name:    Any key from utils_llm.model_map.
        system_prompt: System-level instruction passed to the model.
        cache_dir:     Directory to cache LLM responses on disk.
                       Typically iswc_data/cache/rag/llm/<model>.
                       If None, no caching is performed.
    """

    def __init__(
        self,
        model_name: str,
        system_prompt: str = _SYSTEM_PROMPT,
        cache_dir: Optional[str] = None,
    ) -> None:
        if model_name not in _utils_llm.model_map:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Available: {list(_utils_llm.model_map.keys())}"
            )
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.cache_dir = cache_dir

        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            logger.info("LLM response cache dir: %s", cache_dir)

    # ------------------------------------------------------------------
    # BaseLLM interface
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> LLMResponse:
        """Call the model (or return a cached response) for *prompt*.

        Args:
            prompt: Full human-turn prompt (context + question).

        Returns:
            LLMResponse with generated text.
        """
        # --- cache lookup ---
        cache_file = self._cache_path(prompt)
        if cache_file is not None and os.path.exists(cache_file):
            try:
                with open(cache_file) as f:
                    data = json.load(f)
                logger.debug("LLM cache hit  %s", os.path.basename(cache_file))
                return LLMResponse(text=data["response"], input_tokens=0, output_tokens=0)
            except Exception as exc:
                logger.warning("Could not read LLM cache file %s (%s); re-querying.", cache_file, exc)

        # --- call the model ---
        ok, text = _utils_llm.generate(
            self.model_name,
            prompt,
            system_prompt=self.system_prompt,
        )
        if not ok:
            logger.error("[%s] generation failed: %s", self.model_name, text)
            text = "unknown"
        text = text.strip()

        # --- save to cache ---
        if cache_file is not None:
            try:
                with open(cache_file, "w") as f:
                    json.dump({"response": text}, f, ensure_ascii=False, indent=2)
                logger.debug("LLM cache saved %s", os.path.basename(cache_file))
            except Exception as exc:
                logger.warning("Could not write LLM cache file %s: %s", cache_file, exc)

        return LLMResponse(text=text, input_tokens=0, output_tokens=0)

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_path(self, prompt: str) -> Optional[str]:
        """Return the cache file path for *prompt*, or None if caching is off."""
        if self.cache_dir is None:
            return None
        h = _prompt_hash(self.system_prompt, prompt)
        return os.path.join(self.cache_dir, f"{h}.json")
