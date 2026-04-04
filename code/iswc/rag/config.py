"""
Central configuration for the RAG evaluation framework.

All pipeline variants share the same LLM and dataset settings;
only the retrieval strategy differs between native and SJP RAG.
"""
from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class LLMConfig:
    """LLM backend settings."""
    provider: Literal["anthropic", "openai", "mock"] = "anthropic"
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 512
    temperature: float = 0.0
    api_key: Optional[str] = None          # falls back to env var if None


@dataclass
class RetrieverConfig:
    """Shared retrieval settings."""
    budget: int = 10                       # context budget (facts to retrieve)
    # SJP-specific
    sjp_checkpoint: Optional[str] = None  # path to trained SJP model checkpoint
    sjp_dataset_split: Literal["train", "val", "test"] = "test"
    # Native-specific — Wikidata SPARQL endpoint (public)
    sparql_endpoint: str = "https://query.wikidata.org/sparql"
    sparql_timeout: int = 10              # seconds


@dataclass
class EvalConfig:
    """Evaluation loop settings."""
    dataset: Literal["webqsp", "cwq", "both"] = "both"
    split: Literal["train", "val", "test"] = "test"
    # Path to the HuggingFace cached dataset or local JSON file.
    # If None the dataset is downloaded from the Hub automatically.
    webqsp_path: Optional[str] = None
    cwq_path: Optional[str] = None
    max_samples: Optional[int] = None     # None → entire split
    output_dir: str = "results"
    pipelines: list = field(default_factory=lambda: ["native", "sjp"])


@dataclass
class RAGConfig:
    """Top-level config that composes all sub-configs."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
