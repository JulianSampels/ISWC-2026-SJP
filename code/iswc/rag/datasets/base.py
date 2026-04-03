"""
Shared data structures for KGQA datasets.

Both WebQSP and CWQ samples are normalised into QASample so that
pipelines and evaluation code are dataset-agnostic.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator, List, Optional


@dataclass
class QASample:
    """A single question-answer pair from a KGQA benchmark.

    Attributes:
        question_id:      Dataset-specific unique identifier.
        question:         Natural language question string.
        topic_entities:   Freebase / Wikidata entity IDs that anchor the question
                          (already linked; typically one or two per question).
        answers:          Gold answer strings (lowercased, stripped).
        answer_entities:  Gold answer entity IDs (may be empty for literal answers).
        metadata:         Any extra dataset fields kept for debugging.
    """
    question_id: str
    question: str
    topic_entities: List[str]
    answers: List[str]
    answer_entities: List[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def primary_entity(self) -> Optional[str]:
        """Return the first topic entity, or None if the list is empty."""
        return self.topic_entities[0] if self.topic_entities else None


class QADataset(ABC):
    """Abstract base class for KGQA datasets."""

    @abstractmethod
    def load(self, path: Optional[str] = None, split: str = "test") -> None:
        """Load the dataset from *path* (local file or HuggingFace Hub).

        If *path* is None the implementation should download from the Hub.
        """

    @abstractmethod
    def __len__(self) -> int:
        """Number of samples in the loaded split."""

    @abstractmethod
    def __getitem__(self, idx: int) -> QASample:
        """Return the sample at position *idx*."""

    def __iter__(self) -> Iterator[QASample]:
        for i in range(len(self)):
            yield self[i]
