"""
Shared data structures for KGQA datasets.

Both WebQSP and CWQ samples are normalised into QASample so that
pipelines and evaluation code are dataset-agnostic.
"""
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator, List, Optional
from pathlib import Path
import logging
from typing import List, Optional

from pykeen.triples import TriplesFactory
import numpy as np


logger = logging.getLogger(__name__)


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
    graph: List[List[str]] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def primary_entity(self) -> Optional[str]:
        """Return the first topic entity, or None if the list is empty."""
        return self.topic_entities[0] if self.topic_entities else None



class QADataset(ABC):
    """Abstract base class for KGQA datasets."""
    def __init__(self, name: str) -> None:
        self._samples: List[QASample] = []
        self.name = name

    def load(self, path: Optional[str] = None, split: str = "test") -> None:
        """Load WebQSP.

        Args:
            path:  (a) None → download from HuggingFace Hub.
                   (b) Path to a directory containing {train,val,test}.json
                       (as written by download_qa_datasets.py).
                   (c) Path to a single .json file (original WebQSP release).
            split: One of "train", "val", "test".
        """
        if path is not None:
            p = Path(path)
            json_file = p / f"{split}.json"
            self._samples = self._load_hub_json(json_file)
        else:
            self._samples = self._load_from_hub(None, split)

        logger.info("%s [%s]: loaded %d samples", self.name, split, len(self._samples))

    def _load_hub_json(self, json_file: Path) -> List[QASample]:
        """Load from a local JSON file written by download_qa_datasets.py."""
        with open(json_file, encoding="utf-8") as f:
            rows = json.load(f)
        return [self._hub_row_to_sample(row) for row in rows]

    def _load_from_hub(self, cache_dir: Optional[str], split: str) -> List[QASample]:
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError:
            raise ImportError(
                "The 'datasets' package is required to load from Hub. "
                "Run: pip install datasets"
            )
        hf_split = {"train": "train", "val": "validation", "test": "test"}.get(split, split)
        ds = load_dataset(self._HUB_NAME, split=hf_split, cache_dir=cache_dir)
        return [self._hub_row_to_sample(row) for row in ds]

    @staticmethod
    def _hub_row_to_sample(row: dict) -> QASample:
        topic_entities = row.get("q_entity") or []
        answer_entities = row.get("a_entity") or []
        # Normalise answers to lowercase strings
        answers = [str(a).lower().strip() for a in answer_entities]
        return QASample(
            question_id=str(row["id"]),
            question=row["question"],
            topic_entities=topic_entities,
            answers=answers,
            answer_entities=answer_entities,
            graph=row.get("graph", []),
        )

    def __len__(self) -> int:
        """Number of samples in the loaded split."""
        return len(self._samples)

    def __getitem__(self, idx: int) -> QASample:
        """Return the sample at position *idx*."""
        return self._samples[idx]

    def __iter__(self) -> Iterator[QASample]:
        for i in range(len(self)):
            yield self[i]

    def to_triple_factory(self, idx: int):
        triples = []
        # for sample in self._samples:
        sample = self._samples[idx]
        for triple in sample.graph:
            if len(triple) != 3:
                raise Exception("Invalid triples")
                # if '.' not in triple[1]:
                #     if '#' not in triple[1]:
                #         print(triple)
            h, r, t = triple
            triples.append([str(h), str(r), str(t)])

        logging.info(f"[{self.name}] triples: {len(triples)}")
        triples = np.asarray(triples, dtype=str)
        tf = TriplesFactory.from_labeled_triples(
            triples=triples,
            create_inverse_triples=False,
        )
        return tf
