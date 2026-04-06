"""
ComplexWebQuestions (CWQ) dataset loader.

Supports two sources:
  1. HuggingFace Hub: "rmanluo/RoG-cwq"  (pre-processed, with entity IDs)
  2. Local JSON: original CWQ release format

CWQ features compositional questions (conjunction, comparison, superlative,
nesting) derived from WebQSP SPARQL parses.  Each question can have
multiple gold answers.

Schema (Hub version):
    {
      "id":       "WebQTrn-0_0cfb69ae",
      "question": "...",
      "q_entity": ["m.06w2sn5"],
      "a_entity": ["Jazmyn Bieber"],
    }

Schema (original CWQ JSON):
    {
      "ID":       "...",
      "question": "...",
      "answers":  [{"answer_id": "m.xxx", "answer": "..."}],
      "entities": [{"id": "m.xxx", "friendly_name": "..."}]
    }
"""
import json
import logging
from pathlib import Path
from typing import List, Optional

from .base import QASample, QADataset

logger = logging.getLogger(__name__)

_HUB_NAME = "rmanluo/RoG-cwq"


class CWQDataset(QADataset):
    """ComplexWebQuestions (CWQ) dataset."""

    def __init__(self) -> None:
        self._samples: List[QASample] = []

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, path: Optional[str] = None, split: str = "test") -> None:
        """Load CWQ.

        Args:
            path:  (a) None → download from HuggingFace Hub.
                   (b) Path to a directory containing {train,val,test}.json
                       (as written by download_qa_datasets.py).
                   (c) Path to a single .json file (original CWQ release).
            split: One of "train", "val", "test".
        """
        if path is not None:
            p = Path(path)
            json_file = p / f"{split}.json"
            self._samples = self._load_hub_json(json_file)
        else:
            self._samples = self._load_from_hub(None, split)
        logger.info("CWQ [%s]: loaded %d samples", split, len(self._samples))

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
        ds = load_dataset(_HUB_NAME, split=hf_split, cache_dir=cache_dir)
        return [self._hub_row_to_sample(row) for row in ds]

    @staticmethod
    def _hub_row_to_sample(row: dict) -> QASample:
        topic_entities = row.get("q_entity") or []
        answer_entities = row.get("a_entity") or []
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
        return len(self._samples)

    def __getitem__(self, idx: int) -> QASample:
        return self._samples[idx]
