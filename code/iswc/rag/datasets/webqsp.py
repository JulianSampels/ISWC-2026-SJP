"""
WebQSP dataset loader.

Supports two sources:
  1. HuggingFace Hub: "rmanluo/RoG-webqsp"  (pre-processed, with entity IDs)
  2. Local JSON: original WebQSP release format

The Hub version is preferred because it includes pre-linked entity IDs
that map directly to Freebase MIDs, which our retriever needs.

Schema (Hub version):
    {
      "id":       "WebQTrn-0",
      "question": "what is the name of ...",
      "q_entity": ["m.06w2sn5"],          # topic entity Freebase MIDs
      "a_entity": ["Jazmyn Bieber", ...], # answer strings
    }

Schema (original JSON version):
    {
      "Questions": [{
        "QuestionId": "WebQTrn-0",
        "RawQuestion": "...",
        "QuestionEntities": [{"FreebaseId": "m.06w2sn5"}],
        "Parses": [{"Answers": [{"AnswerArgument": "...", "EntityName": "..."}]}]
      }]
    }
"""
import json
import logging
from pathlib import Path
from typing import List, Optional

from .base import QASample, QADataset

logger = logging.getLogger(__name__)

_HUB_NAME = "rmanluo/RoG-webqsp"


class WebQSPDataset(QADataset):
    """WebQSP question-answer dataset."""

    def __init__(self) -> None:
        self._samples: List[QASample] = []

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

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
            if p.is_dir():
                json_file = p / f"{split}.json"
                self._samples = self._load_hub_json(json_file)
            else:
                self._samples = self._load_original_json(str(p))
        else:
            self._samples = self._load_from_hub(None, split)
        logger.info("WebQSP [%s]: loaded %d samples", split, len(self._samples))

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
        # Normalise answers to lowercase strings
        answers = [str(a).lower().strip() for a in answer_entities]
        return QASample(
            question_id=str(row["id"]),
            question=row["question"],
            topic_entities=topic_entities,
            answers=answers,
            answer_entities=answer_entities,
        )

    def _load_original_json(self, path: str) -> List[QASample]:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        questions = data.get("Questions", data)  # handle both top-level wrappers
        samples = []
        for q in questions:
            qid = q.get("QuestionId", "")
            text = q.get("RawQuestion", q.get("ProcessedQuestion", ""))
            topic_ents = [e["FreebaseId"] for e in q.get("QuestionEntities", [])]
            # Collect answers across all parses (deduplicated)
            seen, answers, answer_ents = set(), [], []
            for parse in q.get("Parses", []):
                for ans in parse.get("Answers", []):
                    arg = ans.get("AnswerArgument", "")
                    name = ans.get("EntityName") or arg
                    key = name.lower().strip()
                    if key not in seen:
                        seen.add(key)
                        answers.append(key)
                        answer_ents.append(arg)
            samples.append(QASample(
                question_id=qid,
                question=text,
                topic_entities=topic_ents,
                answers=answers,
                answer_entities=answer_ents,
            ))
        return samples

    # ------------------------------------------------------------------
    # Sequence interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> QASample:
        return self._samples[idx]
