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


from .base import QADataset


class CWQDataset(QADataset):
    """ComplexWebQuestions (CWQ) dataset."""

    def __init__(self) -> None:
        super().__init__("cwq")
        self._HUB_NAME = "rmanluo/RoG-cwq"
