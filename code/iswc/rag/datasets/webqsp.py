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

from .base import QADataset


class WebQSPDataset(QADataset):
    """WebQSP question-answer dataset."""

    def __init__(self) -> None:
        super().__init__("webqsp")
        self._HUB_NAME = "rmanluo/RoG-webqsp"
