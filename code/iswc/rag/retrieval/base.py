"""
Base types for the retrieval layer.

A retriever's job: given an entity ID (string), return a ranked list
of (head, relation, tail) triples that are relevant to that entity.
The triples are used as structured context by the RAG pipelines.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Triple:
    """A knowledge-graph triple with an optional confidence score.

    Attributes:
        head:     Entity ID or label of the subject.
        relation: Relation label or ID.
        tail:     Entity ID or label of the object.
        score:    Confidence / relevance score assigned by the retriever.
                  Higher = more relevant.  None if the retriever does not score.
    """
    head: str
    relation: str
    tail: str
    score: Optional[float] = None

    def to_text(self) -> str:
        """Human-readable linearisation used as LLM context."""
        return f"({self.head}, {self.relation}, {self.tail})"


class BaseRetriever(ABC):
    """Abstract retriever interface."""

    @abstractmethod
    def retrieve(self, entity_id: str, top_k: int = 10) -> List[Triple]:
        """Return up to *top_k* triples for *entity_id*, ranked by score desc.

        Args:
            entity_id: The head entity for which to retrieve facts.
                       Format depends on the knowledge graph (e.g. Freebase MID
                       like "m.06w2sn5", Wikidata QID like "Q2831", or an
                       integer ID from the SJP vocabulary).
            top_k:     Maximum number of triples to return.

        Returns:
            List of Triple objects, sorted by score descending.
            May be shorter than top_k if fewer triples are available.
        """
