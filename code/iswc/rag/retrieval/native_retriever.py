"""
Native KG retriever — baseline for fair comparison with SJP RAG.

Retrieves 1-hop triples for a topic entity from the Wikidata public SPARQL
endpoint.  No learned scoring is applied; triples are ranked by a simple
heuristic (relation frequency in the returned subgraph).

This represents the straightforward "give the LLM the entity's direct
neighbourhood" baseline — no instance-completion model involved.

Fallback behaviour
------------------
If the SPARQL query fails (network error, timeout, unknown QID) the retriever
returns an empty list rather than crashing, so evaluation can continue.

Entity ID format
----------------
WebQSP and CWQ use Freebase MIDs (e.g. "m.06w2sn5").  Wikidata uses QIDs
(e.g. "Q2831").  The retriever accepts either format:
  - Freebase MID  → converted to Wikidata QID via Wikidata SPARQL lookup.
  - Wikidata QID  → used directly.
"""
import logging
import time
from typing import Dict, List, Optional

import requests

from .base import BaseRetriever, Triple

logger = logging.getLogger(__name__)

_WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
_USER_AGENT = "SJP-RAG-Evaluation/1.0 (research; contact via GitHub)"
_FREEBASE_TO_QID_QUERY = """
SELECT ?item WHERE {{
  ?item wdt:P646 "{freebase_id}" .
}} LIMIT 1
"""
_NEIGHBOUR_QUERY = """
SELECT ?relLabel ?tailLabel WHERE {{
  wd:{qid} ?rel ?tail .
  ?tail a wikibase:Item .
  SERVICE wikibase:label {{
    bd:serviceParam wikibase:language "en" .
    ?rel rdfs:label ?relLabel .
    ?tail rdfs:label ?tailLabel .
  }}
  FILTER(LANG(?relLabel) = "en")
  FILTER(LANG(?tailLabel) = "en")
}} LIMIT {limit}
"""


class NativeKGRetriever(BaseRetriever):
    """1-hop Wikidata neighbour retriever (no learned scoring).

    This is the **native RAG baseline**.  Given a topic entity, it fetches
    all direct (subject, predicate, object) triples from Wikidata and returns
    the top-k by a simple heuristic score (currently: uniform = 1.0, so
    ordering reflects Wikidata's return order).

    Args:
        sparql_endpoint: URL of the SPARQL endpoint (default: Wikidata).
        timeout:         HTTP timeout in seconds.
        cache:           If True, cache SPARQL results in memory to avoid
                         duplicate network calls for the same entity.
    """

    def __init__(
        self,
        sparql_endpoint: str = _WIKIDATA_SPARQL,
        timeout: int = 10,
        cache: bool = True,
    ) -> None:
        self.endpoint = sparql_endpoint
        self.timeout = timeout
        self._cache: Dict[str, List[Triple]] = {} if cache else None  # type: ignore

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, entity_id: str, top_k: int = 10) -> List[Triple]:
        """Return up to *top_k* 1-hop Wikidata triples for *entity_id*.

        Args:
            entity_id: Freebase MID (e.g. "m.06w2sn5") or Wikidata QID
                       (e.g. "Q2831").
            top_k:     Maximum number of triples.

        Returns:
            List of Triple objects (score=1.0 for all; no learned ranking).
        """
        cache_key = f"{entity_id}:{top_k}"
        if self._cache is not None and cache_key in self._cache:
            return self._cache[cache_key]

        qid = self._resolve_qid(entity_id)
        if qid is None:
            logger.warning("Could not resolve entity '%s' to a Wikidata QID.", entity_id)
            return []

        triples = self._fetch_neighbours(qid, limit=top_k)

        if self._cache is not None:
            self._cache[cache_key] = triples
        return triples

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_qid(self, entity_id: str) -> Optional[str]:
        """Translate entity_id to a Wikidata QID.

        Freebase MIDs start with "m." or "g."; Wikidata QIDs start with "Q".
        """
        entity_id = entity_id.strip()
        if entity_id.startswith("Q") and entity_id[1:].isdigit():
            return entity_id  # already a QID

        if entity_id.startswith("/"):
            # "/m/06w2sn5" → "m.06w2sn5"
            entity_id = entity_id.lstrip("/").replace("/", ".")

        # Try Freebase MID lookup via Wikidata P646
        freebase_id = "/" + entity_id.replace(".", "/")
        query = _FREEBASE_TO_QID_QUERY.format(freebase_id=freebase_id)
        rows = self._sparql(query)
        if rows:
            qid_uri = rows[0]["item"]["value"]  # e.g. "http://www.wikidata.org/entity/Q2831"
            return qid_uri.rsplit("/", 1)[-1]
        return None

    def _fetch_neighbours(self, qid: str, limit: int) -> List[Triple]:
        query = _NEIGHBOUR_QUERY.format(qid=qid, limit=limit)
        rows = self._sparql(query)
        triples = []
        for row in rows:
            rel_label = row.get("relLabel", {}).get("value", "")
            tail_label = row.get("tailLabel", {}).get("value", "")
            if rel_label and tail_label:
                triples.append(Triple(
                    head=qid,
                    relation=rel_label,
                    tail=tail_label,
                    score=1.0,   # uniform; no learned ranking
                ))
        return triples

    def _sparql(self, query: str) -> list:
        """Execute a SPARQL query and return the result rows as dicts."""
        headers = {
            "Accept": "application/sparql-results+json",
            "User-Agent": _USER_AGENT,
        }
        try:
            resp = requests.get(
                self.endpoint,
                params={"query": query, "format": "json"},
                headers=headers,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()["results"]["bindings"]
        except Exception as exc:
            logger.warning("SPARQL query failed: %s", exc)
            return []
