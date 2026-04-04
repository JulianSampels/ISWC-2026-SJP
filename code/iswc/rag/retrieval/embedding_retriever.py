"""
Embedding-based retriever — naive RAG baseline.

Given a question and a list of KG triples (pre-extracted from the dataset),
encodes both the question and each triple as dense vectors, then returns the
top-k triples ranked by cosine similarity.

Disk cache
----------
Each unique graph subgraph is cached as a separate file:
    iswc_data/cache/rag/emb/<dataset>/graph_<hash>.pkl

The hash is derived from the graph content (order-independent), so the same
subgraph shared by multiple questions is encoded only once. Question encoding
is a single vector and is always computed fresh (fast).

Cache file format:
    {
        'triple_texts': List[str],   # canonical text for each triple
        'embeddings':   np.ndarray,  # shape (N, embed_dim)
    }

Dependencies
------------
    sentence-transformers  (pip install sentence-transformers)

Fallback
--------
If sentence-transformers is not installed, falls back to BM25 token overlap
scoring (rank-bm25 is already in requirements.txt). BM25 is fast; no disk
cache is used in fallback mode.
"""
import hashlib
import logging
import os
import pickle
from typing import List, Optional

import numpy as np

from .base import Triple

logger = logging.getLogger(__name__)


def _graph_hash(graph: List[List[str]]) -> str:
    """Stable, order-independent hash of a graph's triple set.

    Triples are sorted before hashing so two samples with the same subgraph
    but different ordering produce the same key.
    """
    canonical = "\n".join(
        sorted(f"{t[0]}\t{t[1]}\t{t[2]}" for t in graph if len(t) >= 3)
    )
    return hashlib.sha256(canonical.strip().encode()).hexdigest()[:16]


class EmbeddingRetriever:
    """Cosine-similarity retriever over a per-sample KG subgraph.

    Args:
        model_name: HuggingFace sentence-transformers model ID.
        device:     Torch device string ("cpu", "cuda", etc.).
                    None lets sentence-transformers choose automatically.
        cache_dir:  Directory for per-graph embedding cache files.
                    Typically iswc_data/cache/rag/emb/<dataset>.
                    If None, no disk caching is performed.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self._model = None  # lazy-loaded on first encode

        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            logger.info("Embedding cache dir: %s", cache_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        question: str,
        graph: List[List[str]],
        top_k: int = 10,
    ) -> List[Triple]:
        """Return the top-k most relevant triples for *question*.

        Args:
            question: Natural language question string.
            graph:    List of [head, relation, tail] from the dataset subgraph.
            top_k:    Maximum number of triples to return.

        Returns:
            List of Triple objects sorted by cosine similarity descending.
        """
        if not graph:
            return []
        triples = [Triple(head=t[0], relation=t[1], tail=t[2]) for t in graph if len(t) >= 3]
        if not triples:
            return []

        triple_texts = [t.to_text() for t in triples]

        try:
            scores = self._score_with_embeddings(question, triples, triple_texts)
        except Exception as exc:
            logger.warning("Embedding scoring failed (%s); falling back to BM25.", exc)
            scores = self._score_with_bm25(question, triple_texts)

        for triple, score in zip(triples, scores):
            triple.score = float(score)

        return sorted(triples, key=lambda t: t.score or 0.0, reverse=True)[:top_k]

    # ------------------------------------------------------------------
    # Scoring backends
    # ------------------------------------------------------------------

    def _score_with_embeddings(
        self,
        question: str,
        triples: List[Triple],
        triple_texts: List[str],
    ) -> List[float]:
        """Encode question and graph triples, return cosine similarity scores.

        Graph triple embeddings are loaded from / saved to the per-graph cache
        file.  The question is always encoded fresh (single vector, fast).
        """
        t_embs = self._load_or_encode_graph(triples, triple_texts)

        model = self._get_model()
        q_emb = model.encode(question, convert_to_numpy=True, show_progress_bar=False)

        q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-10)
        t_norms = t_embs / (np.linalg.norm(t_embs, axis=1, keepdims=True) + 1e-10)
        return (t_norms @ q_norm).tolist()

    def _load_or_encode_graph(
        self,
        triples: List[Triple],
        triple_texts: List[str],
    ) -> np.ndarray:
        """Return triple embeddings, loading from cache or encoding as needed."""
        # Derive cache file path from graph content hash
        raw_graph = [[t.head, t.relation, t.tail] for t in triples]
        ghash = _graph_hash(raw_graph)
        cache_file = (
            os.path.join(self.cache_dir, f"graph_{ghash}.pkl")
            if self.cache_dir is not None
            else None
        )

        # Try loading from disk
        if cache_file and os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    cached = pickle.load(f)
                logger.debug("Cache hit  graph_%s (%d triples)", ghash, len(triples))
                return cached["embeddings"]
            except Exception as exc:
                logger.warning("Could not read cache file %s (%s); re-encoding.", cache_file, exc)

        # Encode and save
        logger.debug("Cache miss graph_%s — encoding %d triples", ghash, len(triples))
        model = self._get_model()
        embeddings = model.encode(triple_texts, convert_to_numpy=True, show_progress_bar=False)

        if cache_file:
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump({"triple_texts": triple_texts, "embeddings": embeddings}, f)
                logger.debug("Saved graph_%s → %s", ghash, cache_file)
            except Exception as exc:
                logger.warning("Could not write cache file %s: %s", cache_file, exc)

        return embeddings

    def _score_with_bm25(self, question: str, triple_texts: List[str]) -> List[float]:
        """BM25 fallback — no sentence-transformers required."""
        try:
            from rank_bm25 import BM25Okapi  # type: ignore
        except ImportError:
            logger.warning("rank-bm25 not installed; returning uniform scores.")
            return [1.0] * len(triple_texts)

        tokenized_corpus = [t.lower().split() for t in triple_texts]
        bm25 = BM25Okapi(tokenized_corpus)
        return bm25.get_scores(question.lower().split()).tolist()

    # ------------------------------------------------------------------
    # Model loader
    # ------------------------------------------------------------------

    def _get_model(self):
        """Lazy-load the sentence-transformers model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for EmbeddingRetriever. "
                    "Install it with: pip install sentence-transformers\n"
                    f"Original error: {exc}"
                )
            logger.info("Loading embedding model: %s", self.model_name)
            kwargs = {}
            if self.device is not None:
                kwargs["device"] = self.device
            try:
                self._model = SentenceTransformer(self.model_name, local_files_only=True, **kwargs)
            except OSError:
                self._model = SentenceTransformer(self.model_name, **kwargs)
        return self._model
