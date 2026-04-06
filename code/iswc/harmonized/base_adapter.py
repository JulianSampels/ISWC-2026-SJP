"""Shared abstractions and serialization helpers for harmonized adapters."""

from __future__ import annotations

import csv
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch


logger = logging.getLogger(__name__)


RankedPredictions = Dict[int, List[Tuple[int, int, Optional[float]]]]


class CandidateAdapter(ABC):
    """Base class for harmonized dataset/model/candidate/ranking adapters."""

    name: str

    @abstractmethod
    def prepare_dataset(
        self,
        standardized_dataset_dir: str | Path,
        output_dir: str | Path,
        **kwargs,
    ) -> Dict[str, Any]:
        """Convert canonical dataset splits into method-specific input files."""

    @abstractmethod
    def train_candidate_model(self, **kwargs) -> Dict[str, Any]:
        """Train and persist the model used by candidate generation."""

    @abstractmethod
    def generate_candidates(self, **kwargs) -> RankedPredictions:
        """Generate candidates in canonical ranked format."""

    @abstractmethod
    def train_ranking_model(self, **kwargs) -> Dict[str, Any]:
        """Train and persist the model used by final ranking."""

    @abstractmethod
    def rank_candidates(self, **kwargs) -> RankedPredictions:
        """Rank candidates in canonical ranked format."""


def apply_candidate_budget(
    predictions: RankedPredictions,
    candidate_budget: Optional[int],
) -> RankedPredictions:
    """Apply a per-head top-k cap to ranked predictions."""
    if candidate_budget is None:
        return {int(head): list(ranked) for head, ranked in predictions.items()}

    budget = int(candidate_budget)
    if budget <= 0:
        raise ValueError("candidate_budget must be positive.")

    return {int(head): list(ranked[:budget]) for head, ranked in predictions.items()}


def _dedupe_ranked_pairs(
    ranked_items: Iterable[Tuple[int, int, Optional[float]]]
) -> List[Tuple[int, int, Optional[float]]]:
    seen: set[Tuple[int, int]] = set()
    deduped: List[Tuple[int, int, Optional[float]]] = []
    for relation_id, tail_id, score in ranked_items:
        key = (int(relation_id), int(tail_id))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((key[0], key[1], score))
    return deduped


def _group_from_triples_and_scores(
    triples: torch.Tensor,
    scores: Optional[torch.Tensor] = None,
) -> RankedPredictions:
    triples = triples.detach().cpu()
    if triples.dim() != 2 or triples.size(1) != 3:
        raise ValueError("Expected candidate triples tensor with shape [N, 3].")

    grouped: Dict[int, List[Tuple[int, int, Optional[float]]]] = {}

    score_list: Optional[List[float]] = None
    if scores is not None:
        flat_scores = scores.detach().cpu().reshape(-1)
        if flat_scores.numel() != triples.size(0):
            raise ValueError("Scores tensor length must match number of triples.")
        score_list = flat_scores.tolist()

    for idx, triple in enumerate(triples.tolist()):
        h_id, r_id, t_id = map(int, triple)
        score = float(score_list[idx]) if score_list is not None else None
        grouped.setdefault(h_id, []).append((r_id, t_id, score))

    if score_list is not None:
        for head_id in list(grouped.keys()):
            grouped[head_id].sort(
                key=lambda x: float(x[2]) if x[2] is not None else float("-inf"),
                reverse=True,
            )
            grouped[head_id] = _dedupe_ranked_pairs(grouped[head_id])
    else:
        for head_id in list(grouped.keys()):
            grouped[head_id] = _dedupe_ranked_pairs(grouped[head_id])

    return dict(grouped)


def _normalise_prediction_entry(entry: Any, head_id: int) -> Tuple[int, int, Optional[float]]:
    if isinstance(entry, dict):
        relation_raw = entry.get("relation_id", entry.get("r", entry.get("relation")))
        tail_raw = entry.get("tail_id", entry.get("t", entry.get("tail")))
        if relation_raw is None or tail_raw is None:
            raise ValueError(f"Prediction entry for head {head_id} is missing relation/tail fields: {entry}")
        relation_id = int(relation_raw)
        tail_id = int(tail_raw)
        score_raw = entry.get("score")
        score = float(score_raw) if score_raw is not None else None
        return relation_id, tail_id, score

    if isinstance(entry, (list, tuple)):
        if len(entry) == 4:
            return int(entry[1]), int(entry[2]), float(entry[3])
        if len(entry) == 3:
            return int(entry[0]), int(entry[1]), float(entry[2])
        if len(entry) == 2:
            return int(entry[0]), int(entry[1]), None

    raise ValueError(f"Unsupported prediction entry for head {head_id}: {entry}")


def _load_predictions_from_csv(candidate_path: Path) -> RankedPredictions:
    grouped: Dict[int, List[Tuple[int, int, Optional[float], Optional[int]]]] = {}

    with candidate_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"head_id", "relation_id", "tail_id"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"CSV must include columns {sorted(required)}. "
                f"Observed columns: {reader.fieldnames}"
            )

        for row in reader:
            head_id = int(row["head_id"])
            relation_id = int(row["relation_id"])
            tail_id = int(row["tail_id"])
            score = float(row["score"]) if row.get("score") not in (None, "") else None
            rank = int(row["rank"]) if row.get("rank") not in (None, "") else None
            grouped.setdefault(head_id, []).append((relation_id, tail_id, score, rank))

    predictions: RankedPredictions = {}
    for head_id, items in grouped.items():
        has_rank = any(item[3] is not None for item in items)
        has_score = any(item[2] is not None for item in items)

        if has_rank:
            items.sort(key=lambda x: x[3] if x[3] is not None else 10**18)
        elif has_score:
            items.sort(
                key=lambda x: float(x[2]) if x[2] is not None else float("-inf"),
                reverse=True,
            )

        predictions[head_id] = _dedupe_ranked_pairs((r, t, s) for r, t, s, _ in items)

    return predictions


def _load_predictions_from_json(candidate_path: Path) -> RankedPredictions:
    with candidate_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict) and "predictions" in payload:
        payload = payload["predictions"]

    if not isinstance(payload, dict):
        raise ValueError("JSON predictions must be a dict: head_id -> ranked candidates.")

    predictions: RankedPredictions = {}
    for raw_head_id, items in payload.items():
        head_id = int(raw_head_id)
        parsed = [_normalise_prediction_entry(item, head_id) for item in items]
        predictions[head_id] = _dedupe_ranked_pairs(parsed)

    return predictions


def _load_predictions_from_pt(candidate_path: Path) -> RankedPredictions:
    payload = torch.load(candidate_path, map_location="cpu")

    if isinstance(payload, torch.Tensor):
        return _group_from_triples_and_scores(payload, scores=None)

    if isinstance(payload, dict):
        triples = payload.get("triples", payload.get("candidates"))
        scores = payload.get("scores")
        predictions_obj = payload.get("predictions")

        if triples is not None:
            if not isinstance(triples, torch.Tensor):
                triples = torch.as_tensor(triples, dtype=torch.long)
            if scores is not None and not isinstance(scores, torch.Tensor):
                scores = torch.as_tensor(scores, dtype=torch.float32)
            return _group_from_triples_and_scores(triples, scores)

        if predictions_obj is not None:
            if not isinstance(predictions_obj, dict):
                raise ValueError("'predictions' inside PT payload must be a dict.")
            predictions: RankedPredictions = {}
            for raw_head_id, items in predictions_obj.items():
                head_id = int(raw_head_id)
                parsed = [_normalise_prediction_entry(item, head_id) for item in items]
                predictions[head_id] = _dedupe_ranked_pairs(parsed)
            return predictions

    raise ValueError(
        "Unsupported PT candidate format. Expected one of: tensor [N,3], "
        "dict with 'triples' (+ optional 'scores'), or dict with 'predictions'."
    )


def load_ranked_predictions(candidate_file: str | Path) -> RankedPredictions:
    """Load ranked predictions from .pt, .csv or .json."""
    candidate_path = Path(candidate_file).resolve()
    suffix = candidate_path.suffix.lower()

    if suffix == ".pt":
        return _load_predictions_from_pt(candidate_path)
    if suffix == ".csv":
        return _load_predictions_from_csv(candidate_path)
    if suffix == ".json":
        return _load_predictions_from_json(candidate_path)

    raise ValueError(f"Unsupported candidate file extension '{suffix}'.")


def save_ranked_predictions_csv(
    predictions: RankedPredictions,
    output_file: str | Path,
) -> Path:
    output_path = Path(output_file).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["head_id", "relation_id", "tail_id", "score", "rank"])
        for head_id in sorted(predictions.keys()):
            ranked = predictions[head_id]
            for rank, (relation_id, tail_id, score) in enumerate(ranked, start=1):
                writer.writerow([
                    int(head_id),
                    int(relation_id),
                    int(tail_id),
                    "" if score is None else float(score),
                    rank,
                ])

    logger.info("Saved ranked predictions CSV to %s", output_path)
    return output_path


def save_ranked_predictions_pt(
    predictions: RankedPredictions,
    output_file: str | Path,
) -> Path:
    output_path = Path(output_file).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"predictions": predictions}, output_path)
    logger.info("Saved ranked predictions PT to %s", output_path)
    return output_path


def save_ranked_predictions(
    predictions: RankedPredictions,
    output_file: str | Path,
) -> Path:
    """Save ranked predictions as CSV or PT based on output suffix."""
    output_path = Path(output_file).resolve()
    suffix = output_path.suffix.lower()
    if suffix == ".csv":
        return save_ranked_predictions_csv(predictions, output_path)
    if suffix == ".pt":
        return save_ranked_predictions_pt(predictions, output_path)
    raise ValueError("Output file must end with .csv or .pt")
