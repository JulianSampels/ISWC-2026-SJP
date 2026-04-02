"""Standardized CSV metric computation for harmonized candidate workflows."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch


RankedPredictions = Dict[int, List[Tuple[int, int, Optional[float]]]]
GoldPairsByHead = Dict[int, Set[Tuple[int, int]]]


def _dedupe_ranked_rows(rows: Iterable[Tuple[int, int, Optional[float]]]) -> List[Tuple[int, int, Optional[float]]]:
    seen: Set[Tuple[int, int]] = set()
    deduped: List[Tuple[int, int, Optional[float]]] = []
    for relation_id, tail_id, score in rows:
        key = (int(relation_id), int(tail_id))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((key[0], key[1], score))
    return deduped


def load_ranked_predictions_csv(candidate_file: str | Path) -> RankedPredictions:
    """Load standardized candidate/ranked CSV into grouped predictions."""
    candidate_path = Path(candidate_file).resolve()
    grouped: Dict[int, List[Tuple[int, int, Optional[float], Optional[int], int]]] = {}

    with candidate_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"head_id", "relation_id", "tail_id"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"Candidate CSV must include columns {sorted(required)}. "
                f"Observed: {reader.fieldnames}"
            )

        for input_index, row in enumerate(reader):
            head_id = int(row["head_id"])
            relation_id = int(row["relation_id"])
            tail_id = int(row["tail_id"])
            score = float(row["score"]) if row.get("score") not in (None, "") else None
            rank = int(row["rank"]) if row.get("rank") not in (None, "") else None
            grouped.setdefault(head_id, []).append((relation_id, tail_id, score, rank, input_index))

    predictions: RankedPredictions = {}
    for head_id, rows in grouped.items():
        has_rank = any(rank is not None for _, _, _, rank, _ in rows)
        has_score = any(score is not None for _, _, score, _, _ in rows)

        if has_rank:
            rows.sort(key=lambda item: (item[3] if item[3] is not None else 10**18, item[4]))
        elif has_score:
            rows.sort(
                key=lambda item: (
                    float(item[2]) if item[2] is not None else float("-inf"),
                    -item[4],
                ),
                reverse=True,
            )
        else:
            rows.sort(key=lambda item: item[4])

        predictions[head_id] = _dedupe_ranked_rows((r, t, s) for r, t, s, _, _ in rows)

    return predictions


def load_gold_triples(gold_file: str | Path) -> List[Tuple[int, int, int]]:
    """Load gold triples from CSV or PT file."""
    gold_path = Path(gold_file).resolve()
    suffix = gold_path.suffix.lower()

    if suffix == ".csv":
        triples: List[Tuple[int, int, int]] = []
        with gold_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"head_id", "relation_id", "tail_id"}
            if not required.issubset(set(reader.fieldnames or [])):
                raise ValueError(
                    f"Gold CSV must include columns {sorted(required)}. "
                    f"Observed: {reader.fieldnames}"
                )
            for row in reader:
                triples.append((int(row["head_id"]), int(row["relation_id"]), int(row["tail_id"])))
        return triples

    if suffix == ".pt":
        payload = torch.load(gold_path, map_location="cpu")
        if isinstance(payload, torch.Tensor):
            if payload.dim() != 2 or payload.size(1) != 3:
                raise ValueError("Gold PT tensor must have shape [N, 3].")
            triples: List[Tuple[int, int, int]] = []
            for row in payload.tolist():
                triples.append((int(row[0]), int(row[1]), int(row[2])))
            return triples

        if isinstance(payload, dict):
            triples_obj = payload.get("triples", payload.get("gold_triples"))
            if triples_obj is None:
                raise ValueError("Gold PT dict must contain 'triples' or 'gold_triples'.")
            if not isinstance(triples_obj, torch.Tensor):
                triples_obj = torch.as_tensor(triples_obj, dtype=torch.long)
            if triples_obj.dim() != 2 or triples_obj.size(1) != 3:
                raise ValueError("Gold PT triples must have shape [N, 3].")
            triples: List[Tuple[int, int, int]] = []
            for row in triples_obj.tolist():
                triples.append((int(row[0]), int(row[1]), int(row[2])))
            return triples

        raise ValueError("Unsupported PT payload for gold triples.")

    raise ValueError("Gold triples file must be .csv or .pt")


def to_gold_pairs_by_head(gold_triples: Iterable[Tuple[int, int, int]]) -> GoldPairsByHead:
    grouped: GoldPairsByHead = {}
    for head_id, relation_id, tail_id in gold_triples:
        grouped.setdefault(int(head_id), set()).add((int(relation_id), int(tail_id)))
    return grouped


def average_candidate_set_size(predictions: RankedPredictions) -> float:
    if not predictions:
        return 0.0
    total = sum(len(rows) for rows in predictions.values())
    return float(total) / float(len(predictions))


def candidate_coverage(predictions: RankedPredictions, gold_pairs_by_head: GoldPairsByHead) -> float:
    if not gold_pairs_by_head:
        return 0.0
    covered = 0
    for head_id, gold_pairs in gold_pairs_by_head.items():
        predicted_pairs = {(r, t) for r, t, _ in predictions.get(head_id, [])}
        if predicted_pairs.intersection(gold_pairs):
            covered += 1
    return float(covered) / float(len(gold_pairs_by_head))


def entity_hit_at_k(predictions: RankedPredictions, gold_pairs_by_head: GoldPairsByHead, k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive.")
    if not gold_pairs_by_head:
        return 0.0

    hits = 0
    for head_id, gold_pairs in gold_pairs_by_head.items():
        ranked = predictions.get(head_id, [])[:k]
        top_pairs = {(r, t) for r, t, _ in ranked}
        if top_pairs.intersection(gold_pairs):
            hits += 1
    return float(hits) / float(len(gold_pairs_by_head))


def entity_recall_at_k(predictions: RankedPredictions, gold_pairs_by_head: GoldPairsByHead, k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive.")
    if not gold_pairs_by_head:
        return 0.0

    total = 0.0
    for head_id, gold_pairs in gold_pairs_by_head.items():
        ranked = predictions.get(head_id, [])[:k]
        top_pairs = {(r, t) for r, t, _ in ranked}
        total += float(len(top_pairs.intersection(gold_pairs))) / float(max(len(gold_pairs), 1))
    return total / float(len(gold_pairs_by_head))


def budget_to_first_hit(predictions: RankedPredictions, gold_pairs_by_head: GoldPairsByHead) -> float:
    budgets: List[int] = []
    for head_id, gold_pairs in gold_pairs_by_head.items():
        ranked = predictions.get(head_id, [])
        first_hit: Optional[int] = None
        for rank_index, (relation_id, tail_id, _) in enumerate(ranked, start=1):
            if (relation_id, tail_id) in gold_pairs:
                first_hit = rank_index
                break
        if first_hit is not None:
            budgets.append(first_hit)

    if not budgets:
        return 0.0
    return float(sum(budgets)) / float(len(budgets))


def _filtered_rank_for_positive(
    ranked_rows: Sequence[Tuple[int, int, Optional[float]]],
    gold_pairs: Set[Tuple[int, int]],
    positive_pair: Tuple[int, int],
) -> Optional[int]:
    filtered_rank = 0
    for relation_id, tail_id, _ in ranked_rows:
        pair = (int(relation_id), int(tail_id))
        if pair in gold_pairs and pair != positive_pair:
            continue
        filtered_rank += 1
        if pair == positive_pair:
            return filtered_rank
    return None


def filtered_mrr(predictions: RankedPredictions, gold_pairs_by_head: GoldPairsByHead) -> float:
    reciprocal_sum = 0.0
    total = 0

    for head_id, gold_pairs in gold_pairs_by_head.items():
        ranked = predictions.get(head_id, [])
        for positive_pair in gold_pairs:
            rank = _filtered_rank_for_positive(ranked, gold_pairs, positive_pair)
            reciprocal_sum += 0.0 if rank is None else 1.0 / float(rank)
            total += 1

    if total == 0:
        return 0.0
    return reciprocal_sum / float(total)


def filtered_hits_at_k(predictions: RankedPredictions, gold_pairs_by_head: GoldPairsByHead, k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive.")

    hits = 0
    total = 0
    for head_id, gold_pairs in gold_pairs_by_head.items():
        ranked = predictions.get(head_id, [])
        for positive_pair in gold_pairs:
            rank = _filtered_rank_for_positive(ranked, gold_pairs, positive_pair)
            if rank is not None and rank <= k:
                hits += 1
            total += 1

    if total == 0:
        return 0.0
    return float(hits) / float(total)


def recall_at_k_per_group(predictions: RankedPredictions, gold_pairs_by_head: GoldPairsByHead, k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive.")
    if not gold_pairs_by_head:
        return 0.0

    total = 0.0
    groups = 0
    for head_id, gold_pairs in gold_pairs_by_head.items():
        if not gold_pairs:
            continue
        ranked = predictions.get(head_id, [])[:k]
        predicted_pairs = {(r, t) for r, t, _ in ranked}
        total += float(len(predicted_pairs.intersection(gold_pairs))) / float(len(gold_pairs))
        groups += 1

    if groups == 0:
        return 0.0
    return total / float(groups)


def recall_at_k_total(predictions: RankedPredictions, gold_pairs_by_head: GoldPairsByHead, k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive.")

    positives_total = sum(len(gold_pairs) for gold_pairs in gold_pairs_by_head.values())
    if positives_total == 0:
        return 0.0

    hits = 0
    for head_id, gold_pairs in gold_pairs_by_head.items():
        ranked = predictions.get(head_id, [])[:k]
        predicted_pairs = {(r, t) for r, t, _ in ranked}
        hits += len(predicted_pairs.intersection(gold_pairs))

    return float(hits) / float(positives_total)


def ndcg(predictions: RankedPredictions, gold_pairs_by_head: GoldPairsByHead) -> float:
    if not gold_pairs_by_head:
        return 0.0

    ndcg_values: List[float] = []
    for head_id, gold_pairs in gold_pairs_by_head.items():
        ranked = predictions.get(head_id, [])
        if not gold_pairs:
            continue

        dcg = 0.0
        for rank_index, (relation_id, tail_id, _) in enumerate(ranked, start=1):
            rel = 1.0 if (relation_id, tail_id) in gold_pairs else 0.0
            if rel > 0.0:
                dcg += rel / math.log2(float(rank_index) + 1.0)

        ideal_len = len(gold_pairs)
        idcg = 0.0
        for rank_index in range(1, ideal_len + 1):
            idcg += 1.0 / math.log2(float(rank_index) + 1.0)

        ndcg_values.append(0.0 if idcg == 0.0 else dcg / idcg)

    if not ndcg_values:
        return 0.0
    return float(sum(ndcg_values)) / float(len(ndcg_values))


def mean_average_precision(predictions: RankedPredictions, gold_pairs_by_head: GoldPairsByHead) -> float:
    if not gold_pairs_by_head:
        return 0.0

    ap_values: List[float] = []
    for head_id, gold_pairs in gold_pairs_by_head.items():
        if not gold_pairs:
            continue

        ranked = predictions.get(head_id, [])
        positive_hits = 0
        precision_sum = 0.0

        for rank_index, (relation_id, tail_id, _) in enumerate(ranked, start=1):
            if (relation_id, tail_id) in gold_pairs:
                positive_hits += 1
                precision_sum += float(positive_hits) / float(rank_index)

        ap_values.append(precision_sum / float(len(gold_pairs)))

    if not ap_values:
        return 0.0
    return float(sum(ap_values)) / float(len(ap_values))


def evaluate_candidate_metrics(
    predictions: RankedPredictions,
    gold_pairs_by_head: GoldPairsByHead,
    k_values: Sequence[int],
) -> Dict[str, float]:
    metrics: Dict[str, float] = {
        "avg_candidate_set_size": average_candidate_set_size(predictions),
        "candidate_coverage": candidate_coverage(predictions, gold_pairs_by_head),
        "budget_to_first_hit": budget_to_first_hit(predictions, gold_pairs_by_head),
    }

    for k in k_values:
        metrics[f"entity_hit@{int(k)}"] = entity_hit_at_k(predictions, gold_pairs_by_head, int(k))
        metrics[f"entity_recall@{int(k)}"] = entity_recall_at_k(predictions, gold_pairs_by_head, int(k))

    return metrics


def evaluate_ranked_metrics(
    predictions: RankedPredictions,
    gold_pairs_by_head: GoldPairsByHead,
    k_values: Sequence[int],
) -> Dict[str, float]:
    metrics: Dict[str, float] = {
        "filtered_mrr": filtered_mrr(predictions, gold_pairs_by_head),
        "ndcg": ndcg(predictions, gold_pairs_by_head),
        "map": mean_average_precision(predictions, gold_pairs_by_head),
        "candidate_coverage": candidate_coverage(predictions, gold_pairs_by_head),
    }

    for k in k_values:
        k_int = int(k)
        metrics[f"filtered_hits@{k_int}"] = filtered_hits_at_k(predictions, gold_pairs_by_head, k_int)
        metrics[f"recall@{k_int}_per_group"] = recall_at_k_per_group(predictions, gold_pairs_by_head, k_int)
        metrics[f"recall@{k_int}_total"] = recall_at_k_total(predictions, gold_pairs_by_head, k_int)

    return metrics


def evaluate_candidate_metrics_from_files(
    candidate_csv: str | Path,
    gold_triples_file: str | Path,
    k_values: Sequence[int],
) -> Dict[str, float]:
    predictions = load_ranked_predictions_csv(candidate_csv)
    gold_pairs_by_head = to_gold_pairs_by_head(load_gold_triples(gold_triples_file))
    return evaluate_candidate_metrics(predictions, gold_pairs_by_head, k_values)


def evaluate_ranked_metrics_from_files(
    ranked_csv: str | Path,
    gold_triples_file: str | Path,
    k_values: Sequence[int],
) -> Dict[str, float]:
    predictions = load_ranked_predictions_csv(ranked_csv)
    gold_pairs_by_head = to_gold_pairs_by_head(load_gold_triples(gold_triples_file))
    return evaluate_ranked_metrics(predictions, gold_pairs_by_head, k_values)


def save_metrics_csv(metrics: Dict[str, float], output_file: str | Path) -> Path:
    output_path = Path(output_file).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for metric_name in sorted(metrics.keys()):
            writer.writerow([metric_name, float(metrics[metric_name])])

    return output_path
