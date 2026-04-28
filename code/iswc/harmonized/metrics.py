"""Unified metric computation for harmonized workflows.

This module computes candidate and ranking metrics in a single format:
- DataFrame columns: metric, value
- Metric names are final names (no extra aggregation/k columns)

Torch is used for fast numeric operations in ranking and tie handling.
"""

from __future__ import annotations

import logging
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd
import torch
from tqdm import tqdm


logger = logging.getLogger(__name__)


RankedPredictions = Dict[int, List[Tuple[int, int, Optional[float]]]]
GoldPairsByHead = Dict[int, Set[Tuple[int, int]]]

METRIC_COLUMNS = ["metric", "value"]


def _metric_row(metric: str, value: float) -> Dict[str, float | str]:
    return {"metric": metric, "value": float(value)}


def _validate_k_values(k_values: Sequence[int]) -> Tuple[int, ...]:
    parsed = tuple(sorted({int(k) for k in k_values if int(k) > 0}))
    if not parsed:
        raise ValueError("k_values must contain at least one positive integer")
    return parsed


def _dedupe_ranked_rows(rows: Iterable[Tuple[int, int, Optional[float]]]) -> List[Tuple[int, int, Optional[float]]]:
    """Keep first occurrence of each (relation, tail) pair to avoid duplicate credit."""
    seen: Set[Tuple[int, int]] = set()
    deduped: List[Tuple[int, int, Optional[float]]] = []
    for relation_id, tail_id, score in rows:
        key = (int(relation_id), int(tail_id))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((key[0], key[1], score))
    return deduped


def _coerce_model_row(item: Tuple, head_id: int) -> Tuple[int, int, Optional[float]]:
    """Normalize model output row formats to (relation_id, tail_id, score)."""
    if not isinstance(item, tuple):
        raise ValueError(f"Expected tuple candidate row, got {type(item)}")

    if len(item) == 4:
        _, relation_id, tail_id, score = item
        return int(relation_id), int(tail_id), float(score)

    if len(item) == 3:
        a, b, c = item
        if isinstance(a, Integral) and int(a) == int(head_id) and isinstance(c, Integral):
            return int(b), int(c), None
        if isinstance(c, Real):
            return int(a), int(b), float(c)
        return int(a), int(b), None

    if len(item) == 2:
        relation_id, tail_id = item
        return int(relation_id), int(tail_id), None

    raise ValueError(f"Unsupported candidate tuple format for head {head_id}: {item}")


def _score_tensor_from_rows(rows: List[Tuple[int, int, Optional[float]]]) -> torch.Tensor:
    """Build score tensor.

    If explicit scores are missing, descending surrogate scores preserve
    incoming row order semantics.
    """
    if not rows:
        return torch.empty(0, dtype=torch.float32)

    has_explicit_score = any(score is not None for _, _, score in rows)
    if has_explicit_score:
        values = [float("-inf") if score is None else float(score) for _, _, score in rows]
    else:
        n = len(rows)
        values = [float(n - i) for i in range(n)]
    return torch.tensor(values, dtype=torch.float32)


def load_ranked_predictions_csv(candidate_file: str | Path) -> RankedPredictions:
    """Load standardized ranked CSV and group predictions by head.

    Sorting:
    - rank column if available
    - otherwise score descending if available
    - otherwise input order
    """
    candidate_path = Path(candidate_file).resolve()
    frame = pd.read_csv(candidate_path)

    required = {"head_id", "relation_id", "tail_id"}
    if not required.issubset(set(frame.columns)):
        raise ValueError(f"Candidate CSV must include {sorted(required)}. Observed: {list(frame.columns)}")

    frame["head_id"] = frame["head_id"].astype(int)
    frame["relation_id"] = frame["relation_id"].astype(int)
    frame["tail_id"] = frame["tail_id"].astype(int)

    has_rank = "rank" in frame.columns and frame["rank"].notna().any()
    has_score = "score" in frame.columns and frame["score"].notna().any()

    if has_rank:
        frame = frame.sort_values(["head_id", "rank"], ascending=[True, True], na_position="last")
    elif has_score:
        frame = frame.sort_values(["head_id", "score"], ascending=[True, False], na_position="last")

    predictions: RankedPredictions = {}
    score_col = frame["score"] if "score" in frame.columns else pd.Series([None] * len(frame), index=frame.index)

    for _, sub in frame.groupby("head_id", sort=False):
        head_id = int(sub["head_id"].iloc[0])
        rows = [
            (int(r), int(t), None if pd.isna(s) else float(s))
            for r, t, s in zip(sub["relation_id"], sub["tail_id"], score_col.loc[sub.index])
        ]
        predictions[head_id] = _dedupe_ranked_rows(rows)

    return predictions


def load_gold_pairs_by_head_csv(gold_file: str | Path) -> GoldPairsByHead:
    """Load standardized gold triples CSV and group (relation, tail) by head."""
    gold_path = Path(gold_file).resolve()
    if gold_path.suffix.lower() != ".csv":
        raise ValueError(
            "Gold triples file must be .csv with columns head_id, relation_id, tail_id "
            "in the same ID space as the evaluated predictions."
        )

    frame = pd.read_csv(gold_path)
    required = {"head_id", "relation_id", "tail_id"}
    if not required.issubset(set(frame.columns)):
        raise ValueError(f"Gold CSV must include {sorted(required)}. Observed: {list(frame.columns)}")

    grouped: GoldPairsByHead = {}
    for h, r, t in zip(frame["head_id"], frame["relation_id"], frame["tail_id"]):
        grouped.setdefault(int(h), set()).add((int(r), int(t)))
    return grouped


def to_gold_pairs_by_head(gold_triples: Iterable[Tuple[int, int, int]]) -> GoldPairsByHead:
    """Compatibility helper for in-memory triples used in docs and runners."""
    grouped: GoldPairsByHead = {}
    for head_id, relation_id, tail_id in gold_triples:
        grouped.setdefault(int(head_id), set()).add((int(relation_id), int(tail_id)))
    return grouped


def evaluate_candidate_metrics(
    predictions: RankedPredictions,
    gold_pairs_by_head: GoldPairsByHead,
    k_values: Sequence[int],
) -> pd.DataFrame:
    """Compute candidate generation metrics.

    These metrics evaluate retrieval set quality per head and globally.
    The returned metric names are final and directly consumable.
    """
    del k_values

    n_heads = 0
    n_triples = 0
    total_candidate_size = 0
    total_hits = 0
    coverage_macro_sum = 0.0
    density_macro_sum = 0.0
    b2fh_sum = 0.0
    b2fh_count = 0
    
    max_candidate_len = max((len(rows) for rows in predictions.values()), default=0)

    for head_id, gold_pairs in gold_pairs_by_head.items():
        n_heads += 1
        n_triples += len(gold_pairs)

        # Defensive dedupe: callers may pass in-memory predictions without prior normalization.
        ranked_rows = _dedupe_ranked_rows(predictions.get(head_id, []))
        total_candidate_size += len(ranked_rows)

        labels = torch.tensor([(r, t) in gold_pairs for r, t, _ in ranked_rows], dtype=torch.bool)
        hit_count = int(labels.sum().item())
        total_hits += hit_count

        coverage_h = float(hit_count) / float(max(len(gold_pairs), 1))
        density_h = float(hit_count) / float(len(ranked_rows)) if ranked_rows else 0.0
        coverage_macro_sum += coverage_h
        density_macro_sum += density_h

        if hit_count > 0:
            first_hit = int(torch.nonzero(labels, as_tuple=False)[0].item()) + 1
            b2fh_sum += float(first_hit)
            b2fh_count += 1
        else:
            b2fh_sum += float(max(len(ranked_rows), max_candidate_len) + 1)

    n_heads_safe = max(n_heads, 1)
    n_triples_f = float(n_triples)
    total_candidate_size_f = float(total_candidate_size)

    rows = [
        _metric_row("total candidate size", total_candidate_size_f),
        _metric_row("average candidate size (head)", total_candidate_size_f / float(n_heads_safe)),
        _metric_row(
            "relative candidate size (triple)",
            total_candidate_size_f / n_triples_f if n_triples_f > 0.0 else 0.0,
        ),
        _metric_row("coverage_macro", coverage_macro_sum / float(n_heads_safe)),
        _metric_row("coverage_micro", float(total_hits) / n_triples_f if n_triples_f > 0.0 else 0.0),
        _metric_row("density_macro", density_macro_sum / float(n_heads_safe)),
        _metric_row("density_micro", float(total_hits) / total_candidate_size_f if total_candidate_size_f > 0.0 else 0.0),
        _metric_row("budget_to_first_hit_macro", b2fh_sum / float(n_heads_safe)),
        _metric_row("budget_to_first_hit_coverage_macro", float(b2fh_count) / float(n_heads_safe)),
        _metric_row("n_heads", float(n_heads)),
        _metric_row("n_triples", n_triples_f),
    ]
    return pd.DataFrame(rows, columns=METRIC_COLUMNS)


def evaluate_ranked_metrics(
    predictions: RankedPredictions,
    gold_pairs_by_head: GoldPairsByHead,
    k_values: Sequence[int],
    filter_pairs_by_head: Optional[GoldPairsByHead] = None,
) -> pd.DataFrame:
    """Compute ranking metrics with filtered realistic rank.

    Filtering:
    - The negative pool excludes all known true facts for the head.

    Realistic rank:
    - For each positive score, ties are handled by midpoint rank:
      (optimistic_rank + pessimistic_rank) / 2.
    """
    parsed_k = _validate_k_values(k_values)
    filter_pairs = gold_pairs_by_head if filter_pairs_by_head is None else filter_pairs_by_head

    candidate_df = evaluate_candidate_metrics(
        predictions=predictions,
        gold_pairs_by_head=gold_pairs_by_head,
        k_values=parsed_k,
    )

    reciprocal_rank_sum = 0.0
    reciprocal_rank_count = 0
    n_heads = 0

    recall_macro_sum = {k: 0.0 for k in parsed_k}
    map_macro_sum = {k: 0.0 for k in parsed_k}
    ndcg_macro_sum = {k: 0.0 for k in parsed_k}
    hits_macro_sum = {k: 0.0 for k in parsed_k}
    hits_micro_sum = {k: 0.0 for k in parsed_k}
    max_k = int(parsed_k[-1])

    for head_id, gold_pairs in gold_pairs_by_head.items():
        n_heads += 1
        reciprocal_rank_count += len(gold_pairs)
        # Defensive dedupe keeps candidate and ranking metrics consistent.
        ranked_rows = _dedupe_ranked_rows(predictions.get(head_id, []))
        if not ranked_rows:
            continue

        scores = _score_tensor_from_rows(ranked_rows)
        known_true_pairs = filter_pairs.get(head_id, set())
        eval_positive_list: List[bool] = []
        known_true_list: List[bool] = []
        for relation_id, tail_id, _ in ranked_rows:
            pair = (relation_id, tail_id)
            eval_positive_list.append(pair in gold_pairs)
            known_true_list.append(pair in known_true_pairs)

        if not any(eval_positive_list):
            continue

        eval_positive_mask = torch.tensor(eval_positive_list, dtype=torch.bool)
        known_true_mask = torch.tensor(known_true_list, dtype=torch.bool)

        negative_scores = scores[~known_true_mask]
        positive_scores = scores[eval_positive_mask]

        if positive_scores.numel() > 0:
            if negative_scores.numel() == 0:
                realistic_rank = torch.ones_like(positive_scores, dtype=torch.float32)
            else:
                # Sort negatives ascending once and use binary search for tie-aware ranks.
                sorted_negatives = torch.sort(negative_scores).values
                n_neg = int(sorted_negatives.numel())

                idx_right = torch.searchsorted(sorted_negatives, positive_scores, right=True)
                idx_left = torch.searchsorted(sorted_negatives, positive_scores, right=False)

                optimistic_rank = (n_neg - idx_right) + 1
                pessimistic_rank = (n_neg - idx_left) + 1
                realistic_rank = 0.5 * (optimistic_rank + pessimistic_rank).to(torch.float32)

            reciprocal_rank_sum += float((1.0 / realistic_rank).sum().item())

        k_eff_max = min(max_k, int(scores.numel()))
        if k_eff_max <= 0:
            continue

        topk_idx = torch.topk(scores, k=k_eff_max, largest=True, sorted=True).indices
        y_topk_all = eval_positive_mask[topk_idx].to(torch.float32)

        total_gold = len(gold_pairs)
        for k in parsed_k:
            k_eff = min(int(k), int(y_topk_all.numel()))
            if k_eff <= 0:
                continue

            y_topk = y_topk_all[:k_eff]
            retrieved_positives = float(y_topk.sum().item())

            recall_h = retrieved_positives / float(max(total_gold, 1))
            recall_macro_sum[k] += recall_h
            
            hits_macro_sum[k] += 1.0 if retrieved_positives > 0.0 else 0.0
            hits_micro_sum[k] += retrieved_positives

            if retrieved_positives > 0.0:
                prefix_hits = torch.cumsum(y_topk, dim=0)
                ranks = torch.arange(1, k_eff + 1, dtype=torch.float32)
                precision_at_i = prefix_hits / ranks
                # Denominator for AP is min(total_gold, k) to properly penalize unretrieved positives
                ap_k_h = float((y_topk * precision_at_i).sum().item()) / float(min(total_gold, int(k)))
            else:
                ap_k_h = 0.0
            map_macro_sum[k] += ap_k_h

            discounts = 1.0 / torch.log2(torch.arange(2, int(k) + 2, dtype=torch.float32))
            dcg = float((y_topk * discounts[:k_eff]).sum().item())
            ideal_len = min(total_gold, int(k))
            if ideal_len > 0:
                idcg = float(discounts[:ideal_len].sum().item())
                ndcg_k_h = dcg / idcg if idcg > 0.0 else 0.0
            else:
                ndcg_k_h = 0.0
            ndcg_macro_sum[k] += ndcg_k_h

    n_heads_safe = max(n_heads, 1)
    rank_rows = [_metric_row("mrr_micro", reciprocal_rank_sum / float(reciprocal_rank_count) if reciprocal_rank_count > 0 else 0.0)]
    for k in parsed_k:
        rank_rows.append(_metric_row(f"hits_macro@{k}", hits_macro_sum[k] / float(n_heads_safe)))
        rank_rows.append(_metric_row(f"hits_micro@{k}", hits_micro_sum[k] / float(reciprocal_rank_count) if reciprocal_rank_count > 0 else 0.0))
        rank_rows.append(_metric_row(f"recall@{k}", recall_macro_sum[k] / float(n_heads_safe)))
        rank_rows.append(_metric_row(f"map@{k}", map_macro_sum[k] / float(n_heads_safe)))
        rank_rows.append(_metric_row(f"ndcg@{k}", ndcg_macro_sum[k] / float(n_heads_safe)))

    rank_df = pd.DataFrame(rank_rows, columns=METRIC_COLUMNS)
    return pd.concat([candidate_df, rank_df], ignore_index=True)


def _call_generate_candidates_batch(
    model,
    heads: List[int],
    max_candidates: Optional[int],
    num_workers: int,
) -> Dict[int, List[Tuple]]:
    """Call model.generate_candidates_batch with graceful kwarg fallback."""
    try:
        return model.generate_candidates_batch(
            heads,
            max_candidates=max_candidates,
            chunk_size=512,
            num_workers=num_workers,
        )
    except TypeError:
        try:
            return model.generate_candidates_batch(heads, max_candidates=max_candidates)
        except TypeError:
            return model.generate_candidates_batch(heads)


def evaluate_model(
    model,
    ground_truth: GoldPairsByHead,
    k_values: Sequence[int] = (1, 5, 10, 50),
    cache_dir: Optional[Path] = None,
    max_candidates: Optional[int] = None,
    batch_size: int = 256,
    num_workers: int = 0,
    filter_pairs_by_head: Optional[GoldPairsByHead] = None,
) -> pd.DataFrame:
    """Incrementally generate candidates for heads and evaluate metrics."""
    del cache_dir
    parsed_k = _validate_k_values(k_values)

    heads = [head_id for head_id, gold in ground_truth.items() if gold]
    predictions: RankedPredictions = {}

    for batch_start in tqdm(range(0, len(heads), int(batch_size)), desc="eval batches", unit="batch"):
        batch = heads[batch_start: batch_start + int(batch_size)]
        raw_results = _call_generate_candidates_batch(
            model=model,
            heads=batch,
            max_candidates=max_candidates,
            num_workers=int(num_workers),
        )
        for head_id in batch:
            rows_raw = raw_results.get(head_id, [])
            rows = _dedupe_ranked_rows(_coerce_model_row(item, head_id) for item in rows_raw)
            predictions[head_id] = rows

    metrics_df = evaluate_ranked_metrics(
        predictions=predictions,
        gold_pairs_by_head=ground_truth,
        k_values=parsed_k,
        filter_pairs_by_head=filter_pairs_by_head,
    )

    logger.info("Evaluated metrics: %s", format_metrics_log(metrics_df))
    return metrics_df


def format_metrics_log(metrics_df: pd.DataFrame, precision: int = 6) -> str:
    """Format all metric rows as a stable key=value log string."""
    if metrics_df.empty:
        return ""

    if "metric" not in metrics_df.columns or "value" not in metrics_df.columns:
        raise ValueError("metrics_df must contain 'metric' and 'value' columns")

    metric_map = {str(m): float(v) for m, v in zip(metrics_df["metric"], metrics_df["value"])}
    parts: List[str] = []

    for metric_name in sorted(metric_map):
        value = metric_map[metric_name]
        if math.isnan(value):
            value_str = "nan"
        elif math.isinf(value):
            value_str = "inf" if value > 0 else "-inf"
        else:
            value_str = f"{value:.{int(precision)}g}"
        parts.append(f"{metric_name}={value_str}")

    return " | ".join(parts)


def format_results_table(
    results: Dict[str, pd.DataFrame],
    k_values: Sequence[int] = (1, 5, 10, 50),
) -> str:
    """Build one markdown comparison table from metric DataFrames."""
    del k_values

    metric_names: Set[str] = set()
    metric_maps: Dict[str, Dict[str, float]] = {}

    for name, df in results.items():
        metric_map = {str(m): float(v) for m, v in zip(df["metric"], df["value"])} if not df.empty else {}
        metric_maps[name] = metric_map
        metric_names.update(metric_map.keys())

    if not metric_maps:
        return ""

    ordered_metrics = sorted(metric_names)
    table_rows: List[Dict[str, float | str]] = []
    for name in sorted(metric_maps):
        row: Dict[str, float | str] = {"Configuration": name}
        metric_map = metric_maps[name]
        for metric_name in ordered_metrics:
            row[metric_name] = float(metric_map.get(metric_name, 0.0))
        table_rows.append(row)

    if not table_rows:
        return ""

    table_df = pd.DataFrame(table_rows, columns=["Configuration", *ordered_metrics])
    return table_df.to_markdown(index=False)


def evaluate_candidate_metrics_from_files(
    candidate_csv: str | Path,
    gold_triples_file: str | Path,
    k_values: Sequence[int],
) -> pd.DataFrame:
    predictions = load_ranked_predictions_csv(candidate_csv)
    gold_pairs_by_head = load_gold_pairs_by_head_csv(gold_triples_file)
    return evaluate_candidate_metrics(predictions, gold_pairs_by_head, k_values)


def evaluate_ranked_metrics_from_files(
    ranked_csv: str | Path,
    gold_triples_file: str | Path,
    k_values: Sequence[int],
) -> pd.DataFrame:
    predictions = load_ranked_predictions_csv(ranked_csv)
    gold_pairs_by_head = load_gold_pairs_by_head_csv(gold_triples_file)
    return evaluate_ranked_metrics(predictions, gold_pairs_by_head, k_values)


def save_metrics_csv(metrics_df: pd.DataFrame, output_file: str | Path) -> Path:
    output_path = Path(output_file).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_path, index=False)
    return output_path
