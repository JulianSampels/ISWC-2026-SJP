"""
Shared evaluation pipeline for KG instance completion methods.
==============================================================

Provides evaluate_predictions() — a parallel, checkpoint-aware evaluation
function that computes entity-centric ranking metrics given pre-built
predictions and ground truth.  Used by run_baselines.py, run_gfrt.py, and
any future method runners (reta, sjp, …).

Metrics (all macro-averaged — each entity weighted equally)
-----------------------------------------------------------
EntityHit@K         fraction of entities with ≥1 gold fact in top-K
EntityRecall@K      fraction of gold facts recovered in top-K per entity
NDCG@K              normalised discounted cumulative gain at K
NDCG                normalised discounted cumulative gain over full list
MAP                 mean average precision
MRR                 mean reciprocal rank over per-entity gold facts
B2FH                mean budget-to-first-hit (entities with ≥1 hit)
B2FH-Coverage       fraction of entities where ≥1 gold fact is found
Coverage@S          EntityHit@S for large S values
AvgCandidateSize    average ranked-list length per entity
CandidateCoverage   fraction of gold facts that appear anywhere in lists
"""
from __future__ import annotations

import json
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import time

from tqdm import tqdm

from iswc.evaluation.entity_metrics import (
    entity_hit_at_k      as _entity_hit_at_k,
    entity_recall_at_k   as _entity_recall_at_k,
    ndcg_at_k            as _ndcg_at_k,
    ndcg                 as _ndcg,
    mean_reciprocal_rank as _mean_reciprocal_rank,
    average_precision    as _average_precision,
    budget_to_first_hit  as _budget_to_first_hit,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-head metric worker
# ---------------------------------------------------------------------------
# Globals are set in the main process before forking so all workers see
# k_values / coverage_sizes via copy-on-write — only head IDs and ranked
# lists travel through the IPC pipe.

_eval_k_values:       Tuple[int, ...] = ()
_eval_coverage_sizes: Tuple[int, ...] = ()


def _eval_one_head(args: Tuple) -> Dict:
    """
    Compute all ranking metrics for a single head entity.

    Accepts (h, ranked, gold) where:
        ranked — List[Tuple[int, int]]  (r, t) pairs in ranked order
        gold   — Set[Tuple[int, int]]   ground-truth (r, t) pairs

    Returns a partial-sums dict that the main process aggregates.
    """
    h, ranked, gold = args
    k_values       = _eval_k_values
    coverage_sizes = _eval_coverage_sizes

    entity_hit    = {k: _entity_hit_at_k(ranked, gold, k)    for k in k_values}
    entity_recall = {k: _entity_recall_at_k(ranked, gold, k) for k in k_values}
    mrr           = _mean_reciprocal_rank(ranked, gold)
    b2fh          = _budget_to_first_hit(ranked, gold)   # int or None
    ndcg_k        = {k: _ndcg_at_k(ranked, gold, k)     for k in k_values}
    ndcg_full     = _ndcg(ranked, gold)
    map_val       = _average_precision(ranked, gold)
    coverage      = {s: _entity_hit_at_k(ranked, gold, s) for s in coverage_sizes}

    ranked_set = set(ranked)
    n_covered  = len(ranked_set & gold)

    return {
        "entity_hit":    entity_hit,
        "entity_recall": entity_recall,
        "ndcg":          ndcg_k,
        "ndcg_full":     ndcg_full,
        "map":           map_val,
        "mrr":           mrr,
        "b2fh":          b2fh,
        "coverage":      coverage,
        "n_triples":     len(gold),
        "n_cands":       len(ranked),
        "n_covered":     n_covered,
    }


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
EVAL_CKPT_NAME = "eval_checkpoint.json"


def _save_eval_checkpoint(path: Path, state: Dict) -> None:
    """Atomically write evaluation state to disk (write-then-rename)."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(path)


def _load_eval_checkpoint(path: Path) -> Optional[Dict]:
    """Return the checkpoint dict, or None if the file does not exist."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        logger.warning(f"  Could not read checkpoint {path} — starting fresh.")
        return None


def _accumulators_to_state(
    batch_start: int,
    entity_hit_sum: Dict, entity_recall_sum: Dict, ndcg_sum: Dict,
    coverage: Dict, mrr_sum: float, map_sum: float,
    ndcg_full_sum: float, b2fh_sum: float, b2fh_count: int,
    n_triples: int, n_heads_evaluated: int, n_cands_sum: int, n_covered_sum: int,
    k_values: Tuple, coverage_sizes: Tuple,
) -> Dict:
    """Serialise accumulator dicts (int keys → str for JSON)."""
    return {
        "batch_start":       batch_start,
        "entity_hit_sum":    {str(k): v for k, v in entity_hit_sum.items()},
        "entity_recall_sum": {str(k): v for k, v in entity_recall_sum.items()},
        "ndcg_sum":          {str(k): v for k, v in ndcg_sum.items()},
        "coverage":          {str(s): v for s, v in coverage.items()},
        "mrr_sum":           mrr_sum,
        "map_sum":           map_sum,
        "ndcg_full_sum":     ndcg_full_sum,
        "b2fh_sum":          b2fh_sum,
        "b2fh_count":        b2fh_count,
        "n_triples":         n_triples,
        "n_heads_evaluated": n_heads_evaluated,
        "n_cands_sum":       n_cands_sum,
        "n_covered_sum":     n_covered_sum,
        "k_values":          list(k_values),
        "coverage_sizes":    list(coverage_sizes),
    }


def _state_to_accumulators(state: Dict, k_values: Tuple, coverage_sizes: Tuple) -> Dict:
    """Restore accumulator dicts from a checkpoint (str keys → int)."""
    return {
        "batch_start":       state["batch_start"],
        "entity_hit_sum":    {int(k): v for k, v in state["entity_hit_sum"].items()},
        "entity_recall_sum": {int(k): v for k, v in state["entity_recall_sum"].items()},
        "ndcg_sum":          {int(k): v for k, v in state["ndcg_sum"].items()},
        "coverage":          {int(s): v for s, v in state["coverage"].items()},
        "mrr_sum":           state["mrr_sum"],
        "map_sum":           state["map_sum"],
        "ndcg_full_sum":     state["ndcg_full_sum"],
        "b2fh_sum":          state["b2fh_sum"],
        "b2fh_count":        state["b2fh_count"],
        "n_triples":         state["n_triples"],
        "n_heads_evaluated": state["n_heads_evaluated"],
        "n_cands_sum":       state.get("n_cands_sum", 0),
        "n_covered_sum":     state.get("n_covered_sum", 0),
    }


# ---------------------------------------------------------------------------
# Core evaluation function (takes pre-built predictions — no CUDA involved)
# ---------------------------------------------------------------------------

# def evaluate_predictions(
#     predictions:     Dict[int, List[Tuple[int, int]]],
#     ground_truth:    Dict[int, Set[Tuple[int, int]]],
#     k_values:        Tuple[int, ...] = (1, 5, 10, 50),
#     coverage_sizes:  Tuple[int, ...] = (10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000),
#     eval_workers:    int             = 32,
#     checkpoint_path: Optional[Path]  = None,
#     batch_size:      int             = 256,
# ) -> Dict[str, float]:
#     """
#     Evaluate ranked predictions against ground truth.

#     All metrics are macro-averaged (each entity weighted equally).
#     The fork pool is created here — callers must NOT hold an active CUDA
#     context when this function is entered (i.e. finish all GPU work first).

#     Parameters
#     ----------
#     predictions:
#         Dict[head_id -> list of (r, t) pairs in ranked order].
#     ground_truth:
#         Dict[head_id -> set of gold (r, t) pairs].
#     k_values:
#         K values to compute entity-centric metrics at.
#     coverage_sizes:
#         S values for Coverage@S metrics (large S, candidate retrieval quality).
#     eval_workers:
#         Number of parallel worker processes.  0 or 1 = sequential.
#     checkpoint_path:
#         Path to checkpoint file.  If provided, state is saved after every batch
#         and the run can be resumed on interruption.  Pass None to disable.
#     batch_size:
#         Heads processed per checkpoint batch.

#     Returns
#     -------
#     Dict with keys:
#         entity_hit@K, entity_recall@K, ndcg@K  — for each K in k_values
#         ndcg, map, mrr
#         b2fh, b2fh_coverage
#         avg_candidate_size, candidate_coverage
#         coverage@S                              — for each S in coverage_sizes
#         n_triples, n_heads
#     """
#     heads_todo = [h for h in ground_truth if ground_truth[h]]

#     # ── Checkpoint: resume if interrupted ────────────────────────────────────
#     ckpt = _load_eval_checkpoint(checkpoint_path) if checkpoint_path is not None else None

#     if ckpt and ckpt.get("k_values") == list(k_values) \
#              and ckpt.get("coverage_sizes") == list(coverage_sizes):
#         restored          = _state_to_accumulators(ckpt, k_values, coverage_sizes)
#         resume_from       = restored["batch_start"]
#         entity_hit_sum    = restored["entity_hit_sum"]
#         entity_recall_sum = restored["entity_recall_sum"]
#         ndcg_sum          = restored["ndcg_sum"]
#         coverage          = restored["coverage"]
#         mrr_sum           = restored["mrr_sum"]
#         map_sum           = restored["map_sum"]
#         ndcg_full_sum     = restored["ndcg_full_sum"]
#         b2fh_sum          = restored["b2fh_sum"]
#         b2fh_count        = restored["b2fh_count"]
#         n_triples         = restored["n_triples"]
#         n_heads_evaluated = restored["n_heads_evaluated"]
#         n_cands_sum       = restored["n_cands_sum"]
#         n_covered_sum     = restored["n_covered_sum"]
#         logger.info(
#             f"  Resuming evaluation from batch {resume_from} "
#             f"({n_heads_evaluated} heads already done)."
#         )
#     else:
#         if ckpt:
#             logger.warning(
#                 "  Checkpoint found but k_values/coverage_sizes differ — starting fresh."
#             )
#         resume_from       = 0
#         entity_hit_sum    = {k: 0.0 for k in k_values}
#         entity_recall_sum = {k: 0.0 for k in k_values}
#         ndcg_sum          = {k: 0.0 for k in k_values}
#         coverage          = {s: 0.0 for s in coverage_sizes}
#         mrr_sum           = 0.0
#         map_sum           = 0.0
#         ndcg_full_sum     = 0.0
#         b2fh_sum          = 0.0
#         b2fh_count        = 0
#         n_triples         = 0
#         n_heads_evaluated = 0
#         n_cands_sum       = 0
#         n_covered_sum     = 0

#     batches = list(range(0, len(heads_todo), batch_size))

#     # Set globals before forking so workers inherit them via copy-on-write.
#     global _eval_k_values, _eval_coverage_sizes
#     _eval_k_values       = k_values
#     _eval_coverage_sizes = coverage_sizes

#     n_proc = min(eval_workers, len(heads_todo)) if eval_workers > 0 else 0
#     pool   = mp.get_context("fork").Pool(processes=n_proc) if n_proc > 1 else None

#     try:
#         start_idx = batches.index(resume_from) if resume_from in batches else 0
#         for batch_start in tqdm(batches, desc="eval batches", unit="batch", initial=start_idx):
#             if batch_start < resume_from:
#                 continue

#             batch     = heads_todo[batch_start : batch_start + batch_size]
#             task_args = [
#                 (h, predictions.get(h, []), ground_truth[h])
#                 for h in batch
#                 if ground_truth.get(h)
#             ]

#             t0 = time.time()
#             if pool is not None:
#                 partial_results = pool.map(_eval_one_head, task_args)
#             else:
#                 partial_results = [_eval_one_head(a) for a in task_args]
#             logger.info(f"  Metrics ready in {time.time()-t0:.1f}s")

#             for pr in partial_results:
#                 for k in k_values:
#                     entity_hit_sum[k]    += pr["entity_hit"][k]
#                     entity_recall_sum[k] += pr["entity_recall"][k]
#                     ndcg_sum[k]          += pr["ndcg"][k]
#                 map_sum       += pr["map"]
#                 mrr_sum       += pr["mrr"]
#                 ndcg_full_sum += pr["ndcg_full"]
#                 if pr["b2fh"] is not None:
#                     b2fh_sum   += pr["b2fh"]
#                     b2fh_count += 1
#                 for s in coverage_sizes:
#                     coverage[s] += pr["coverage"][s]
#                 n_triples         += pr["n_triples"]
#                 n_heads_evaluated += 1
#                 n_cands_sum       += pr["n_cands"]
#                 n_covered_sum     += pr["n_covered"]

#             if checkpoint_path is not None:
#                 _save_eval_checkpoint(checkpoint_path, _accumulators_to_state(
#                     batch_start + batch_size,
#                     entity_hit_sum, entity_recall_sum, ndcg_sum, coverage,
#                     mrr_sum, map_sum, ndcg_full_sum, b2fh_sum, b2fh_count,
#                     n_triples, n_heads_evaluated, n_cands_sum, n_covered_sum,
#                     k_values, coverage_sizes,
#                 ))

#     finally:
#         if pool is not None:
#             pool.close()
#             pool.join()

#     if checkpoint_path is not None and checkpoint_path.exists():
#         checkpoint_path.unlink()
#         logger.info("  Evaluation complete — checkpoint removed.")

#     # ── Aggregate ─────────────────────────────────────────────────────────────
#     n_h = n_heads_evaluated or 1

#     metrics: Dict[str, float] = {}
#     for k in k_values:
#         metrics[f"entity_hit@{k}"]    = entity_hit_sum[k]    / n_h
#         metrics[f"entity_recall@{k}"] = entity_recall_sum[k] / n_h
#         metrics[f"ndcg@{k}"]          = ndcg_sum[k]          / n_h
#     metrics["ndcg"]               = ndcg_full_sum / n_h
#     metrics["map"]                = map_sum        / n_h
#     metrics["mrr"]                = mrr_sum        / n_h
#     metrics["b2fh"]               = b2fh_sum / b2fh_count if b2fh_count > 0 else float("inf")
#     metrics["b2fh_coverage"]      = b2fh_count / n_h
#     metrics["avg_candidate_size"] = n_cands_sum  / n_h
#     metrics["candidate_coverage"] = n_covered_sum / n_triples if n_triples > 0 else 0.0
#     for s in coverage_sizes:
#         metrics[f"coverage@{s}"] = coverage[s] / n_h
#     metrics["n_triples"] = float(n_triples)
#     metrics["n_heads"]   = float(n_heads_evaluated)

#     logger.info(
#         f"Evaluated {n_heads_evaluated} entities. "
#         f"MRR={metrics['mrr']:.4f}, "
#         f"EntityHit@{k_values[-1]}={metrics.get(f'entity_hit@{k_values[-1]}', 0):.4f}, "
#         f"B2FH={metrics['b2fh']:.1f}, "
#         f"CandCov={metrics['candidate_coverage']:.4f}."
#     )
#     return metrics


# ---------------------------------------------------------------------------
# Incremental model-aware evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    ground_truth:    Dict[int, Set[Tuple[int, int]]],
    k_values:        Tuple[int, ...] = (1, 5, 10, 50),
    coverage_sizes:  Tuple[int, ...] = (10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000),
    cache_dir:       Optional[Path]  = None,
    max_candidates:  Optional[int]   = None,
    batch_size:      int             = 256,
    num_workers:     int             = 32,
) -> Dict[str, float]:
    """
    Incremental evaluation: generate candidates and compute metrics batch by
    batch so the full candidate set never has to reside in memory at once.

    The fork pool for metric computation is created once before the loop.
    Candidate generation runs in the main process (GPU), while metric workers
    are CPU-only and never call CUDA — so the two can safely coexist as long
    as generate_candidates_batch does NOT itself spawn additional forked
    workers that use CUDA (pass num_workers=0 to the model call if needed).

    Parameters
    ----------
    model:
        Object whose generate_candidates_batch() is called by default.
        Ignored when generate_fn is provided.
    generate_fn:
        Optional callable ``(batch: List[int]) -> Dict[int, List[Tuple]]``
        that returns raw candidates per head.  Tuples may be (r, t) or
        (h, r, t, score) — both are handled.  Use this to plug in models
        with a different interface (e.g. GFRTFilter).
    cache_dir:
        Directory for checkpoint file.  Pass None to disable checkpointing.
    """
    heads_todo = [h for h in ground_truth if ground_truth[h]]

    # ── Checkpoint ────────────────────────────────────────────────────────────
    ckpt_path: Optional[Path] = None
    if cache_dir is not None:
        cache_dir.mkdir(exist_ok=True, parents=True)
        ckpt_path = cache_dir / EVAL_CKPT_NAME

    ckpt = _load_eval_checkpoint(ckpt_path) if ckpt_path is not None else None

    if ckpt and ckpt.get("k_values") == list(k_values) \
             and ckpt.get("coverage_sizes") == list(coverage_sizes):
        restored          = _state_to_accumulators(ckpt, k_values, coverage_sizes)
        resume_from       = restored["batch_start"]
        entity_hit_sum    = restored["entity_hit_sum"]
        entity_recall_sum = restored["entity_recall_sum"]
        ndcg_sum          = restored["ndcg_sum"]
        coverage          = restored["coverage"]
        mrr_sum           = restored["mrr_sum"]
        map_sum           = restored["map_sum"]
        ndcg_full_sum     = restored["ndcg_full_sum"]
        b2fh_sum          = restored["b2fh_sum"]
        b2fh_count        = restored["b2fh_count"]
        n_triples         = restored["n_triples"]
        n_heads_evaluated = restored["n_heads_evaluated"]
        n_cands_sum       = restored["n_cands_sum"]
        n_covered_sum     = restored["n_covered_sum"]
        logger.info(
            f"  Resuming from batch {resume_from} "
            f"({n_heads_evaluated} heads already done)."
        )
    else:
        if ckpt:
            logger.warning("  Checkpoint k_values/coverage_sizes differ — starting fresh.")
        resume_from       = 0
        entity_hit_sum    = {k: 0.0 for k in k_values}
        entity_recall_sum = {k: 0.0 for k in k_values}
        ndcg_sum          = {k: 0.0 for k in k_values}
        coverage          = {s: 0.0 for s in coverage_sizes}
        mrr_sum           = 0.0
        map_sum           = 0.0
        ndcg_full_sum     = 0.0
        b2fh_sum          = 0.0
        b2fh_count        = 0
        n_triples         = 0
        n_heads_evaluated = 0
        n_cands_sum       = 0
        n_covered_sum     = 0

    batches = list(range(0, len(heads_todo), batch_size))

    # ── Set globals and create pool once (workers are CPU-only) ───────────────
    global _eval_k_values, _eval_coverage_sizes
    _eval_k_values       = k_values
    _eval_coverage_sizes = coverage_sizes

    n_proc = min(num_workers, len(heads_todo)) if num_workers > 0 else 0
    pool   = mp.get_context("fork").Pool(processes=n_proc) if n_proc > 1 else None

    try:
        start_idx = batches.index(resume_from) if resume_from in batches else 0
        for batch_start in tqdm(batches, desc="eval batches", unit="batch", initial=start_idx):
            if batch_start < resume_from:
                continue

            batch = heads_todo[batch_start : batch_start + batch_size]

            # ── Generate candidates for this batch (CUDA, main process) ───────
            t0 = time.time()
            # if generate_fn is not None:
            #     raw_results = generate_fn(batch)
            # else:
            raw_results = model.generate_candidates_batch(
                batch,
                max_candidates=max_candidates,
                chunk_size=512,
                num_workers=0,   # no nested fork to avoid CUDA-after-fork
            )
            logger.info(f"  Candidate generation ready in {time.time()-t0:.1f}s")

            # Convert to (r, t) pairs, deduplicating in ranked order
            batch_predictions: Dict[int, List[Tuple[int, int]]] = {}
            for h, cands in raw_results.items():
                seen: set = set()
                ranked: List[Tuple[int, int]] = []
                for item in cands:
                    r, t = (item[0], item[1]) if len(item) == 2 else (item[1], item[2])
                    if (r, t) not in seen:
                        seen.add((r, t))
                        ranked.append((r, t))
                batch_predictions[h] = ranked

            # ── Evaluate this batch (CPU, pool workers) ────────────────────────
            t0 = time.time()
            task_args = [
                (h, batch_predictions.get(h, []), ground_truth[h])
                for h in batch
                if ground_truth.get(h)
            ]
            if pool is not None:
                partial_results = pool.map(_eval_one_head, task_args)
            else:
                partial_results = [_eval_one_head(a) for a in task_args]
            logger.info(f"  Metrics ready in {time.time()-t0:.1f}s")

            for pr in partial_results:
                for k in k_values:
                    entity_hit_sum[k]    += pr["entity_hit"][k]
                    entity_recall_sum[k] += pr["entity_recall"][k]
                    ndcg_sum[k]          += pr["ndcg"][k]
                map_sum       += pr["map"]
                mrr_sum       += pr["mrr"]
                ndcg_full_sum += pr["ndcg_full"]
                if pr["b2fh"] is not None:
                    b2fh_sum   += pr["b2fh"]
                    b2fh_count += 1
                for s in coverage_sizes:
                    coverage[s] += pr["coverage"][s]
                n_triples         += pr["n_triples"]
                n_heads_evaluated += 1
                n_cands_sum       += pr["n_cands"]
                n_covered_sum     += pr["n_covered"]

            if ckpt_path is not None:
                _save_eval_checkpoint(ckpt_path, _accumulators_to_state(
                    batch_start + batch_size,
                    entity_hit_sum, entity_recall_sum, ndcg_sum, coverage,
                    mrr_sum, map_sum, ndcg_full_sum, b2fh_sum, b2fh_count,
                    n_triples, n_heads_evaluated, n_cands_sum, n_covered_sum,
                    k_values, coverage_sizes,
                ))

    finally:
        if pool is not None:
            pool.close()
            pool.join()

    if ckpt_path is not None and ckpt_path.exists():
        ckpt_path.unlink()
        logger.info("  Evaluation complete — checkpoint removed.")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    n_h = n_heads_evaluated or 1

    metrics: Dict[str, float] = {}
    for k in k_values:
        metrics[f"entity_hit@{k}"]    = entity_hit_sum[k]    / n_h
        metrics[f"entity_recall@{k}"] = entity_recall_sum[k] / n_h
        metrics[f"ndcg@{k}"]          = ndcg_sum[k]          / n_h
    metrics["ndcg"]               = ndcg_full_sum / n_h
    metrics["map"]                = map_sum        / n_h
    metrics["mrr"]                = mrr_sum        / n_h
    metrics["b2fh"]               = b2fh_sum / b2fh_count if b2fh_count > 0 else float("inf")
    metrics["b2fh_coverage"]      = b2fh_count / n_h
    metrics["avg_candidate_size"] = n_cands_sum  / n_h
    metrics["candidate_coverage"] = n_covered_sum / n_triples if n_triples > 0 else 0.0
    for s in coverage_sizes:
        metrics[f"coverage@{s}"] = coverage[s] / n_h
    metrics["n_triples"] = float(n_triples)
    metrics["n_heads"]   = float(n_heads_evaluated)

    logger.info(
        f"Evaluated {n_heads_evaluated} entities. "
        f"MRR={metrics['mrr']:.4f}, "
        f"EntityHit@{k_values[-1]}={metrics.get(f'entity_hit@{k_values[-1]}', 0):.4f}, "
        f"B2FH={metrics['b2fh']:.1f}, "
        f"CandCov={metrics['candidate_coverage']:.4f}."
    )
    return metrics


# ---------------------------------------------------------------------------
# Results table formatter
# ---------------------------------------------------------------------------

def format_results_table(
    results:        Dict[str, Dict[str, float]],
    k_values:       Tuple[int, ...] = (1, 5, 10, 50),
    coverage_sizes:  Tuple[int, ...] = (10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000),
) -> str:
    """
    Format a comparison table for multiple configurations.

    Parameters
    ----------
    results:
        Dict[config_name -> metrics dict] as returned by evaluate_predictions().
    k_values:
        K values used in the evaluation (for column selection).
    coverage_sizes:
        S values to show in the Coverage@S columns.
    """
    show_k   = [k for k in (1, 10, 50) if k in k_values]
    col_keys = (
        [f"entity_hit@{k}"    for k in show_k] +
        [f"entity_recall@{k}" for k in show_k] +
        [f"ndcg@{k}"          for k in show_k] +
        ["ndcg", "map", "mrr", "b2fh", "b2fh_coverage"] +
        [f"coverage@{s}" for s in coverage_sizes] +
        ["n_triples"]
    )
    col_hdrs = (
        [f"EHit@{k}"    for k in show_k] +
        [f"ERecall@{k}" for k in show_k] +
        [f"nDCG@{k}"    for k in show_k] +
        ["nDCG", "MAP", "MRR", "B2FH", "B2FH-Cov"] +
        [f"Cov@{s}"  for s in coverage_sizes] +
        ["Triples"]
    )
    widths = [max(8, len(h) + 1) for h in col_hdrs]
    name_w = max(30, max((len(n) for n in results), default=0) + 2)

    header  = f"{'Configuration':<{name_w}}" + "".join(f"{h:>{w}}" for h, w in zip(col_hdrs, widths))
    divider = "-" * len(header)
    lines   = ["\n=== Instance Completion Results ===", header, divider]

    for name, m in results.items():
        cells = []
        for key, w in zip(col_keys, widths):
            v = m.get(key, 0.0)
            if key == "n_triples":
                cells.append(f"{int(v):>{w}}")
            else:
                val_str = f"{v:.4f}"
                cells.append(f"{val_str:>{w}}")
        lines.append(f"{name:<{name_w}}" + "".join(cells))

    lines.append("")
    return "\n".join(lines)
