"""
Baseline Experiment Runner — Instance Completion on KG Benchmarks
=================================================================

Runs all combinations of relation predictor × KGC model and reports
ranking metrics on the test split.

Evaluation
----------
For each head entity h in the test set:
    - Ground truth: all (r, t) pairs in test_triples with that head
    - Pipeline generates top-N ranked (h, r, t) candidates

Metrics reported (all macro-averaged — each entity weighted equally):
    EntityHit@K    — fraction of entities with ≥1 gold fact in top-K
    EntityRecall@K — fraction of gold facts recovered in top-K per entity
    NDCG@K         — normalised discounted cumulative gain at K
    NDCG           — normalised discounted cumulative gain over full ranked list
    MAP            — mean average precision
    MRR            — mean reciprocal rank over per-entity gold facts
    B2FH           — mean budget-to-first-hit (over entities with ≥1 hit)
    B2FH-Coverage  — fraction of entities where any gold fact is found
    Coverage@S     — EntityHit@S for large S (candidate retrieval quality)

Usage
-----
::

    cd code/
    python -m iswc.baselines.run_baselines \\
        --dataset fb15k237 \\
        --rel-predictors bpr recoin \\
        --kgc-models transe transh \\
        --k-r 10 --k-t 50 --embed-dim 128 --epochs 200

    # Quick smoke-test (few entities, few epochs)
    python -m iswc.baselines.run_baselines \\
        --dataset fb15k237 \\
        --rel-predictors bpr \\
        --kgc-models transe \\
        --epochs 5 --test-limit 50
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import hashlib
import multiprocessing as mp
import pickle
import sys
import time
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from tqdm import tqdm
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

import torch

# ---------------------------------------------------------------------------
# Path setup — ensure the repo's code/ directory is importable
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_CODE_DIR = _HERE.parents[2]          # code/iswc/baselines/ → code/
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

_SJP_DIR = _CODE_DIR / "SJP_code" / "PathE"
if str(_SJP_DIR) not in sys.path:
    sys.path.insert(0, str(_SJP_DIR))

from iswc.baselines import (
    BPRRelationPredictor,
    RecoinRelationPredictor,
    PyKEENTrainer,
    InstanceCompletionPipeline,
)
from iswc.evaluation.entity_metrics import (
    entity_hit_at_k      as _entity_hit_at_k,
    entity_recall_at_k   as _entity_recall_at_k,
    ndcg_at_k            as _ndcg_at_k,
    ndcg                 as _ndcg,
    mean_reciprocal_rank as _mean_reciprocal_rank,
    average_precision    as _average_precision,
    budget_to_first_hit  as _budget_to_first_hit,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _hash_heads(heads: List[int]) -> str:
    """Stable, order-independent hash of a graph's triple set.

    Triples are sorted before hashing so two samples with the same subgraph
    but different ordering produce the same key.
    """
    heads = sorted(heads)
    heads = [str(x) for x in heads]
    canonical = ','.join(heads)
    return hashlib.sha256(canonical.strip().encode()).hexdigest()[:16]


SCHEMA = pa.schema([
    ("key", pa.int64()),
    (
        "values",
        pa.list_(
            pa.struct([
                ("x", pa.int64()),
                ("y", pa.int64()),
                ("z", pa.int64()),
                ("score", pa.float64()),
            ])
        ),
    ),
])


def save_candidates(candidates: dict[int, list[(int, int, int, float)]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    keys = list(candidates.keys())
    values = [
        [{"x": a, "y": b, "z": c, "score": d} for (a, b, c, d) in records]
        for records in candidates.values()
    ]

    table = pa.table(
        {
            "key": pa.array(keys, type=pa.int64()),
            "values": pa.array(
                values,
                type=pa.list_(
                    pa.struct([
                        ("x", pa.int64()),
                        ("y", pa.int64()),
                        ("z", pa.int64()),
                        ("score", pa.float64()),
                    ])
                ),
            ),
        },
        schema=SCHEMA,
    )

    pq.write_table(table, path, compression="zstd")


# from pathlib import Path
# import pandas as pd
def load_candidates(path: Path) -> Dict[int, List[(int, int, int, float)]]:
    table = pq.read_table(path, columns=["key", "values"])
    d = table.to_pylist()

    return {
        row["key"]: [
            (item["x"], item["y"], item["z"], item["score"])
            for item in row["values"]
        ]
        for row in d
    }

# ---------------------------------------------------------------------------
# Evaluation checkpoint helpers
# ---------------------------------------------------------------------------
_EVAL_CKPT_NAME = "eval_checkpoint.json"


def _save_eval_checkpoint(path: Path, state: Dict) -> None:
    """Atomically write evaluation state to disk (write-then-rename)."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(path)               # atomic on POSIX; safe on Windows Python ≥3.3


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
    n_triples: int, n_heads_evaluated: int,
    k_values: Tuple, coverage_sizes: Tuple,
) -> Dict:
    """Serialise accumulator dicts (int keys → str for JSON)."""
    return {
        "batch_start":        batch_start,
        "entity_hit_sum":     {str(k): v for k, v in entity_hit_sum.items()},
        "entity_recall_sum":  {str(k): v for k, v in entity_recall_sum.items()},
        "ndcg_sum":           {str(k): v for k, v in ndcg_sum.items()},
        "coverage":           {str(s): v for s, v in coverage.items()},
        "mrr_sum":            mrr_sum,
        "map_sum":            map_sum,
        "ndcg_full_sum":      ndcg_full_sum,
        "b2fh_sum":           b2fh_sum,
        "b2fh_count":         b2fh_count,
        "n_triples":          n_triples,
        "n_heads_evaluated":  n_heads_evaluated,
        "k_values":           list(k_values),
        "coverage_sizes":     list(coverage_sizes),
    }


def _state_to_accumulators(state: Dict, k_values: Tuple, coverage_sizes: Tuple) -> Dict:
    """Restore accumulator dicts from a checkpoint (str keys → int)."""
    return {
        "batch_start":        state["batch_start"],
        "entity_hit_sum":     {int(k): v for k, v in state["entity_hit_sum"].items()},
        "entity_recall_sum":  {int(k): v for k, v in state["entity_recall_sum"].items()},
        "ndcg_sum":           {int(k): v for k, v in state["ndcg_sum"].items()},
        "coverage":           {int(s): v for s, v in state["coverage"].items()},
        "mrr_sum":            state["mrr_sum"],
        "map_sum":            state["map_sum"],
        "ndcg_full_sum":      state["ndcg_full_sum"],
        "b2fh_sum":           state["b2fh_sum"],
        "b2fh_count":         state["b2fh_count"],
        "n_triples":          state["n_triples"],
        "n_heads_evaluated":  state["n_heads_evaluated"],
    }


# ---------------------------------------------------------------------------
# Per-head metric worker
# ---------------------------------------------------------------------------
# Globals are written in the main process before forking so all workers see
# the current batch's data via copy-on-write — only head IDs (ints) travel
# through the IPC pipe, avoiding expensive candidate-list serialization.

_eval_candidates:     Dict[int, List[Tuple[int, int, int, float]]] = {}
_eval_ground_truth:   Dict[int, Set[Tuple[int, int]]]              = {}
_eval_k_values:       Tuple[int, ...]                              = ()
_eval_coverage_sizes: Tuple[int, ...]                              = ()


def _eval_one_head(args: Tuple) -> Dict:
    """
    Compute all ranking metrics for a single head entity.

    Accepts (h, candidates, gold) as a tuple so the pool can be kept alive
    across batches — only k_values/coverage_sizes are read from globals
    (set once before the pool is created and never changed).
    Returns a partial-sums dict that the main process aggregates.
    """
    h, candidates, gold = args
    k_values       = _eval_k_values
    coverage_sizes = _eval_coverage_sizes

    n_gold  = len(gold)
    # Convert 4-tuple candidates to ranked (r, t) list (deduplicated, order preserved)
    seen: set = set()
    ranked: List[Tuple[int, int]] = []
    for _, r, t, _ in candidates:
        if (r, t) not in seen:
            seen.add((r, t))
            ranked.append((r, t))

    # Entity-level metrics (macro) — from entity_metrics
    entity_hit    = {k: _entity_hit_at_k(ranked, gold, k)    for k in k_values}
    entity_recall = {k: _entity_recall_at_k(ranked, gold, k) for k in k_values}
    mrr           = _mean_reciprocal_rank(ranked, gold)
    b2fh          = _budget_to_first_hit(ranked, gold)   # int or None

    # NDCG@K and NDCG (full list) — from entity_metrics
    ndcg_k    = {k: _ndcg_at_k(ranked, gold, k) for k in k_values}
    ndcg_full = _ndcg(ranked, gold)

    # MAP — from entity_metrics
    map_val = _average_precision(ranked, gold)

    # Coverage@S — entity_hit at coverage_sizes (same definition, separate size list)
    coverage = {s: _entity_hit_at_k(ranked, gold, s) for s in coverage_sizes}

    return {
        "entity_hit":    entity_hit,
        "entity_recall": entity_recall,
        "ndcg":          ndcg_k,
        "ndcg_full":     ndcg_full,
        "map":           map_val,
        "mrr":           mrr,
        "b2fh":          b2fh,
        "coverage":      coverage,
        "n_triples":     n_gold,
    }


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def prepare_candidates(
    pipeline: InstanceCompletionPipeline,
    test_heads: List[int],
    ground_truth: Dict[int, Set[Tuple[int, int]]],
    cache_dir: Path,
    max_candidates: Optional[int] = None,
    head_batch_size: int = 256,
    chunk_size: int = 512,
    num_workers: int = 0,
) -> None:
    """
    Pre-generate and cache candidates for all test heads.

    Heads whose cache file already exists are skipped.  The remaining heads
    are processed in sub-batches of `head_batch_size` using
    pipeline.generate_candidates_batch(), which scores all (h, r) pairs in
    that sub-batch in a single chunked GPU pass (controlled by `chunk_size`).
    Within each sub-batch, Stage-1 relation prediction is optionally
    parallelised across `num_workers` CPU processes.

    Each head's candidate list is stored as an individual pickle file:
        cache_dir / head_{h}.pkl  →  List[(head, rel, tail, score)]
    """

    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata_file = cache_dir / '00metadata.json'
    heads_todo = [h for h in test_heads if ground_truth.get(h)]

    if metadata_file.exists():
        metadata = json.loads(metadata_file.read_text())
        heads_todo = [h for h in heads_todo if h not in metadata.keys()]
    else:
        metadata = {}

    if not heads_todo:
        logger.info("  All candidates already cached — skipping generation.")
        return

    logger.info(
        f"  Generating candidates for {len(heads_todo)} heads "
        f"(head_batch_size={head_batch_size}, chunk_size={chunk_size}, max_candidates={max_candidates})…"
    )

    save_executor = concurrent.futures.ProcessPoolExecutor(max_workers=4)
    save_futures: list[concurrent.futures.Future] = []

    try:
        for batch_start in tqdm(range(0, len(heads_todo), head_batch_size),
                                desc="candidate batches", unit="batch"):
            batch = heads_todo[batch_start : batch_start + head_batch_size]

            t0 = time.time()
            batch_results = pipeline.generate_candidates_batch(
                batch,
                max_candidates=max_candidates,
                chunk_size=chunk_size,
                num_workers=num_workers,
            )
            logger.info(f"  Candidate generation ready in {time.time()-t0:.1f}s")

            cache_file = cache_dir / f'{_hash_heads(batch)}.parquet'
            for h in batch:
                metadata[h] = str(cache_file)
            metadata_file.write_text(json.dumps(metadata, indent=2))

            # Dispatch save to background process — overlaps with all future generation
            future = save_executor.submit(save_candidates, batch_results, cache_file)
            save_futures.append(future)
            logger.info(f"  Save dispatched asynchronously → {cache_file.name}")

        # Wait for all saves to complete
        t0 = time.time()
        for future in concurrent.futures.as_completed(save_futures):
            future.result()  # re-raises any exception from the worker
        logger.info(f"  All saves complete in {time.time()-t0:.1f}s")
    finally:
        save_executor.shutdown(wait=False)

    logger.info(f"  Candidate generation complete ({len(heads_todo)} heads).")


def evaluate_pipeline(
    pipeline: InstanceCompletionPipeline,
    test_heads: List[int],
    ground_truth: Dict[int, Set[Tuple[int, int]]],
    k_values: Tuple[int, ...] = (1, 5, 10, 50),
    coverage_sizes: Tuple[int, ...] = (10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000),
    cache_dir: Optional[Path] = None,
    max_candidates: Optional[int] = None,
    head_batch_size: int = 256,
    chunk_size: int = 512,
    num_workers: int = 0,
    eval_workers: int = 32,
) -> Dict[str, float]:
    """
    Comprehensive ranking evaluation of the instance completion pipeline.

    Candidates are generated via prepare_candidates() (GPU-batched, cached to
    disk), then loaded per-head for metric computation.  Heads whose cache
    file is missing after generation are skipped gracefully.

    All metrics are macro-averaged (each entity weighted equally), using
    entity_metrics.py as the authoritative implementation.

    EntityHit@K    — fraction of entities with ≥1 gold in top-K       (macro)
    EntityRecall@K — fraction of gold facts recovered in top-K         (macro)
    NDCG@K         — normalised DCG at K                               (macro)
    NDCG           — normalised DCG over full ranked list               (macro)
    MAP            — mean average precision                             (macro)
    MRR            — mean reciprocal rank over gold facts per entity    (macro)
    B2FH           — mean budget-to-first-hit (entities with ≥1 hit)  (macro)
    B2FH-Coverage  — fraction of entities where ≥1 gold is found
    Coverage@S     — same as EntityHit@S for large S values

    Returns
    -------
    Dict with keys entity_hit@K, entity_recall@K, ndcg@K, ndcg, map, mrr,
    b2fh, b2fh_coverage, coverage@S, n_triples, n_heads.
    """
    # prepare_candidates(
    #     pipeline, test_heads, ground_truth, cache_dir,
    #     max_candidates=max_candidates,
    #     head_batch_size=head_batch_size,
    #     chunk_size=chunk_size,
    #     num_workers=num_workers,
    # )

    # ── Checkpoint: resume if interrupted ────────────────────────────────────
    ckpt_path = cache_dir / _EVAL_CKPT_NAME
    cache_dir.mkdir(exist_ok=True, parents=True)
    ckpt      = _load_eval_checkpoint(ckpt_path)

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
        logger.info(
            f"  Resuming evaluation from batch {resume_from} "
            f"({n_heads_evaluated} heads already done)."
        )
    else:
        if ckpt:
            logger.warning(
                "  Checkpoint found but k_values/coverage_sizes differ — starting fresh."
            )
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

    heads_todo = [h for h in test_heads if ground_truth.get(h)]
    batches    = list(range(0, len(heads_todo), head_batch_size))

    # Set constant globals before forking so workers inherit them at pool
    # creation time — these never change between batches.
    global _eval_k_values, _eval_coverage_sizes
    _eval_k_values       = k_values
    _eval_coverage_sizes = coverage_sizes

    n_proc = min(eval_workers, len(heads_todo)) if eval_workers > 0 else 0
    pool   = mp.get_context("fork").Pool(processes=n_proc) if n_proc > 1 else None

    # ── Per-head evaluation ───────────────────────────────────────────────────
    try:
        for batch_start in tqdm(batches, desc="eval batches", unit="batch",
                                initial=batches.index(resume_from) if resume_from in batches else 0):
            if batch_start < resume_from:
                continue                # already processed in a previous run

            batch = heads_todo[batch_start : batch_start + head_batch_size]
            t0 = time.time()
            batch_results = pipeline.generate_candidates_batch(
                batch,
                max_candidates=max_candidates,
                chunk_size=chunk_size,
                num_workers=num_workers,
            )
            logger.info(f"  Candidate generation ready in {time.time()-t0:.1f}s")

            # ── Parallel per-head metric computation ──────────────────────────
            # Per-batch candidates and ground truth are passed as arguments so
            # the persistent pool (forked once before the loop) can process any
            # batch without relying on globals that changed since fork time.
            t0 = time.time()
            heads_in_batch = [h for h in batch_results if ground_truth.get(h)]
            task_args      = [(h, batch_results[h], ground_truth[h]) for h in heads_in_batch]

            if pool is not None:
                partial_results = pool.map(_eval_one_head, task_args)
            else:
                partial_results = [_eval_one_head(a) for a in task_args]

            # ── Aggregate partial results ──────────────────────────────────────
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

            logger.info(f"  Metrics ready in {time.time()-t0:.1f}s")

            # ── Save checkpoint after every batch ─────────────────────────────
            next_batch_start = batch_start + head_batch_size
            _save_eval_checkpoint(ckpt_path, _accumulators_to_state(
                next_batch_start,
                entity_hit_sum, entity_recall_sum, ndcg_sum, coverage,
                mrr_sum, map_sum, ndcg_full_sum, b2fh_sum, b2fh_count,
                n_triples, n_heads_evaluated,
                k_values, coverage_sizes,
            ))

    finally:
        if pool is not None:
            pool.close()
            pool.join()

    # ── Remove checkpoint on clean completion ─────────────────────────────────
    if ckpt_path.exists():
        ckpt_path.unlink()
        logger.info(f"  Evaluation complete — checkpoint removed.")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    n_h = n_heads_evaluated or 1

    metrics: Dict[str, float] = {}
    for k in k_values:
        metrics[f"entity_hit@{k}"]    = entity_hit_sum[k]    / n_h
        metrics[f"entity_recall@{k}"] = entity_recall_sum[k] / n_h
        metrics[f"ndcg@{k}"]          = ndcg_sum[k]          / n_h
    metrics["ndcg"]         = ndcg_full_sum / n_h
    metrics["map"]          = map_sum       / n_h
    metrics["mrr"]          = mrr_sum       / n_h   # macro: each entity weighted equally
    metrics["b2fh"]         = b2fh_sum / b2fh_count if b2fh_count > 0 else float("inf")
    metrics["b2fh_coverage"] = b2fh_count / n_h
    for s in coverage_sizes:
        metrics[f"coverage@{s}"] = coverage[s] / n_h
    metrics["n_triples"] = float(n_triples)
    metrics["n_heads"]   = float(n_heads_evaluated)
    return metrics


def format_results_table(
    results: Dict[str, Dict[str, float]],
    k_values: Tuple[int, ...] = (1, 5, 10, 50),
    coverage_sizes: Tuple[int, ...] = (10, 50, 100, 200, 500),
) -> str:
    # Show a representative subset; the full dict is in the JSON output.
    show_k = [k for k in (1, 10, 50) if k in k_values]
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
    name_w = max(30, max(len(n) for n in results) + 2)

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


# ---------------------------------------------------------------------------
# Relation predictor factory
# ---------------------------------------------------------------------------

def build_relation_predictor(
    name: str,
    train_triples: torch.Tensor,
    num_entities: int,
    num_relations: int,
    bpr_epochs: int,
    bpr_embed_dim: int,
    device: torch.device,
    dataset_name: str,
    cache_dir: Path,
    entity_types: Optional[Dict[int, List[int]]] = None,
):
    """Train and return a relation predictor by name (bpr / recoin)."""
    if name == "bpr":
        logger.info(f"Training BPR relation predictor (embed_dim={bpr_embed_dim}, epochs={bpr_epochs}) …")
        pred = BPRRelationPredictor(
            train_triples=train_triples,
            num_entities=num_entities,
            num_relations=num_relations,
            embed_dim=bpr_embed_dim,
            device=device,
            dataset_name=dataset_name,
            cache_dir=cache_dir,
        )
        pred.train(num_epochs=bpr_epochs, verbose=True)
        return pred

    if name == "recoin":
        if not entity_types:
            logger.warning(
                "RecoinRelationPredictor: no entity_types provided — "
                "falling back to global relation frequency (equivalent to frequency baseline)."
            )
        logger.info("Building Recoin relation predictor …")
        return RecoinRelationPredictor(
            train_triples=train_triples,
            dataset_name=dataset_name,
            cache_dir=cache_dir,
            entity_types=entity_types or {},
        )

    raise ValueError(f"Unknown relation predictor '{name}'. Choose from: bpr, recoin.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run instance-completion baseline experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset",         default="fb15k237",
                   help="Dataset name passed to KgLoader.")
    p.add_argument("--rel-predictors",  nargs="+", default=["bpr", "recoin"],
                   choices=["bpr", "recoin"],
                   help="Relation predictors to evaluate.")
    p.add_argument("--kgc-models",      nargs="+", default=["transe", "transh"],
                   help="PyKEEN KGC model names to evaluate.")
    p.add_argument("--k-r",             type=int, default=100,
                   help="Top-k relations per head entity (Stage 1).")
    p.add_argument("--k-t",             type=int, default=10000,
                   help="Top-k tails per (head, relation) query (Stage 2).")
    p.add_argument("--alpha",           type=float, default=0.5,
                   help="Score mixing weight: alpha*rel + (1-alpha)*tail.")
    p.add_argument("--embed-dim",       type=int, default=128,
                   help="Embedding dimension for both BPR and KGC models.")
    p.add_argument("--epochs",          type=int, default=100,
                   help="Training epochs for KGC models.")
    p.add_argument("--bpr-epochs",      type=int, default=None,
                   help="BPR training epochs (defaults to --epochs).")
    p.add_argument("--max-candidates",  type=int, default=100_0000,
                   help="Max candidates returned per head entity.")
    p.add_argument("--test-limit",      type=int, default=None,
                   help="Limit number of test head entities (for quick runs).")
    p.add_argument("--head-batch-size", type=int, default=256,
                   help="Heads per generate_candidates_batch call (controls Stage-1 memory).")
    p.add_argument("--chunk-size",      type=int, default=512,
                   help="(h,r) pairs per score_t GPU call. Peak GPU mem ≈ chunk×entities×4B.")
    p.add_argument("--num-workers",     type=int, default=0,
                   help="CPU processes for Stage-1 relation prediction. "
                        "0=sequential. Set >0 only with a CPU-based relation predictor.")
    p.add_argument("--eval-workers",    type=int, default=32,
                   help="CPU processes for per-head metric computation in evaluate_pipeline.")
    p.add_argument("--output-dir",      default=None,
                   help="Directory to save per-run JSON results. Skipped if not set.")
    p.add_argument("--device",          default=None,
                   help="Torch device (e.g. 'cuda', 'cpu'). Auto-detected if not set.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(args.device) if args.device else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bpr_epochs = args.bpr_epochs or args.epochs
    dataset_name = args.dataset
    model_cache_dir = Path('iswc_data/cache/models')
    logger.info(f"Device: {device}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Relation predictors: {args.rel_predictors}")
    logger.info(f"KGC models: {args.kgc_models}")
    logger.info(f"k_r={args.k_r}, k_t={args.k_t}, alpha={args.alpha}")

    # ── Load KG ──────────────────────────────────────────────────────────────
    logger.info("Loading knowledge graph …")
    from pathe.kgloader import KgLoader
    kg = KgLoader(args.dataset)

    train_triples: torch.Tensor = kg.train_triples        # (N, 3) LongTensor
    test_triples:  torch.Tensor = kg.test_no_inv          # use originals (no inverses)
    num_entities:  int          = kg.num_nodes_total
    num_relations: int          = kg.triple_factory.training.num_relations

    logger.info(
        f"  Entities: {num_entities}, Relations: {num_relations}, "
        f"Train triples: {len(train_triples)}, Test triples: {len(test_triples)}"
    )
    # ── Build ground truth ────────────────────────────────────────────────────
    ground_truth: Dict[int, Set[Tuple[int, int]]] = defaultdict(set)
    for h, r, t in test_triples.tolist():
        ground_truth[h].add((r, t))

    test_heads = sorted(ground_truth.keys())
    if args.test_limit:
        test_heads = test_heads[: args.test_limit]
        logger.info(f"  Test limit applied: {len(test_heads)} heads.")
    else:
        logger.info(f"  Unique test heads: {len(test_heads)}")

    # ── Build entity types for Recoin (extracted from type-like triples) ──────
    # FB15K-237 encodes entity types via specific relations; without a string
    # label map we cannot identify them here, so Recoin falls back to global
    # frequency.  Pass --entity-types-file to override (future extension).
    entity_types: Dict[int, List[int]] = {}

    # ── Output setup ──────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    # ── Experiment loop ───────────────────────────────────────────────────────
    all_results: Dict[str, Dict[str, float]] = {}

    for rel_name, kgc_name in product(args.rel_predictors, args.kgc_models):
        config_name = f"{rel_name}+{kgc_name}"
        logger.info(f"\n{'='*60}")
        logger.info(f"Configuration: {config_name}")
        logger.info(f"{'='*60}")

        # -- Stage 1: relation predictor --
        t0 = time.time()
        rel_pred = build_relation_predictor(
            name=rel_name,
            train_triples=train_triples,
            num_entities=num_entities,
            num_relations=num_relations,
            bpr_epochs=bpr_epochs,
            bpr_embed_dim=args.embed_dim,
            device=device,
            entity_types=entity_types,
            dataset_name=dataset_name,
            cache_dir=model_cache_dir,
        )
        logger.info(f"  Relation predictor ready in {time.time()-t0:.1f}s")

        # -- Stage 2: KGC tail predictor (cached) --
        t0 = time.time()
        logger.info(f"  Building KGC model: {kgc_name} …")
        kgc_wrapper = PyKEENTrainer.from_kg_loader(
            model_name=kgc_name,
            kg_loader=kg,
            embed_dim=args.embed_dim,
            device=device,
            dataset_name=args.dataset,
            cache_dir=model_cache_dir,
        ).train(num_epochs=args.epochs)
        logger.info(f"  KGC model ready in {time.time()-t0:.1f}s")

        # -- Pipeline --
        pipeline = InstanceCompletionPipeline(
            relation_predictor=rel_pred,
            kgc_model=kgc_wrapper,
            k_r=args.k_r,
            k_t=args.k_t,
            alpha=args.alpha,
            device=device,
        )

        # -- Evaluate --
        logger.info(f"  Evaluating on {len(test_heads)} test heads …")
        t0 = time.time()
        metrics = evaluate_pipeline(
            pipeline=pipeline,
            test_heads=test_heads,
            ground_truth=ground_truth,
            cache_dir=Path(f'iswc_data/cache/candidates/{kgc_name}_{rel_name}_{args.dataset}_r{args.k_r}_t{args.k_t}'),
            max_candidates=args.max_candidates,
            head_batch_size=args.head_batch_size,
            chunk_size=args.chunk_size,
            num_workers=args.num_workers,
            eval_workers=args.eval_workers,
        )
        elapsed = time.time() - t0
        logger.info(
            f"  Done in {elapsed:.1f}s | "
            f"EHit@10={metrics.get('entity_hit@10', 0):.4f}  "
            f"ERecall@10={metrics.get('entity_recall@10', 0):.4f}  "
            f"nDCG@10={metrics.get('ndcg@10', 0):.4f}  "
            f"nDCG={metrics.get('ndcg', 0):.4f}  "
            f"MAP={metrics.get('map', 0):.4f}  "
            f"MRR={metrics.get('mrr', 0):.4f}  "
            f"B2FH={metrics.get('b2fh', float('inf')):.1f}  "
            f"Cov@100={metrics.get('coverage@100', 0):.4f}  "
            f"(triples={int(metrics['n_triples'])})"
        )

        all_results[config_name] = metrics

        # -- Save per-run JSON --
        if output_dir:
            run_meta = {
                "config":        config_name,
                "dataset":       args.dataset,
                "rel_predictor": rel_name,
                "kgc_model":     kgc_name,
                "k_r":           args.k_r,
                "k_t":           args.k_t,
                "alpha":         args.alpha,
                "embed_dim":     args.embed_dim,
                "epochs":        args.epochs,
                "test_heads":    len(test_heads),
                "metrics":       metrics,
            }
            out_path = output_dir / f"{args.dataset}_{config_name}_r{args.k_r}_t{args.k_t}.json"
            out_path.write_text(json.dumps(run_meta, indent=2))
            logger.info(f"  Saved → {out_path}")

    # ── Print summary table ───────────────────────────────────────────────────
    print(format_results_table(all_results))

    if output_dir:
        summary_path = output_dir / f"{args.dataset}_r{args.k_r}_t{args.k_t}_summary.json"
        summary_path.write_text(json.dumps(all_results, indent=2))
        logger.info(f"Summary saved → {summary_path}")


if __name__ == "__main__":
    main()
