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

Metrics reported (from iswc.harmonized.metrics):
    Candidate: total/average/relative size, coverage, density, B2FH
    Ranking: MRR, Recall@K, MAP@K, nDCG@K

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
from typing import Any, Dict, List, Optional, Set, Tuple
from tqdm import tqdm
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

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
from iswc.harmonized.metrics import evaluate_model, format_metrics_log, format_results_table


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
# _EVAL_CKPT_NAME = "eval_checkpoint.json"

#                 n_heads_evaluated += 1

#             logger.info(f"  Metrics ready in {time.time()-t0:.1f}s")

#             # ── Save checkpoint after every batch ─────────────────────────────
#             next_batch_start = batch_start + head_batch_size
#             _save_eval_checkpoint(ckpt_path, _accumulators_to_state(
#                 next_batch_start,
#                 entity_hit_sum, entity_recall_sum, ndcg_sum, coverage,
#                 mrr_sum, map_sum, ndcg_full_sum, b2fh_sum, b2fh_count,
#                 n_triples, n_heads_evaluated,
#                 k_values, coverage_sizes,
#             ))

#     finally:
#         if pool is not None:
#             pool.close()
#             pool.join()

#     # ── Remove checkpoint on clean completion ─────────────────────────────────
#     if ckpt_path.exists():
#         ckpt_path.unlink()
#         logger.info(f"  Evaluation complete — checkpoint removed.")

#     # ── Aggregate ─────────────────────────────────────────────────────────────
#     n_h = n_heads_evaluated or 1

#     metrics: Dict[str, float] = {}
#     for k in k_values:
#         metrics[f"entity_hit@{k}"]    = entity_hit_sum[k]    / n_h
#         metrics[f"entity_recall@{k}"] = entity_recall_sum[k] / n_h
#         metrics[f"ndcg@{k}"]          = ndcg_sum[k]          / n_h
#     metrics["ndcg"]         = ndcg_full_sum / n_h
#     metrics["map"]          = map_sum       / n_h
#     metrics["mrr"]          = mrr_sum       / n_h   # macro: each entity weighted equally
#     metrics["b2fh"]         = b2fh_sum / b2fh_count if b2fh_count > 0 else float("inf")
#     metrics["b2fh_coverage"] = b2fh_count / n_h
#     for s in coverage_sizes:
#         metrics[f"coverage@{s}"] = coverage[s] / n_h
#     metrics["n_triples"] = float(n_triples)
#     metrics["n_heads"]   = float(n_heads_evaluated)
#     return metrics


# def format_results_table(
#     results: Dict[str, Dict[str, float]],
#     k_values: Tuple[int, ...] = (1, 5, 10, 50),
#     coverage_sizes: Tuple[int, ...] = (10, 50, 100, 200, 500),
# ) -> str:
#     # Show a representative subset; the full dict is in the JSON output.
#     show_k = [k for k in (1, 10, 50) if k in k_values]
#     col_keys = (
#         [f"entity_hit@{k}"    for k in show_k] +
#         [f"entity_recall@{k}" for k in show_k] +
#         [f"ndcg@{k}"          for k in show_k] +
#         ["ndcg", "map", "mrr", "b2fh", "b2fh_coverage"] +
#         [f"coverage@{s}" for s in coverage_sizes] +
#         ["n_triples"]
#     )
#     col_hdrs = (
#         [f"EHit@{k}"    for k in show_k] +
#         [f"ERecall@{k}" for k in show_k] +
#         [f"nDCG@{k}"    for k in show_k] +
#         ["nDCG", "MAP", "MRR", "B2FH", "B2FH-Cov"] +
#         [f"Cov@{s}"  for s in coverage_sizes] +
#         ["Triples"]
#     )
#     widths = [max(8, len(h) + 1) for h in col_hdrs]
#     name_w = max(30, max(len(n) for n in results) + 2)

#     header  = f"{'Configuration':<{name_w}}" + "".join(f"{h:>{w}}" for h, w in zip(col_hdrs, widths))
#     divider = "-" * len(header)
#     lines   = ["\n=== Instance Completion Results ===", header, divider]

#     for name, m in results.items():
#         cells = []
#         for key, w in zip(col_keys, widths):
#             v = m.get(key, 0.0)
#             if key == "n_triples":
#                 cells.append(f"{int(v):>{w}}")
#             else:
#                 val_str = f"{v:.4f}"
#                 cells.append(f"{val_str:>{w}}")
#         lines.append(f"{name:<{name_w}}" + "".join(cells))

#     lines.append("")
#     return "\n".join(lines)


# ---------------------------------------------------------------------------
# Relation predictor factory
# ---------------------------------------------------------------------------
def evaluate_pipeline(
    pipeline: InstanceCompletionPipeline,
    ground_truth: Dict[int, Set[Tuple[int, int]]],
    cache_dir: Optional[Path] = None,
    max_candidates: Optional[int] = None,
    batch_size: int = 256,
    num_workers: int = 32,
) -> pd.DataFrame:
    return evaluate_model(model=pipeline,
                          ground_truth=ground_truth,
                          num_workers=num_workers,
                          cache_dir=cache_dir,
                          max_candidates=max_candidates,
                          batch_size=batch_size)



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
    p.add_argument("--batch-size", type=int, default=256,
                   help="Heads per generate_candidates_batch call (controls Stage-1 memory).")
    # p.add_argument("--chunk-size",      type=int, default=512,
    #                help="(h,r) pairs per score_t GPU call. Peak GPU mem ≈ chunk×entities×4B.")
    p.add_argument("--num-workers",     type=int, default=32,
                   help="CPU processes for Stage-1 relation prediction. "
                        "0=sequential. Set >0 only with a CPU-based relation predictor.")
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
    all_results: Dict[str, pd.DataFrame] = {}

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
            # test_heads=test_heads,
            ground_truth=ground_truth,
            cache_dir=Path(f'iswc_data/cache/candidates/{kgc_name}_{rel_name}_{args.dataset}_r{args.k_r}_t{args.k_t}'),
            max_candidates=args.max_candidates,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        elapsed = time.time() - t0
        logger.info("  Done in %.1fs | %s", elapsed, format_metrics_log(metrics))

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
                "metrics":       metrics.to_dict(orient="records"),
            }
            out_path = output_dir / f"{args.dataset}_{config_name}_r{args.k_r}_t{args.k_t}.json"
            out_path.write_text(json.dumps(run_meta, indent=2))
            logger.info(f"  Saved → {out_path}")

    # ── Print summary table ───────────────────────────────────────────────────
    print(format_results_table(all_results))

    if output_dir:
        summary_path = output_dir / f"{args.dataset}_r{args.k_r}_t{args.k_t}_summary.json"
        summary_payload = {name: df.to_dict(orient="records") for name, df in all_results.items()}
        summary_path.write_text(json.dumps(summary_payload, indent=2))
        logger.info(f"Summary saved → {summary_path}")


if __name__ == "__main__":
    main()
