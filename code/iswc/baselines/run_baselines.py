"""
Baseline Experiment Runner — Instance Completion on KG Benchmarks
=================================================================

Runs all combinations of relation predictor × KGC model and reports
per-triple Hits@K metrics on the test split.

Evaluation
----------
For each head entity h in the test set:
    - Ground truth: all (r, t) pairs in test_triples with that head
    - Pipeline generates top-N ranked (h, r, t) candidates
    - For each ground-truth triple (h, r, t), check if it appears
      in the top-K candidates (K ∈ {1, 5, 10, 50})

Metrics reported (per-triple, micro-averaged across all test triples):
    Hits@1, Hits@5, Hits@10, Hits@50, MRR

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
import json
import logging
import sys
import time
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_pipeline(
    pipeline: InstanceCompletionPipeline,
    test_heads: List[int],
    ground_truth: Dict[int, Set[Tuple[int, int]]],
    max_candidates: int,
    k_values: Tuple[int, ...] = (1, 5, 10, 50),
) -> Dict[str, float]:
    """
    Per-triple evaluation of the instance completion pipeline.

    For each test head h:
        candidates = pipeline.generate_candidates(h, max_candidates)
        For each ground-truth (r, t) of h:
            rank = position of (r, t) in the candidate list (1-indexed)
            → contributes to Hits@K if rank ≤ K, and to MRR as 1/rank

    Returns
    -------
    Dict with keys hits@1, hits@5, hits@10, hits@50, mrr, n_triples, n_heads.
    """
    hits = {k: 0 for k in k_values}
    rr_sum = 0.0
    n_triples = 0
    n_heads_evaluated = 0

    for i, h in enumerate(test_heads):
        gold = ground_truth.get(h)
        if not gold:
            continue

        candidates = pipeline.generate_candidates(h, max_candidates=max_candidates)
        # Build (r, t) → rank mapping (1-indexed)
        rank_of: Dict[Tuple[int, int], int] = {}
        for rank, (_, r, t, _) in enumerate(candidates, start=1):
            rt = (r, t)
            if rt not in rank_of:   # keep best rank if duplicates
                rank_of[rt] = rank

        for r, t in gold:
            n_triples += 1
            rank = rank_of.get((r, t))
            if rank is not None:
                for k in k_values:
                    if rank <= k:
                        hits[k] += 1
                rr_sum += 1.0 / rank

        n_heads_evaluated += 1
        if (i + 1) % 200 == 0:
            logger.info(f"    evaluated {i + 1}/{len(test_heads)} heads …")

    metrics: Dict[str, float] = {}
    for k in k_values:
        metrics[f"hits@{k}"] = hits[k] / n_triples if n_triples > 0 else 0.0
    metrics["mrr"]       = rr_sum / n_triples if n_triples > 0 else 0.0
    metrics["n_triples"] = float(n_triples)
    metrics["n_heads"]   = float(n_heads_evaluated)
    return metrics


def format_results_table(
    results: Dict[str, Dict[str, float]],
    k_values: Tuple[int, ...] = (1, 5, 10, 50),
) -> str:
    k_cols    = [f"hits@{k}" for k in k_values]
    col_hdrs  = [f"H@{k}" for k in k_values] + ["MRR", "Triples"]
    col_keys  = k_cols + ["mrr", "n_triples"]
    widths    = [max(8, len(h) + 1) for h in col_hdrs]
    name_w    = max(30, max(len(n) for n in results) + 2)

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
        )
        pred.train(num_epochs=bpr_epochs, verbose=False)
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
    p.add_argument("--k-r",             type=int, default=10,
                   help="Top-k relations per head entity (Stage 1).")
    p.add_argument("--k-t",             type=int, default=50,
                   help="Top-k tails per (head, relation) query (Stage 2).")
    p.add_argument("--alpha",           type=float, default=0.5,
                   help="Score mixing weight: alpha*rel + (1-alpha)*tail.")
    p.add_argument("--embed-dim",       type=int, default=128,
                   help="Embedding dimension for both BPR and KGC models.")
    p.add_argument("--epochs",          type=int, default=200,
                   help="Training epochs for KGC models.")
    p.add_argument("--bpr-epochs",      type=int, default=None,
                   help="BPR training epochs (defaults to --epochs).")
    p.add_argument("--max-candidates",  type=int, default=500,
                   help="Max candidates returned per head entity.")
    p.add_argument("--test-limit",      type=int, default=None,
                   help="Limit number of test head entities (for quick runs).")
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
    import ipdb; ipdb.set_trace()
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
            max_candidates=args.max_candidates,
        )
        elapsed = time.time() - t0
        logger.info(
            f"  Done in {elapsed:.1f}s | "
            f"H@1={metrics['hits@1']:.4f}  H@10={metrics['hits@10']:.4f}  "
            f"MRR={metrics['mrr']:.4f}  (triples={int(metrics['n_triples'])})"
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
            out_path = output_dir / f"{args.dataset}_{config_name}.json"
            out_path.write_text(json.dumps(run_meta, indent=2))
            logger.info(f"  Saved → {out_path}")

    # ── Print summary table ───────────────────────────────────────────────────
    print(format_results_table(all_results))

    if output_dir:
        summary_path = output_dir / f"{args.dataset}_summary.json"
        summary_path.write_text(json.dumps(all_results, indent=2))
        logger.info(f"Summary saved → {summary_path}")


if __name__ == "__main__":
    main()
