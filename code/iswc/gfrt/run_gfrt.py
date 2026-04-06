"""
GFRT Experiment Runner — Multi-View Filter for Relation-Free KG Completion
===========================================================================
Trains the GFRT model (MVF paper, Li et al. 2023) from scratch on a dataset
loaded via KgLoader and evaluates on the test set using entity-centric metrics.

Usage
-----
::

    cd code/
    python -m iswc.gfrt.run_gfrt \\
        --dataset fb15k237 \\
        --epochs 200 \\
        --embed-dim 64 \\
        --num-layers 2 \\
        --k-r 250 \\
        --k-t 15000 \\

    # Quick smoke-test
    python -m iswc.gfrt.run_gfrt \\
        --dataset fb15k237 \\
        --epochs 5 \\
        --test-limit 100
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import torch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE     = Path(__file__).resolve()
_CODE_DIR = _HERE.parents[2]   # code/iswc/gfrt/ → code/
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

_SJP_DIR = _CODE_DIR / "SJP_code" / "PathE"
if str(_SJP_DIR) not in sys.path:
    sys.path.insert(0, str(_SJP_DIR))

from iswc.gfrt import (
    GFRTModel,
    GFRTFilter,
    GFRTTrainer,
    build_gfrt_pipeline,
)
from iswc.evaluation.entity_metrics import (
    evaluate_entity_centric,
    format_results_table,
    MetricResults,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_predictions(
    gfrt_filter: GFRTFilter,
    test_heads: List[int],
    candidate_budget: int,
    batch_size: int = 64,
) -> Dict[int, List[Tuple[int, int]]]:
    """
    Generate (r, t) ranked predictions for each test head.

    Returns Dict[head_id -> list of (r, t) in ranked order].
    """
    predictions: Dict[int, List[Tuple[int, int]]] = {}
    total = len(test_heads)

    for start in range(0, total, batch_size):
        batch = test_heads[start: start + batch_size]
        raw = gfrt_filter.generate_candidates_batch(batch, candidate_budget)
        for h, cands in raw.items():
            # cands: list of (head, relation, tail, score) sorted by score desc
            predictions[h] = [(r, t) for (_, r, t, _) in cands]

        if (start // batch_size) % 10 == 0:
            logger.info(f"  Generated candidates for {min(start+batch_size, total)}/{total} heads")

    return predictions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate GFRT for relation-free KG completion."
    )
    parser.add_argument("--dataset",          default="fb15k237",
                        help="Dataset name for KgLoader.")
    parser.add_argument("--epochs",           type=int,   default=100,
                        help="Number of training epochs (paper: {50,100}).")
    parser.add_argument("--embed-dim",        type=int,   default=100,
                        help="Embedding dimensionality (paper: {50,100}).")
    parser.add_argument("--num-layers",       type=int,   default=2,
                        help="Number of GNN layers.")
    parser.add_argument("--top-k1",           type=int,   default=None,
                        help="Top-k1 similar entities per entity (graph construction). "
                             "Default: 100 for FB15K237/JF17k, 20 for UMLS.")
    parser.add_argument("--top-k2",           type=int,   default=None,
                        help="Top-k2 similar relations per relation (graph construction). "
                             "Default: 30 for FB15K237/JF17k, 10 for UMLS.")
    parser.add_argument("--k-r",              type=int,   default=20,
                        help="Top-k relations to select per head (candidate generation).")
    parser.add_argument("--k-t",              type=int,   default=100,
                        help="Top-k tail entities to select per head (candidate generation).")
    parser.add_argument("--candidate-budget", type=int,   default=200,
                        help="Maximum (r, t) candidates returned per head.")
    parser.add_argument("--batch-size",       type=int,   default=256,
                        help="Training batch size (triples per step).")
    parser.add_argument("--lr-intra",         type=float, default=0.01,
                        help="Learning rate for intra-view GNN parameters.")
    parser.add_argument("--lr-inter",         type=float, default=0.001,
                        help="Learning rate for inter-view alignment.")
    parser.add_argument("--margin",           type=float, default=1.0,
                        help="Hinge margin for intra-view loss.")
    parser.add_argument("--log-every",        type=int,   default=10,
                        help="Log training loss every N epochs.")
    parser.add_argument("--test-limit",       type=int,   default=None,
                        help="Restrict evaluation to first N test heads (smoke-test).")
    parser.add_argument("--output-dir",       type=str,   default=None,
                        help="Directory to save results JSON.")
    parser.add_argument("--save-model",       type=str,   default=None,
                        help="Path to save the trained model checkpoint.")
    parser.add_argument("--load-model",       type=str,   default=None,
                        help="Path to load a previously saved model checkpoint (skips training).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Dataset: {args.dataset}")

    # ── Load KG ──────────────────────────────────────────────────────────────
    logger.info("Loading knowledge graph …")
    from pathe.kgloader import KgLoader
    kg = KgLoader(args.dataset)

    train_triples: torch.Tensor = kg.train
    test_triples:  torch.Tensor = kg.test
    num_entities:  int          = kg.num_nodes_total
    num_relations: int          = kg.triple_factory.training.num_relations

    logger.info(
        f"  Entities: {num_entities}, Relations: {num_relations}, "
        f"Train: {len(train_triples)}, Test: {len(test_triples)}"
    )

    # ── Build ground truth ────────────────────────────────────────────────────
    ground_truth: Dict[int, Set[Tuple[int, int]]] = defaultdict(set)
    for h, r, t in test_triples.tolist():
        ground_truth[h].add((r, t))

    test_heads = sorted(ground_truth.keys())
    if args.test_limit:
        test_heads = test_heads[: args.test_limit]
        logger.info(f"  Test limit: {len(test_heads)} heads.")
    else:
        logger.info(f"  Unique test heads: {len(test_heads)}")

    # ── Dataset-specific graph construction defaults (paper Table 2) ─────────
    _UMLS_DATASETS = {"umls"}
    is_umls = args.dataset.lower() in _UMLS_DATASETS
    top_k1 = args.top_k1 if args.top_k1 is not None else (20  if is_umls else 100)
    top_k2 = args.top_k2 if args.top_k2 is not None else (10  if is_umls else 30)
    logger.info(f"  Graph construction: top_k1={top_k1}, top_k2={top_k2}")
    args.candidate_budget = args.k_r * args.k_t

    args.save_model = args.load_model = f'iswc_data/cache/models/gfrt_{args.dataset}.pt'
    # ── Build graphs and model ────────────────────────────────────────────────
    logger.info("Building GFRT graphs …")
    t0 = time.time()
    model, graph_H, graph_T = build_gfrt_pipeline(
        train_triples=train_triples,
        num_entities=num_entities,
        num_relations=num_relations,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        top_k1=top_k1,
        top_k2=top_k2,
        margin=args.margin,
        device=device,
    )
    logger.info(f"  Graphs built in {time.time()-t0:.1f}s")
    logger.info(
        f"  G_H: {graph_H.ee_src.numel()} ee-edges, "
        f"{graph_H.rr_src.numel()} rr-edges, "
        f"{graph_H.er_src.numel()} er-edges"
    )
    logger.info(
        f"  G_T: {graph_T.ee_src.numel()} ee-edges, "
        f"{graph_T.rr_src.numel()} rr-edges, "
        f"{graph_T.er_src.numel()} er-edges"
    )

    # ── Load or Train ─────────────────────────────────────────────────────────
    if args.load_model and Path(args.load_model).exists():
        logger.info(f"Loading model from {args.load_model} …")
        model = GFRTModel.load(args.load_model, device=device)
    else:
        logger.info(f"Training GFRT for {args.epochs} epochs …")
        trainer = GFRTTrainer(
            model=model,
            graph_H=graph_H,
            graph_T=graph_T,
            train_triples=train_triples,
            device=device,
            lr_intra=args.lr_intra,
            lr_inter=args.lr_inter,
        )

        t0 = time.time()
        for epoch in range(1, args.epochs + 1):
            losses = trainer.train_epoch(batch_size=args.batch_size)
            if epoch % args.log_every == 0 or epoch == 1:
                logger.info(
                    f"  Epoch {epoch:4d}/{args.epochs}  "
                    f"L_H={losses['loss_H']:.4f}  "
                    f"L_T={losses['loss_T']:.4f}  "
                    f"L_cross={losses['loss_cross']:.4f}"
                )

        logger.info(f"  Training completed in {time.time()-t0:.1f}s")

        if args.save_model:
            model.save(args.save_model)

    trainer = GFRTTrainer(
        model=model,
        graph_H=graph_H,
        graph_T=graph_T,
        train_triples=train_triples,
        device=device,
    )

    # ── Generate candidates ───────────────────────────────────────────────────
    logger.info("Generating candidates for test entities …")
    h_emb, rH_emb, t_emb, rT_emb = trainer.get_embeddings()

    gfrt_filter = GFRTFilter(
        h_emb=h_emb,
        rH_emb=rH_emb,
        t_emb=t_emb,
        rT_emb=rT_emb,
        model=model,
        train_triples=train_triples,
        top_m_relations=args.k_r,
        top_n_tails=args.k_t,
    )

    t0 = time.time()
    predictions = _build_predictions(
        gfrt_filter,
        test_heads,
        candidate_budget=args.candidate_budget,
    )
    logger.info(f"  Candidate generation done in {time.time()-t0:.1f}s")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    logger.info("Evaluating …")
    results = evaluate_entity_centric(
        predictions=predictions,
        gold_triples=test_triples,
        k_values=[1, 3, 5, 10, 20, 50, 100],
        test_heads=test_heads,
    )

    # ── Print results ─────────────────────────────────────────────────────────
    table = format_results_table({"GFRT": results}, k=10)
    print("\n" + table)

    print("\n── Full metrics ──────────────────────────────────────────────────")
    print(f"  MRR:              {results.mrr:.4f}")
    print(f"  Hits@1:           {results.hits_at_1:.4f}")
    print(f"  Hits@3:           {results.hits_at_3:.4f}")
    print(f"  Hits@10:          {results.hits_at_10:.4f}")
    print(f"  MAP:              {results.map:.4f}")
    print(f"  NDCG (full):      {results.ndcg:.4f}")
    for k in [5, 10, 20, 50]:
        print(
            f"  NDCG@{k:<3}:         {results.ndcg_at_k.get(k, 0):.4f}  "
            f"EntityHit@{k:<3}: {results.entity_hit_at_k.get(k, 0):.4f}  "
            f"EntityRecall@{k:<3}: {results.entity_recall_at_k.get(k, 0):.4f}"
        )
    print(f"  B2FH:             {results.budget_to_first_hit:.2f}")
    print(f"  B2FH-Coverage:    {results.budget_to_first_hit_coverage:.4f}")
    print(f"  Cand.Size (avg):  {results.avg_candidate_size:.1f}")
    print(f"  Cand.Coverage:    {results.candidate_coverage:.4f}")

    # ── Save results ──────────────────────────────────────────────────────────
    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        results_dict = {
            "dataset":    args.dataset,
            "epochs":     args.epochs,
            "embed_dim":  args.embed_dim,
            "num_layers": args.num_layers,
            "top_k1":     top_k1,
            "top_k2":     top_k2,
            "k_r":        args.k_r,
            "k_t":        args.k_t,
            "margin":     args.margin,
            "budget":     args.candidate_budget,
            "metrics": {
                "mrr":           results.mrr,
                "hits_at_1":     results.hits_at_1,
                "hits_at_3":     results.hits_at_3,
                "hits_at_10":    results.hits_at_10,
                "map":           results.map,
                "ndcg":          results.ndcg,
                "ndcg_at_k":     results.ndcg_at_k,
                "entity_hit_at_k":    results.entity_hit_at_k,
                "entity_recall_at_k": results.entity_recall_at_k,
                "b2fh":          results.budget_to_first_hit,
                "b2fh_coverage": results.budget_to_first_hit_coverage,
                "avg_candidate_size": results.avg_candidate_size,
                "candidate_coverage": results.candidate_coverage,
            },
        }

        results_path = out / f"gfrt_{args.dataset}.json"
        results_path.write_text(json.dumps(results_dict, indent=2))
        logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
