"""
run_eval.py — Main evaluation script for native vs SJP RAG comparison.

Runs one or both pipelines over WebQSP and/or CWQ and prints a metric table.
Results are also saved to JSON for further analysis.

Quick start
-----------
# Compare both pipelines on both datasets (placeholder SJP model):
    python -m iswc.rag.run_eval

# Only SJP pipeline on WebQSP, 50 test samples:
    python -m iswc.rag.run_eval --dataset webqsp --pipeline sjp --max-samples 50

# Use a real SJP checkpoint:
    python -m iswc.rag.run_eval \\
        --sjp-checkpoint path/to/checkpoint.ckpt \\
        --entity-map     path/to/entity2id.json \\
        --relation-map   path/to/relation2id.json

Environment variables
---------------------
    DEEPSEEK_API_KEY    Required for deepseek-chat / deepseek-reasoner.
    GEMINI_API_KEY      Required for gemini-2.0-flash.
    OPENAI_API_KEY      Required for gpt-* models.
    (Ollama models require no key — they use a local endpoint.)

Output files (written to --output-dir, default ./results/)
-----------------------------------------------------------
    {dataset}_{pipeline}_{split}_predictions.json   Per-sample predictions
    {dataset}_{pipeline}_{split}_metrics.json       Aggregate metrics
    comparison_{dataset}_{split}.txt                Side-by-side table
"""
import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_id_map(path: Optional[str], invert: bool = False) -> dict:
    """Load a JSON ID-map file.

    Args:
        path:   Path to a JSON file mapping str → int or int → str.
        invert: If True, swap keys and values.

    Returns:
        Loaded dict, or empty dict if path is None or file missing.
    """
    if path is None or not os.path.exists(path):
        return {}
    with open(path) as f:
        d = json.load(f)
    if invert:
        return {v: k for k, v in d.items()}
    return d


def _save_json(data, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved → %s", path)


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------

def build_llm(args, dataset_name: str = "", pipeline_name: str = ""):
    """Construct the shared LLM from CLI args.

    Cache dir: iswc_data/cache/rag/llm/<model>/<dataset>/<pipeline>/budget<N>
    Each unique prompt is saved as <prompt_hash>.json inside that directory.
    """
    from .llm.utils_llm_adapter import UtilsLLM

    model_slug = re.sub(r"[^a-zA-Z0-9_-]", "_", args.model)
    cache_dir = os.path.join(
        "iswc_data", "cache", "rag", "llm",
        model_slug, dataset_name, pipeline_name, f"budget{args.budget}",
    )

    return UtilsLLM(model_name=args.model, cache_dir=cache_dir)


def build_native_pipeline(args, llm, dataset_obj):
    """Build the native (embedding cosine-similarity) RAG pipeline."""
    from .retrieval.embedding_retriever import EmbeddingRetriever
    from .pipelines.native_rag import NativeRAGPipeline

    # Auto-construct cache dir: iswc_data/cache/rag/emb/<dataset>
    cache_dir = Path("iswc_data/cache/rag/emb") / dataset_obj.name
    cache_dir.mkdir(parents=True, exist_ok=True)

    retriever = EmbeddingRetriever(
        model_name=getattr(args, "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
        device=getattr(args, "device", None),
        cache_dir=str(cache_dir),
    )
    return NativeRAGPipeline(retriever=retriever, llm=llm)


def build_gfrt_pipeline(args, llm, dataset_obj):
    """Build the GFRT-enhanced RAG pipeline.

    Loads a trained GFRT checkpoint if it exists at --gfrt-model-dir/<dataset>.pt,
    otherwise trains from scratch and saves the checkpoint there.
    """
    # from .retrieval.gfrt_retriever import build_gfrt_retrievers
    from .pipelines.gfrt_rag import GFRTRAGPipeline
    model_dir = Path(getattr(args, "gfrt_model_dir", "iswc_data/cache"))

    # retrievers = build_gfrt_retrievers(
    #     dataset_obj=dataset_obj,
    #     model_dir=str(model_dir),
    #     device=str(getattr(args, "device", None) or "cpu"),
    #     k_r=getattr(args, "gfrt_k_r", 20),
    #     k_t=getattr(args, "gfrt_k_t", 100),
    #     epochs=getattr(args, "gfrt_epochs", 100),
    # )
    return GFRTRAGPipeline(model_dir=model_dir, llm=llm)


def build_sjp_pipeline(args, llm):
    """Build the SJP-enhanced RAG pipeline.

    If --sjp-checkpoint is not provided, the pipeline runs in placeholder mode
    and retrieves nothing (useful for pipeline testing without a trained model).
    """
    from .retrieval.fact_suggester import SJPFactSuggester
    from .pipelines.sjp_rag import SJPRAGPipeline

    cache_dir = Path("iswc_data/cache/rag/sjp") / dataset_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load vocabulary maps if provided
    entity_id_map = _load_id_map(args.entity_map)                    # str → int
    id_to_entity = _load_id_map(args.entity_map, invert=True)        # int → str
    id_to_relation = _load_id_map(args.relation_map, invert=True)    # int → str

    model, dataset, cand_gen, relation_maps = None, None, None, None

    if args.sjp_checkpoint:
        model, dataset, cand_gen, relation_maps = _load_sjp_model(args)

    retriever = SJPFactSuggester(
        model=model,
        dataset=dataset,
        candidate_generator=cand_gen,
        relation_maps=relation_maps,
        entity_id_map=entity_id_map,
        id_to_entity=id_to_entity,
        id_to_relation=id_to_relation,
        device=args.device,
    )
    return SJPRAGPipeline(retriever=retriever, llm=llm)


def _load_sjp_model(args):
    """Load a trained SJP model checkpoint and the corresponding dataset.

    This function is intentionally minimal — it delegates to the SJP codebase.
    Users may need to extend it to match their specific training configuration.

    Returns:
        (model, dataset, candidate_generator, relation_maps)
    """
    logger.info("Loading SJP model from %s …", args.sjp_checkpoint)
    try:
        import torch
        from pathe.wrappers import PathEModelWrapperUniqueHeads           # type: ignore
        from pathe.candidates import CandidateGeneratorGlobalWithTail     # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Could not import SJP model code. "
            "Make sure SJP_code/PathE is on your PYTHONPATH.\n"
            f"Original error: {exc}"
        )

    model = PathEModelWrapperUniqueHeads.load_from_checkpoint(args.sjp_checkpoint)
    model.eval()

    # Dataset must be pre-built — pass the path via --sjp-dataset-path
    if not args.sjp_dataset_path:
        raise ValueError(
            "--sjp-dataset-path is required when --sjp-checkpoint is provided. "
            "Point it to the serialised UniqueHeadEntityMultiPathDataset directory."
        )
    dataset = torch.load(args.sjp_dataset_path)

    cand_gen = CandidateGeneratorGlobalWithTail(
        p=args.candidates_threshold_p,
        q=args.candidates_quantile_q,
        temperature=args.candidates_temperature,
        alpha=args.candidates_alpha,
        beta=args.candidates_beta,
        per_group_cap=args.budget,
    )

    relation_maps = dataset.relation_maps
    return model, dataset, cand_gen, relation_maps


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def run_pipeline_on_dataset(pipeline, dataset_obj, args) -> List[dict]:
    """Run *pipeline* on all samples and return per-sample prediction dicts."""
    from .pipelines.base import PipelineResult

    n = min(args.max_samples, len(dataset_obj)) if args.max_samples else len(dataset_obj)
    results = []
    for i in tqdm(range(n), desc=f"{pipeline.name}", unit="q"):
        sample = dataset_obj[i]
        result: PipelineResult = pipeline.run(sample, budget=args.budget)
        results.append({
            "question_id":       result.question_id,
            "question":          result.question,
            "predicted_answers": result.predicted_answers,
            "gold_answers":      result.gold_answers,
            "raw_response":      result.raw_response,
            "num_retrieved":     len(result.retrieved_triples),
            "context":           result.context_text,
        })

    return results


def evaluate_and_report(
    pipeline_name: str,
    raw_results: List[dict],
    output_dir: str,
    dataset_name: str,
    split: str,
    budget: int = 10,
) -> Dict[str, float]:
    """Compute metrics, save files, return aggregate dict."""
    from .pipelines.base import PipelineResult
    from .evaluation.metrics import evaluate_results, aggregate_metrics

    # Re-construct minimal PipelineResult objects for the metric functions
    pr_list = [
        PipelineResult(
            question_id=r["question_id"],
            question=r["question"],
            predicted_answers=r["predicted_answers"],
            gold_answers=r["gold_answers"],
            raw_response=r.get("raw_response", ""),
        )
        for r in raw_results
    ]
    per_sample = evaluate_results(pr_list)
    agg = aggregate_metrics(per_sample)

    prefix = f"{dataset_name}_{pipeline_name}_{split}_budget{budget}"
    _save_json(raw_results, os.path.join(output_dir, f"{prefix}_predictions.json"))
    _save_json({"per_sample": per_sample, "aggregate": agg},
               os.path.join(output_dir, f"{prefix}_metrics.json"))

    return agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate native RAG vs SJP RAG on WebQSP / CWQ.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Dataset ---
    parser.add_argument("--dataset", choices=["webqsp", "cwq", "both"], default="webqsp",
                        help="Dataset to evaluate on.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test",
                        help="Dataset split.")
    parser.add_argument("--webqsp-path", default='./iswc_data/standard/webqsp',
                        help="Local path to WebQSP JSON or HF cache dir. "
                             "Downloads from Hub if omitted.")
    parser.add_argument("--cwq-path", default='./iswc_data/standard/cwq',
                        help="Local path to CWQ JSON or HF cache dir. "
                             "Downloads from Hub if omitted.")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit evaluation to this many samples (for fast tests).")

    # --- Pipeline selection ---
    parser.add_argument("--pipeline", choices=["native", "sjp", "gfrt", "both", "all"], default="both",
                        help="Which pipeline(s) to run. 'both'=native+sjp, 'all'=native+sjp+gfrt.")

    # --- LLM ---
    parser.add_argument("--model", default="ollama-qwen3-8b",
                        help="Model name (must be a key in utils_llm.model_map, e.g. "
                             "ollama-llama3.1-8b,ollama-qwen3-8b,ollama-qwen3-32b,ollama-qwen3-1.7b,ollama-gemma2-9b).")

    # --- Retrieval ---
    parser.add_argument("--budget", type=int, default=10,
                        help="Number of triples to retrieve as context (budget). "
                             "Results are saved per-budget to support budget sweep experiments.")
    parser.add_argument("--sparql-endpoint", default=_WIKIDATA_SPARQL,
                        help="SPARQL endpoint URL for native retriever.")
    parser.add_argument("--sparql-timeout", type=int, default=10,
                        help="SPARQL query timeout in seconds.")
    parser.add_argument("--embedding-model",
                        default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Sentence-transformers model for native RAG embedding retrieval.")
    parser.add_argument("--embedding-cache-dir",
                        default=None,
                        help="Override the default embedding cache directory. "
                             "Default: iswc_data/cache/rag/emb/<dataset>. "
                             "Each graph subgraph is cached as graph_<hash>.pkl.")

    # --- GFRT model ---
    parser.add_argument("--gfrt-model-dir", default="iswc_data/cache",
                        help="Directory to save/load GFRT model checkpoints "
                             "(saved as gfrt_<dataset>.pt).")
    parser.add_argument("--gfrt-k-r", type=int, default=20,
                        help="Top-k relations per head for GFRT candidate generation.")
    parser.add_argument("--gfrt-k-t", type=int, default=100,
                        help="Top-k tails per (head, relation) for GFRT candidate generation.")
    parser.add_argument("--gfrt-epochs", type=int, default=100,
                        help="Training epochs for GFRT (ignored when loading from checkpoint).")

    # --- SJP model (optional; enables real model instead of placeholder) ---
    parser.add_argument("--sjp-checkpoint", default=None,
                        help="Path to trained SJP model checkpoint (.ckpt).")
    parser.add_argument("--sjp-dataset-path", default=None,
                        help="Path to serialised UniqueHeadEntityMultiPathDataset (.pt).")
    parser.add_argument("--entity-map", default=None,
                        help="JSON file mapping entity string ID → SJP integer ID.")
    parser.add_argument("--relation-map", default=None,
                        help="JSON file mapping relation string → SJP integer ID.")
    parser.add_argument("--device", default=None,
                        help="Torch device for SJP model (e.g. 'cuda', 'cpu').")
    # SJP candidate generator hyper-parameters
    parser.add_argument("--candidates-threshold-p", type=float, default=0.0)
    parser.add_argument("--candidates-quantile-q", type=float, default=0.9)
    parser.add_argument("--candidates-temperature", type=float, default=1.0)
    parser.add_argument("--candidates-alpha", type=float, default=0.5)
    parser.add_argument("--candidates-beta", type=float, default=0.3)

    # --- Output ---
    parser.add_argument("--output-dir", default="results",
                        help="Directory to write prediction and metric JSON files.")

    args = parser.parse_args(argv)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Datasets to evaluate
    datasets_to_run = (
        ["webqsp", "cwq"] if args.dataset == "both" else [args.dataset]
    )
    # Pipelines to run
    if args.pipeline == "both":
        pipelines_to_run = ["native", "sjp"]
    elif args.pipeline == "all":
        pipelines_to_run = ["native", "sjp", "gfrt"]
    else:
        pipelines_to_run = [args.pipeline]

    from .evaluation.metrics import format_metrics_table

    for ds_name in datasets_to_run:
        logger.info("═" * 60)
        logger.info("Dataset: %s  |  split: %s", ds_name.upper(), args.split)
        logger.info("═" * 60)

        # Load dataset
        dataset_obj = _load_dataset(ds_name, args)

        agg_by_pipeline: Dict[str, Dict] = {}

        for pipe_name in pipelines_to_run:
            logger.info("── Pipeline: %s ──", pipe_name)
            # Each (dataset, pipeline) pair gets its own LLM cache directory
            llm = build_llm(args, dataset_name=ds_name, pipeline_name=pipe_name)
            if pipe_name == "native":
                pipeline = build_native_pipeline(args, llm, dataset_obj=dataset_obj)
            elif pipe_name == "gfrt":
                # pipeline = build_gfrt_pipeline(args, llm, dataset_name=ds_name)
                pipeline = build_gfrt_pipeline(args, llm, dataset_obj=dataset_obj)
            else:
                pipeline = build_sjp_pipeline(args, llm)
            raw = run_pipeline_on_dataset(pipeline, dataset_obj, args)
            agg = evaluate_and_report(pipe_name, raw, args.output_dir, ds_name, args.split, budget=args.budget)
            agg_by_pipeline[pipe_name] = agg
            logger.info("%s  →  Hits@1=%.3f  F1=%.3f  EM=%.3f",
                        pipe_name, agg.get("hits@1", 0), agg.get("f1", 0),
                        agg.get("exact_match", 0))

        # Print comparison table
        table = format_metrics_table(
            agg_by_pipeline,
            title=f"{ds_name.upper()} [{args.split}] — {args.model}",
        )
        print(table)

        # Save table
        table_path = os.path.join(args.output_dir, f"comparison_{ds_name}_{args.split}_budget{args.budget}.txt")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(table_path, "w") as f:
            f.write(table)
        logger.info("Comparison table saved → %s", table_path)

    logger.info("Done.")


def _load_dataset(name: str, args):
    """Load WebQSP or CWQ by name."""
    if name == "webqsp":
        from .datasets.webqsp import WebQSPDataset
        ds = WebQSPDataset()
        ds.load(path=args.webqsp_path, split=args.split)
    else:
        from .datasets.cwq import CWQDataset
        ds = CWQDataset()
        ds.load(path=args.cwq_path, split=args.split)
    return ds


# Import constant for default SPARQL endpoint (used in argparse default)
_WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"


if __name__ == "__main__":
    main()
