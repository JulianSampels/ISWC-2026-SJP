"""
Harmonized interface for SJP and RETA dataset/model/candidate/ranking workflows.

This module intentionally stays thin and only orchestrates standardized adapter calls:
1) Generate standardized datasets from KgLoader dataset names.
2) Convert standardized datasets into SJP- and RETA-specific input layouts.
3) Train candidate-generation models.
4) Generate candidate sets.
5) Train ranking models.
6) Rank candidates.
7) Evaluate and compare outputs with shared metrics.

Run as a CLI:
    python -m iswc.harmonized.interface --help
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


def parse_k_values(raw: str) -> List[int]:
    parsed = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
    if not parsed:
        raise ValueError("k-values cannot be empty.")
    values = [int(chunk) for chunk in parsed]
    if any(v <= 0 for v in values):
        raise ValueError("All k-values must be positive integers.")
    return values


def _normalise_delimiter(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    if raw == "":
        return None
    if raw == "\\t":
        return "\t"
    return raw


def _resolve_default_reta_code_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "RETA_code"


def _validate_reta_top_nfilters(value: int, command: str) -> None:
    if value > 0:
        raise ValueError(
            f"--top-nfilters must be <= 0 for RETA {command}. "
            "RETA's filtering path expects non-positive alpha-style thresholds "
            "(e.g. -10) in this codebase."
        )


def _evaluate_metrics(stage: str, input_file: str | Path, gold_triples: str | Path, k_values: Sequence[int]):
    from iswc.harmonized.metrics import (
        evaluate_candidate_metrics_from_files,
        evaluate_ranked_metrics_from_files,
    )

    if stage == "candidates":
        return evaluate_candidate_metrics_from_files(
            candidate_csv=input_file,
            gold_triples_file=gold_triples,
            k_values=list(k_values),
        )
    return evaluate_ranked_metrics_from_files(
        ranked_csv=input_file,
        gold_triples_file=gold_triples,
        k_values=list(k_values),
    )


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Harmonized SJP/RETA interface for standardized dataset generation, "
            "adapter translation, model training, candidate generation, ranking, and evaluation."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    generate_standard = sub.add_parser(
        "generate-standard-dataset",
        help="Step 1: Generate standardized train/valid/test text files from a built-in KgLoader dataset name.",
    )
    generate_standard.add_argument("--dataset-name", required=True)
    generate_standard.add_argument("--output-dir", required=True)
    generate_standard.add_argument("--inverse-mode", choices=["manual", "automatic", "none"], default="none")
    generate_standard.add_argument("--overwrite", action="store_true", default=False)

    prepare_dataset = sub.add_parser(
        "prepare-dataset",
        help="Step 2: Prepare adapter-specific dataset files from a standardized dataset.",
    )
    prepare_dataset.add_argument("--adapter", choices=["sjp", "reta"], required=True)
    prepare_dataset.add_argument("--standard-dataset-dir", required=True)
    prepare_dataset.add_argument("--output-dir", required=True)
    prepare_dataset.add_argument("--triple-order", choices=["hrt", "htr"], default="hrt")
    prepare_dataset.add_argument("--delimiter", default=None, help="Optional delimiter. Use \\t for tab.")
    prepare_dataset.add_argument("--has-header", action="store_true", default=False)
    prepare_dataset.add_argument("--overwrite", action="store_true", default=False)
    prepare_dataset.add_argument("--num-paths-per-entity", type=int, default=20)
    prepare_dataset.add_argument("--num-steps", type=int, default=10)
    prepare_dataset.add_argument("--parallel", action=argparse.BooleanOptionalAction, default=True)
    prepare_dataset.add_argument("--inverse-mode", choices=["manual", "automatic", "none"], default="manual")
    prepare_dataset.add_argument("--default-entity-type", default="Thing")
    prepare_dataset.add_argument("--sjp-code-dir", default=None)
    prepare_dataset.add_argument("--reta-code-dir", default=None)

    train_candidate = sub.add_parser(
        "train-candidate-model",
        help="Step 3: Train model(s) used for candidate generation.",
    )
    train_candidate.add_argument("--adapter", choices=["sjp", "reta"], required=True)
    train_candidate.add_argument("--path-dataset-dir", default=None, help="SJP path dataset root (train/val/test).")
    train_candidate.add_argument("--path-setup", default="20_10")
    train_candidate.add_argument("--cmd", choices=["train", "resume", "test"], default="train")
    train_candidate.add_argument("--log-dir", default="./logs/harmonized")
    train_candidate.add_argument("--expname", default="harmonized_sjp")
    train_candidate.add_argument("--num-workers", type=int, default=1)
    train_candidate.add_argument("--max-epochs", type=int, default=100)
    train_candidate.add_argument("--tuple-checkpoint", default=None)
    train_candidate.add_argument("--skip-phase1", action="store_true", default=False)
    train_candidate.add_argument("--candidate-budget", type=int, default=500)
    train_candidate.add_argument("--sjp-code-dir", default=None)
    train_candidate.add_argument("--reta-code-dir", default=None)
    train_candidate.add_argument("--reta-data-dir", default=None, help="Path to RETA prepared dataset directory.")

    generate_candidates = sub.add_parser(
        "generate-candidates",
        help="Step 4: Generate standardized candidate CSV via the selected adapter.",
    )
    generate_candidates.add_argument("--adapter", choices=["sjp", "reta"], required=True)
    generate_candidates.add_argument("--output-file", required=True, help="Standardized candidate CSV output path.")
    generate_candidates.add_argument("--candidate-budget", type=int, default=500)
    generate_candidates.add_argument("--path-dataset-dir", default=None, help="SJP path dataset root (train/val/test).")
    generate_candidates.add_argument("--candidate-model-path", default=None, help="Trained candidate model path.")
    generate_candidates.add_argument("--path-setup", default="20_10")
    generate_candidates.add_argument("--log-dir", default="./logs/harmonized")
    generate_candidates.add_argument("--expname", default="harmonized_sjp")
    generate_candidates.add_argument("--num-workers", type=int, default=1)
    generate_candidates.add_argument("--max-epochs", type=int, default=100)
    generate_candidates.add_argument("--sjp-code-dir", default=None)
    generate_candidates.add_argument("--reta-code-dir", default=None)
    generate_candidates.add_argument("--reta-data-dir", default=None, help="Path to RETA prepared dataset directory.")
    generate_candidates.add_argument("--entities-evaluated", default="both", choices=["both", "one", "none"])
    generate_candidates.add_argument("--top-nfilters", type=int, default=-10)
    generate_candidates.add_argument("--at-least", type=int, default=2)
    generate_candidates.add_argument("--sparsifier", type=int, default=1)
    generate_candidates.add_argument("--build-type-dictionaries", default="True", choices=["True", "False"])
    generate_candidates.add_argument("--device", default="cuda:0")
    generate_candidates.add_argument("--max-facts", type=int, default=None)
    generate_candidates.add_argument("--map-to-sjp-dataset-dir", default=None)

    train_ranking = sub.add_parser(
        "train-ranking-model",
        help="Step 5: Train model(s) used for final ranking.",
    )
    train_ranking.add_argument("--adapter", choices=["sjp", "reta"], required=True)
    train_ranking.add_argument("--path-dataset-dir", default=None, help="SJP path dataset root (train/val/test).")
    train_ranking.add_argument("--candidate-model-path", default=None, help="Trained SJP candidate model path.")
    train_ranking.add_argument("--path-setup", default="20_10")
    train_ranking.add_argument("--cmd", choices=["train", "resume", "test"], default="train")
    train_ranking.add_argument("--log-dir", default="./logs/harmonized")
    train_ranking.add_argument("--expname", default="harmonized_sjp")
    train_ranking.add_argument("--num-workers", type=int, default=1)
    train_ranking.add_argument("--max-epochs", type=int, default=100)
    train_ranking.add_argument("--triple-checkpoint", default=None)
    train_ranking.add_argument("--skip-phase2", action="store_true", default=False)
    train_ranking.add_argument("--candidate-budget", type=int, default=500)
    train_ranking.add_argument("--sjp-code-dir", default=None)
    train_ranking.add_argument("--reta-code-dir", default=None)
    train_ranking.add_argument("--reta-data-dir", default=None, help="Path to RETA prepared dataset directory.")
    train_ranking.add_argument("--model-output-dir", default=None, help="Directory where RETA model is saved.")
    train_ranking.add_argument("--epochs", type=int, default=300)
    train_ranking.add_argument("--batchsize", type=int, default=128)
    train_ranking.add_argument("--num-filters", type=int, default=200)
    train_ranking.add_argument("--embsize", type=int, default=100)
    train_ranking.add_argument("--learningrate", type=float, default=0.0002)
    train_ranking.add_argument("--with-types", default="True", choices=["True", "False"])
    train_ranking.add_argument("--gpu-ids", default="0")
    train_ranking.add_argument("--at-least", type=int, default=2)
    train_ranking.add_argument("--top-nfilters", type=int, default=-10)
    train_ranking.add_argument("--build-type-dictionaries", default="True", choices=["True", "False"])
    train_ranking.add_argument("--sparsifier", type=int, default=1)
    train_ranking.add_argument("--entities-evaluated", default="both", choices=["both", "one", "none"])
    train_ranking.add_argument("--num-negative-samples", type=int, default=1)
    train_ranking.add_argument("--negative-strategy", type=float, default=0.0)
    train_ranking.add_argument("--load", default="False", choices=["False", "preload", "True"])
    train_ranking.add_argument("--model-to-be-trained", default="")

    rank_candidates = sub.add_parser(
        "rank-candidates",
        help="Step 6: Rank standardized candidates via the selected adapter.",
    )
    rank_candidates.add_argument("--adapter", choices=["sjp", "reta"], required=True)
    rank_candidates.add_argument("--candidate-file", required=True, help="Standardized candidate CSV input path.")
    rank_candidates.add_argument("--output-file", required=True, help="Standardized ranked CSV output path.")
    rank_candidates.add_argument("--ranking-model-path", required=True, help="Trained ranking model path.")
    rank_candidates.add_argument("--candidate-budget", type=int, default=500)
    rank_candidates.add_argument("--path-dataset-dir", default=None, help="SJP path dataset root (train/val/test).")
    rank_candidates.add_argument("--path-setup", default="20_10")
    rank_candidates.add_argument("--log-dir", default="./logs/harmonized")
    rank_candidates.add_argument("--expname", default="harmonized_sjp")
    rank_candidates.add_argument("--num-workers", type=int, default=1)
    rank_candidates.add_argument("--max-epochs", type=int, default=100)
    rank_candidates.add_argument("--sjp-code-dir", default=None)
    rank_candidates.add_argument("--reta-code-dir", default=None)
    rank_candidates.add_argument("--reta-data-dir", default=None, help="Path to RETA prepared dataset directory.")
    rank_candidates.add_argument("--entities-evaluated", default="both", choices=["both", "one", "none"])
    rank_candidates.add_argument("--top-nfilters", type=int, default=-10)
    rank_candidates.add_argument("--at-least", type=int, default=2)
    rank_candidates.add_argument("--sparsifier", type=int, default=1)
    rank_candidates.add_argument("--build-type-dictionaries", default="True", choices=["True", "False"])
    rank_candidates.add_argument("--device", default="cuda:0")
    rank_candidates.add_argument("--max-facts", type=int, default=None)
    rank_candidates.add_argument("--map-to-sjp-dataset-dir", default=None)

    evaluate = sub.add_parser(
        "evaluate",
        help="Step 7a: Evaluate one candidate/ranked CSV against gold triples.",
    )
    evaluate.add_argument("--stage", choices=["candidates", "ranking"], required=True)
    evaluate.add_argument("--input-file", required=True)
    evaluate.add_argument("--gold-triples", required=True)
    evaluate.add_argument("--k-values", default="1,3,5,10")
    evaluate.add_argument("--name", default="Method")
    evaluate.add_argument("--output-csv", default=None)

    compare = sub.add_parser(
        "compare",
        help="Step 7b: Compare multiple candidate/ranked CSV files against gold triples.",
    )
    compare.add_argument("--stage", choices=["candidates", "ranking"], required=True)
    compare.add_argument("--gold-triples", required=True)
    compare.add_argument("--k-values", default="1,3,5,10")
    compare.add_argument(
        "--method",
        action="append",
        nargs=2,
        metavar=("NAME", "INPUT_FILE"),
        required=True,
        help="Add a method by name and input CSV path. Can be repeated.",
    )
    compare.add_argument("--output-json", default=None)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = _build_cli()
    args = parser.parse_args(argv)

    if args.command == "generate-standard-dataset":
        from iswc.harmonized.dataset import generate_standardized_dataset_from_kgloader

        summary = generate_standardized_dataset_from_kgloader(
            dataset_name=args.dataset_name,
            output_dir=args.output_dir,
            inverse_mode=args.inverse_mode,
            overwrite=args.overwrite,
        )
        summary["workflow_step"] = "generate-standard-dataset"
        print(json.dumps(summary, indent=2))
        return

    if args.command == "prepare-dataset":
        from iswc.harmonized.adapters import RETAAdapter, SJPAdapter

        delimiter = _normalise_delimiter(args.delimiter)
        if args.adapter == "sjp":
            adapter = SJPAdapter(sjp_code_dir=args.sjp_code_dir)
            summary = adapter.prepare_dataset(
                standardized_dataset_dir=args.standard_dataset_dir,
                output_dir=args.output_dir,
                num_paths_per_entity=args.num_paths_per_entity,
                num_steps=args.num_steps,
                parallel=bool(args.parallel),
                inverse_mode=args.inverse_mode,
                triple_order=args.triple_order,
                delimiter=delimiter,
                has_header=args.has_header,
                overwrite=args.overwrite,
            )
        else:
            reta_code_dir = Path(args.reta_code_dir).resolve() if args.reta_code_dir is not None else _resolve_default_reta_code_dir()
            adapter = RETAAdapter(reta_code_dir=reta_code_dir, reta_data_dir=args.output_dir)
            summary = adapter.prepare_dataset(
                standardized_dataset_dir=args.standard_dataset_dir,
                output_dir=args.output_dir,
                default_entity_type=args.default_entity_type,
                triple_order=args.triple_order,
                delimiter=delimiter,
                has_header=args.has_header,
                overwrite=args.overwrite,
            )

        summary["workflow_step"] = "prepare-dataset"
        summary["adapter"] = args.adapter.upper()
        print(json.dumps(summary, indent=2))
        return

    if args.command == "train-candidate-model":
        from iswc.harmonized.adapters import RETAAdapter, SJPAdapter

        if args.adapter == "sjp":
            if args.path_dataset_dir is None:
                raise ValueError("--path-dataset-dir is required when --adapter sjp")
            adapter = SJPAdapter(sjp_code_dir=args.sjp_code_dir)
            summary = adapter.train_candidate_model(
                path_dataset_dir=args.path_dataset_dir,
                path_setup=args.path_setup,
                cmd=args.cmd,
                log_dir=args.log_dir,
                expname=args.expname,
                num_workers=args.num_workers,
                max_epochs=args.max_epochs,
                tuple_checkpoint=args.tuple_checkpoint,
                skip_phase1=args.skip_phase1,
                candidate_budget=args.candidate_budget,
            )
        else:
            reta_code_dir = Path(args.reta_code_dir).resolve() if args.reta_code_dir is not None else _resolve_default_reta_code_dir()
            if args.reta_data_dir is None:
                raise ValueError("--reta-data-dir is required when --adapter reta")
            adapter = RETAAdapter(reta_code_dir=reta_code_dir, reta_data_dir=args.reta_data_dir)
            summary = adapter.train_candidate_model()

        summary["workflow_step"] = "train-candidate-model"
        print(json.dumps(summary, indent=2))
        return

    if args.command == "generate-candidates":
        from iswc.harmonized.adapters import RETAAdapter, SJPAdapter

        if Path(args.output_file).suffix.lower() != ".csv":
            raise ValueError("--output-file must end with .csv for generate-candidates")

        if args.adapter == "sjp":
            if args.path_dataset_dir is None:
                raise ValueError("--path-dataset-dir is required when --adapter sjp")
            if args.candidate_model_path is None:
                raise ValueError("--candidate-model-path is required when --adapter sjp")

            adapter = SJPAdapter(sjp_code_dir=args.sjp_code_dir)
            predictions = adapter.generate_candidates(
                output_file=args.output_file,
                candidate_budget=args.candidate_budget,
                path_dataset_dir=args.path_dataset_dir,
                candidate_model_path=args.candidate_model_path,
                path_setup=args.path_setup,
                log_dir=args.log_dir,
                expname=args.expname,
                num_workers=args.num_workers,
                max_epochs=args.max_epochs,
            )
            summary = {
                "workflow_step": "generate-candidates",
                "adapter": "SJP",
                "output_file": str(Path(args.output_file).resolve()),
                "candidate_budget": int(args.candidate_budget),
                "num_heads": len(predictions),
                "path_dataset_dir": str(Path(args.path_dataset_dir).resolve()),
                "candidate_model_path": str(Path(args.candidate_model_path).resolve()),
            }
            print(json.dumps(summary, indent=2))
            return

        reta_code_dir = Path(args.reta_code_dir).resolve() if args.reta_code_dir is not None else _resolve_default_reta_code_dir()
        if args.reta_data_dir is None:
            raise ValueError("--reta-data-dir is required when --adapter reta")
        _validate_reta_top_nfilters(int(args.top_nfilters), "generate-candidates")

        adapter = RETAAdapter(reta_code_dir=reta_code_dir, reta_data_dir=args.reta_data_dir)
        predictions = adapter.generate_candidates(
            output_file=args.output_file,
            candidate_budget=args.candidate_budget,
            entities_evaluated=args.entities_evaluated,
            top_nfilters=args.top_nfilters,
            at_least=args.at_least,
            sparsifier=args.sparsifier,
            build_type_dictionaries=args.build_type_dictionaries,
            device=args.device,
            max_facts=args.max_facts,
            map_to_sjp_dataset_dir=args.map_to_sjp_dataset_dir,
        )
        summary = {
            "workflow_step": "generate-candidates",
            "adapter": "RETA",
            "output_file": str(Path(args.output_file).resolve()),
            "candidate_budget": int(args.candidate_budget),
            "num_heads": len(predictions),
            "reta_data_dir": str(Path(args.reta_data_dir).resolve()),
        }
        print(json.dumps(summary, indent=2))
        return

    if args.command == "train-ranking-model":
        from iswc.harmonized.adapters import RETAAdapter, SJPAdapter

        if args.adapter == "sjp":
            if args.path_dataset_dir is None:
                raise ValueError("--path-dataset-dir is required when --adapter sjp")
            if args.candidate_model_path is None:
                raise ValueError("--candidate-model-path is required when --adapter sjp")

            adapter = SJPAdapter(sjp_code_dir=args.sjp_code_dir)
            summary = adapter.train_ranking_model(
                path_dataset_dir=args.path_dataset_dir,
                candidate_model_path=args.candidate_model_path,
                path_setup=args.path_setup,
                cmd=args.cmd,
                log_dir=args.log_dir,
                expname=args.expname,
                num_workers=args.num_workers,
                max_epochs=args.max_epochs,
                triple_checkpoint=args.triple_checkpoint,
                candidate_budget=args.candidate_budget,
                skip_phase2=args.skip_phase2,
            )
        else:
            reta_code_dir = Path(args.reta_code_dir).resolve() if args.reta_code_dir is not None else _resolve_default_reta_code_dir()
            if args.reta_data_dir is None:
                raise ValueError("--reta-data-dir is required when --adapter reta")
            if args.model_output_dir is None:
                raise ValueError("--model-output-dir is required when --adapter reta")
            _validate_reta_top_nfilters(int(args.top_nfilters), "train-ranking-model")

            adapter = RETAAdapter(reta_code_dir=reta_code_dir, reta_data_dir=args.reta_data_dir)
            summary = adapter.train_ranking_model(
                model_output_dir=args.model_output_dir,
                epochs=args.epochs,
                batchsize=args.batchsize,
                num_filters=args.num_filters,
                embsize=args.embsize,
                learningrate=args.learningrate,
                with_types=args.with_types,
                gpu_ids=args.gpu_ids,
                at_least=args.at_least,
                top_nfilters=args.top_nfilters,
                build_type_dictionaries=args.build_type_dictionaries,
                sparsifier=args.sparsifier,
                entities_evaluated=args.entities_evaluated,
                num_negative_samples=args.num_negative_samples,
                negative_strategy=args.negative_strategy,
                load=args.load,
                model_to_be_trained=args.model_to_be_trained,
            )

        summary["workflow_step"] = "train-ranking-model"
        print(json.dumps(summary, indent=2))
        return

    if args.command == "rank-candidates":
        from iswc.harmonized.adapters import RETAAdapter, SJPAdapter

        if Path(args.candidate_file).suffix.lower() != ".csv":
            raise ValueError("--candidate-file must end with .csv for rank-candidates")
        if Path(args.output_file).suffix.lower() != ".csv":
            raise ValueError("--output-file must end with .csv for rank-candidates")

        if args.adapter == "sjp":
            if args.path_dataset_dir is None:
                raise ValueError("--path-dataset-dir is required when --adapter sjp")
            adapter = SJPAdapter(sjp_code_dir=args.sjp_code_dir)
            predictions = adapter.rank_candidates(
                candidate_file=args.candidate_file,
                output_file=args.output_file,
                candidate_budget=args.candidate_budget,
                path_dataset_dir=args.path_dataset_dir,
                ranking_model_path=args.ranking_model_path,
                path_setup=args.path_setup,
                log_dir=args.log_dir,
                expname=args.expname,
                num_workers=args.num_workers,
                max_epochs=args.max_epochs,
            )
            summary = {
                "workflow_step": "rank-candidates",
                "adapter": "SJP",
                "candidate_file": str(Path(args.candidate_file).resolve()),
                "output_file": str(Path(args.output_file).resolve()),
                "candidate_budget": int(args.candidate_budget),
                "num_heads": len(predictions),
                "ranking_model_path": str(Path(args.ranking_model_path).resolve()),
            }
            print(json.dumps(summary, indent=2))
            return

        reta_code_dir = Path(args.reta_code_dir).resolve() if args.reta_code_dir is not None else _resolve_default_reta_code_dir()
        if args.reta_data_dir is None:
            raise ValueError("--reta-data-dir is required when --adapter reta")
        _validate_reta_top_nfilters(int(args.top_nfilters), "rank-candidates")

        adapter = RETAAdapter(reta_code_dir=reta_code_dir, reta_data_dir=args.reta_data_dir)
        predictions = adapter.rank_candidates(
            candidate_file=args.candidate_file,
            ranking_model_path=args.ranking_model_path,
            output_file=args.output_file,
            candidate_budget=args.candidate_budget,
            entities_evaluated=args.entities_evaluated,
            top_nfilters=args.top_nfilters,
            at_least=args.at_least,
            sparsifier=args.sparsifier,
            build_type_dictionaries=args.build_type_dictionaries,
            device=args.device,
            max_facts=args.max_facts,
            map_to_sjp_dataset_dir=args.map_to_sjp_dataset_dir,
        )
        summary = {
            "workflow_step": "rank-candidates",
            "adapter": "RETA",
            "candidate_file": str(Path(args.candidate_file).resolve()),
            "output_file": str(Path(args.output_file).resolve()),
            "candidate_budget": int(args.candidate_budget),
            "num_heads": len(predictions),
            "reta_data_dir": str(Path(args.reta_data_dir).resolve()),
            "ranking_model_path": str(Path(args.ranking_model_path).resolve()),
        }
        print(json.dumps(summary, indent=2))
        return

    if args.command == "evaluate":
        if Path(args.input_file).suffix.lower() != ".csv":
            raise ValueError("--input-file must end with .csv")

        k_values = parse_k_values(args.k_values)
        metrics_df = _evaluate_metrics(
            stage=args.stage,
            input_file=args.input_file,
            gold_triples=args.gold_triples,
            k_values=k_values,
        )

        metrics = metrics_df.to_dict(orient="records")

        payload = {
            "workflow_step": "evaluate",
            "name": args.name,
            "stage": args.stage,
            "input_file": str(Path(args.input_file).resolve()),
            "gold_triples": str(Path(args.gold_triples).resolve()),
            "metrics": metrics,
        }

        if args.output_csv is not None:
            from iswc.harmonized.metrics import save_metrics_csv

            save_metrics_csv(metrics_df, args.output_csv)
            payload["output_csv"] = str(Path(args.output_csv).resolve())

        print(json.dumps(payload, indent=2))
        return

    if args.command == "compare":
        k_values = parse_k_values(args.k_values)
        compared: Dict[str, List[Dict[Any, Any]]] = {}

        for name, input_file in args.method:
            metrics_df = _evaluate_metrics(
                stage=args.stage,
                input_file=input_file,
                gold_triples=args.gold_triples,
                k_values=k_values,
            )
            compared[name] = metrics_df.to_dict(orient="records")

        payload: Dict[str, Any] = {
            "workflow_step": "compare",
            "stage": args.stage,
            "gold_triples": str(Path(args.gold_triples).resolve()),
            "k_values": list(k_values),
            "results": compared,
        }

        if args.output_json is not None:
            output_path = Path(args.output_json).resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            payload["output_json"] = str(output_path)

        print(json.dumps(payload, indent=2))
        return

    raise ValueError(f"Unsupported command {args.command}")


if __name__ == "__main__":
    main()
