"""Public exports for the harmonized SJP/RETA interface package."""

__all__ = [
    "CandidateAdapter",
    "SJPAdapter",
    "RETAAdapter",
    "resolve_standardized_dataset",
    "load_standardized_dataset_triples",
    "generate_standardized_dataset_from_kgloader",
    "export_standardized_dataset_to_sjp",
    "export_standard_dataset_to_reta",
    "extract_reta_filter_candidates",
    "rank_reta_candidates",
    "load_ranked_predictions",
    "apply_candidate_budget",
    "save_ranked_predictions",
    "save_ranked_predictions_csv",
    "save_ranked_predictions_pt",
    "evaluate_candidates_csv",
    "evaluate_ranked_candidates_csv",
    "evaluate_candidate_metrics_from_files",
    "evaluate_ranked_metrics_from_files",
    "save_metrics_csv",
]


def __getattr__(name: str):
    if name in {"CandidateAdapter", "SJPAdapter", "RETAAdapter"}:
        from .adapters import CandidateAdapter, RETAAdapter, SJPAdapter

        return {
            "CandidateAdapter": CandidateAdapter,
            "SJPAdapter": SJPAdapter,
            "RETAAdapter": RETAAdapter,
        }[name]

    if name in {
        "resolve_standardized_dataset",
        "load_standardized_dataset_triples",
        "generate_standardized_dataset_from_kgloader",
        "export_standardized_dataset_to_sjp",
    }:
        from .dataset import (
            export_standardized_dataset_to_sjp,
            generate_standardized_dataset_from_kgloader,
            load_standardized_dataset_triples,
            resolve_standardized_dataset,
        )

        return {
            "resolve_standardized_dataset": resolve_standardized_dataset,
            "load_standardized_dataset_triples": load_standardized_dataset_triples,
            "generate_standardized_dataset_from_kgloader": generate_standardized_dataset_from_kgloader,
            "export_standardized_dataset_to_sjp": export_standardized_dataset_to_sjp,
        }[name]

    if name in {
        "export_standard_dataset_to_reta",
        "extract_reta_filter_candidates",
        "rank_reta_candidates",
        "load_ranked_predictions",
        "apply_candidate_budget",
        "save_ranked_predictions",
        "save_ranked_predictions_csv",
        "save_ranked_predictions_pt",
        "evaluate_candidates_csv",
        "evaluate_ranked_candidates_csv",
    }:
        from .interface import (
            apply_candidate_budget,
            evaluate_candidates_csv,
            evaluate_ranked_candidates_csv,
            export_standard_dataset_to_reta,
            extract_reta_filter_candidates,
            load_ranked_predictions,
            rank_reta_candidates,
            save_ranked_predictions,
            save_ranked_predictions_csv,
            save_ranked_predictions_pt,
        )

        return {
            "export_standard_dataset_to_reta": export_standard_dataset_to_reta,
            "extract_reta_filter_candidates": extract_reta_filter_candidates,
            "rank_reta_candidates": rank_reta_candidates,
            "load_ranked_predictions": load_ranked_predictions,
            "apply_candidate_budget": apply_candidate_budget,
            "save_ranked_predictions": save_ranked_predictions,
            "save_ranked_predictions_csv": save_ranked_predictions_csv,
            "save_ranked_predictions_pt": save_ranked_predictions_pt,
            "evaluate_candidates_csv": evaluate_candidates_csv,
            "evaluate_ranked_candidates_csv": evaluate_ranked_candidates_csv,
        }[name]

    if name in {
        "evaluate_candidate_metrics_from_files",
        "evaluate_ranked_metrics_from_files",
        "save_metrics_csv",
    }:
        from .metrics import (
            evaluate_candidate_metrics_from_files,
            evaluate_ranked_metrics_from_files,
            save_metrics_csv,
        )

        return {
            "evaluate_candidate_metrics_from_files": evaluate_candidate_metrics_from_files,
            "evaluate_ranked_metrics_from_files": evaluate_ranked_metrics_from_files,
            "save_metrics_csv": save_metrics_csv,
        }[name]

    raise AttributeError(f"module 'iswc.harmonized' has no attribute '{name}'")
