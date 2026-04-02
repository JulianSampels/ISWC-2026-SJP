"""Public exports for the harmonized SJP/RETA interface package."""

__all__ = [
    "CandidateAdapter",
    "SJPAdapter",
    "RETAAdapter",
    "resolve_standardized_dataset",
    "load_standardized_dataset_triples",
    "generate_standardized_dataset_from_kgloader",
    "export_standardized_dataset_to_sjp",
    "build_reta_dictionaries",
    "export_standard_dataset_to_reta",
    "extract_reta_filter_candidates",
    "rank_reta_candidates",
    "load_ranked_predictions",
    "apply_candidate_budget",
    "save_ranked_predictions",
    "save_ranked_predictions_csv",
    "save_ranked_predictions_pt",
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
        "build_reta_dictionaries",
        "export_standard_dataset_to_reta",
        "extract_reta_filter_candidates",
        "rank_reta_candidates",
        "load_ranked_predictions",
        "apply_candidate_budget",
        "save_ranked_predictions",
        "save_ranked_predictions_csv",
        "save_ranked_predictions_pt",
    }:
        from .interface import (
            apply_candidate_budget,
            build_reta_dictionaries,
            export_standard_dataset_to_reta,
            extract_reta_filter_candidates,
            load_ranked_predictions,
            rank_reta_candidates,
            save_ranked_predictions,
            save_ranked_predictions_csv,
            save_ranked_predictions_pt,
        )

        return {
            "build_reta_dictionaries": build_reta_dictionaries,
            "export_standard_dataset_to_reta": export_standard_dataset_to_reta,
            "extract_reta_filter_candidates": extract_reta_filter_candidates,
            "rank_reta_candidates": rank_reta_candidates,
            "load_ranked_predictions": load_ranked_predictions,
            "apply_candidate_budget": apply_candidate_budget,
            "save_ranked_predictions": save_ranked_predictions,
            "save_ranked_predictions_csv": save_ranked_predictions_csv,
            "save_ranked_predictions_pt": save_ranked_predictions_pt,
        }[name]

    raise AttributeError(f"module 'iswc.harmonized' has no attribute '{name}'")
