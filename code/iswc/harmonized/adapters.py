"""Compatibility exports for harmonized adapters.

New code should import from:
- iswc.harmonized.base_adapter
- iswc.harmonized.sjp_adapter
- iswc.harmonized.reta_adapter
"""

from .base_adapter import (
    CandidateAdapter,
    RankedPredictions,
    apply_candidate_budget,
    load_ranked_predictions,
    save_ranked_predictions,
)
from .reta_adapter import (
    RETAAdapter,
    build_reta_dictionaries,
    export_standard_dataset_to_reta,
    extract_reta_filter_candidates,
    rank_reta_candidates,
)
from .sjp_adapter import SJPAdapter


__all__ = [
    "CandidateAdapter",
    "RankedPredictions",
    "SJPAdapter",
    "RETAAdapter",
    "apply_candidate_budget",
    "load_ranked_predictions",
    "save_ranked_predictions",
    "build_reta_dictionaries",
    "export_standard_dataset_to_reta",
    "extract_reta_filter_candidates",
    "rank_reta_candidates",
]
