from .bpr_recoin import (
    BaseRelationPredictor,
    BPRRelationPredictor,
    RecoinRelationPredictor,
)

from .kgc_tails import (
    PyKEENWrapper,
    PyKEENTrainer,
    build_kgc_model,
    PYKEEN_MODELS,
    MODEL_REGISTRY,   # alias for PYKEEN_MODELS
)

from .pipeline import InstanceCompletionPipeline

try:
    from .gfrt import build_gfrt_pipeline, GFRTTrainer, GFRTFilter
    _GFRT_AVAILABLE = True
except ImportError:
    _GFRT_AVAILABLE = False

__all__ = [
    # Adapted frequency baselines (RQ1)
    "RelationFirstBaseline",
    "IndependentCombinationBaseline",
    "TailFirstBaseline",
    # Relation predictors (Stage 1)
    "BaseRelationPredictor",
    "BPRRelationPredictor",
    "RecoinRelationPredictor",
    # PyKEEN-backed KGC tail prediction (Stage 2)
    "PyKEENWrapper",
    "PyKEENTrainer",
    "build_kgc_model",
    "PYKEEN_MODELS",
    "MODEL_REGISTRY",
    # Two-stage instance completion pipeline
    "InstanceCompletionPipeline",
    # GFRT graph baseline
    "build_gfrt_pipeline",
    "GFRTTrainer",
    "GFRTFilter",
]
