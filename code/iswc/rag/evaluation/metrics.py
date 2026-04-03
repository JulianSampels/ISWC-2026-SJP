"""
Standard KGQA evaluation metrics for RAG pipelines.

Metric definitions
------------------

Hits@K
    For each question, check whether any gold answer appears in the top-K
    predicted answers.  Averaged across questions.

    The standard metric on WebQSP (Yih et al., 2016) and the primary metric
    used in many KG-RAG papers (He et al., 2021; Sun et al., 2023).

Exact Match (EM)
    1 if the top-1 predicted answer exactly matches any gold answer
    (after normalisation), else 0.  Averaged across questions.

    Strict version of Hits@1; penalises partial string matches.

Answer F1 / Precision / Recall
    Treat the set of predicted answers and the set of gold answers as bags
    and compute token-overlap F1 between them (micro-averaged).

    The standard metric on CWQ (Talmor & Berant, 2018) and SQuAD-style
    evaluation.  Handles multi-answer questions naturally.

    Normalisation applied before comparison:
      - lowercase
      - strip leading/trailing whitespace
      - remove articles ("a", "an", "the")
      - collapse multiple spaces

Usage example
-------------
    from evaluation.metrics import evaluate_results, aggregate_metrics

    per_sample = evaluate_results(pipeline_results)
    summary = aggregate_metrics(per_sample)
    print(summary)
    # {
    #   "hits@1": 0.612,
    #   "hits@3": 0.721,
    #   "exact_match": 0.518,
    #   "f1": 0.634,
    #   "precision": 0.671,
    #   "recall": 0.601,
    #   "n": 1628
    # }
"""
import re
import string
from collections import Counter
from typing import Dict, List, Optional

from ..pipelines.base import PipelineResult


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

_ARTICLES = {"a", "an", "the"}
_PUNCT = set(string.punctuation)


def _normalise(text: str) -> str:
    """Lowercase, strip punctuation, remove articles, collapse whitespace."""
    text = text.lower().strip()
    # Remove punctuation
    text = "".join(ch if ch not in _PUNCT else " " for ch in text)
    # Remove articles
    tokens = [t for t in text.split() if t not in _ARTICLES]
    return " ".join(tokens)


def _tokenise(text: str) -> List[str]:
    return _normalise(text).split()


# ---------------------------------------------------------------------------
# Per-sample metrics
# ---------------------------------------------------------------------------

def hits_at_k(predicted: List[str], gold: List[str], k: int = 1) -> float:
    """Return 1.0 if any gold answer matches any of the top-k predictions.

    Args:
        predicted: Ordered list of predicted answers (most confident first).
        gold:      List of gold answer strings.
        k:         How many predictions to consider.

    Returns:
        1.0 (hit) or 0.0 (miss).
    """
    gold_norm = {_normalise(g) for g in gold}
    for pred in predicted[:k]:
        if _normalise(pred) in gold_norm:
            return 1.0
    return 0.0


def exact_match(predicted: List[str], gold: List[str]) -> float:
    """Return 1.0 if the top-1 prediction exactly matches any gold answer.

    Args:
        predicted: List of predicted answers (top-1 is predicted[0]).
        gold:      List of gold answer strings.

    Returns:
        1.0 or 0.0.
    """
    if not predicted:
        return 0.0
    pred_norm = _normalise(predicted[0])
    return float(any(_normalise(g) == pred_norm for g in gold))


def answer_precision(predicted: List[str], gold: List[str]) -> float:
    """Token-level precision: fraction of predicted tokens that are in gold.

    For multi-answer questions the best-matching gold answer is used.

    Args:
        predicted: List of predicted answer strings.
        gold:      List of gold answer strings.

    Returns:
        Precision in [0, 1].
    """
    if not predicted:
        return 0.0
    pred_tokens = Counter(_tokenise(" ".join(predicted)))
    if sum(pred_tokens.values()) == 0:
        return 0.0
    best = 0.0
    for g in gold:
        gold_tokens = Counter(_tokenise(g))
        common = sum((pred_tokens & gold_tokens).values())
        p = common / sum(pred_tokens.values())
        best = max(best, p)
    return best


def answer_recall(predicted: List[str], gold: List[str]) -> float:
    """Token-level recall: fraction of gold tokens that are in predicted.

    For multi-answer questions the best-matching gold answer is used.

    Args:
        predicted: List of predicted answer strings.
        gold:      List of gold answer strings.

    Returns:
        Recall in [0, 1].
    """
    if not gold:
        return 1.0
    pred_tokens = Counter(_tokenise(" ".join(predicted)))
    best = 0.0
    for g in gold:
        gold_tokens = Counter(_tokenise(g))
        if sum(gold_tokens.values()) == 0:
            continue
        common = sum((pred_tokens & gold_tokens).values())
        r = common / sum(gold_tokens.values())
        best = max(best, r)
    return best


def answer_f1(predicted: List[str], gold: List[str]) -> float:
    """Harmonic mean of token-level precision and recall.

    This is the standard F1 metric used on CWQ and WebQSP.

    Args:
        predicted: List of predicted answer strings.
        gold:      List of gold answer strings.

    Returns:
        F1 in [0, 1].
    """
    p = answer_precision(predicted, gold)
    r = answer_recall(predicted, gold)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_results(results: List[PipelineResult]) -> List[Dict[str, float]]:
    """Compute all metrics for each PipelineResult.

    Args:
        results: List of PipelineResult objects from a pipeline run.

    Returns:
        List of dicts, one per sample, with keys:
          hits@1, hits@3, hits@5, exact_match, f1, precision, recall,
          num_retrieved (number of triples returned by the retriever).
    """
    per_sample = []
    for r in results:
        per_sample.append({
            "question_id":   r.question_id,
            "hits@1":        hits_at_k(r.predicted_answers, r.gold_answers, k=1),
            "hits@3":        hits_at_k(r.predicted_answers, r.gold_answers, k=3),
            "hits@5":        hits_at_k(r.predicted_answers, r.gold_answers, k=5),
            "exact_match":   exact_match(r.predicted_answers, r.gold_answers),
            "f1":            answer_f1(r.predicted_answers, r.gold_answers),
            "precision":     answer_precision(r.predicted_answers, r.gold_answers),
            "recall":        answer_recall(r.predicted_answers, r.gold_answers),
            "num_retrieved": len(r.retrieved_triples),
        })
    return per_sample


def aggregate_metrics(per_sample: List[Dict[str, float]]) -> Dict[str, float]:
    """Average per-sample metrics across the dataset.

    Args:
        per_sample: Output of evaluate_results().

    Returns:
        Dict with macro-averaged metrics and sample count (key "n").
    """
    if not per_sample:
        return {"n": 0}
    metric_keys = ["hits@1", "hits@3", "hits@5", "exact_match", "f1", "precision", "recall"]
    agg: Dict[str, float] = {"n": float(len(per_sample))}
    for key in metric_keys:
        agg[key] = sum(s[key] for s in per_sample) / len(per_sample)
    return agg


def format_metrics_table(
    results_by_pipeline: Dict[str, Dict[str, float]],
    title: str = "Evaluation Results",
) -> str:
    """Format a comparison table of aggregated metrics across pipelines.

    Args:
        results_by_pipeline: Dict mapping pipeline name → aggregate metrics dict.
        title:               Table title string.

    Returns:
        A formatted string table ready to print or log.

    Example output::

        === Evaluation Results ===
        Pipeline        Hits@1   Hits@3   Hits@5   EM       F1       P        R        N
        native_rag      0.612    0.721    0.758    0.518    0.634    0.671    0.601    1628
        sjp_rag         0.683    0.779    0.812    0.591    0.702    0.735    0.672    1628
    """
    cols = ["hits@1", "hits@3", "hits@5", "exact_match", "f1", "precision", "recall", "n"]
    col_labels = ["Hits@1", "Hits@3", "Hits@5", "EM", "F1", "P", "R", "N"]

    header = f"{'Pipeline':<20}" + "".join(f"{lbl:<9}" for lbl in col_labels)
    divider = "-" * len(header)

    rows = [f"\n=== {title} ===", header, divider]
    for pipeline, metrics in results_by_pipeline.items():
        cells = []
        for col in cols:
            val = metrics.get(col, 0.0)
            if col == "n":
                cells.append(f"{int(val):<9}")
            else:
                cells.append(f"{val:.3f}    ")
        rows.append(f"{pipeline:<20}" + "".join(cells))
    rows.append("")
    return "\n".join(rows)
