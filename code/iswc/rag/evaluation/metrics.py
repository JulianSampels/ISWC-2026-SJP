"""
Standard KGQA evaluation metrics for RAG pipelines.

Metric definitions
------------------

Hits@K
    For each question, 1 if any gold answer appears in the top-K predicted
    answers, else 0.  Averaged across questions (macro = micro for binary
    values, so the simple mean is correct).

    The standard metric on WebQSP (Yih et al., 2016).

Exact Match (EM)
    1 if the top-1 predicted answer exactly matches any gold answer
    (after normalisation), else 0.  Averaged across questions.

Answer F1 / Precision / Recall (set-level, micro-averaged)
    For each question, treat predictions and gold answers as *sets* of
    normalised answer strings (not token bags):

        TP_i = |predicted_set_i  ∩  gold_set_i|
        P_i  = TP_i / |predicted_set_i|
        R_i  = TP_i / |gold_set_i|

    Then micro-aggregate across all questions:

        micro-Precision = Σ TP_i / Σ |predicted_set_i|
        micro-Recall    = Σ TP_i / Σ |gold_set_i|
        micro-F1        = 2 · micro-P · micro-R / (micro-P + micro-R)

    Why set-level?
        KGQA answers are entity names, not text spans.  Concatenating all
        predictions into one string and comparing against the best single
        gold answer (the old approach) artificially inflates recall for
        multi-answer questions (e.g. gold = {Paris, Rome, Berlin},
        predicted = {Paris} → old recall = 1.0, correct recall = 1/3).

    Why micro instead of macro (simple mean of per-sample F1)?
        Micro-averaging weights by question "difficulty" (number of answers).
        Macro-averaging treats a 1-answer question and a 10-answer question
        identically, even though the per-sample F1 values are not comparable.
        Micro is the standard for multi-answer KGQA benchmarks (CWQ, WebQSP).

    Note: per-sample F1 values are still stored (for per-question analysis)
    but aggregate_metrics uses micro-averaged F1/P/R.

Normalisation applied before all comparisons
    - lowercase
    - strip leading/trailing whitespace
    - remove articles ("a", "an", "the")
    - remove punctuation
    - collapse multiple spaces

Usage example
-------------
    from evaluation.metrics import evaluate_results, aggregate_metrics

    per_sample = evaluate_results(pipeline_results)
    summary = aggregate_metrics(per_sample)
    print(summary)
    # {
    #   "hits@1": 0.612,     # macro (= micro for binary)
    #   "hits@3": 0.721,
    #   "exact_match": 0.518,
    #   "f1": 0.634,         # micro-averaged set-level
    #   "precision": 0.671,  # micro-averaged set-level
    #   "recall": 0.601,     # micro-averaged set-level
    #   "n": 1628
    # }
"""
import re
import string
from collections import Counter
from typing import Dict, List, Optional, Tuple

from ..pipelines.base import PipelineResult


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

_ARTICLES = {"a", "an", "the"}
_PUNCT = set(string.punctuation)


def _normalise(text: str) -> str:
    """Lowercase, strip punctuation, remove articles, collapse whitespace."""
    text = text.lower().strip()
    text = "".join(ch if ch not in _PUNCT else " " for ch in text)
    tokens = [t for t in text.split() if t not in _ARTICLES]
    return " ".join(tokens)


def _tokenise(text: str) -> List[str]:
    return _normalise(text).split()


# ---------------------------------------------------------------------------
# Per-sample metrics
# ---------------------------------------------------------------------------

def hits_at_k(predicted: List[str], gold: List[str], k: int = 1) -> float:
    """
    Return 1.0 if any gold answer matches any of the top-k predictions.

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
    """
    Return 1.0 if the top-1 prediction exactly matches any gold answer.

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


def _set_f1_counts(
    predicted: List[str], gold: List[str]
) -> Tuple[int, int, int]:
    """
    Set-level TP / predicted-set size / gold-set size for one question.

    Uses normalised answer strings as atomic units (not token bags).  This
    is the appropriate granularity for entity-level KGQA answers.

    Returns:
        (tp, pred_n, gold_n) where
            tp     = |predicted_set ∩ gold_set|
            pred_n = |predicted_set|
            gold_n = |gold_set|
    """
    pred_set = {_normalise(p) for p in predicted} - {""}
    gold_set = {_normalise(g) for g in gold} - {""}
    tp = len(pred_set & gold_set)
    return tp, len(pred_set), len(gold_set)


def answer_f1(predicted: List[str], gold: List[str]) -> float:
    """
    Set-level F1 for a single question.

    Args:
        predicted: List of predicted answer strings.
        gold:      List of gold answer strings.

    Returns:
        F1 in [0, 1].
    """
    tp, pred_n, gold_n = _set_f1_counts(predicted, gold)
    if pred_n == 0 and gold_n == 0:
        return 1.0
    if pred_n == 0 or gold_n == 0:
        return 0.0
    p = tp / pred_n
    r = tp / gold_n
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def answer_precision(predicted: List[str], gold: List[str]) -> float:
    """
    Set-level precision for a single question.

    Args:
        predicted: List of predicted answer strings.
        gold:      List of gold answer strings.

    Returns:
        Precision in [0, 1].
    """
    tp, pred_n, _ = _set_f1_counts(predicted, gold)
    return tp / pred_n if pred_n > 0 else 0.0


def answer_recall(predicted: List[str], gold: List[str]) -> float:
    """
    Set-level recall for a single question.

    Args:
        predicted: List of predicted answer strings.
        gold:      List of gold answer strings.

    Returns:
        Recall in [0, 1].
    """
    tp, _, gold_n = _set_f1_counts(predicted, gold)
    return tp / gold_n if gold_n > 0 else (1.0 if not gold else 0.0)


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_results(results: List[PipelineResult]) -> List[Dict]:
    """
    Compute all metrics for each PipelineResult.

    Per-sample dicts include both individual metric values (for per-question
    analysis) and the raw counts needed for micro-aggregation.

    Args:
        results: List of PipelineResult objects from a pipeline run.

    Returns:
        List of dicts, one per sample.  Keys:
            question_id, hits@1, hits@3, hits@5, exact_match,
            f1, precision, recall,      ← per-sample (for inspection)
            _tp, _pred_n, _gold_n,      ← raw counts for micro-aggregation
            num_retrieved
    """
    per_sample = []
    for r in results:
        tp, pred_n, gold_n = _set_f1_counts(r.predicted_answers, r.gold_answers)
        per_sample.append({
            "question_id":   r.question_id,
            "hits@1":        hits_at_k(r.predicted_answers, r.gold_answers, k=1),
            "hits@3":        hits_at_k(r.predicted_answers, r.gold_answers, k=3),
            "hits@5":        hits_at_k(r.predicted_answers, r.gold_answers, k=5),
            "exact_match":   exact_match(r.predicted_answers, r.gold_answers),
            # Per-sample F1/P/R (for per-question analysis / debugging)
            "f1":            answer_f1(r.predicted_answers, r.gold_answers),
            "precision":     answer_precision(r.predicted_answers, r.gold_answers),
            "recall":        answer_recall(r.predicted_answers, r.gold_answers),
            # Raw counts for correct micro-aggregation in aggregate_metrics()
            "_tp":           tp,
            "_pred_n":       pred_n,
            "_gold_n":       gold_n,
            "num_retrieved": len(r.retrieved_triples),
        })
    return per_sample


def aggregate_metrics(per_sample: List[Dict]) -> Dict[str, float]:
    """
    Aggregate per-sample metrics across the dataset.

    Aggregation strategy
    --------------------
    hits@K, exact_match
        Macro-average of binary (0/1) values.  For binary metrics
        macro-average == micro-average (sum/n = total_hits/n), so this
        is the globally correct fraction of questions answered correctly.

    f1, precision, recall
        Micro-averaged at the answer-set level, using the raw TP / pred_n /
        gold_n counts stored by evaluate_results():

            micro-P  = Σ _tp_i / Σ _pred_n_i
            micro-R  = Σ _tp_i / Σ _gold_n_i
            micro-F1 = 2 · micro-P · micro-R / (micro-P + micro-R)

        This counts every answer equally regardless of which question it
        came from, and correctly handles multi-answer questions.

    Args:
        per_sample: Output of evaluate_results().

    Returns:
        Dict with aggregated metrics and sample count (key "n").
    """
    if not per_sample:
        return {"n": 0}

    n = len(per_sample)
    agg: Dict[str, float] = {"n": float(n)}

    # --- Binary metrics: macro-average is correct (= micro for 0/1 values) ---
    for key in ("hits@1", "hits@3", "hits@5", "exact_match"):
        agg[key] = sum(s[key] for s in per_sample) / n

    # --- F1 / P / R: micro-average from accumulated set-level counts ---
    total_tp     = sum(s["_tp"]     for s in per_sample)
    total_pred_n = sum(s["_pred_n"] for s in per_sample)
    total_gold_n = sum(s["_gold_n"] for s in per_sample)

    micro_p = total_tp / total_pred_n if total_pred_n > 0 else 0.0
    micro_r = total_tp / total_gold_n if total_gold_n > 0 else 0.0
    micro_f1 = (
        2 * micro_p * micro_r / (micro_p + micro_r)
        if micro_p + micro_r > 0 else 0.0
    )

    agg["precision"] = micro_p
    agg["recall"]    = micro_r
    agg["f1"]        = micro_f1

    return agg


def format_metrics_table(
    results_by_pipeline: Dict[str, Dict[str, float]],
    title: str = "Evaluation Results",
) -> str:
    """
    Format a comparison table of aggregated metrics across pipelines.

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
