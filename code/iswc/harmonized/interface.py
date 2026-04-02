"""
Harmonized interface for SJP and RETA dataset/candidate/ranking workflows.

This module provides a cleaned harmonized workflow:
1) Generate standardized datasets from KgLoader dataset names.
2) Convert standardized datasets into SJP- and RETA-specific input layouts.
3) Build RETA dictionaries_and_facts.bin without modifying RETA_code.
4) Load/save standardized candidate and ranked CSV files.
5) Generate candidates (SJP phase 1+2, RETA-Filter).
6) Rank candidates (SJP phase 3, RETA-Grader).

Run as a CLI:
    python -m iswc.harmonized.interface --help
"""

from __future__ import annotations

import argparse
import ast
import csv
import importlib
import json
import logging
import pickle
import shutil
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


logger = logging.getLogger(__name__)


# Make "code/iswc" importable from this module location when needed.
# code/iswc/harmonized/interface.py -> parents[2] == code/
_CODE_DIR = Path(__file__).resolve().parents[2]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from iswc.evaluation import MetricResults, evaluate_entity_centric, format_results_table  # noqa: E402
from iswc.harmonized.dataset import (  # noqa: E402
    generate_standardized_dataset_from_kgloader,
    load_standardized_dataset_triples,
    resolve_standardized_dataset,
)


SJP_SPLITS = ("train", "val", "test")


def _split_dir(path_dataset_dir: Path, split: str) -> Path:
    if split not in SJP_SPLITS:
        raise ValueError(f"Unsupported split '{split}'. Expected one of {SJP_SPLITS}.")
    return path_dataset_dir / split


def _load_id2label_mapping(path: Path) -> Dict[int, str]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    parsed: Dict[int, str] = {}
    for key, value in raw.items():
        parsed[int(key)] = str(value)
    return parsed


def _validate_contiguous_id_space(label_to_id: Dict[str, int], name: str) -> None:
    ids = sorted(label_to_id.values())
    if not ids:
        raise ValueError(f"{name} mapping is empty.")
    expected = list(range(ids[-1] + 1))
    if ids != expected:
        raise ValueError(
            f"{name} ids must be contiguous from 0..N for RETA compatibility. "
            f"Observed min={ids[0]}, max={ids[-1]}, size={len(ids)}."
        )


def _load_sjp_canonical_maps(path_dataset_dir: Path) -> Tuple[Dict[str, int], Dict[str, int]]:
    entity_label_to_id: Dict[str, int] = {}
    relation_label_to_id: Dict[str, int] = {}

    for split in SJP_SPLITS:
        split_path = _split_dir(path_dataset_dir, split)
        id2entity = _load_id2label_mapping(split_path / "id2entity.json")
        id2relation = _load_id2label_mapping(split_path / "id2relation.json")

        for entity_id, label in id2entity.items():
            if label in entity_label_to_id and entity_label_to_id[label] != entity_id:
                raise ValueError(
                    f"Entity label '{label}' has inconsistent ids across splits: "
                    f"{entity_label_to_id[label]} vs {entity_id}."
                )
            entity_label_to_id[label] = entity_id

        for relation_id, label in id2relation.items():
            if label in relation_label_to_id and relation_label_to_id[label] != relation_id:
                raise ValueError(
                    f"Relation label '{label}' has inconsistent ids across splits: "
                    f"{relation_label_to_id[label]} vs {relation_id}."
                )
            relation_label_to_id[label] = relation_id

    _validate_contiguous_id_space(entity_label_to_id, "Entity")
    _validate_contiguous_id_space(relation_label_to_id, "Relation")

    return entity_label_to_id, relation_label_to_id


def _parse_nary_line(line: str) -> Dict[str, Any]:
    line = line.strip()
    if not line:
        return {}
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        # RETA inputs are often python-literal-like, so keep this fallback.
        return ast.literal_eval(line)


def _load_nary_file(
    path: Path,
    values_indexes: Optional[Dict[str, int]] = None,
    roles_indexes: Optional[Dict[str, int]] = None,
) -> Tuple[List[Dict[Tuple[int, ...], List[int]]], Dict[str, int], Dict[str, int]]:
    values_indexes = dict(values_indexes or {})
    roles_indexes = dict(roles_indexes or {})

    next_value = (max(values_indexes.values()) + 1) if values_indexes else 0
    next_role = (max(roles_indexes.values()) + 1) if roles_indexes else 0

    data: List[Dict[Tuple[int, ...], List[int]]] = []

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            record = _parse_nary_line(raw_line)
            if not record:
                continue

            arity = int(record.get("N", 2))
            while len(data) < arity - 1:
                data.append({})

            encoded: Tuple[int, ...] = tuple()
            for role, value in record.items():
                if role == "N":
                    continue

                role_id = roles_indexes.get(role)
                if role_id is None:
                    role_id = next_role
                    roles_indexes[role] = role_id
                    next_role += 1

                values = [value] if isinstance(value, str) else list(value)
                for value_label in values:
                    value_id = values_indexes.get(value_label)
                    if value_id is None:
                        value_id = next_value
                        values_indexes[value_label] = value_id
                        next_value += 1
                    encoded += (int(role_id), int(value_id))

            data[arity - 2][encoded] = [1]

    return data, values_indexes, roles_indexes


def _build_role_val_from_train(
    nary_train_path: Path,
    values_indexes: Dict[str, int],
    roles_indexes: Dict[str, int],
) -> Dict[int, List[int]]:
    role_val: Dict[int, set[int]] = defaultdict(set)

    with nary_train_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            record = _parse_nary_line(raw_line)
            if not record:
                continue

            for role, value in record.items():
                if role == "N":
                    continue
                role_id = int(roles_indexes[role])
                values = [value] if isinstance(value, str) else list(value)
                for value_label in values:
                    role_val[role_id].add(int(values_indexes[value_label]))

    return {role_id: sorted(list(values)) for role_id, values in role_val.items()}


def build_reta_dictionaries(
    reta_data_dir: str | Path,
    values_indexes: Optional[Dict[str, int]] = None,
    roles_indexes: Optional[Dict[str, int]] = None,
    output_filename: str = "dictionaries_and_facts.bin",
) -> Path:
    """
    Build RETA dictionaries_and_facts.bin from n-ary files.

    This is an in-repo replacement for invoking RETA's TensorFlow-based
    builddata.py script and avoids modifying RETA_code.
    """
    reta_data_path = Path(reta_data_dir).resolve()

    train_facts, values_indexes, roles_indexes = _load_nary_file(
        reta_data_path / "n-ary_train.json",
        values_indexes=values_indexes,
        roles_indexes=roles_indexes,
    )
    valid_facts, values_indexes, roles_indexes = _load_nary_file(
        reta_data_path / "n-ary_valid.json",
        values_indexes=values_indexes,
        roles_indexes=roles_indexes,
    )
    test_facts, values_indexes, roles_indexes = _load_nary_file(
        reta_data_path / "n-ary_test.json",
        values_indexes=values_indexes,
        roles_indexes=roles_indexes,
    )

    role_val = _build_role_val_from_train(
        reta_data_path / "n-ary_train.json",
        values_indexes=values_indexes,
        roles_indexes=roles_indexes,
    )

    payload = {
        "train_facts": train_facts,
        "valid_facts": valid_facts,
        "test_facts": test_facts,
        "values_indexes": values_indexes,
        "roles_indexes": roles_indexes,
        "role_val": role_val,
    }

    output_path = reta_data_path / output_filename
    with output_path.open("wb") as handle:
        pickle.dump(payload, handle)

    logger.info("Saved RETA dictionary bundle to %s", output_path)
    return output_path


def _build_label_to_id_from_standard(
    triples_by_split: Dict[str, List[Tuple[str, str, str]]],
) -> Tuple[Dict[str, int], Dict[str, int]]:
    entity_label_to_id: Dict[str, int] = {}
    relation_label_to_id: Dict[str, int] = {}

    for split in ("train", "valid", "test"):
        if split not in triples_by_split:
            continue
        for head, relation, tail in triples_by_split[split]:
            if head not in entity_label_to_id:
                entity_label_to_id[head] = len(entity_label_to_id)
            if tail not in entity_label_to_id:
                entity_label_to_id[tail] = len(entity_label_to_id)
            if relation not in relation_label_to_id:
                relation_label_to_id[relation] = len(relation_label_to_id)

    _validate_contiguous_id_space(entity_label_to_id, "Entity")
    _validate_contiguous_id_space(relation_label_to_id, "Relation")
    return entity_label_to_id, relation_label_to_id


def export_standard_dataset_to_reta(
    standardized_dataset_dir: str | Path,
    output_dir: str | Path,
    default_entity_type: str = "Thing",
    build_reta_bin: bool = True,
    triple_order: str = "hrt",
    delimiter: Optional[str] = None,
    has_header: bool = False,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Export a canonical harmonized dataset into RETA-compatible files.

    The input dataset is expected in internet-style split files (train/valid/test)
    with labeled triples.
    """
    standardized = resolve_standardized_dataset(standardized_dataset_dir)
    reta_output = Path(output_dir).resolve()

    if reta_output.exists() and overwrite:
        if reta_output.is_dir():
            shutil.rmtree(reta_output)
        else:
            reta_output.unlink()
    reta_output.mkdir(parents=True, exist_ok=True)

    triples_by_split = load_standardized_dataset_triples(
        standardized_dataset_dir=standardized.root,
        triple_order=triple_order,
        delimiter=delimiter,
        has_header=has_header,
    )
    entity_label_to_id, relation_label_to_id = _build_label_to_id_from_standard(triples_by_split)

    all_relation_labels_for_type_file: List[str] = []
    split_map = {
        "train": "train",
        "valid": "valid",
        "test": "test",
    }

    for split, reta_split in split_map.items():
        nary_path = reta_output / f"n-ary_{reta_split}.json"
        txt_path = reta_output / f"{reta_split}.txt"

        with nary_path.open("w", encoding="utf-8") as nary_file, txt_path.open("w", encoding="utf-8") as txt_file:
            for head, relation, tail in triples_by_split[split]:
                fact = {relation: [head, tail], "N": 2}
                nary_file.write(json.dumps(fact, ensure_ascii=True) + "\n")
                txt_file.write(f"{head}\t{tail}\t{relation}\n")
                all_relation_labels_for_type_file.append(relation)

    entity_type_file = reta_output / "entity2types_ttv.txt"
    with entity_type_file.open("w", encoding="utf-8") as handle:
        for label in sorted(entity_label_to_id.keys(), key=lambda x: entity_label_to_id[x]):
            handle.write(f"{label}\t{default_entity_type}\n")

    type_relation_file = reta_output / "type2relation2type_ttv.txt"
    with type_relation_file.open("w", encoding="utf-8") as handle:
        for relation_label in all_relation_labels_for_type_file:
            handle.write(f"{default_entity_type}\t{relation_label}\t{default_entity_type}\n")

    with (reta_output / "entity2id.json").open("w", encoding="utf-8") as handle:
        json.dump(entity_label_to_id, handle, indent=2)
    with (reta_output / "relation2id.json").open("w", encoding="utf-8") as handle:
        json.dump(relation_label_to_id, handle, indent=2)

    dictionary_path: Optional[Path] = None
    if build_reta_bin:
        dictionary_path = build_reta_dictionaries(
            reta_data_dir=reta_output,
            values_indexes=entity_label_to_id,
            roles_indexes=relation_label_to_id,
        )

    summary = {
        "standardized_dataset_dir": str(standardized.root),
        "reta_output_dir": str(reta_output),
        "num_entities": len(entity_label_to_id),
        "num_relations": len(relation_label_to_id),
        "default_entity_type": default_entity_type,
        "built_reta_dictionary": bool(build_reta_bin),
        "dictionary_path": str(dictionary_path) if dictionary_path is not None else None,
    }
    metadata_path = reta_output / "harmonized_export_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    logger.info(
        "Exported standardized dataset at %s to RETA-compatible directory %s",
        standardized.root,
        reta_output,
    )
    return summary


def apply_candidate_budget(
    predictions: Dict[int, List[Tuple[int, int, Optional[float]]]],
    candidate_budget: Optional[int],
) -> Dict[int, List[Tuple[int, int, Optional[float]]]]:
    """Apply a per-head top-k cap to ranked predictions."""
    if candidate_budget is None:
        return {int(head): list(ranked) for head, ranked in predictions.items()}

    budget = int(candidate_budget)
    if budget <= 0:
        raise ValueError("candidate_budget must be positive.")

    return {
        int(head): list(ranked[:budget])
        for head, ranked in predictions.items()
    }


def _dedupe_ranked_pairs(
    ranked_items: Iterable[Tuple[int, int, Optional[float]]]
) -> List[Tuple[int, int, Optional[float]]]:
    seen: set[Tuple[int, int]] = set()
    deduped: List[Tuple[int, int, Optional[float]]] = []
    for relation_id, tail_id, score in ranked_items:
        key = (int(relation_id), int(tail_id))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((key[0], key[1], score))
    return deduped


def _group_from_triples_and_scores(
    triples: torch.Tensor,
    scores: Optional[torch.Tensor] = None,
) -> Dict[int, List[Tuple[int, int, Optional[float]]]]:
    triples = triples.detach().cpu()
    if triples.dim() != 2 or triples.size(1) != 3:
        raise ValueError("Expected candidate triples tensor with shape [N, 3].")

    grouped: Dict[int, List[Tuple[int, int, Optional[float]]]] = defaultdict(list)

    score_list: Optional[List[float]] = None
    if scores is not None:
        flat_scores = scores.detach().cpu().reshape(-1)
        if flat_scores.numel() != triples.size(0):
            raise ValueError("Scores tensor length must match number of triples.")
        score_list = flat_scores.tolist()

    for idx, triple in enumerate(triples.tolist()):
        h_id, r_id, t_id = map(int, triple)
        score = float(score_list[idx]) if score_list is not None else None
        grouped[h_id].append((r_id, t_id, score))

    if score_list is not None:
        for head_id in list(grouped.keys()):
            grouped[head_id].sort(
                key=lambda x: float(x[2]) if x[2] is not None else float("-inf"),
                reverse=True,
            )
            grouped[head_id] = _dedupe_ranked_pairs(grouped[head_id])
    else:
        for head_id in list(grouped.keys()):
            grouped[head_id] = _dedupe_ranked_pairs(grouped[head_id])

    return dict(grouped)


def _normalise_prediction_entry(entry: Any, head_id: int) -> Tuple[int, int, Optional[float]]:
    if isinstance(entry, dict):
        relation_raw = entry.get("relation_id", entry.get("r", entry.get("relation")))
        tail_raw = entry.get("tail_id", entry.get("t", entry.get("tail")))
        if relation_raw is None or tail_raw is None:
            raise ValueError(f"Prediction entry for head {head_id} is missing relation/tail fields: {entry}")
        relation_id = int(relation_raw)
        tail_id = int(tail_raw)
        score_raw = entry.get("score")
        score = float(score_raw) if score_raw is not None else None
        return relation_id, tail_id, score

    if isinstance(entry, (list, tuple)):
        if len(entry) == 4:
            # assume (h, r, t, score)
            return int(entry[1]), int(entry[2]), float(entry[3])
        if len(entry) == 3:
            # assume (r, t, score)
            return int(entry[0]), int(entry[1]), float(entry[2])
        if len(entry) == 2:
            # assume (r, t)
            return int(entry[0]), int(entry[1]), None

    raise ValueError(f"Unsupported prediction entry for head {head_id}: {entry}")


def _load_predictions_from_csv(candidate_path: Path) -> Dict[int, List[Tuple[int, int, Optional[float]]]]:
    grouped: Dict[int, List[Tuple[int, int, Optional[float], Optional[int]]]] = defaultdict(list)

    with candidate_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"head_id", "relation_id", "tail_id"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"CSV must include columns {sorted(required)}. "
                f"Observed columns: {reader.fieldnames}"
            )

        for row in reader:
            head_id = int(row["head_id"])
            relation_id = int(row["relation_id"])
            tail_id = int(row["tail_id"])
            score = float(row["score"]) if row.get("score") not in (None, "") else None
            rank = int(row["rank"]) if row.get("rank") not in (None, "") else None
            grouped[head_id].append((relation_id, tail_id, score, rank))

    predictions: Dict[int, List[Tuple[int, int, Optional[float]]]] = {}
    for head_id, items in grouped.items():
        has_rank = any(item[3] is not None for item in items)
        has_score = any(item[2] is not None for item in items)

        if has_rank:
            items.sort(key=lambda x: x[3] if x[3] is not None else 10**18)
        elif has_score:
            items.sort(
                key=lambda x: float(x[2]) if x[2] is not None else float("-inf"),
                reverse=True,
            )

        predictions[head_id] = _dedupe_ranked_pairs((r, t, s) for r, t, s, _ in items)

    return predictions


def _load_predictions_from_json(candidate_path: Path) -> Dict[int, List[Tuple[int, int, Optional[float]]]]:
    with candidate_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict) and "predictions" in payload:
        payload = payload["predictions"]

    if not isinstance(payload, dict):
        raise ValueError("JSON predictions must be a dict: head_id -> ranked candidates.")

    predictions: Dict[int, List[Tuple[int, int, Optional[float]]]] = {}
    for raw_head_id, items in payload.items():
        head_id = int(raw_head_id)
        parsed = [_normalise_prediction_entry(item, head_id) for item in items]
        predictions[head_id] = _dedupe_ranked_pairs(parsed)

    return predictions


def _load_predictions_from_pt(candidate_path: Path) -> Dict[int, List[Tuple[int, int, Optional[float]]]]:
    payload = torch.load(candidate_path, map_location="cpu")

    if isinstance(payload, torch.Tensor):
        return _group_from_triples_and_scores(payload, scores=None)

    if isinstance(payload, dict):
        triples = payload.get("triples", payload.get("candidates"))
        scores = payload.get("scores")
        predictions_obj = payload.get("predictions")

        if triples is not None:
            if not isinstance(triples, torch.Tensor):
                triples = torch.as_tensor(triples, dtype=torch.long)
            if scores is not None and not isinstance(scores, torch.Tensor):
                scores = torch.as_tensor(scores, dtype=torch.float32)
            return _group_from_triples_and_scores(triples, scores)

        if predictions_obj is not None:
            if not isinstance(predictions_obj, dict):
                raise ValueError("'predictions' inside PT payload must be a dict.")
            predictions: Dict[int, List[Tuple[int, int, Optional[float]]]] = {}
            for raw_head_id, items in predictions_obj.items():
                head_id = int(raw_head_id)
                parsed = [_normalise_prediction_entry(item, head_id) for item in items]
                predictions[head_id] = _dedupe_ranked_pairs(parsed)
            return predictions

    raise ValueError(
        "Unsupported PT candidate format. Expected one of: tensor [N,3], "
        "dict with 'triples' (+ optional 'scores'), or dict with 'predictions'."
    )


def load_ranked_predictions(candidate_file: str | Path) -> Dict[int, List[Tuple[int, int, Optional[float]]]]:
    """
    Load ranked candidate predictions from .pt, .csv or .json.

    Returns:
        Dict[head_id -> list[(relation_id, tail_id, score_or_none)]], ranked.
    """
    candidate_path = Path(candidate_file).resolve()
    suffix = candidate_path.suffix.lower()

    if suffix == ".pt":
        return _load_predictions_from_pt(candidate_path)
    if suffix == ".csv":
        return _load_predictions_from_csv(candidate_path)
    if suffix == ".json":
        return _load_predictions_from_json(candidate_path)

    raise ValueError(f"Unsupported candidate file extension '{suffix}'.")


def save_ranked_predictions_csv(
    predictions: Dict[int, List[Tuple[int, int, Optional[float]]]],
    output_file: str | Path,
) -> Path:
    output_path = Path(output_file).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["head_id", "relation_id", "tail_id", "score", "rank"])
        for head_id in sorted(predictions.keys()):
            ranked = predictions[head_id]
            for rank, (relation_id, tail_id, score) in enumerate(ranked, start=1):
                writer.writerow([
                    int(head_id),
                    int(relation_id),
                    int(tail_id),
                    "" if score is None else float(score),
                    rank,
                ])

    logger.info("Saved ranked predictions CSV to %s", output_path)
    return output_path


def save_ranked_predictions_pt(
    predictions: Dict[int, List[Tuple[int, int, Optional[float]]]],
    output_file: str | Path,
) -> Path:
    output_path = Path(output_file).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"predictions": predictions}, output_path)
    logger.info("Saved ranked predictions PT to %s", output_path)
    return output_path


def save_ranked_predictions(
    predictions: Dict[int, List[Tuple[int, int, Optional[float]]]],
    output_file: str | Path,
) -> Path:
    """Save ranked predictions as CSV or PT based on output suffix."""
    output_path = Path(output_file).resolve()
    suffix = output_path.suffix.lower()
    if suffix == ".csv":
        return save_ranked_predictions_csv(predictions, output_path)
    if suffix == ".pt":
        return save_ranked_predictions_pt(predictions, output_path)
    raise ValueError("Output file must end with .csv or .pt")


def _to_metric_input(
    predictions: Dict[int, List[Tuple[int, int, Optional[float]]]]
) -> Dict[int, List[Tuple[int, int]]]:
    metric_input: Dict[int, List[Tuple[int, int]]] = {}
    for head_id, ranked in predictions.items():
        metric_input[int(head_id)] = [(int(rel_id), int(tail_id)) for rel_id, tail_id, _ in ranked]
    return metric_input


def parse_k_values(raw: str) -> List[int]:
    parsed = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
    if not parsed:
        raise ValueError("k-values cannot be empty.")
    values = [int(chunk) for chunk in parsed]
    if any(v <= 0 for v in values):
        raise ValueError("All k-values must be positive integers.")
    return values


def evaluate_candidate_file(
    candidate_file: str | Path,
    gold_triples_file: str | Path,
    k_values: Sequence[int],
) -> MetricResults:
    predictions = load_ranked_predictions(candidate_file)
    metric_input = _to_metric_input(predictions)
    gold_triples = torch.load(Path(gold_triples_file).resolve(), map_location="cpu")
    return evaluate_entity_centric(metric_input, gold_triples, list(k_values))


def _import_reta_main_module(reta_code_dir: str | Path):
    reta_code_path = Path(reta_code_dir).resolve()
    if str(reta_code_path) not in sys.path:
        sys.path.insert(0, str(reta_code_path))
    return importlib.import_module("main_reta_plus")


def _prepare_reta_runtime(
    reta_code_dir: str | Path,
    reta_data_dir: str | Path,
    entities_evaluated: str,
    top_nfilters: int,
    at_least: int,
    sparsifier: int,
    build_type_dictionaries: str,
    device: str,
) -> Dict[str, Any]:
    reta = _import_reta_main_module(reta_code_dir)
    reta_data_path = Path(reta_data_dir).resolve()

    with (reta_data_path / "dictionaries_and_facts.bin").open("rb") as handle:
        data_info = pickle.load(handle)

    test = data_info["test_facts"]
    relation2id = data_info["roles_indexes"]
    entity2id = data_info["values_indexes"]

    id2entity = {int(entity_id): label for label, entity_id in entity2id.items()}
    id2relation = {int(relation_id): label for label, relation_id in relation2id.items()}

    type2id, id2type = reta.build_type2id_v2(str(reta_data_path))
    unk_type_id = len(type2id)
    type2id["UNK"] = unk_type_id
    id2type[unk_type_id] = "UNK"

    entity_name_to_types, entity_id_to_types, _, _ = reta.build_entity2types_dictionaries(
        str(reta_data_path), entity2id
    )

    head2relation2tails = reta.build_head2relation2tails(
        str(reta_data_path), entity2id, relation2id, entity_id_to_types, entities_evaluated
    )

    type_id_to_frequency = reta.build_typeId2frequency(str(reta_data_path), type2id)

    head_tail_to_types, entity_id_to_type_ids = reta.build_headTail2hTypetType(
        str(reta_data_path),
        entity2id,
        type2id,
        entity_name_to_types,
        sparsifier,
        type_id_to_frequency,
        build_type_dictionaries,
    )

    _, test, _ = reta.add_type_pair_to_fact([], test, [], head_tail_to_types, entity_id_to_type_ids, unk_type_id)

    type2relation_type_frequency = reta.build_type2relationType2frequency(
        str(reta_data_path), build_type_dictionaries
    )

    type_head_tail_entity_matrix, tail_type_relation_head_type_tensor = reta.build_tensor_matrix(
        str(reta_data_path),
        entity2id,
        relation2id,
        entity_name_to_types,
        top_nfilters,
        type2relation_type_frequency,
        entity_id_to_type_ids,
        type2id,
        id2type,
        device,
        entities_evaluated,
    )

    entity2sparsified_types = reta.build_entity2sparsifiedTypes(
        type_id_to_frequency,
        entity_id_to_types,
        type2id,
        sparsifier,
        unk_type_id,
        id2type,
    )

    flat_test_facts: List[Any] = []
    head_to_template_fact: Dict[int, List[int]] = {}
    for grouped in test:
        for fact in grouped:
            cast_fact = list(fact)
            flat_test_facts.append(cast_fact)
            head_id = int(cast_fact[1])
            if head_id not in head_to_template_fact:
                head_to_template_fact[head_id] = cast_fact

    return {
        "reta": reta,
        "reta_data_path": reta_data_path,
        "relation2id": relation2id,
        "entity2id": entity2id,
        "id2entity": id2entity,
        "id2relation": id2relation,
        "type2id": type2id,
        "id2type": id2type,
        "unk_type_id": unk_type_id,
        "entity_name_to_types": entity_name_to_types,
        "entity_id_to_types": entity_id_to_types,
        "head2relation2tails": head2relation2tails,
        "type_id_to_frequency": type_id_to_frequency,
        "type2relation_type_frequency": type2relation_type_frequency,
        "type_head_tail_entity_matrix": type_head_tail_entity_matrix,
        "tail_type_relation_head_type_tensor": tail_type_relation_head_type_tensor,
        "entity2sparsified_types": entity2sparsified_types,
        "flat_test_facts": flat_test_facts,
        "head_to_template_fact": head_to_template_fact,
        "at_least": int(at_least),
        "sparsifier": int(sparsifier),
        "entities_evaluated": entities_evaluated,
    }


def _build_reta_facts_from_candidate_pairs(
    reta: Any,
    fact: List[int],
    candidate_pairs: List[Tuple[int, int]],
    entity2sparsified_types: Dict[int, List[int]],
    unk_type_id: int,
) -> Tuple[List[np.ndarray], Dict[int, Tuple[int, int]]]:
    if not candidate_pairs:
        return [], {}

    all_relations = [int(relation_id) for relation_id, _ in candidate_pairs]
    all_tails = [int(tail_id) for _, tail_id in candidate_pairs]

    tiled_fact = np.array(fact * len(candidate_pairs)).reshape(len(candidate_pairs), -1)
    tiled_fact[:, 3] = all_tails
    tiled_fact[:, 0] = all_relations
    tiled_fact[:, 2] = all_relations

    current_head_entity = int(tiled_fact[0][1])
    head_sparsified_types = entity2sparsified_types.get(current_head_entity, [unk_type_id])

    tmp_t_types = [list(entity2sparsified_types.get(entity_id, [])) for entity_id in list(tiled_fact[:, 3])]
    for index, types in enumerate(tmp_t_types):
        if not types:
            tmp_t_types[index] = [unk_type_id]

    tmp_t_types_0 = np.array([types[0] for types in tmp_t_types])

    tmp_array = tiled_fact[:, :4]
    for head_type in head_sparsified_types:
        tmp_array = np.c_[tmp_array, np.full((tiled_fact.shape[0], 1), np.array(head_type)), tmp_t_types_0.T]
    new_tiled_fact = list(tmp_array)

    multi_type_indices = [index for index, types in enumerate(tmp_t_types) if len(types) > 1]
    for index in multi_type_indices:
        for tail_type in tmp_t_types[index][1:]:
            for head_type in head_sparsified_types:
                new_tiled_fact[index] = np.concatenate((new_tiled_fact[index], np.array([head_type, tail_type])))

    grouped_facts, idx2relation_tail = reta.sort_testing_facts_according_to_arity_fast(new_tiled_fact, candidate_pairs)
    return grouped_facts, idx2relation_tail


def extract_reta_filter_candidates(
    reta_code_dir: str | Path,
    reta_data_dir: str | Path,
    output_file: str | Path,
    entities_evaluated: str = "both",
    top_nfilters: int = -10,
    at_least: int = 2,
    sparsifier: int = 2,
    build_type_dictionaries: str = "True",
    device: str = "cuda:0",
    max_facts: Optional[int] = None,
    map_to_sjp_dataset_dir: Optional[str | Path] = None,
    candidate_budget: Optional[int] = None,
) -> Dict[int, List[Tuple[int, int, Optional[float]]]]:
    runtime = _prepare_reta_runtime(
        reta_code_dir=reta_code_dir,
        reta_data_dir=reta_data_dir,
        entities_evaluated=entities_evaluated,
        top_nfilters=top_nfilters,
        at_least=at_least,
        sparsifier=sparsifier,
        build_type_dictionaries=build_type_dictionaries,
        device=device,
    )

    reta = runtime["reta"]
    id2entity = runtime["id2entity"]
    relation2id = runtime["relation2id"]
    entity2id = runtime["entity2id"]
    type2relation_type_frequency = runtime["type2relation_type_frequency"]
    type_head_tail_entity_matrix = runtime["type_head_tail_entity_matrix"]
    tail_type_relation_head_type_tensor = runtime["tail_type_relation_head_type_tensor"]
    entity_name_to_types = runtime["entity_name_to_types"]
    type2id = runtime["type2id"]
    at_least = runtime["at_least"]
    sparsifier = runtime["sparsifier"]
    type_id_to_frequency = runtime["type_id_to_frequency"]
    entity_id_to_types = runtime["entity_id_to_types"]
    unk_type_id = runtime["unk_type_id"]
    id2type = runtime["id2type"]
    entity2sparsified_types = runtime["entity2sparsified_types"]
    flat_test_facts = runtime["flat_test_facts"]
    head2relation2tails = runtime["head2relation2tails"]
    reta_data_path = runtime["reta_data_path"]

    predictions: Dict[int, List[Tuple[int, int, Optional[float]]]] = {}
    visited_heads: set[int] = set()
    processed = 0

    for raw_fact in flat_test_facts:
        if max_facts is not None and processed >= max_facts:
            break
        processed += 1

        fact = list(raw_fact)
        head_id = int(fact[1])
        if head_id in visited_heads:
            continue
        visited_heads.add(head_id)

        if head_id not in head2relation2tails:
            continue
        if head_id not in id2entity:
            continue

        head_name = id2entity[head_id]
        ranked: List[Tuple[int, int, Optional[float]]] = []

        if head_name in entity_name_to_types:
            filtered_facts, idx2relation_tail = reta.get_reta_filtered_results(
                head_id,
                id2entity,
                type2relation_type_frequency,
                top_nfilters,
                3,
                fact,
                None,
                int(len(fact) / 2),
                device,
                type_head_tail_entity_matrix,
                tail_type_relation_head_type_tensor,
                entity_name_to_types,
                type2id,
                relation2id,
                at_least,
                sparsifier,
                type_id_to_frequency,
                entity_id_to_types,
                unk_type_id,
                id2type,
                entity2sparsified_types,
                entities_evaluated,
                [],
            )
            if filtered_facts and idx2relation_tail:
                for index in sorted(idx2relation_tail.keys()):
                    relation_id, tail_id = idx2relation_tail[index]
                    ranked.append((int(relation_id), int(tail_id), None))

        if not ranked and entities_evaluated == "none":
            ranked = [
                (int(relation_id), int(tail_id), None)
                for relation_id in runtime["id2relation"].keys()
                for tail_id in runtime["id2entity"].keys()
            ]

        if ranked:
            predictions[head_id] = _dedupe_ranked_pairs(ranked)

    if map_to_sjp_dataset_dir is not None:
        entity_id_map, relation_id_map = _build_reta_to_sjp_id_maps(
            reta_data_dir=reta_data_path,
            sjp_dataset_dir=map_to_sjp_dataset_dir,
        )
        predictions = _map_predictions_with_id_maps(predictions, entity_id_map, relation_id_map)

    predictions = apply_candidate_budget(predictions, candidate_budget)
    save_ranked_predictions(predictions, output_file)
    logger.info("Extracted RETA filter candidates for %d heads", len(predictions))
    return predictions


def rank_reta_candidates(
    reta_code_dir: str | Path,
    reta_data_dir: str | Path,
    candidate_file: str | Path,
    model_path: str | Path,
    output_file: str | Path,
    entities_evaluated: str = "both",
    top_nfilters: int = -10,
    at_least: int = 2,
    sparsifier: int = 2,
    build_type_dictionaries: str = "True",
    device: str = "cuda:0",
    max_facts: Optional[int] = None,
    map_to_sjp_dataset_dir: Optional[str | Path] = None,
    candidate_budget: Optional[int] = None,
) -> Dict[int, List[Tuple[int, int, Optional[float]]]]:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "RETA ranking requires CUDA because RETA forward() uses tensor.cuda(device)."
        )

    runtime = _prepare_reta_runtime(
        reta_code_dir=reta_code_dir,
        reta_data_dir=reta_data_dir,
        entities_evaluated=entities_evaluated,
        top_nfilters=top_nfilters,
        at_least=at_least,
        sparsifier=sparsifier,
        build_type_dictionaries=build_type_dictionaries,
        device=device,
    )

    reta = runtime["reta"]
    reta_data_path = runtime["reta_data_path"]
    model = torch.load(Path(model_path).resolve(), map_location=device)
    model.eval()
    model.to(device)

    candidate_predictions = load_ranked_predictions(candidate_file)
    ranked_predictions: Dict[int, List[Tuple[int, int, Optional[float]]]] = {}

    processed_heads = 0
    for head_id, candidate_rows in candidate_predictions.items():
        if max_facts is not None and processed_heads >= max_facts:
            break
        processed_heads += 1

        template_fact = runtime["head_to_template_fact"].get(int(head_id))
        if template_fact is None:
            continue

        candidate_pairs = [(int(relation_id), int(tail_id)) for relation_id, tail_id, _ in candidate_rows]
        candidate_pairs = list(dict.fromkeys(candidate_pairs))
        if not candidate_pairs:
            continue

        grouped_facts, idx2relation_tail = _build_reta_facts_from_candidate_pairs(
            reta=reta,
            fact=template_fact,
            candidate_pairs=candidate_pairs,
            entity2sparsified_types=runtime["entity2sparsified_types"],
            unk_type_id=runtime["unk_type_id"],
        )
        if not grouped_facts or not idx2relation_tail:
            continue

        score_lists = reta.get_scores_from_reta_filtered_results(grouped_facts, model, device)
        if not score_lists:
            continue

        scores = np.concatenate(score_lists).ravel()
        sorted_indices = (-scores).argsort()

        ranked_rows: List[Tuple[int, int, Optional[float]]] = []
        for index in sorted_indices:
            relation_id, tail_id = idx2relation_tail[int(index)]
            ranked_rows.append((int(relation_id), int(tail_id), float(scores[int(index)])))

        ranked_predictions[int(head_id)] = _dedupe_ranked_pairs(ranked_rows)

    if map_to_sjp_dataset_dir is not None:
        entity_id_map, relation_id_map = _build_reta_to_sjp_id_maps(
            reta_data_dir=reta_data_path,
            sjp_dataset_dir=map_to_sjp_dataset_dir,
        )
        ranked_predictions = _map_predictions_with_id_maps(ranked_predictions, entity_id_map, relation_id_map)

    ranked_predictions = apply_candidate_budget(ranked_predictions, candidate_budget)
    save_ranked_predictions(ranked_predictions, output_file)
    logger.info("Ranked RETA candidates for %d heads", len(ranked_predictions))
    return ranked_predictions


def _build_reta_to_sjp_id_maps(
    reta_data_dir: str | Path,
    sjp_dataset_dir: str | Path,
) -> Tuple[Dict[int, int], Dict[int, int]]:
    reta_data_path = Path(reta_data_dir).resolve()
    with (reta_data_path / "dictionaries_and_facts.bin").open("rb") as handle:
        data_info = pickle.load(handle)

    reta_entity_label_to_id: Dict[str, int] = data_info["values_indexes"]
    reta_relation_label_to_id: Dict[str, int] = data_info["roles_indexes"]

    sjp_entity_label_to_id, sjp_relation_label_to_id = _load_sjp_canonical_maps(Path(sjp_dataset_dir).resolve())

    entity_id_map: Dict[int, int] = {}
    for label, reta_id in reta_entity_label_to_id.items():
        if label not in sjp_entity_label_to_id:
            raise KeyError(f"Entity label '{label}' in RETA mapping is missing from SJP mapping.")
        entity_id_map[int(reta_id)] = int(sjp_entity_label_to_id[label])

    relation_id_map: Dict[int, int] = {}
    for label, reta_id in reta_relation_label_to_id.items():
        if label not in sjp_relation_label_to_id:
            raise KeyError(f"Relation label '{label}' in RETA mapping is missing from SJP mapping.")
        relation_id_map[int(reta_id)] = int(sjp_relation_label_to_id[label])

    return entity_id_map, relation_id_map


def _map_predictions_with_id_maps(
    predictions: Dict[int, List[Tuple[int, int, Optional[float]]]],
    entity_id_map: Dict[int, int],
    relation_id_map: Dict[int, int],
) -> Dict[int, List[Tuple[int, int, Optional[float]]]]:
    mapped: Dict[int, List[Tuple[int, int, Optional[float]]]] = {}
    for head_id, ranked in predictions.items():
        new_head = entity_id_map.get(int(head_id))
        if new_head is None:
            continue
        mapped_ranked: List[Tuple[int, int, Optional[float]]] = []
        for relation_id, tail_id, score in ranked:
            new_relation = relation_id_map.get(int(relation_id))
            new_tail = entity_id_map.get(int(tail_id))
            if new_relation is None or new_tail is None:
                continue
            mapped_ranked.append((new_relation, new_tail, score))
        mapped[new_head] = mapped_ranked
    return mapped


def _results_to_jsonable(results: MetricResults) -> Dict[str, Any]:
    return asdict(results)


def run_compare(
    methods: Dict[str, str],
    gold_triples_file: str | Path,
    k_values: Sequence[int],
    output_json: Optional[str | Path] = None,
) -> Dict[str, MetricResults]:
    evaluated: Dict[str, MetricResults] = {}
    for name, candidate_file in methods.items():
        evaluated[name] = evaluate_candidate_file(candidate_file, gold_triples_file, k_values)

    print(format_results_table(evaluated, k=max(k_values)))

    if output_json is not None:
        payload = {name: _results_to_jsonable(result) for name, result in evaluated.items()}
        output_path = Path(output_json).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        logger.info("Saved comparison metrics to %s", output_path)

    return evaluated


def _normalise_delimiter(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    if raw == "":
        return None
    if raw == "\\t":
        return "\t"
    return raw


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Harmonized SJP/RETA interface for standardized dataset generation, "
            "adapter translation, candidate generation, and ranking."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    generate_standard = sub.add_parser(
        "generate-standard-dataset",
        help="Generate standardized train/valid/test text files from a built-in KgLoader dataset name.",
    )
    generate_standard.add_argument("--dataset-name", required=True)
    generate_standard.add_argument("--output-dir", required=True)
    generate_standard.add_argument("--inverse-mode", choices=["manual", "automatic", "none"], default="none")
    generate_standard.add_argument("--overwrite", action="store_true", default=False)

    prepare_dataset = sub.add_parser(
        "prepare-dataset",
        help="Prepare predictor-specific dataset files from a standardized dataset via adapter.",
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
    prepare_dataset.add_argument("--skip-reta-bin", action="store_true")
    prepare_dataset.add_argument("--sjp-code-dir", default=None)
    prepare_dataset.add_argument("--reta-code-dir", default=None)

    generate_candidates = sub.add_parser(
        "generate-candidates",
        help="Generate standardized candidate CSV via the selected adapter.",
    )
    generate_candidates.add_argument("--adapter", choices=["sjp", "reta"], required=True)
    generate_candidates.add_argument("--output-file", required=True, help="Standardized CSV output path.")
    generate_candidates.add_argument("--candidate-budget", type=int, default=500)
    generate_candidates.add_argument("--path-dataset-dir", default=None, help="SJP path dataset root (train/val/test).")
    generate_candidates.add_argument("--path-setup", default="20_10")
    generate_candidates.add_argument("--cmd", choices=["train", "resume", "test"], default="train")
    generate_candidates.add_argument("--log-dir", default="./logs/harmonized")
    generate_candidates.add_argument("--expname", default="harmonized_sjp")
    generate_candidates.add_argument("--num-workers", type=int, default=1)
    generate_candidates.add_argument("--max-epochs", type=int, default=100)
    generate_candidates.add_argument("--tuple-checkpoint", default=None)
    generate_candidates.add_argument("--skip-phase1", action="store_true", default=False)
    generate_candidates.add_argument("--sjp-code-dir", default=None)
    generate_candidates.add_argument("--reta-code-dir", default=None)
    generate_candidates.add_argument("--reta-data-dir", default=None, help="Path to RETA prepared dataset directory.")
    generate_candidates.add_argument("--entities-evaluated", default="both", choices=["both", "one", "none"])
    generate_candidates.add_argument("--top-nfilters", type=int, default=-10)
    generate_candidates.add_argument("--at-least", type=int, default=2)
    generate_candidates.add_argument("--sparsifier", type=int, default=2)
    generate_candidates.add_argument("--build-type-dictionaries", default="True", choices=["True", "False"])
    generate_candidates.add_argument("--device", default="cuda:0")
    generate_candidates.add_argument("--max-facts", type=int, default=None)
    generate_candidates.add_argument("--map-to-sjp-dataset-dir", default=None)

    rank_candidates = sub.add_parser(
        "rank-candidates",
        help="Rank standardized candidate CSV via the selected adapter.",
    )
    rank_candidates.add_argument("--adapter", choices=["sjp", "reta"], required=True)
    rank_candidates.add_argument("--candidate-file", required=True, help="Standardized candidate CSV input path.")
    rank_candidates.add_argument("--output-file", required=True, help="Standardized ranked CSV output path.")
    rank_candidates.add_argument("--candidate-budget", type=int, default=500)
    rank_candidates.add_argument("--path-dataset-dir", default=None, help="SJP path dataset root (train/val/test).")
    rank_candidates.add_argument("--path-setup", default="20_10")
    rank_candidates.add_argument("--cmd", choices=["train", "resume", "test"], default="train")
    rank_candidates.add_argument("--log-dir", default="./logs/harmonized")
    rank_candidates.add_argument("--expname", default="harmonized_sjp")
    rank_candidates.add_argument("--num-workers", type=int, default=1)
    rank_candidates.add_argument("--max-epochs", type=int, default=100)
    rank_candidates.add_argument("--triple-checkpoint", default=None)
    rank_candidates.add_argument("--skip-phase2", action="store_true", default=False)
    rank_candidates.add_argument("--sjp-code-dir", default=None)
    rank_candidates.add_argument("--reta-code-dir", default=None)
    rank_candidates.add_argument("--reta-data-dir", default=None, help="Path to RETA prepared dataset directory.")
    rank_candidates.add_argument("--model-path", default=None, help="Path to trained RETA model file.")
    rank_candidates.add_argument("--entities-evaluated", default="both", choices=["both", "one", "none"])
    rank_candidates.add_argument("--top-nfilters", type=int, default=-10)
    rank_candidates.add_argument("--at-least", type=int, default=2)
    rank_candidates.add_argument("--sparsifier", type=int, default=2)
    rank_candidates.add_argument("--build-type-dictionaries", default="True", choices=["True", "False"])
    rank_candidates.add_argument("--device", default="cuda:0")
    rank_candidates.add_argument("--max-facts", type=int, default=None)
    rank_candidates.add_argument("--map-to-sjp-dataset-dir", default=None)

    return parser


def _resolve_default_reta_code_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "RETA_code"


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = _build_cli()
    args = parser.parse_args(argv)

    if args.command == "generate-standard-dataset":
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
                build_reta_bin=not args.skip_reta_bin,
                triple_order=args.triple_order,
                delimiter=delimiter,
                has_header=args.has_header,
                overwrite=args.overwrite,
            )

        summary["workflow_step"] = "prepare-dataset"
        summary["adapter"] = args.adapter.upper()
        print(json.dumps(summary, indent=2))
        return

    if args.command == "generate-candidates":
        from iswc.harmonized.adapters import RETAAdapter, SJPAdapter

        if Path(args.output_file).suffix.lower() != ".csv":
            raise ValueError("--output-file must end with .csv for generate-candidates")

        if args.adapter == "sjp":
            adapter = SJPAdapter(sjp_code_dir=args.sjp_code_dir)
            if args.path_dataset_dir is None:
                raise ValueError("--path-dataset-dir is required when --adapter sjp")
            predictions = adapter.generate_candidates(
                output_file=args.output_file,
                candidate_budget=args.candidate_budget,
                path_dataset_dir=args.path_dataset_dir,
                path_setup=args.path_setup,
                cmd=args.cmd,
                log_dir=args.log_dir,
                expname=args.expname,
                num_workers=args.num_workers,
                max_epochs=args.max_epochs,
                tuple_checkpoint=args.tuple_checkpoint,
                skip_phase1=args.skip_phase1,
            )
            summary = {
                "workflow_step": "generate-candidates",
                "adapter": "SJP",
                "output_file": str(Path(args.output_file).resolve()),
                "candidate_budget": int(args.candidate_budget),
                "num_heads": len(predictions),
                "path_dataset_dir": str(Path(args.path_dataset_dir).resolve()),
            }
            print(json.dumps(summary, indent=2))
            return

        reta_code_dir = Path(args.reta_code_dir).resolve() if args.reta_code_dir is not None else _resolve_default_reta_code_dir()
        if args.reta_data_dir is None:
            raise ValueError("--reta-data-dir is required when --adapter reta")
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

    if args.command == "rank-candidates":
        from iswc.harmonized.adapters import RETAAdapter, SJPAdapter

        if Path(args.candidate_file).suffix.lower() != ".csv":
            raise ValueError("--candidate-file must end with .csv for rank-candidates")
        if Path(args.output_file).suffix.lower() != ".csv":
            raise ValueError("--output-file must end with .csv for rank-candidates")

        if args.adapter == "sjp":
            adapter = SJPAdapter(sjp_code_dir=args.sjp_code_dir)
            if args.path_dataset_dir is None:
                raise ValueError("--path-dataset-dir is required when --adapter sjp")
            predictions = adapter.rank_candidates(
                candidate_file=args.candidate_file,
                output_file=args.output_file,
                candidate_budget=args.candidate_budget,
                path_dataset_dir=args.path_dataset_dir,
                path_setup=args.path_setup,
                cmd=args.cmd,
                log_dir=args.log_dir,
                expname=args.expname,
                num_workers=args.num_workers,
                max_epochs=args.max_epochs,
                triple_checkpoint=args.triple_checkpoint,
                skip_phase2=args.skip_phase2,
            )
            summary = {
                "workflow_step": "rank-candidates",
                "adapter": "SJP",
                "candidate_file": str(Path(args.candidate_file).resolve()),
                "output_file": str(Path(args.output_file).resolve()),
                "candidate_budget": int(args.candidate_budget),
                "num_heads": len(predictions),
            }
            print(json.dumps(summary, indent=2))
            return

        reta_code_dir = Path(args.reta_code_dir).resolve() if args.reta_code_dir is not None else _resolve_default_reta_code_dir()
        if args.reta_data_dir is None:
            raise ValueError("--reta-data-dir is required when --adapter reta")
        if args.model_path is None:
            raise ValueError("--model-path is required when --adapter reta")

        adapter = RETAAdapter(reta_code_dir=reta_code_dir, reta_data_dir=args.reta_data_dir)
        predictions = adapter.rank_candidates(
            candidate_file=args.candidate_file,
            model_path=args.model_path,
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
            "model_path": str(Path(args.model_path).resolve()),
        }
        print(json.dumps(summary, indent=2))
        return

    raise ValueError(f"Unsupported command {args.command}")


if __name__ == "__main__":
    main()
