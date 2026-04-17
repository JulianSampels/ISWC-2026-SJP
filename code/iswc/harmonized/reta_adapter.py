"""RETA adapter implementation for the harmonized workflow."""

from __future__ import annotations

import ast
import csv
import importlib
import json
import logging
import pickle
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .base_adapter import (
    CandidateAdapter,
    RankedPredictions,
    apply_candidate_budget,
    load_ranked_predictions,
    save_ranked_predictions,
)
from .dataset import load_standardized_dataset_triples, resolve_standardized_dataset


logger = logging.getLogger(__name__)


SJP_SPLITS = ("train", "val", "test")


def _dedupe_ranked_pairs(
    ranked_items: List[Tuple[int, int, Optional[float]]]
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
    role_val: Dict[int, set[int]] = {}

    with nary_train_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            record = _parse_nary_line(raw_line)
            if not record:
                continue

            for role, value in record.items():
                if role == "N":
                    continue
                role_id = int(roles_indexes[role])
                if role_id not in role_val:
                    role_val[role_id] = set()
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
    """Build RETA dictionaries_and_facts.bin from exported n-ary files."""
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


def _write_id_triples_csv(path: Path, triples: List[Tuple[int, int, int]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["head_id", "relation_id", "tail_id"])
        for head_id, relation_id, tail_id in triples:
            writer.writerow([int(head_id), int(relation_id), int(tail_id)])
            count += 1
    return count


def export_standard_dataset_to_reta(
    standardized_dataset_dir: str | Path,
    output_dir: str | Path,
    default_entity_type: str = "Thing",
    triple_order: str = "hrt",
    delimiter: Optional[str] = None,
    has_header: bool = False,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Export canonical dataset files to RETA input layout."""
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
    split_map = {"train": "train", "valid": "valid", "test": "test"}

    for split, reta_split in split_map.items():
        nary_path = reta_output / f"n-ary_{reta_split}.json"
        txt_path = reta_output / f"{reta_split}.txt"

        with nary_path.open("w", encoding="utf-8") as nary_file, txt_path.open("w", encoding="utf-8") as txt_file:
            for head, relation, tail in triples_by_split[split]:
                fact = {relation: [head, tail], "N": 2}
                nary_file.write(json.dumps(fact, ensure_ascii=True) + "\n")
                txt_file.write(f"{head}\t{tail}\t{relation}\n")
                all_relation_labels_for_type_file.append(relation)

    # 1. Gather structural heuristic types from the training graph
    entity_to_types: Dict[str, set[str]] = {label: set() for label in entity_label_to_id.keys()}
    if "train" in triples_by_split:
        for head, relation, tail in triples_by_split["train"]:
            entity_to_types[head].add(f"{relation}_domain")
            entity_to_types[tail].add(f"{relation}_range")

    # 2. Write the heuristic types to the entity mappings
    # If an entity is absent from the train set (inductive or isolated), we fall back to a default placeholder
    entity_type_file = reta_output / "entity2types_ttv.txt"
    fallback_type_used = False
    with entity_type_file.open("w", encoding="utf-8") as handle:
        for label in sorted(entity_label_to_id.keys(), key=lambda x: entity_label_to_id[x]):
            types = entity_to_types[label]
            if not types:
                handle.write(f"{label}\t{default_entity_type}\n")
                fallback_type_used = True
            for t in sorted(types):
                handle.write(f"{label}\t{t}\n")

    # 3. Link the heuristic relation schemas together
    type_relation_file = reta_output / "type2relation2type_ttv.txt"
    with type_relation_file.open("w", encoding="utf-8") as handle:
        # Keep fallback schema entries only when fallback type was actually assigned.
        unique_relations = sorted(set(all_relation_labels_for_type_file))
        for relation_label in unique_relations:
            # Main heuristics
            handle.write(f"{relation_label}_domain\t{relation_label}\t{relation_label}_range\n")
            # Fallback heuristics for validation/test nodes that missed training
            if fallback_type_used:
                handle.write(f"{default_entity_type}\t{relation_label}\t{default_entity_type}\n")

    with (reta_output / "entity2id.json").open("w", encoding="utf-8") as handle:
        json.dump(entity_label_to_id, handle, indent=2)
    with (reta_output / "relation2id.json").open("w", encoding="utf-8") as handle:
        json.dump(relation_label_to_id, handle, indent=2)

    gold_test_rows = [
        (
            int(entity_label_to_id[head]),
            int(relation_label_to_id[relation]),
            int(entity_label_to_id[tail]),
        )
        for head, relation, tail in triples_by_split["test"]
    ]
    _write_id_triples_csv(reta_output / "gold_test.csv", gold_test_rows)

    summary = {
        "standardized_dataset_dir": str(standardized.root),
        "reta_output_dir": str(reta_output),
        "num_entities": len(entity_label_to_id),
        "num_relations": len(relation_label_to_id),
        "default_entity_type": default_entity_type,
        "gold_triples_file": str(reta_output / "gold_test.csv"),
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
) -> RankedPredictions:
    """Generate RETA filter candidates (phase before model scoring)."""
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

    predictions: RankedPredictions = {}
    visited_heads: set[int] = set()
    processed = 0
    skipped_no_head2relation = 0
    skipped_no_entity_id = 0
    skipped_no_types = 0
    heads_with_candidates = 0

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
            skipped_no_head2relation += 1
            continue
        if head_id not in id2entity:
            skipped_no_entity_id += 1
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
        else:
            skipped_no_types += 1

        if ranked:
            heads_with_candidates += 1
            predictions[head_id] = _dedupe_ranked_pairs(ranked)

    if map_to_sjp_dataset_dir is not None:
        entity_id_map, relation_id_map = _build_reta_to_sjp_id_maps(
            reta_data_dir=reta_data_path,
            sjp_dataset_dir=map_to_sjp_dataset_dir,
        )
        predictions = _map_predictions_with_id_maps(predictions, entity_id_map, relation_id_map)

    predictions = apply_candidate_budget(predictions, candidate_budget)
    save_ranked_predictions(predictions, output_file)
    logger.info(
        "RETA filter diagnostics | processed_facts=%d unique_heads=%d pass_heads=%d "
        "skip_no_head2rels=%d skip_no_id2entity=%d skip_no_types=%d",
        processed,
        len(visited_heads),
        heads_with_candidates,
        skipped_no_head2relation,
        skipped_no_entity_id,
        skipped_no_types,
    )
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
) -> RankedPredictions:
    """Rank RETA candidates with a trained RETA model."""
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
    ranked_predictions: RankedPredictions = {}

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
    predictions: RankedPredictions,
    entity_id_map: Dict[int, int],
    relation_id_map: Dict[int, int],
) -> RankedPredictions:
    mapped: RankedPredictions = {}
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


@dataclass
class RETAAdapter(CandidateAdapter):
    """Adapter for RETA dataset conversion and candidate/ranking workflows."""

    reta_code_dir: str | Path
    reta_data_dir: str | Path
    name: str = "RETA"

    def _resolve_reta_data_dir(self) -> Path:
        return Path(self.reta_data_dir).resolve()

    def _load_reta_id_maps(self) -> Tuple[Optional[Dict[str, int]], Optional[Dict[str, int]]]:
        reta_data_path = self._resolve_reta_data_dir()
        entity_file = reta_data_path / "entity2id.json"
        relation_file = reta_data_path / "relation2id.json"

        if not entity_file.is_file() or not relation_file.is_file():
            return None, None

        with entity_file.open("r", encoding="utf-8") as handle:
            values_indexes = {str(key): int(value) for key, value in json.load(handle).items()}
        with relation_file.open("r", encoding="utf-8") as handle:
            roles_indexes = {str(key): int(value) for key, value in json.load(handle).items()}
        return values_indexes, roles_indexes

    def _ensure_reta_dictionary_bundle(self, force_rebuild: bool = False) -> Path:
        reta_data_path = self._resolve_reta_data_dir()
        dictionary_path = reta_data_path / "dictionaries_and_facts.bin"
        if dictionary_path.is_file() and not force_rebuild:
            return dictionary_path

        values_indexes, roles_indexes = self._load_reta_id_maps()
        return build_reta_dictionaries(
            reta_data_dir=reta_data_path,
            values_indexes=values_indexes,
            roles_indexes=roles_indexes,
        )

    def _resolve_latest_reta_model(self, model_output_dir: str | Path) -> Path:
        output_dir = Path(model_output_dir).resolve()
        if not output_dir.is_dir():
            raise FileNotFoundError(f"RETA model output directory does not exist: {output_dir}")
        files = [path for path in output_dir.iterdir() if path.is_file()]
        if not files:
            raise FileNotFoundError(f"No RETA model files found in: {output_dir}")
        return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)[0]

    def prepare_dataset(
        self,
        standardized_dataset_dir: str | Path,
        output_dir: str | Path,
        default_entity_type: str = "Thing",
        triple_order: str = "hrt",
        delimiter: Optional[str] = None,
        has_header: bool = False,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """Convert canonical dataset into RETA inputs and build dictionary bundle."""
        summary = export_standard_dataset_to_reta(
            standardized_dataset_dir=standardized_dataset_dir,
            output_dir=output_dir,
            default_entity_type=default_entity_type,
            triple_order=triple_order,
            delimiter=delimiter,
            has_header=has_header,
            overwrite=overwrite,
        )
        self.reta_data_dir = summary["reta_output_dir"]
        dictionary_path = self._ensure_reta_dictionary_bundle(force_rebuild=True)
        summary["built_reta_dictionary"] = True
        summary["dictionary_path"] = str(dictionary_path)
        return summary

    def train_candidate_model(self, **kwargs) -> Dict[str, Any]:
        """RETA candidate generation is filter-based and does not train a model."""
        self._ensure_reta_dictionary_bundle(force_rebuild=False)
        return {
            "adapter": self.name,
            "task": "train-candidate-model",
            "model_path": None,
            "message": "RETA candidate generation uses deterministic filtering and has no trainable candidate model.",
            "reta_data_dir": str(self._resolve_reta_data_dir()),
        }

    def generate_candidates(
        self,
        output_file: str | Path,
        candidate_budget: int,
        entities_evaluated: str = "both",
        top_nfilters: int = -10,
        at_least: int = 2,
        sparsifier: int = 2,
        build_type_dictionaries: str = "True",
        device: str = "cuda:0",
        max_facts: Optional[int] = None,
        map_to_sjp_dataset_dir: Optional[str | Path] = None,
    ) -> RankedPredictions:
        """Generate RETA filter candidates in canonical format."""
        self._ensure_reta_dictionary_bundle(force_rebuild=False)
        return extract_reta_filter_candidates(
            reta_code_dir=self.reta_code_dir,
            reta_data_dir=self.reta_data_dir,
            output_file=output_file,
            candidate_budget=int(candidate_budget),
            entities_evaluated=entities_evaluated,
            top_nfilters=top_nfilters,
            at_least=at_least,
            sparsifier=sparsifier,
            build_type_dictionaries=build_type_dictionaries,
            device=device,
            max_facts=max_facts,
            map_to_sjp_dataset_dir=map_to_sjp_dataset_dir,
        )

    def train_ranking_model(
        self,
        model_output_dir: str | Path,
        epochs: int = 1000,
        batchsize: int = 128,
        num_filters: int = 100,
        embsize: int = 100,
        learningrate: float = 0.0001,
        with_types: str = "True",
        gpu_ids: str = "0",
        at_least: int = 2,
        top_nfilters: int = -10,
        build_type_dictionaries: str = "False",
        sparsifier: int = 2,
        entities_evaluated: str = "both",
        num_negative_samples: int = 1,
        negative_strategy: float = 0,
        load: str = "False",
        model_to_be_trained: str = "",
    ) -> Dict[str, Any]:
        """Train RETA model by invoking RETA_code/main_reta_plus.py unchanged."""
        self._ensure_reta_dictionary_bundle(force_rebuild=False)

        reta_code = Path(self.reta_code_dir).resolve()
        data_dir = self._resolve_reta_data_dir()
        out_dir = Path(model_output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        command = [
            sys.executable,
            "main_reta_plus.py",
            "--indir",
            str(data_dir),
            "--outdir",
            str(out_dir),
            "--load",
            str(load),
            "--withTypes",
            str(with_types),
            "--epochs",
            str(int(epochs)),
            "--batchsize",
            str(int(batchsize)),
            "--num_filters",
            str(int(num_filters)),
            "--embsize",
            str(int(embsize)),
            "--learningrate",
            str(float(learningrate)),
            "--gpu_ids",
            str(gpu_ids),
            "--num_negative_samples",
            str(int(num_negative_samples)),
            "--atLeast",
            str(int(at_least)),
            "--topNfilters",
            str(int(top_nfilters)),
            "--buildTypeDictionaries",
            str(build_type_dictionaries),
            "--sparsifier",
            str(int(sparsifier)),
            "--entitiesEvaluated",
            str(entities_evaluated),
            "--negative_strategy",
            str(float(negative_strategy)),
        ]

        if str(load) == "preload" and model_to_be_trained:
            command.extend(["--modelToBeTrained", str(Path(model_to_be_trained).resolve())])

        subprocess.run(command, cwd=reta_code, check=True)
        model_path = self._resolve_latest_reta_model(out_dir)

        return {
            "adapter": self.name,
            "task": "train-ranking-model",
            "model_path": str(model_path.resolve()),
            "reta_data_dir": str(data_dir),
            "model_output_dir": str(out_dir),
            "runner_command": " ".join(command),
        }

    def rank_candidates(
        self,
        candidate_file: str | Path,
        ranking_model_path: str | Path,
        output_file: str | Path,
        candidate_budget: int,
        entities_evaluated: str = "both",
        top_nfilters: int = -10,
        at_least: int = 2,
        sparsifier: int = 2,
        build_type_dictionaries: str = "True",
        device: str = "cuda:0",
        max_facts: Optional[int] = None,
        map_to_sjp_dataset_dir: Optional[str | Path] = None,
    ) -> RankedPredictions:
        """Rank RETA candidates with RETA grader in canonical format."""
        self._ensure_reta_dictionary_bundle(force_rebuild=False)
        return rank_reta_candidates(
            reta_code_dir=self.reta_code_dir,
            reta_data_dir=self.reta_data_dir,
            candidate_file=candidate_file,
            model_path=ranking_model_path,
            output_file=output_file,
            candidate_budget=candidate_budget,
            entities_evaluated=entities_evaluated,
            top_nfilters=top_nfilters,
            at_least=at_least,
            sparsifier=sparsifier,
            build_type_dictionaries=build_type_dictionaries,
            device=device,
            max_facts=max_facts,
            map_to_sjp_dataset_dir=map_to_sjp_dataset_dir,
        )
