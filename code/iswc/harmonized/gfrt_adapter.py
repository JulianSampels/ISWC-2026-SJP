"""GFRT adapter implementation for the harmonized workflow."""

from __future__ import annotations

import csv
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

from iswc.gfrt import GFRTFilter, GFRTModel, GFRTTrainer, build_gfrt_pipeline

from .base_adapter import (
    CandidateAdapter,
    RankedPredictions,
    apply_candidate_budget,
    load_ranked_predictions,
    save_ranked_predictions,
)
from .dataset import load_standardized_dataset_triples, resolve_standardized_dataset


logger = logging.getLogger(__name__)


def _build_label_to_id_maps(
    triples_by_split: Mapping[str, Sequence[Tuple[str, str, str]]],
) -> Tuple[Dict[str, int], Dict[str, int]]:
    entity2id: Dict[str, int] = {}
    relation2id: Dict[str, int] = {}

    for split in ("train", "valid", "test"):
        for head_label, relation_label, tail_label in triples_by_split.get(split, []):
            if head_label not in entity2id:
                entity2id[head_label] = len(entity2id)
            if tail_label not in entity2id:
                entity2id[tail_label] = len(entity2id)
            if relation_label not in relation2id:
                relation2id[relation_label] = len(relation2id)

    if not entity2id:
        raise ValueError("Cannot build GFRT maps from an empty dataset.")
    if not relation2id:
        raise ValueError("Cannot build GFRT maps because no relations were found.")

    return entity2id, relation2id


def _encode_labeled_triples(
    triples: Sequence[Tuple[str, str, str]],
    entity2id: Dict[str, int],
    relation2id: Dict[str, int],
) -> List[Tuple[int, int, int]]:
    rows: List[Tuple[int, int, int]] = []
    for head_label, relation_label, tail_label in triples:
        rows.append((
            int(entity2id[head_label]),
            int(relation2id[relation_label]),
            int(entity2id[tail_label]),
        ))
    return rows


def _write_id_triples_csv(path: Path, triples: Iterable[Tuple[int, int, int]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["head_id", "relation_id", "tail_id"])
        for head_id, relation_id, tail_id in triples:
            writer.writerow([int(head_id), int(relation_id), int(tail_id)])
            count += 1
    return count


def _to_tensor(rows: Sequence[Tuple[int, int, int]]) -> torch.Tensor:
    if not rows:
        return torch.empty((0, 3), dtype=torch.long)
    return torch.as_tensor(rows, dtype=torch.long)


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_prepared_dir(path_dataset_dir: str | Path) -> Path:
    prepared_dir = Path(path_dataset_dir).resolve()
    if not prepared_dir.is_dir():
        raise FileNotFoundError(f"GFRT prepared dataset directory does not exist: {prepared_dir}")
    return prepared_dir


def _load_split_tensor(prepared_dir: Path, split: str) -> torch.Tensor:
    split_pt = prepared_dir / f"{split}.pt"
    if not split_pt.is_file():
        raise FileNotFoundError(f"Missing GFRT split tensor file: {split_pt}")

    tensor = torch.load(split_pt, map_location="cpu")
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"Expected tensor in {split_pt}, got {type(tensor)}")
    tensor = tensor.to(dtype=torch.long, device="cpu")
    if tensor.dim() != 2 or tensor.size(1) != 3:
        raise ValueError(f"Expected shape [N, 3] in {split_pt}, observed {tuple(tensor.shape)}")
    return tensor


def _load_training_runtime(path_dataset_dir: str | Path) -> Dict[str, Any]:
    prepared_dir = _resolve_prepared_dir(path_dataset_dir)

    entity2id_path = prepared_dir / "entity2id.json"
    relation2id_path = prepared_dir / "relation2id.json"
    if not entity2id_path.is_file() or not relation2id_path.is_file():
        raise FileNotFoundError(
            "GFRT prepared dataset is missing entity/relation maps. "
            "Expected files: entity2id.json and relation2id.json"
        )

    entity2id_raw = _load_json(entity2id_path)
    relation2id_raw = _load_json(relation2id_path)
    entity2id = {str(label): int(idx) for label, idx in entity2id_raw.items()}
    relation2id = {str(label): int(idx) for label, idx in relation2id_raw.items()}

    train_triples = _load_split_tensor(prepared_dir, "train")
    valid_triples = _load_split_tensor(prepared_dir, "valid")
    test_triples = _load_split_tensor(prepared_dir, "test")

    return {
        "prepared_dir": prepared_dir,
        "entity2id": entity2id,
        "relation2id": relation2id,
        "train_triples": train_triples,
        "valid_triples": valid_triples,
        "test_triples": test_triples,
        "num_entities": len(entity2id),
        "num_relations": len(relation2id),
    }


def _resolve_model_metadata_path(model_path: Path) -> Path:
    return model_path.with_suffix(".meta.json")


def _load_model_metadata(model_path: str | Path) -> Dict[str, Any]:
    path = _resolve_model_metadata_path(Path(model_path).resolve())
    if not path.is_file():
        return {}
    return _load_json(path)


def _default_model_path(log_dir: str | Path, expname: str) -> Path:
    return Path(log_dir).resolve() / expname / "gfrt_model.pt"


def _to_device(device: Optional[str] = None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = str(device).strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("Requested CUDA device '%s' is unavailable; using CPU.", device)
        return torch.device("cpu")
    return torch.device(device)


def _prepare_filter_runtime(
    path_dataset_dir: str | Path,
    model_path: str | Path,
    top_k1: Optional[int],
    top_k2: Optional[int],
    device: Optional[str],
) -> Dict[str, Any]:
    runtime = _load_training_runtime(path_dataset_dir)
    metadata = _load_model_metadata(model_path)

    resolved_top_k1 = int(top_k1) if top_k1 is not None else int(metadata.get("top_k1", 100))
    resolved_top_k2 = int(top_k2) if top_k2 is not None else int(metadata.get("top_k2", 30))
    run_device = _to_device(device)

    model, graph_h, graph_t = build_gfrt_pipeline(
        train_triples=runtime["train_triples"],
        num_entities=int(runtime["num_entities"]),
        num_relations=int(runtime["num_relations"]),
        embed_dim=int(metadata.get("embed_dim", 100)),
        num_layers=int(metadata.get("num_layers", 2)),
        top_k1=resolved_top_k1,
        top_k2=resolved_top_k2,
        margin=float(metadata.get("margin", 1.0)),
        device=run_device,
    )
    del model

    loaded_model = GFRTModel.load(str(Path(model_path).resolve()), device=run_device)
    trainer = GFRTTrainer(
        model=loaded_model,
        graph_H=graph_h,
        graph_T=graph_t,
        train_triples=runtime["train_triples"],
        device=run_device,
    )
    h_emb, r_h_emb, t_emb, r_t_emb = trainer.get_embeddings()

    runtime.update(
        {
            "model": loaded_model,
            "graph_h": graph_h,
            "graph_t": graph_t,
            "device": run_device,
            "top_k1": resolved_top_k1,
            "top_k2": resolved_top_k2,
            "h_emb": h_emb,
            "r_h_emb": r_h_emb,
            "t_emb": t_emb,
            "r_t_emb": r_t_emb,
        }
    )
    return runtime


@dataclass
class GFRTAdapter(CandidateAdapter):
    """Adapter for GFRT dataset conversion, model training, and inference."""

    name: str = "GFRT"

    def prepare_dataset(
        self,
        standardized_dataset_dir: str | Path,
        output_dir: str | Path,
        triple_order: str = "hrt",
        delimiter: Optional[str] = None,
        has_header: bool = False,
        overwrite: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """Encode standardized labeled triples into numeric GFRT-ready split tensors."""
        del kwargs
        standardized = resolve_standardized_dataset(standardized_dataset_dir)
        output_path = Path(output_dir).resolve()

        if output_path.exists() and overwrite:
            if output_path.is_dir():
                shutil.rmtree(output_path)
            else:
                output_path.unlink()
        elif output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_path}. "
                "Use --overwrite to replace it."
            )

        output_path.mkdir(parents=True, exist_ok=True)

        triples_by_split = load_standardized_dataset_triples(
            standardized_dataset_dir=standardized.root,
            triple_order=triple_order,
            delimiter=delimiter,
            has_header=has_header,
        )
        entity2id, relation2id = _build_label_to_id_maps(triples_by_split)

        id_rows: Dict[str, List[Tuple[int, int, int]]] = {
            split: _encode_labeled_triples(triples, entity2id, relation2id)
            for split, triples in triples_by_split.items()
        }

        split_counts: Dict[str, int] = {}
        for split in ("train", "valid", "test"):
            split_rows = id_rows[split]

            split_csv = output_path / f"{split}_ids.csv"
            split_counts[split] = _write_id_triples_csv(split_csv, split_rows)

            split_tensor = _to_tensor(split_rows)
            torch.save(split_tensor, output_path / f"{split}.pt")

        # Convenience gold file for harmonized metric CLI.
        _write_id_triples_csv(output_path / "gold_test.csv", id_rows["test"])

        id2entity = {int(idx): label for label, idx in entity2id.items()}
        id2relation = {int(idx): label for label, idx in relation2id.items()}

        with (output_path / "entity2id.json").open("w", encoding="utf-8") as handle:
            json.dump(entity2id, handle, indent=2)
        with (output_path / "relation2id.json").open("w", encoding="utf-8") as handle:
            json.dump(relation2id, handle, indent=2)
        with (output_path / "id2entity.json").open("w", encoding="utf-8") as handle:
            json.dump(id2entity, handle, indent=2)
        with (output_path / "id2relation.json").open("w", encoding="utf-8") as handle:
            json.dump(id2relation, handle, indent=2)

        summary = {
            "adapter": self.name,
            "task": "prepare-dataset",
            "standardized_dataset_dir": str(standardized.root),
            "gfrt_output_dir": str(output_path),
            "num_entities": len(entity2id),
            "num_relations": len(relation2id),
            "split_counts": split_counts,
            "gold_triples_file": str(output_path / "gold_test.csv"),
        }
        metadata_path = output_path / "harmonized_export_metadata.json"
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        summary["metadata_file"] = str(metadata_path)
        return summary

    def train_candidate_model(
        self,
        path_dataset_dir: str | Path,
        log_dir: str | Path = "./logs/harmonized",
        expname: str = "harmonized_gfrt",
        max_epochs: int = 100,
        embed_dim: int = 100,
        num_layers: int = 2,
        top_k1: int = 100,
        top_k2: int = 30,
        batch_size: int = 256,
        lr_intra: float = 0.01,
        lr_inter: float = 0.001,
        margin: float = 1.0,
        log_every: int = 10,
        model_path: Optional[str | Path] = None,
        load_model_path: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        """Train GFRT and save the trained model as candidate-generation checkpoint."""
        runtime = _load_training_runtime(path_dataset_dir)
        device = _to_device(None)

        target_model_path = Path(model_path).resolve() if model_path is not None else _default_model_path(log_dir, expname)
        target_model_path.parent.mkdir(parents=True, exist_ok=True)

        model, graph_h, graph_t = build_gfrt_pipeline(
            train_triples=runtime["train_triples"],
            num_entities=int(runtime["num_entities"]),
            num_relations=int(runtime["num_relations"]),
            embed_dim=int(embed_dim),
            num_layers=int(num_layers),
            top_k1=int(top_k1),
            top_k2=int(top_k2),
            margin=float(margin),
            device=device,
        )

        load_path = Path(load_model_path).resolve() if load_model_path is not None else None
        source_metadata: Dict[str, Any] = {}
        if load_path is not None and load_path.is_file():
            model = GFRTModel.load(str(load_path), device=device)
            source_metadata = _load_model_metadata(load_path)
        else:
            trainer = GFRTTrainer(
                model=model,
                graph_H=graph_h,
                graph_T=graph_t,
                train_triples=runtime["train_triples"],
                device=device,
                lr_intra=float(lr_intra),
                lr_inter=float(lr_inter),
            )

            epochs = max(int(max_epochs), 1)
            log_interval = max(int(log_every), 1)
            for epoch in range(1, epochs + 1):
                losses = trainer.train_epoch(batch_size=max(int(batch_size), 1))
                if epoch == 1 or epoch % log_interval == 0 or epoch == epochs:
                    logger.info(
                        "GFRT epoch %d/%d | L_H=%.4f L_T=%.4f L_cross=%.4f",
                        epoch,
                        epochs,
                        losses["loss_H"],
                        losses["loss_T"],
                        losses["loss_cross"],
                    )

        model.save(str(target_model_path))

        metadata = {
            "adapter": self.name,
            "path_dataset_dir": str(Path(path_dataset_dir).resolve()),
            "num_entities": int(runtime["num_entities"]),
            "num_relations": int(runtime["num_relations"]),
            "embed_dim": int(getattr(model, "embed_dim", embed_dim)),
            "num_layers": int(len(model.gnn_head.layers)),
            "margin": float(getattr(model, "margin_intra", margin)),
            "top_k1": int(source_metadata.get("top_k1", top_k1)),
            "top_k2": int(source_metadata.get("top_k2", top_k2)),
            "batch_size": int(batch_size),
            "lr_intra": float(lr_intra),
            "lr_inter": float(lr_inter),
            "max_epochs": int(max_epochs),
            "device": str(device),
        }
        metadata_path = _resolve_model_metadata_path(target_model_path)
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

        return {
            "adapter": self.name,
            "task": "train-candidate-model",
            "model_path": str(target_model_path),
            "model_metadata_path": str(metadata_path),
            "path_dataset_dir": str(Path(path_dataset_dir).resolve()),
            "device": str(device),
        }

    def generate_candidates(
        self,
        output_file: str | Path,
        candidate_budget: int,
        path_dataset_dir: str | Path,
        candidate_model_path: str | Path,
        top_m_relations: int = 20,
        top_n_tails: int = 100,
        top_k1: Optional[int] = None,
        top_k2: Optional[int] = None,
        device: Optional[str] = None,
        max_facts: Optional[int] = None,
    ) -> RankedPredictions:
        """Generate GFRT candidates from test heads and save canonical ranked output."""
        runtime = _prepare_filter_runtime(
            path_dataset_dir=path_dataset_dir,
            model_path=candidate_model_path,
            top_k1=top_k1,
            top_k2=top_k2,
            device=device,
        )

        gfrt_filter = GFRTFilter(
            h_emb=runtime["h_emb"],
            rH_emb=runtime["r_h_emb"],
            t_emb=runtime["t_emb"],
            rT_emb=runtime["r_t_emb"],
            model=runtime["model"],
            train_triples=runtime["train_triples"],
            top_m_relations=max(int(top_m_relations), 1),
            top_n_tails=max(int(top_n_tails), 1),
        )

        test_heads = sorted({int(row[0]) for row in runtime["test_triples"].tolist()})
        if max_facts is not None:
            test_heads = test_heads[: max(int(max_facts), 0)]

        predictions: RankedPredictions = {}
        for head_id in test_heads:
            rows = gfrt_filter.generate_candidates(head=head_id, max_candidates=int(candidate_budget))
            predictions[head_id] = [
                (int(relation_id), int(tail_id), float(score))
                for _, relation_id, tail_id, score in rows
            ]

        predictions = apply_candidate_budget(predictions, int(candidate_budget))
        save_ranked_predictions(predictions, output_file)
        return predictions

    def train_ranking_model(
        self,
        path_dataset_dir: str | Path,
        candidate_model_path: Optional[str | Path] = None,
        log_dir: str | Path = "./logs/harmonized",
        expname: str = "harmonized_gfrt",
        max_epochs: int = 100,
        embed_dim: int = 100,
        num_layers: int = 2,
        top_k1: int = 100,
        top_k2: int = 30,
        batch_size: int = 256,
        lr_intra: float = 0.01,
        lr_inter: float = 0.001,
        margin: float = 1.0,
        log_every: int = 10,
    ) -> Dict[str, Any]:
        """GFRT uses one model for candidate generation and ranking.

        If candidate_model_path is not provided, this trains a new model and
        returns it as the ranking model path.
        """
        if candidate_model_path is not None:
            model_path = Path(candidate_model_path).resolve()
            if not model_path.is_file():
                raise FileNotFoundError(f"GFRT model file not found: {model_path}")
            return {
                "adapter": self.name,
                "task": "train-ranking-model",
                "model_path": str(model_path),
                "path_dataset_dir": str(Path(path_dataset_dir).resolve()),
                "message": "GFRT uses the same trained model for candidate generation and ranking.",
            }

        trained = self.train_candidate_model(
            path_dataset_dir=path_dataset_dir,
            log_dir=log_dir,
            expname=expname,
            max_epochs=max_epochs,
            embed_dim=embed_dim,
            num_layers=num_layers,
            top_k1=top_k1,
            top_k2=top_k2,
            batch_size=batch_size,
            lr_intra=lr_intra,
            lr_inter=lr_inter,
            margin=margin,
            log_every=log_every,
        )
        trained["task"] = "train-ranking-model"
        trained["message"] = "GFRT uses the same trained model for candidate generation and ranking."
        return trained

    def rank_candidates(
        self,
        candidate_file: str | Path,
        output_file: str | Path,
        candidate_budget: int,
        path_dataset_dir: str | Path,
        ranking_model_path: str | Path,
        top_k1: Optional[int] = None,
        top_k2: Optional[int] = None,
        device: Optional[str] = None,
        max_facts: Optional[int] = None,
    ) -> RankedPredictions:
        """Rank provided candidates with GFRT scoring and save canonical output."""
        runtime = _prepare_filter_runtime(
            path_dataset_dir=path_dataset_dir,
            model_path=ranking_model_path,
            top_k1=top_k1,
            top_k2=top_k2,
            device=device,
        )

        candidate_predictions = load_ranked_predictions(candidate_file)
        ranked_predictions: RankedPredictions = {}

        processed_heads = 0
        for head_id in sorted(candidate_predictions.keys()):
            if max_facts is not None and processed_heads >= max(int(max_facts), 0):
                break
            processed_heads += 1

            rows = candidate_predictions[head_id]
            if not rows:
                continue

            relation_ids = torch.as_tensor([int(r) for r, _, _ in rows], dtype=torch.long, device=runtime["device"])
            tail_ids = torch.as_tensor([int(t) for _, t, _ in rows], dtype=torch.long, device=runtime["device"])
            head_ids = torch.full(
                (relation_ids.size(0),),
                int(head_id),
                dtype=torch.long,
                device=runtime["device"],
            )

            scores = runtime["model"].score_candidates(
                heads=head_ids,
                relations=relation_ids,
                tails=tail_ids,
                h_emb=runtime["h_emb"],
                rH_emb=runtime["r_h_emb"],
                t_emb=runtime["t_emb"],
                rT_emb=runtime["r_t_emb"],
            )

            order = torch.argsort(scores, descending=True)
            ranked_rows: List[Tuple[int, int, Optional[float]]] = []
            for idx in order.tolist():
                relation_id, tail_id, _ = rows[int(idx)]
                ranked_rows.append((int(relation_id), int(tail_id), float(scores[int(idx)].item())))
            ranked_predictions[int(head_id)] = ranked_rows

        ranked_predictions = apply_candidate_budget(ranked_predictions, int(candidate_budget))
        save_ranked_predictions(ranked_predictions, output_file)
        return ranked_predictions
