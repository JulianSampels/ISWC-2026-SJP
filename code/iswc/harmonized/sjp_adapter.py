"""SJP adapter implementation for the harmonized workflow."""

from __future__ import annotations

import importlib
import sys
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from .base_adapter import (
    CandidateAdapter,
    RankedPredictions,
    apply_candidate_budget,
    load_ranked_predictions,
    save_ranked_predictions,
)
from .dataset import export_standardized_dataset_to_sjp


@dataclass
class SJPAdapter(CandidateAdapter):
    """Adapter for SJP dataset conversion and phase-based training/inference."""

    sjp_code_dir: str | Path | None = None
    name: str = "SJP"

    def _resolve_sjp_code_dir(self) -> Path:
        if self.sjp_code_dir is not None:
            return Path(self.sjp_code_dir).resolve()
        return Path(__file__).resolve().parents[2] / "SJP_code"

    def _resolve_path_setup(self, train_path: Path, requested: str) -> str:
        if any(path.name.endswith("_paths.csv") and path.name.startswith(requested) for path in train_path.iterdir()):
            return requested

        candidates = sorted(
            path.name for path in train_path.iterdir()
            if path.is_file() and path.name.endswith("_paths.csv")
        )
        for filename in candidates:
            if filename.endswith("_er_paths.csv"):
                return filename[: -len("_er_paths.csv")]
            return filename[: -len("_paths.csv")]

        raise FileNotFoundError(
            f"Could not infer path_setup from {train_path}. No *_paths.csv files were found."
        )

    def _import_sjp_modules(self):
        sjp_code_dir = self._resolve_sjp_code_dir()
        if str(sjp_code_dir) not in sys.path:
            sys.path.insert(0, str(sjp_code_dir))

        pathe_trainer = importlib.import_module("PathE.pathe.pathe_trainer")
        data_utils = importlib.import_module("PathE.pathe.data_utils")
        du = importlib.import_module("PathE.pathe.data_utils")
        triple_lib = importlib.import_module("PathE.pathe.triple_lib")
        pathdata = importlib.import_module("PathE.pathe.pathdata")
        return pathe_trainer, data_utils, du, triple_lib, pathdata

    @staticmethod
    def _group_candidate_tensor(
        candidates: torch.Tensor,
        scores: Optional[torch.Tensor],
    ) -> RankedPredictions:
        grouped: Dict[int, List[tuple[int, int, Optional[float]]]] = {}
        if candidates.dim() != 2 or candidates.size(1) != 3:
            raise ValueError("Expected candidate tensor with shape [N, 3].")

        score_values: Optional[List[float]] = None
        if scores is not None:
            flat_scores = scores.detach().cpu().reshape(-1)
            if flat_scores.numel() != candidates.size(0):
                raise ValueError("Candidate scores must align with candidate triples.")
            score_values = [float(x) for x in flat_scores.tolist()]

        for index, triple in enumerate(candidates.detach().cpu().tolist()):
            head_id, relation_id, tail_id = map(int, triple)
            score = score_values[index] if score_values is not None else None
            grouped.setdefault(head_id, []).append((relation_id, tail_id, score))

        if score_values is not None:
            for head_id in list(grouped.keys()):
                grouped[head_id].sort(
                    key=lambda item: float(item[2]) if item[2] is not None else float("-inf"),
                    reverse=True,
                )

        return grouped

    @staticmethod
    def _predictions_to_tensor(predictions: RankedPredictions) -> torch.Tensor:
        rows: List[List[int]] = []
        for head_id, ranked in predictions.items():
            for relation_id, tail_id, _ in ranked:
                rows.append([int(head_id), int(relation_id), int(tail_id)])

        if not rows:
            raise ValueError("Candidate file is empty. At least one candidate triple is required.")
        return torch.as_tensor(rows, dtype=torch.long)

    @staticmethod
    def _latest_checkpoint(checkpoint_dir: Path, before: set[Path]) -> Optional[Path]:
        if not checkpoint_dir.is_dir():
            return None
        ckpts = sorted(checkpoint_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
        for ckpt in ckpts:
            if ckpt not in before:
                return ckpt
        return ckpts[0] if ckpts else None

    def _build_split_join_args(
        self,
        path_dataset_dir: str | Path,
        path_setup: str,
        cmd: str,
        log_dir: str | Path,
        expname: str,
        candidate_budget: int,
        num_workers: int,
        max_epochs: int,
        tuple_checkpoint: Optional[str | Path],
        triple_checkpoint: Optional[str | Path],
        skip_phase1: bool,
        skip_phase2: bool,
    ) -> Namespace:
        path_root = Path(path_dataset_dir).resolve()
        train_path = path_root / "train"
        valid_path = path_root / "val"
        test_path = path_root / "test"

        for required in (train_path, valid_path, test_path):
            if not required.is_dir():
                raise FileNotFoundError(f"Missing SJP split directory: {required}")

        effective_setup = self._resolve_path_setup(train_path, str(path_setup))

        log_root = Path(log_dir).resolve()
        checkpoint_dir = log_root / expname / "checkpoints"
        figure_dir = log_root / expname / "figures"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        figure_dir.mkdir(parents=True, exist_ok=True)

        return Namespace(
            cmd=cmd,
            model="SplitJoinPredictPathE",
            node_projector="dummy",
            train_paths=str(train_path),
            valid_paths=str(valid_path),
            test_paths=str(test_path),
            path_setup=str(effective_setup),
            max_ppt=50,
            augmentation_factor=20,
            num_workers=max(int(num_workers), 0),
            batch_size=16,
            val_batch_size=16,
            device="cuda" if torch.cuda.is_available() else "cpu",
            num_devices=1,
            debug=False,
            wandb_project=None,
            wandb_id=None,
            tuple_monitor="valid_mrr",
            triple_monitor="valid_link_mrr",
            checkpoint_dir=str(checkpoint_dir),
            chekpoint_ksteps=None,
            max_epochs=max(int(max_epochs), 1),
            val_check_interval=1.0,
            check_val_every_n_epoch=1,
            patience=10,
            use_manual_optimization=True,
            link_head_detached=False,
            accumulate_gradient=1,
            class_weights=False,
            phase1_rp_loss_fn="bce",
            phase1_tp_loss_fn="bce",
            lp_loss_fn="bce",
            loss_weight=0.5,
            full_test=False,
            num_negatives=0,
            val_num_negatives=0,
            candidate_generator="global_with_tail",
            candidates_threshold_p=None,
            candidates_quantile_q=None,
            candidates_temperature=1.0,
            candidates_alpha=0.5,
            candidates_beta=0.5,
            candidates_cap=int(candidate_budget),
            candidates_normalize_mode="global_joint",
            group_strategy=[0],
            figure_dir=str(figure_dir),
            save_candidates=True,
            save_candidates_csv=True,
            candidate_output_dir=str(log_root / expname / "phase2_candidates"),
            skip_phase1=bool(skip_phase1),
            skip_phase2=bool(skip_phase2),
            tuple_checkpoint=str(Path(tuple_checkpoint).resolve()) if tuple_checkpoint is not None else None,
            triple_checkpoint=str(Path(triple_checkpoint).resolve()) if triple_checkpoint is not None else None,
            log_dir=str(log_root),
            expname=expname,
            version=0,
        )

    def _prepare_phase_inputs(self, args: Namespace) -> Dict[str, Any]:
        pathe_trainer, data_utils, du, triple_lib, pathdata = self._import_sjp_modules()

        train_tuples, val_tuples, test_tuples = data_utils.load_tuple_tensors(
            args.train_paths,
            args.valid_paths,
            args.test_paths,
        )
        train_triples, val_triples, test_triples = data_utils.load_triple_tensors(
            args.train_paths,
            args.valid_paths,
            args.test_paths,
        )
        paths, relcon, _ = du.load_unrolled_setup(args.train_paths, args.path_setup)

        train_rel2inv = du.load_relation2inverse_relation_from_file(args.train_paths)
        val_rel2inv = du.load_relation2inverse_relation_from_file(args.valid_paths)
        test_rel2inv = du.load_relation2inverse_relation_from_file(args.test_paths)

        unique_rels = triple_lib.get_unique_relations(train_triples, val_triples, test_triples)
        tokens_to_idxs = pathdata.create_vocabulary_from_relations(unique_rels.tolist(), ["MSK"])

        return {
            "pathe_trainer": pathe_trainer,
            "triple_lib": triple_lib,
            "paths": paths,
            "relcon": relcon,
            "tokens_to_idxs": tokens_to_idxs,
            "train_tuples": train_tuples,
            "val_tuples": val_tuples,
            "test_tuples": test_tuples,
            "train_triples": train_triples,
            "val_triples": val_triples,
            "test_triples": test_triples,
            "train_rel2inv": train_rel2inv,
            "val_rel2inv": val_rel2inv,
            "test_rel2inv": test_rel2inv,
        }

    def prepare_dataset(
        self,
        standardized_dataset_dir: str | Path,
        output_dir: str | Path,
        num_paths_per_entity: int = 20,
        num_steps: int = 10,
        parallel: bool = True,
        inverse_mode: str = "manual",
        triple_order: str = "hrt",
        delimiter: Optional[str] = None,
        has_header: bool = False,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """Convert canonical dataset into SJP path dataset structure."""
        if inverse_mode != "manual":
            raise ValueError(
                "SJP adapter requires inverse_mode='manual' because PathE expects explicit "
                "inverse relation mapping files in the translated dataset."
            )
        return export_standardized_dataset_to_sjp(
            standardized_dataset_dir=standardized_dataset_dir,
            output_dir=output_dir,
            num_paths_per_entity=num_paths_per_entity,
            num_steps=num_steps,
            parallel=parallel,
            inverse_mode=inverse_mode,
            triple_order=triple_order,
            delimiter=delimiter,
            has_header=has_header,
            overwrite=overwrite,
        )

    def train_candidate_model(
        self,
        path_dataset_dir: str | Path,
        path_setup: str = "20_10",
        cmd: str = "train",
        log_dir: str | Path = "./logs/harmonized",
        expname: str = "harmonized_sjp",
        num_workers: int = 1,
        max_epochs: int = 100,
        tuple_checkpoint: Optional[str | Path] = None,
        skip_phase1: bool = False,
        candidate_budget: int = 500,
    ) -> Dict[str, Any]:
        """Train SJP phase-1 model and return saved tuple checkpoint path."""
        args = self._build_split_join_args(
            path_dataset_dir=path_dataset_dir,
            path_setup=path_setup,
            cmd=cmd,
            log_dir=log_dir,
            expname=expname,
            candidate_budget=candidate_budget,
            num_workers=num_workers,
            max_epochs=max_epochs,
            tuple_checkpoint=tuple_checkpoint,
            triple_checkpoint=None,
            skip_phase1=skip_phase1,
            skip_phase2=True,
        )

        phase = self._prepare_phase_inputs(args)
        p1_artifacts = phase["pathe_trainer"].run_phase_1_property_prediction(
            args,
            phase["train_tuples"],
            phase["val_tuples"],
            phase["test_tuples"],
            phase["train_triples"],
            phase["val_triples"],
            phase["test_triples"],
            phase["paths"],
            phase["relcon"],
            phase["tokens_to_idxs"],
            phase["train_rel2inv"],
            phase["val_rel2inv"],
            phase["test_rel2inv"],
        )

        tuple_ckpt = p1_artifacts[2]
        if not tuple_ckpt:
            raise RuntimeError("SJP candidate model training did not produce a tuple checkpoint.")

        return {
            "adapter": self.name,
            "task": "train-candidate-model",
            "model_path": str(Path(tuple_ckpt).resolve()),
            "path_dataset_dir": str(Path(path_dataset_dir).resolve()),
            "log_dir": str(Path(log_dir).resolve()),
            "expname": expname,
        }

    def generate_candidates(
        self,
        output_file: str | Path,
        candidate_budget: int,
        path_dataset_dir: str | Path,
        candidate_model_path: str | Path,
        path_setup: str = "20_10",
        log_dir: str | Path = "./logs/harmonized",
        expname: str = "harmonized_sjp",
        num_workers: int = 1,
        max_epochs: int = 100,
    ) -> RankedPredictions:
        """Generate SJP candidates using a trained phase-1 model checkpoint."""
        args = self._build_split_join_args(
            path_dataset_dir=path_dataset_dir,
            path_setup=path_setup,
            cmd="test",
            log_dir=log_dir,
            expname=expname,
            candidate_budget=candidate_budget,
            num_workers=num_workers,
            max_epochs=max_epochs,
            tuple_checkpoint=candidate_model_path,
            triple_checkpoint=None,
            skip_phase1=True,
            skip_phase2=False,
        )

        phase = self._prepare_phase_inputs(args)
        p1_artifacts = phase["pathe_trainer"].run_phase_1_property_prediction(
            args,
            phase["train_tuples"],
            phase["val_tuples"],
            phase["test_tuples"],
            phase["train_triples"],
            phase["val_triples"],
            phase["test_triples"],
            phase["paths"],
            phase["relcon"],
            phase["tokens_to_idxs"],
            phase["train_rel2inv"],
            phase["val_rel2inv"],
            phase["test_rel2inv"],
        )
        trainer_t, pl_model_t, tuple_ckpt, tr_loader, va_loader, te_loader, train_set_t, valid_set_t, test_set_t = p1_artifacts

        _, _, cand_test, cand_scores = phase["pathe_trainer"].run_phase_2_candidate_generation(
            args,
            trainer_t,
            pl_model_t,
            tuple_ckpt,
            tr_loader,
            va_loader,
            te_loader,
            train_set_t,
            valid_set_t,
            test_set_t,
            phase["train_triples"],
            phase["val_triples"],
            phase["test_triples"],
        )

        candidates_test, _ = cand_test
        predictions = self._group_candidate_tensor(candidates_test, cand_scores)
        predictions = apply_candidate_budget(predictions, int(candidate_budget))
        save_ranked_predictions(predictions, output_file)
        return predictions

    def train_ranking_model(
        self,
        path_dataset_dir: str | Path,
        candidate_model_path: str | Path,
        path_setup: str = "20_10",
        cmd: str = "train",
        log_dir: str | Path = "./logs/harmonized",
        expname: str = "harmonized_sjp",
        num_workers: int = 1,
        max_epochs: int = 100,
        triple_checkpoint: Optional[str | Path] = None,
        candidate_budget: int = 500,
        skip_phase2: bool = False,
    ) -> Dict[str, Any]:
        """Train SJP phase-3 model and return saved triple checkpoint path."""
        args = self._build_split_join_args(
            path_dataset_dir=path_dataset_dir,
            path_setup=path_setup,
            cmd=cmd,
            log_dir=log_dir,
            expname=expname,
            candidate_budget=candidate_budget,
            num_workers=num_workers,
            max_epochs=max_epochs,
            tuple_checkpoint=candidate_model_path,
            triple_checkpoint=triple_checkpoint,
            skip_phase1=True,
            skip_phase2=skip_phase2,
        )

        phase = self._prepare_phase_inputs(args)
        p1_artifacts = phase["pathe_trainer"].run_phase_1_property_prediction(
            args,
            phase["train_tuples"],
            phase["val_tuples"],
            phase["test_tuples"],
            phase["train_triples"],
            phase["val_triples"],
            phase["test_triples"],
            phase["paths"],
            phase["relcon"],
            phase["tokens_to_idxs"],
            phase["train_rel2inv"],
            phase["val_rel2inv"],
            phase["test_rel2inv"],
        )
        trainer_t, pl_model_t, tuple_ckpt, tr_loader, va_loader, te_loader, train_set_t, valid_set_t, test_set_t = p1_artifacts

        cand_train, cand_val, cand_test, cand_scores = phase["pathe_trainer"].run_phase_2_candidate_generation(
            args,
            trainer_t,
            pl_model_t,
            tuple_ckpt,
            tr_loader,
            va_loader,
            te_loader,
            train_set_t,
            valid_set_t,
            test_set_t,
            phase["train_triples"],
            phase["val_triples"],
            phase["test_triples"],
        )

        checkpoint_dir = Path(args.checkpoint_dir)
        before = set(checkpoint_dir.glob("*.ckpt")) if checkpoint_dir.is_dir() else set()

        phase["pathe_trainer"].run_phase_3_triple_classification(
            args,
            cand_train,
            cand_val,
            cand_test,
            phase["paths"],
            phase["relcon"],
            phase["tokens_to_idxs"],
            phase["train_triples"],
            phase["val_triples"],
            phase["test_triples"],
            cand_scores,
        )

        triple_ckpt_path = self._latest_checkpoint(checkpoint_dir, before)
        if triple_ckpt_path is None:
            raise RuntimeError("SJP ranking model training did not produce a checkpoint.")

        return {
            "adapter": self.name,
            "task": "train-ranking-model",
            "model_path": str(triple_ckpt_path.resolve()),
            "path_dataset_dir": str(Path(path_dataset_dir).resolve()),
            "log_dir": str(Path(log_dir).resolve()),
            "expname": expname,
        }

    def rank_candidates(
        self,
        candidate_file: str | Path,
        output_file: str | Path,
        candidate_budget: int,
        path_dataset_dir: str | Path,
        ranking_model_path: str | Path,
        path_setup: str = "20_10",
        log_dir: str | Path = "./logs/harmonized",
        expname: str = "harmonized_sjp",
        num_workers: int = 1,
        max_epochs: int = 100,
    ) -> RankedPredictions:
        """Apply SJP phase-3 model to provided candidates and save standardized output.

        PathE phase-3 helper currently returns evaluation metrics, not per-triple
        score tensors. The saved ranking therefore preserves candidate ordering.
        """
        args = self._build_split_join_args(
            path_dataset_dir=path_dataset_dir,
            path_setup=path_setup,
            cmd="test",
            log_dir=log_dir,
            expname=expname,
            candidate_budget=candidate_budget,
            num_workers=num_workers,
            max_epochs=max_epochs,
            tuple_checkpoint=None,
            triple_checkpoint=ranking_model_path,
            skip_phase1=False,
            skip_phase2=True,
        )

        phase = self._prepare_phase_inputs(args)

        provided_predictions = load_ranked_predictions(candidate_file)
        candidate_tensor = self._predictions_to_tensor(provided_predictions)

        cand_train = torch.unique(phase["train_triples"], dim=0)
        cand_val = torch.unique(phase["val_triples"], dim=0)
        cand_test = torch.unique(candidate_tensor, dim=0)

        train_labels = phase["triple_lib"].build_labels_for_triples(cand_train, phase["train_triples"])
        val_labels = phase["triple_lib"].build_labels_for_triples(cand_val, phase["val_triples"])
        test_labels = phase["triple_lib"].build_labels_for_triples(cand_test, phase["test_triples"])
        candidate_scores_test = torch.zeros(cand_test.size(0), dtype=torch.float32)

        phase["pathe_trainer"].run_phase_3_triple_classification(
            args,
            (cand_train, train_labels),
            (cand_val, val_labels),
            (cand_test, test_labels),
            phase["paths"],
            phase["relcon"],
            phase["tokens_to_idxs"],
            phase["train_triples"],
            phase["val_triples"],
            phase["test_triples"],
            candidate_scores_test,
        )

        ranked_predictions = apply_candidate_budget(provided_predictions, int(candidate_budget))
        save_ranked_predictions(ranked_predictions, output_file)
        return ranked_predictions
