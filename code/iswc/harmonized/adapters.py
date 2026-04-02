"""Adapter layer for the harmonized SJP/RETA interface."""

from __future__ import annotations

import importlib
from argparse import Namespace
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .dataset import export_standardized_dataset_to_sjp
from .interface import (
    apply_candidate_budget,
    export_standard_dataset_to_reta,
    extract_reta_filter_candidates,
    load_ranked_predictions,
    rank_reta_candidates,
    save_ranked_predictions,
)


RankedPredictions = Dict[int, List[Tuple[int, int, Optional[float]]]]


class CandidateAdapter(ABC):
    """Base class for harmonized dataset/candidate/ranking adapters."""

    name: str

    @abstractmethod
    def prepare_dataset(
        self,
        standardized_dataset_dir: str | Path,
        output_dir: str | Path,
        **kwargs,
    ) -> Dict[str, Any]:
        """Convert canonical dataset splits into method-specific input files."""

    @abstractmethod
    def generate_candidates(self, **kwargs) -> RankedPredictions:
        """Generate candidates in canonical format."""

    @abstractmethod
    def rank_candidates(self, **kwargs) -> RankedPredictions:
        """Rank candidates in canonical format."""


@dataclass
class SJPAdapter(CandidateAdapter):
    """Adapter for SJP dataset conversion and phase-based generation/ranking."""

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
        grouped: Dict[int, List[Tuple[int, int, Optional[float]]]] = {}
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

    def generate_candidates(
        self,
        output_file: str | Path,
        candidate_budget: int,
        path_dataset_dir: str | Path,
        path_setup: str = "20_10",
        cmd: str = "train",
        log_dir: str | Path = "./logs/harmonized",
        expname: str = "harmonized_sjp",
        num_workers: int = 1,
        max_epochs: int = 100,
        tuple_checkpoint: Optional[str | Path] = None,
        skip_phase1: bool = False,
    ) -> RankedPredictions:
        """
        Run SJP phase 1 + phase 2 and export standardized candidates.
        """
        pathe_trainer, data_utils, du, triple_lib, pathdata = self._import_sjp_modules()

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
            skip_phase2=False,
        )

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

        p1_artifacts = pathe_trainer.run_phase_1_property_prediction(
            args,
            train_tuples,
            val_tuples,
            test_tuples,
            train_triples,
            val_triples,
            test_triples,
            paths,
            relcon,
            tokens_to_idxs,
            train_rel2inv,
            val_rel2inv,
            test_rel2inv,
        )
        trainer_t, pl_model_t, tuple_ckpt, tr_loader, va_loader, te_loader, train_set_t, valid_set_t, test_set_t = p1_artifacts

        _, _, cand_test, cand_scores = pathe_trainer.run_phase_2_candidate_generation(
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
            train_triples,
            val_triples,
            test_triples,
        )

        candidates_test, _ = cand_test
        predictions = self._group_candidate_tensor(candidates_test, cand_scores)
        predictions = apply_candidate_budget(predictions, int(candidate_budget))
        save_ranked_predictions(predictions, output_file)
        return predictions

    def rank_candidates(
        self,
        candidate_file: str | Path,
        output_file: str | Path,
        candidate_budget: int,
        path_dataset_dir: str | Path,
        path_setup: str = "20_10",
        cmd: str = "train",
        log_dir: str | Path = "./logs/harmonized",
        expname: str = "harmonized_sjp",
        num_workers: int = 1,
        max_epochs: int = 100,
        triple_checkpoint: Optional[str | Path] = None,
        skip_phase2: bool = False,
    ) -> RankedPredictions:
        """
        Run SJP phase 3 on standardized candidates and export ranked output.

        PathE's phase-3 helper does not emit per-triple scores, so the exported
        canonical ranking preserves input order/scores while ensuring phase-3 is
        executed on the provided candidate set.
        """
        pathe_trainer, data_utils, du, triple_lib, pathdata = self._import_sjp_modules()

        args = self._build_split_join_args(
            path_dataset_dir=path_dataset_dir,
            path_setup=path_setup,
            cmd=cmd,
            log_dir=log_dir,
            expname=expname,
            candidate_budget=candidate_budget,
            num_workers=num_workers,
            max_epochs=max_epochs,
            tuple_checkpoint=None,
            triple_checkpoint=triple_checkpoint,
            skip_phase1=False,
            skip_phase2=skip_phase2,
        )

        train_triples, val_triples, test_triples = data_utils.load_triple_tensors(
            args.train_paths,
            args.valid_paths,
            args.test_paths,
        )
        paths, relcon, _ = du.load_unrolled_setup(args.train_paths, args.path_setup)

        unique_rels = triple_lib.get_unique_relations(train_triples, val_triples, test_triples)
        tokens_to_idxs = pathdata.create_vocabulary_from_relations(unique_rels.tolist(), ["MSK"])

        provided_predictions = load_ranked_predictions(candidate_file)
        candidate_tensor = self._predictions_to_tensor(provided_predictions)

        cand_train = torch.unique(train_triples, dim=0)
        cand_val = torch.unique(val_triples, dim=0)
        cand_test = torch.unique(candidate_tensor, dim=0)

        train_labels = triple_lib.build_labels_for_triples(cand_train, train_triples)
        val_labels = triple_lib.build_labels_for_triples(cand_val, val_triples)
        test_labels = triple_lib.build_labels_for_triples(cand_test, test_triples)
        candidate_scores_test = torch.zeros(cand_test.size(0), dtype=torch.float32)

        pathe_trainer.run_phase_3_triple_classification(
            args,
            (cand_train, train_labels),
            (cand_val, val_labels),
            (cand_test, test_labels),
            paths,
            relcon,
            tokens_to_idxs,
            train_triples,
            val_triples,
            test_triples,
            candidate_scores_test,
        )

        ranked_predictions = apply_candidate_budget(provided_predictions, int(candidate_budget))
        save_ranked_predictions(ranked_predictions, output_file)
        return ranked_predictions


@dataclass
class RETAAdapter(CandidateAdapter):
    """Adapter for RETA dataset conversion and ranked candidate extraction."""

    reta_code_dir: str | Path
    reta_data_dir: str | Path
    name: str = "RETA"

    def prepare_dataset(
        self,
        standardized_dataset_dir: str | Path,
        output_dir: str | Path,
        default_entity_type: str = "Thing",
        build_reta_bin: bool = True,
        triple_order: str = "hrt",
        delimiter: Optional[str] = None,
        has_header: bool = False,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """Convert canonical dataset into RETA n-ary/type/dictionary inputs."""
        summary = export_standard_dataset_to_reta(
            standardized_dataset_dir=standardized_dataset_dir,
            output_dir=output_dir,
            default_entity_type=default_entity_type,
            build_reta_bin=build_reta_bin,
            triple_order=triple_order,
            delimiter=delimiter,
            has_header=has_header,
            overwrite=overwrite,
        )
        self.reta_data_dir = summary["reta_output_dir"]
        return summary

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

    def rank_candidates(
        self,
        candidate_file: str | Path,
        model_path: str | Path,
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
        return rank_reta_candidates(
            reta_code_dir=self.reta_code_dir,
            reta_data_dir=self.reta_data_dir,
            candidate_file=candidate_file,
            model_path=model_path,
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
