"""Adapter layer for the harmonized SJP/RETA interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..evaluation import MetricResults
from .dataset import export_standardized_dataset_to_sjp
from .interface import (
    apply_candidate_budget,
    evaluate_candidate_file,
    export_standard_dataset_to_reta,
    extract_reta_candidates,
    load_ranked_predictions,
    parse_k_values,
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
    def load_predictions(self, candidate_file: str | Path) -> RankedPredictions:
        """Load ranked candidates from a method-specific artifact."""

    @abstractmethod
    def generate_candidates(self, **kwargs) -> RankedPredictions:
        """Generate (or standardize) candidates in canonical ranked format."""

    def generate_final_ranking(self, **kwargs) -> RankedPredictions:
        """Generate final rankings in canonical format.

        Many pipelines use the same primitive for candidate generation and
        final ranking. Adapters can override this when needed.
        """
        return self.generate_candidates(**kwargs)

    def evaluate(
        self,
        candidate_file: str | Path,
        gold_triples_file: str | Path,
        k_values: Sequence[int] | str,
    ) -> MetricResults:
        """Evaluate candidates with shared entity-centric metrics."""
        if isinstance(k_values, str):
            k_values = parse_k_values(k_values)
        return evaluate_candidate_file(candidate_file, gold_triples_file, list(k_values))


@dataclass
class SJPAdapter(CandidateAdapter):
    """Adapter for SJP dataset conversion and phase-2 candidate generation."""

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

    def load_predictions(self, candidate_file: str | Path) -> RankedPredictions:
        return load_ranked_predictions(candidate_file)

    def run_phase2_submodule(
        self,
        path_dataset_dir: str | Path,
        candidate_budget: int,
        path_setup: str = "20_10",
        cmd: str = "train",
        log_dir: str | Path = "./logs/harmonized",
        expname: str = "harmonized_sjp",
        candidate_output_dir: Optional[str | Path] = None,
        tuple_checkpoint: Optional[str | Path] = None,
        triple_checkpoint: Optional[str | Path] = None,
        additional_runner_args: Optional[Sequence[str]] = None,
        python_executable: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run SJP Split-Join-Predict and persist phase-2 candidates."""
        path_root = Path(path_dataset_dir).resolve()
        train_path = path_root / "train"
        valid_path = path_root / "val"
        test_path = path_root / "test"

        for required in (train_path, valid_path, test_path):
            if not required.is_dir():
                raise FileNotFoundError(f"Missing SJP split directory: {required}")

        effective_path_setup = self._resolve_path_setup(train_path, str(path_setup))

        out_dir = Path(candidate_output_dir).resolve() if candidate_output_dir is not None else Path(log_dir).resolve() / expname / "phase2_candidates"
        out_dir.mkdir(parents=True, exist_ok=True)

        if python_executable is None:
            python_executable = sys.executable

        runner_args = [
            python_executable,
            "-m",
            "PathE.pathe.runner",
            cmd,
            "SplitJoinPredictPathE",
            "--log_dir",
            str(Path(log_dir).resolve()),
            "--expname",
            expname,
            "--train_paths",
            str(train_path),
            "--valid_paths",
            str(valid_path),
            "--test_paths",
            str(test_path),
            "--path_setup",
            str(effective_path_setup),
            "--candidates_cap",
            str(int(candidate_budget)),
            "--save_candidates",
            "--candidate_output_dir",
            str(out_dir),
            "--use_manual_optimization",
        ]

        if tuple_checkpoint is not None:
            runner_args.extend(["--tuple_checkpoint", str(Path(tuple_checkpoint).resolve())])
        if triple_checkpoint is not None:
            runner_args.extend(["--triple_checkpoint", str(Path(triple_checkpoint).resolve())])
        if additional_runner_args:
            runner_args.extend([str(item) for item in additional_runner_args])

        worker_index: Optional[int] = None
        for index, value in enumerate(runner_args[:-1]):
            if value == "--num_workers":
                worker_index = index + 1

        if worker_index is None:
            runner_args.extend(["--num_workers", "1"])
        else:
            try:
                parsed_workers = int(runner_args[worker_index])
            except ValueError:
                parsed_workers = 1
            if parsed_workers < 1:
                runner_args[worker_index] = "1"

        if "--ent_aggregation" not in runner_args:
            runner_args.extend(["--ent_aggregation", "transformer"])

        sjp_cwd = self._resolve_sjp_code_dir()
        subprocess.run(runner_args, cwd=sjp_cwd, check=True)

        phase2_candidate_file = out_dir / "phase2_candidates_test.pt"
        if not phase2_candidate_file.exists():
            raise FileNotFoundError(
                f"Expected phase-2 candidate output was not found: {phase2_candidate_file}"
            )

        return {
            "adapter": self.name,
            "cmd": cmd,
            "path_dataset_dir": str(path_root),
            "candidate_budget": int(candidate_budget),
            "path_setup": str(effective_path_setup),
            "candidate_output_dir": str(out_dir),
            "phase2_candidate_file": str(phase2_candidate_file),
            "runner_command": " ".join(shlex.quote(arg) for arg in runner_args),
            "runner_cwd": str(sjp_cwd),
        }

    def generate_candidates(
        self,
        output_file: str | Path,
        candidate_budget: int,
        phase2_candidate_file: Optional[str | Path] = None,
        run_submodule: bool = False,
        path_dataset_dir: Optional[str | Path] = None,
        path_setup: str = "20_10",
        cmd: str = "train",
        log_dir: str | Path = "./logs/harmonized",
        expname: str = "harmonized_sjp",
        candidate_output_dir: Optional[str | Path] = None,
        tuple_checkpoint: Optional[str | Path] = None,
        triple_checkpoint: Optional[str | Path] = None,
        additional_runner_args: Optional[Sequence[str]] = None,
        python_executable: Optional[str] = None,
    ) -> RankedPredictions:
        """
        Generate canonical ranked candidates from SJP phase-2 artifacts.

        If `run_submodule=True`, this method first invokes the SJP runner with
        `--candidates_cap=<candidate_budget>` and then standardizes the saved
        `phase2_candidates_test.pt` file.
        """
        if phase2_candidate_file is None and run_submodule:
            if path_dataset_dir is None:
                raise ValueError("path_dataset_dir is required when run_submodule=True")
            run_summary = self.run_phase2_submodule(
                path_dataset_dir=path_dataset_dir,
                candidate_budget=candidate_budget,
                path_setup=path_setup,
                cmd=cmd,
                log_dir=log_dir,
                expname=expname,
                candidate_output_dir=candidate_output_dir,
                tuple_checkpoint=tuple_checkpoint,
                triple_checkpoint=triple_checkpoint,
                additional_runner_args=additional_runner_args,
                python_executable=python_executable,
            )
            phase2_candidate_file = run_summary["phase2_candidate_file"]

        if phase2_candidate_file is None:
            raise ValueError("phase2_candidate_file is required unless run_submodule=True")

        predictions = self.load_predictions(phase2_candidate_file)
        predictions = apply_candidate_budget(predictions, int(candidate_budget))
        save_ranked_predictions(predictions, output_file)
        return predictions

    def generate_final_ranking(
        self,
        output_file: str | Path,
        candidate_budget: int,
        phase2_candidate_file: Optional[str | Path] = None,
        run_submodule: bool = False,
        path_dataset_dir: Optional[str | Path] = None,
        path_setup: str = "20_10",
        cmd: str = "train",
        log_dir: str | Path = "./logs/harmonized",
        expname: str = "harmonized_sjp",
        candidate_output_dir: Optional[str | Path] = None,
        tuple_checkpoint: Optional[str | Path] = None,
        triple_checkpoint: Optional[str | Path] = None,
        additional_runner_args: Optional[Sequence[str]] = None,
        python_executable: Optional[str] = None,
    ) -> RankedPredictions:
        """Produce final canonical rankings using SJP phase-2 scored candidates."""
        return self.generate_candidates(
            output_file=output_file,
            candidate_budget=candidate_budget,
            phase2_candidate_file=phase2_candidate_file,
            run_submodule=run_submodule,
            path_dataset_dir=path_dataset_dir,
            path_setup=path_setup,
            cmd=cmd,
            log_dir=log_dir,
            expname=expname,
            candidate_output_dir=candidate_output_dir,
            tuple_checkpoint=tuple_checkpoint,
            triple_checkpoint=triple_checkpoint,
            additional_runner_args=additional_runner_args,
            python_executable=python_executable,
        )


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

    def load_predictions(self, candidate_file: str | Path) -> RankedPredictions:
        return load_ranked_predictions(candidate_file)

    def generate_candidates(
        self,
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
        """Generate canonical RETA candidate rankings from a trained model."""
        return extract_reta_candidates(
            reta_code_dir=self.reta_code_dir,
            reta_data_dir=self.reta_data_dir,
            model_path=model_path,
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

    def generate_final_ranking(
        self,
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
        """Produce final RETA rankings in canonical format."""
        return self.generate_candidates(
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

    def extract_candidates(
        self,
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
        candidate_budget: int = 500,
    ) -> RankedPredictions:
        """Backward-compatible alias for RETA candidate extraction."""
        return self.generate_candidates(
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
