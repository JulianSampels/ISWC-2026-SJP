"""Canonical dataset support for the harmonized SJP/RETA interface."""

from __future__ import annotations

import json
import os
import shutil
import sys
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch


SPLITS = ("train", "valid", "test")

_SPLIT_ALIASES = {
    "train": ("train",),
    "valid": ("valid", "val", "validation", "dev"),
    "test": ("test",),
}

_SPLIT_EXTENSIONS = (".txt", ".tsv", ".csv")


@dataclass(frozen=True)
class StandardizedDataset:
    """Canonical triple-file dataset with train/valid/test splits."""

    root: Path
    train_file: Path
    valid_file: Path
    test_file: Path

    def split_file(self, split: str) -> Path:
        if split == "train":
            return self.train_file
        if split == "valid":
            return self.valid_file
        if split == "test":
            return self.test_file
        raise ValueError(f"Unsupported split '{split}'.")


def _resolve_split_file(dataset_root: Path, split: str) -> Path:
    aliases = _SPLIT_ALIASES[split]
    for alias in aliases:
        for extension in _SPLIT_EXTENSIONS:
            candidate = dataset_root / f"{alias}{extension}"
            if candidate.is_file():
                return candidate

    folder = dataset_root / split
    if folder.is_dir():
        for alias in aliases:
            for extension in _SPLIT_EXTENSIONS:
                candidate = folder / f"{alias}{extension}"
                if candidate.is_file():
                    return candidate

    searched_names: List[str] = []
    for alias in aliases:
        for extension in _SPLIT_EXTENSIONS:
            searched_names.append(f"{alias}{extension}")
            searched_names.append(f"{split}/{alias}{extension}")

    raise FileNotFoundError(
        f"Could not find split '{split}' in {dataset_root}. "
        f"Looked for: {', '.join(searched_names)}"
    )


def resolve_standardized_dataset(dataset_dir: str | Path) -> StandardizedDataset:
    """Resolve a canonical internet-style dataset directory."""
    root = Path(dataset_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    train_file = _resolve_split_file(root, "train")
    valid_file = _resolve_split_file(root, "valid")
    test_file = _resolve_split_file(root, "test")

    return StandardizedDataset(
        root=root,
        train_file=train_file,
        valid_file=valid_file,
        test_file=test_file,
    )


def _guess_delimiter(line: str) -> Optional[str]:
    if "\t" in line:
        return "\t"
    if line.count(",") >= 2:
        return ","
    return None


def _parse_triples_line(
    line: str,
    triple_order: str = "hrt",
    delimiter: Optional[str] = None,
) -> Tuple[str, str, str]:
    stripped = line.strip()
    if not stripped:
        raise ValueError("Encountered empty triple line.")

    use_delimiter = delimiter if delimiter is not None else _guess_delimiter(stripped)
    if use_delimiter is None:
        parts = stripped.split()
    else:
        parts = [chunk.strip() for chunk in stripped.split(use_delimiter)]

    if len(parts) < 3:
        raise ValueError(f"Could not parse triple from line: {line!r}")
    if len(parts) > 3:
        head = parts[0]
        tail = parts[-1]
        middle = use_delimiter.join(parts[1:-1]) if use_delimiter is not None else " ".join(parts[1:-1])
        parts = [head, middle, tail]

    if triple_order == "hrt":
        head, relation, tail = parts[0], parts[1], parts[2]
    elif triple_order == "htr":
        head, tail, relation = parts[0], parts[1], parts[2]
    else:
        raise ValueError("triple_order must be either 'hrt' or 'htr'.")

    return str(head), str(relation), str(tail)


def load_labeled_triples(
    triple_file: str | Path,
    triple_order: str = "hrt",
    delimiter: Optional[str] = None,
    has_header: bool = False,
) -> List[Tuple[str, str, str]]:
    """Load (head, relation, tail) triples from a text-like file."""
    path = Path(triple_file).resolve()
    triples: List[Tuple[str, str, str]] = []

    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            if index == 0 and has_header:
                continue
            triples.append(_parse_triples_line(line, triple_order=triple_order, delimiter=delimiter))

    if not triples:
        raise ValueError(f"No triples found in {path}.")

    return triples


def _write_split_file(path: Path, triples: Iterable[Tuple[str, str, str]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for head, relation, tail in triples:
            handle.write(f"{head}\t{relation}\t{tail}\n")
            count += 1
    return count


def _ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {path}. "
                "Use overwrite=True to replace it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_standardized_dataset(
    triples_by_split: Dict[str, Sequence[Tuple[str, str, str]]],
    output_dir: str | Path,
    metadata: Optional[Dict[str, Any]] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Write a canonical harmonized dataset (train.txt, valid.txt, test.txt)."""
    output_root = Path(output_dir).resolve()
    _ensure_output_dir(output_root, overwrite=overwrite)

    stats: Dict[str, Any] = {
        "dataset_dir": str(output_root),
        "files": {},
        "counts": {},
    }

    for split in SPLITS:
        if split not in triples_by_split:
            raise KeyError(f"Missing split '{split}' in triples_by_split.")
        split_path = output_root / f"{split}.txt"
        count = _write_split_file(split_path, triples_by_split[split])
        stats["files"][split] = str(split_path)
        stats["counts"][split] = count

    if metadata is not None:
        metadata_path = output_root / "metadata.json"
        payload = dict(metadata)
        payload.setdefault("counts", stats["counts"])
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        stats["metadata_file"] = str(metadata_path)

    return stats


def canonicalize_downloaded_dataset(
    source_dir: str | Path,
    output_dir: str | Path,
    triple_order: str = "hrt",
    delimiter: Optional[str] = None,
    has_header: bool = False,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Normalize a downloaded dataset into canonical harmonized split files."""
    source = resolve_standardized_dataset(source_dir)

    triples_by_split = {
        "train": load_labeled_triples(
            source.train_file,
            triple_order=triple_order,
            delimiter=delimiter,
            has_header=has_header,
        ),
        "valid": load_labeled_triples(
            source.valid_file,
            triple_order=triple_order,
            delimiter=delimiter,
            has_header=has_header,
        ),
        "test": load_labeled_triples(
            source.test_file,
            triple_order=triple_order,
            delimiter=delimiter,
            has_header=has_header,
        ),
    }

    metadata = {
        "source_type": "downloaded",
        "source_dir": str(source.root),
        "triple_order": triple_order,
        "delimiter": delimiter,
        "has_header": bool(has_header),
    }
    return write_standardized_dataset(
        triples_by_split=triples_by_split,
        output_dir=output_dir,
        metadata=metadata,
        overwrite=overwrite,
    )


def _triples_to_label_rows(
    triples: torch.Tensor,
    id_to_entity: Dict[int, str],
    id_to_relation: Dict[int, str],
) -> List[Tuple[str, str, str]]:
    rows: List[Tuple[str, str, str]] = []
    for head_id, relation_id, tail_id in triples.tolist():
        rows.append((
            str(id_to_entity[int(head_id)]),
            str(id_to_relation[int(relation_id)]),
            str(id_to_entity[int(tail_id)]),
        ))
    return rows


def _import_sjp_modules() -> Tuple[Any, Any]:
    code_dir = Path(__file__).resolve().parents[2]
    sjp_code_dir = code_dir / "SJP_code"
    if str(sjp_code_dir) not in sys.path:
        sys.path.insert(0, str(sjp_code_dir))

    from PathE.pathe.kgloader import KgLoader  # type: ignore
    from PathE.pathe.pathdataset import PathDataset  # type: ignore

    return KgLoader, PathDataset


@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _backup_if_exists(path: Path) -> Optional[Path]:
    if not path.exists():
        return None
    backup = path.parent / f".harmonized_backup_{path.name}_{uuid.uuid4().hex}"
    path.rename(backup)
    return backup


def _restore_from_backup(path: Path, backup: Optional[Path]) -> None:
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    if backup is not None and backup.exists():
        backup.rename(path)


def _write_test_dataset_tsv(
    standardized_dataset: StandardizedDataset,
    staging_dir: Path,
    triple_order: str,
    delimiter: Optional[str],
    has_header: bool,
) -> Dict[str, int]:
    staging_dir.mkdir(parents=True, exist_ok=True)

    split_map = {
        "train": standardized_dataset.train_file,
        "valid": standardized_dataset.valid_file,
        "test": standardized_dataset.test_file,
    }
    target_names = {
        "train": "train.tsv",
        "valid": "val.tsv",
        "test": "test.tsv",
    }

    counts: Dict[str, int] = {}
    for split, source_file in split_map.items():
        triples = load_labeled_triples(
            source_file,
            triple_order=triple_order,
            delimiter=delimiter,
            has_header=has_header,
        )

        # KgLoader('test-dataset') expects files with columns: head, tail, relation.
        output_file = staging_dir / target_names[split]
        with output_file.open("w", encoding="utf-8") as handle:
            for head, relation, tail in triples:
                handle.write(f"{head}\t{tail}\t{relation}\n")
        counts[split] = len(triples)

    return counts


def export_standardized_dataset_to_sjp(
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
    """
    Convert a canonical dataset into SJP PathE path dataset format.

    The conversion uses SJP submodule utilities (`KgLoader`, `PathDataset`) to
    preserve native path-generation behavior.
    """
    inverse_mode = inverse_mode.strip().lower()
    if inverse_mode != "manual":
        raise ValueError(
            "SJP translation requires inverse_mode='manual' so relation2inverseRelation "
            "files and inverse ids are generated in the expected PathE format."
        )

    standardized = resolve_standardized_dataset(standardized_dataset_dir)
    output_path = Path(output_dir).resolve()
    _ensure_output_dir(output_path, overwrite=overwrite)

    code_dir = Path(__file__).resolve().parents[2]
    sjp_code_dir = code_dir / "SJP_code"
    path_e_dir = sjp_code_dir / "PathE"

    staging_input = sjp_code_dir / "data" / "test-dataset"
    generated_dataset_dir = path_e_dir / "data" / "path_datasets" / "test-dataset"

    backup_staging = _backup_if_exists(staging_input)
    backup_generated = _backup_if_exists(generated_dataset_dir)

    generated_counts: Dict[str, int] = {}
    generated_root: Optional[Path] = None

    try:
        generated_counts = _write_test_dataset_tsv(
            standardized_dataset=standardized,
            staging_dir=staging_input,
            triple_order=triple_order,
            delimiter=delimiter,
            has_header=has_header,
        )

        KgLoader, PathDataset = _import_sjp_modules()

        automatically_add_inverse = False
        manually_add_inverse = True

        with _working_directory(path_e_dir):
            dataset = KgLoader(
                "test-dataset",
                automatically_add_inverse=automatically_add_inverse,
                manually_add_inverse=manually_add_inverse,
            )
            try:
                path_handler = PathDataset(
                    dataset=dataset,
                    num_paths_per_entity=int(num_paths_per_entity),
                    num_steps=int(num_steps),
                    parallel=bool(parallel),
                )
            except ValueError as exc:
                # PathE parallel random walks can fail on tiny graphs when the
                # internally computed chunk size becomes zero.
                if bool(parallel) and "range() arg 3 must not be zero" in str(exc):
                    path_handler = PathDataset(
                        dataset=dataset,
                        num_paths_per_entity=int(num_paths_per_entity),
                        num_steps=int(num_steps),
                        parallel=False,
                    )
                else:
                    raise
            generated_root = Path(path_handler.dataset_dir).resolve()

        if generated_root is None or not generated_root.exists():
            raise RuntimeError("SJP dataset generation did not produce an output directory.")

        shutil.copytree(generated_root, output_path, dirs_exist_ok=True)

        summary = {
            "standardized_dataset_dir": str(standardized.root),
            "sjp_output_dir": str(output_path),
            "num_paths_per_entity": int(num_paths_per_entity),
            "num_steps": int(num_steps),
            "parallel": bool(parallel),
            "inverse_mode": inverse_mode,
            "split_counts": generated_counts,
        }
        metadata_path = output_path / "harmonized_standard_source.json"
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

        summary["metadata_file"] = str(metadata_path)
        return summary

    finally:
        _restore_from_backup(staging_input, backup_staging)
        _restore_from_backup(generated_dataset_dir, backup_generated)


def generate_standardized_dataset_from_kgloader(
    dataset_name: str,
    output_dir: str | Path,
    inverse_mode: str = "none",
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Generate canonical split files from a built-in KgLoader dataset."""
    inverse_mode = inverse_mode.strip().lower()
    if inverse_mode not in {"manual", "automatic", "none"}:
        raise ValueError("inverse_mode must be one of: manual, automatic, none")

    KgLoader, _ = _import_sjp_modules()

    automatically_add_inverse = inverse_mode == "automatic"
    manually_add_inverse = inverse_mode == "manual"

    loader = KgLoader(
        dataset_name,
        automatically_add_inverse=automatically_add_inverse,
        manually_add_inverse=manually_add_inverse,
    )

    train_triples = loader.train_no_inv if hasattr(loader, "train_no_inv") else loader.train_triples
    valid_triples = loader.val_no_inv if hasattr(loader, "val_no_inv") else loader.val_triples
    test_triples = loader.test_no_inv if hasattr(loader, "test_no_inv") else loader.test_triples

    id_to_entity = {
        int(key): str(value)
        for key, value in loader.triple_factory.training.entity_id_to_label.items()
    }
    id_to_relation = {
        int(key): str(value)
        for key, value in loader.triple_factory.training.relation_id_to_label.items()
    }

    triples_by_split = {
        "train": _triples_to_label_rows(train_triples, id_to_entity, id_to_relation),
        "valid": _triples_to_label_rows(valid_triples, id_to_entity, id_to_relation),
        "test": _triples_to_label_rows(test_triples, id_to_entity, id_to_relation),
    }

    metadata = {
        "source_type": "kgloader",
        "dataset_name": dataset_name,
        "inverse_mode": inverse_mode,
    }

    summary = write_standardized_dataset(
        triples_by_split=triples_by_split,
        output_dir=output_dir,
        metadata=metadata,
        overwrite=overwrite,
    )

    summary["num_entities"] = len(id_to_entity)
    summary["num_relations"] = len(id_to_relation)
    return summary


def load_standardized_dataset_triples(
    standardized_dataset_dir: str | Path,
    triple_order: str = "hrt",
    delimiter: Optional[str] = None,
    has_header: bool = False,
) -> Dict[str, List[Tuple[str, str, str]]]:
    """Load all canonical splits as labeled triples."""
    standardized = resolve_standardized_dataset(standardized_dataset_dir)

    return {
        "train": load_labeled_triples(
            standardized.train_file,
            triple_order=triple_order,
            delimiter=delimiter,
            has_header=has_header,
        ),
        "valid": load_labeled_triples(
            standardized.valid_file,
            triple_order=triple_order,
            delimiter=delimiter,
            has_header=has_header,
        ),
        "test": load_labeled_triples(
            standardized.test_file,
            triple_order=triple_order,
            delimiter=delimiter,
            has_header=has_header,
        ),
    }
