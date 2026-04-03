"""
download_qa_datasets.py — Download WebQSP and CWQ into iswc_data/standard/.

Downloads both datasets from HuggingFace Hub (rmanluo/RoG-webqsp and
rmanluo/RoG-cwq) and saves them as JSON files under:

    iswc_data/standard/webqsp/{train,val,test}.json
    iswc_data/standard/cwq/{train,val,test}.json

Usage
-----
    python -m iswc.download_qa_datasets
    python -m iswc.download_qa_datasets --output-dir /custom/path
    python -m iswc.download_qa_datasets --dataset webqsp
"""
import argparse
import json
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_DEFAULT_OUT = _THIS_DIR.parent / "iswc_data" / "standard"

_DATASETS = {
    "webqsp": "rmanluo/RoG-webqsp",
    "cwq":    "rmanluo/RoG-cwq",
}

_SPLITS = {
    "train": "train",
    "val":   "validation",
    "test":  "test",
}


def download_dataset(name: str, hub_name: str, out_root: Path) -> None:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        sys.exit("Missing dependency: pip install datasets")

    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, hf_split in _SPLITS.items():
        out_file = out_dir / f"{split_name}.json"
        if out_file.exists():
            print(f"  [{name}/{split_name}] already exists, skipping.")
            continue

        print(f"  [{name}/{split_name}] downloading from {hub_name} …", end=" ", flush=True)
        ds = load_dataset(hub_name, split=hf_split)
        rows = [dict(row) for row in ds]
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"done. ({len(rows):,} samples → {out_file})")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Download WebQSP and CWQ QA datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Root directory to write datasets into. "
             f"Default: {_DEFAULT_OUT}",
    )
    parser.add_argument(
        "--dataset", choices=["webqsp", "cwq", "both"], default="webqsp",
        help="Which dataset(s) to download.",
    )
    args = parser.parse_args(argv)

    out_root = Path(args.output_dir) if args.output_dir else _DEFAULT_OUT
    print(f"Output root: {out_root}\n")

    to_download = (
        list(_DATASETS.items())
        if args.dataset == "both"
        else [(args.dataset, _DATASETS[args.dataset])]
    )

    for name, hub_name in to_download:
        print(f"Dataset: {name}  ({hub_name})")
        download_dataset(name, hub_name, out_root)
        print()

    print("All done.")


if __name__ == "__main__":
    main()
