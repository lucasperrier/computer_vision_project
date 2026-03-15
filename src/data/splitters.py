from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Tuple

import pandas as pd


DEFAULT_SPLIT_RATIOS: Tuple[float, float, float] = (0.70, 0.15, 0.15)


def stable_hash_to_unit_interval(value: str, seed: int = 42) -> float:
    payload = f"{seed}::{value}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    int_value = int(digest[:16], 16)
    max_value = float(16**16 - 1)
    return int_value / max_value


def validate_split_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    if min(train_ratio, val_ratio, test_ratio) <=0:
        raise ValueError("All split ratios must be > 0.")
    total = train_ratio + val_ratio + test_ratio
    if abs(total-1.0) > 1e-8:
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.6f}.")
    

def assign_split(
        relative_path: str,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        seed: int = 42,
) -> str:
    _ = test_ratio
    score = stable_hash_to_unit_interval(relative_path, seed=seed)
    if score < train_ratio:
        return "train"
    if score < train_ratio + val_ratio:
        return "val"
    return "test"


def build_splits_from_manifest(
        manifest_df: pd.DataFrame,
        train_ratio: float = DEFAULT_SPLIT_RATIOS[0],
        val_ratio: float = DEFAULT_SPLIT_RATIOS[1],
        test_ratio: float = DEFAULT_SPLIT_RATIOS[2],
        seed: int = 42,
        stratify_by: str | None = "label",
) -> pd.DataFrame:
    required_columns = {"relative_path"}
    missing = required_columns - set(manifest_df.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
    
    validate_split_ratios(train_ratio, val_ratio, test_ratio)

    df = manifest_df.copy()

    if stratify_by is not None and stratify_by in df.columns:
        split_series = []
        for _, group_df in df.groupby(stratify_by, dropna=False, sort=False):
            group_scores = group_df["relative_path"].astype(str).map(
                lambda x: stable_hash_to_unit_interval(x, seed=seed)
            )
            group_split = pd.Series(index=group_df.index, dtype="object")
            group_split[group_scores < train_ratio] = "train"
            group_split[(group_scores >= train_ratio) & (group_scores < train_ratio + val_ratio)] = "val"
            group_split[group_scores >= train_ratio + val_ratio] = "test"
            split_series.append(group_split)
        df["split"] = pd.concat(split_series).sort_index()
    else:
        df["split"] = df["relative_path"].astype(str).map(
            lambda x: assign_split(
            relative_path=x,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
    )
        
    return df.sort_values(["split", "relative_path"]).reset_index(drop=True)


def write_split_files(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name in ("train", "val", "test"):
        split_df = df[df["split"] == split_name].copy()
        split_df.to_csv(output_dir / f"{split_name}.csv", index=False)

    robustness_df = df.copy()
    robustness_df.to_csv(output_dir / "robustness.csv", index=False)


def load_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Manifest file not found: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError("Manifest format not supported. Use .csv or .parquet.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate deterministic train/val/test split files.")
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("data/processed/manifests/manifest.csv"),
        help="Path to the input manifest (.csv or .parquet).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/splits"),
        help="Output directory for train.csv, val.csv, test.csv, robustness.csv.",
    )
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_SPLIT_RATIOS[0], help="Train ratio.")
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_SPLIT_RATIOS[1], help="Validation ratio.")
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_SPLIT_RATIOS[2], help="Test ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Seed used in deterministic hash splitting.")
    parser.add_argument(
        "--stratify-by",
        type=str,
        default="label",
        help="Optional column to stratify by (e.g. label). Use empty string to disable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stratify_by = args.stratify_by if args.stratify_by else None

    manifest_df = load_manifest(args.manifest_path)
    split_df = build_splits_from_manifest(
        manifest_df=manifest_df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        stratify_by=stratify_by,
    )
    write_split_files(split_df, args.output_dir)

    counts = split_df["split"].value_counts().to_dict()
    print(f"Loaded manifest rows: {len(manifest_df)}")
    print(f"Split counts: {counts}")
    print(f"Wrote split files to: {args.output_dir}")


if __name__ == "__main__":
    main()