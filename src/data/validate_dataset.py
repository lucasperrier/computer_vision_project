from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image, UnidentifiedImageError


PATH_CANDIDATES = ("path", "image_path", "relative_path")
REQUIRED_BASE_COLUMNS = {"label"}


@dataclass
class ValidationReport:
    total_rows: int
    duplicate_rows: int
    duplicate_paths: int
    unreadable_files: int
    corrupt_images: int
    missing_files: int
    class_counts: dict
    split_overlap_pairs: dict


def resolve_path_column(df: pd.DataFrame) -> str | None:
    for c in PATH_CANDIDATES:
        if c in df.columns:
            return c
    return None


def check_required_columns(df: pd.DataFrame, name: str) -> tuple[list[str], str | None]:
    errors: list[str] = []
    path_col = resolve_path_column(df)

    if path_col is None:
        errors.append(f"[{name}] missing path column. Expected one of {PATH_CANDIDATES}")
    missing = REQUIRED_BASE_COLUMNS - set(df.columns)
    if missing:
        errors.append(f"[{name}] missing required columns: {sorted(missing)}")

    return errors, path_col


def check_duplicates(df: pd.DataFrame, name: str, path_col: str | None) -> tuple[list[str], int, int]:
    errors = []
    dup_rows = int(df.duplicated().sum())
    dup_paths = int(df.duplicated(subset=[path_col]).sum()) if path_col else 0

    if dup_rows > 0:
        errors.append(f"[{name}] duplicated rows: {dup_rows}")
    if dup_paths > 0:
        errors.append(f"[{name}] duplicated {path_col} values: {dup_paths}")

    return errors, dup_rows, dup_paths


def check_files_and_images(paths: Iterable[Path]) -> tuple[list[str], int, int, int]:
    errors = []
    missing_files = 0
    unreadable = 0
    corrupt = 0

    for p in paths:
        if not p.exists():
            missing_files += 1
            continue
        if not p.is_file():
            unreadable += 1
            continue
        try:
            with Image.open(p) as img:
                img.verify()
        except UnidentifiedImageError:
            corrupt += 1
        except Exception:
            unreadable += 1

    if missing_files:
        errors.append(f"missing files: {missing_files}")
    if unreadable:
        errors.append(f"unreadable files: {unreadable}")
    if corrupt:
        errors.append(f"corrupt/unidentified images: {corrupt}")

    return errors, missing_files, unreadable, corrupt


def check_split_overlap(split_to_df: dict[str, pd.DataFrame], split_to_path_col: dict[str, str]) -> tuple[list[str], dict]:
    errors = []
    overlaps = {}
    names = list(split_to_df.keys())

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            col_a, col_b = split_to_path_col[a], split_to_path_col[b]
            set_a = set(split_to_df[a][col_a].astype(str))
            set_b = set(split_to_df[b][col_b].astype(str))
            inter = set_a.intersection(set_b)
            overlaps[f"{a}-{b}"] = len(inter)
            if inter:
                errors.append(f"split overlap detected between {a} and {b}: {len(inter)} shared files")

    return errors, overlaps


def class_balance(df: pd.DataFrame) -> dict:
    if "label" not in df.columns:
        return {}
    counts = df["label"].value_counts(dropna=False).to_dict()
    return {str(k): int(v) for k, v in counts.items()}


def load_csv(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{name} file not found: {path}")
    return pd.read_csv(path)


def validate(
    manifest_path: Path,
    train_path: Path | None = None,
    val_path: Path | None = None,
    test_path: Path | None = None,
    robustness_path: Path | None = None,
    raw_root: Path | None = None,
) -> tuple[ValidationReport, list[str]]:
    errors: list[str] = []

    manifest = load_csv(manifest_path, "manifest")
    req_errs, manifest_path_col = check_required_columns(manifest, "manifest")
    errors += req_errs

    dup_errs, dup_rows, dup_paths = check_duplicates(manifest, "manifest", manifest_path_col)
    errors += dup_errs

    missing = unreadable = corrupt = 0
    if manifest_path_col is not None:
        raw_root = raw_root or Path(".")
        file_paths = []
        for p in manifest[manifest_path_col].astype(str).tolist():
            pp = Path(p)
            file_paths.append(pp if pp.is_absolute() else (raw_root / pp))
        file_errs, missing, unreadable, corrupt = check_files_and_images(file_paths)
        errors += file_errs

    splits: dict[str, pd.DataFrame] = {}
    split_path_cols: dict[str, str] = {}

    for name, p in {
        "train": train_path,
        "val": val_path,
        "test": test_path,
        "robustness": robustness_path,
    }.items():
        if p is not None and p.exists():
            df = load_csv(p, name)
            s_errs, s_path_col = check_required_columns(df, name)
            errors += s_errs
            d_errs, _, _ = check_duplicates(df, name, s_path_col)
            errors += d_errs
            if s_path_col is not None:
                splits[name] = df
                split_path_cols[name] = s_path_col

    overlap_pairs = {}
    if len(splits) >= 2:
        overlap_errs, overlap_pairs = check_split_overlap(splits, split_path_cols)
        errors += overlap_errs

    report = ValidationReport(
        total_rows=len(manifest),
        duplicate_rows=dup_rows,
        duplicate_paths=dup_paths,
        unreadable_files=unreadable,
        corrupt_images=corrupt,
        missing_files=missing,
        class_counts=class_balance(manifest),
        split_overlap_pairs=overlap_pairs,
    )
    return report, errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate dataset integrity and splits.")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to manifest.csv")
    parser.add_argument("--train", type=Path, default=None, help="Path to train.csv")
    parser.add_argument("--val", type=Path, default=None, help="Path to val.csv")
    parser.add_argument("--test", type=Path, default=None, help="Path to test.csv")
    parser.add_argument("--robustness", type=Path, default=None, help="Path to robustness.csv")
    parser.add_argument("--raw-root", type=Path, default=Path("."), help="Base dir for relative paths")
    args = parser.parse_args()

    report, errors = validate(
        manifest_path=args.manifest,
        train_path=args.train,
        val_path=args.val,
        test_path=args.test,
        robustness_path=args.robustness,
        raw_root=args.raw_root,
    )

    print("=== Dataset Validation Report ===")
    print(f"total_rows: {report.total_rows}")
    print(f"duplicate_rows: {report.duplicate_rows}")
    print(f"duplicate_paths: {report.duplicate_paths}")
    print(f"missing_files: {report.missing_files}")
    print(f"unreadable_files: {report.unreadable_files}")
    print(f"corrupt_images: {report.corrupt_images}")
    print(f"class_counts: {report.class_counts}")
    if report.split_overlap_pairs:
        print(f"split_overlap_pairs: {report.split_overlap_pairs}")

    if errors:
        print("\nValidation FAILED:")
        for e in errors:
            print(f"- {e}")
        return 1

    print("\nValidation PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())