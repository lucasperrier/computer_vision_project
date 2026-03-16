import sys
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from src.data.validate_dataset import validate


def _make_image(path: Path, size=(8, 8), color=(255, 255, 255)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color)
    img.save(path)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_validate_passes_with_valid_manifest_and_splits(tmp_path: Path):
    raw_root = tmp_path / "data" / "raw"
    img1_rel = Path("ccic/Positive/img1.jpg")
    img2_rel = Path("ccic/Negative/img2.jpg")
    _make_image(raw_root / img1_rel)
    _make_image(raw_root / img2_rel)

    manifest_path = tmp_path / "manifest.csv"
    _write_csv(
        manifest_path,
        [
            {"relative_path": str(img1_rel), "label": "crack"},
            {"relative_path": str(img2_rel), "label": "non_crack"},
        ],
    )

    train_path = tmp_path / "train.csv"
    val_path = tmp_path / "val.csv"
    test_path = tmp_path / "test.csv"
    robustness_path = tmp_path / "robustness.csv"

    _write_csv(train_path, [{"relative_path": str(img1_rel), "label": "crack"}])
    _write_csv(val_path, [{"relative_path": str(img2_rel), "label": "non_crack"}])
    pd.DataFrame(columns=["relative_path", "label"]).to_csv(test_path, index=False)
    _write_csv(
        robustness_path,
        [
            {"relative_path": str(img1_rel), "label": "crack"},
            {"relative_path": str(img2_rel), "label": "non_crack"},
        ],
    )

    report, errors = validate(
        manifest_path=manifest_path,
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        raw_root=raw_root,
    )

    assert errors == []
    assert report.total_rows == 2
    assert report.missing_files == 0
    assert report.corrupt_images == 0
    assert report.unreadable_files == 0


def test_validate_fails_on_missing_files(tmp_path: Path):
    raw_root = tmp_path / "data" / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)

    manifest_path = tmp_path / "manifest.csv"
    _write_csv(
        manifest_path,
        [{"relative_path": "does/not/exist.jpg", "label": "crack"}],
    )

    report, errors = validate(manifest_path=manifest_path, raw_root=raw_root)

    assert report.missing_files == 1
    assert any("missing files: 1" in e for e in errors)


def test_validate_detects_split_overlap(tmp_path: Path):
    raw_root = tmp_path / "data" / "raw"
    rel = Path("ccic/Positive/shared.jpg")
    _make_image(raw_root / rel)

    manifest_path = tmp_path / "manifest.csv"
    _write_csv(manifest_path, [{"relative_path": str(rel), "label": "crack"}])

    train_path = tmp_path / "train.csv"
    val_path = tmp_path / "val.csv"
    _write_csv(train_path, [{"relative_path": str(rel), "label": "crack"}])
    _write_csv(val_path, [{"relative_path": str(rel), "label": "crack"}])

    report, errors = validate(
        manifest_path=manifest_path,
        train_path=train_path,
        val_path=val_path,
        raw_root=raw_root,
    )

    assert report.split_overlap_pairs.get("train-val") == 1
    assert any("split overlap detected between train and val" in e for e in errors)


def test_validate_detects_corrupt_image(tmp_path: Path):
    raw_root = tmp_path / "data" / "raw"
    rel = Path("ccic/Positive/corrupt.jpg")
    bad_path = raw_root / rel
    bad_path.parent.mkdir(parents=True, exist_ok=True)
    bad_path.write_bytes(b"not-an-image")

    manifest_path = tmp_path / "manifest.csv"
    _write_csv(manifest_path, [{"relative_path": str(rel), "label": "crack"}])

    report, errors = validate(manifest_path=manifest_path, raw_root=raw_root)

    assert report.corrupt_images == 1
    assert any("corrupt/unidentified images: 1" in e for e in errors)