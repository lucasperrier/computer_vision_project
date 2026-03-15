import pandas as pd
import pytest

from src.data.splitters import build_splits_from_manifest, validate_split_ratios, write_split_files


def test_build_splits_is_deterministic_for_same_seed():
    df = pd.DataFrame(
        {
            "relative_path": [f"ds/class_a/img_{i}.jpg" for i in range(100)],
            "label": ["crack" if i % 2 == 0 else "non_crack" for i in range(100)],
        }
    )
    out1 = build_splits_from_manifest(df, seed=42, stratify_by="label")
    out2 = build_splits_from_manifest(df, seed=42, stratify_by="label")
    assert out1["split"].tolist() == out2["split"].tolist()


def test_build_splits_changes_with_seed():
    df = pd.DataFrame(
        {
            "relative_path": [f"ds/class_a/img_{i}.jpg" for i in range(100)],
            "label": ["crack" if i % 2 == 0 else "non_crack" for i in range(100)],
        }
    )
    out1 = build_splits_from_manifest(df, seed=42, stratify_by="label")
    out2 = build_splits_from_manifest(df, seed=43, stratify_by="label")
    assert out1["split"].tolist() != out2["split"].tolist()


def test_validate_split_ratios_rejects_invalid_values():
    with pytest.raises(ValueError):
        validate_split_ratios(0.7, 0.2, 0.2)
    with pytest.raises(ValueError):
        validate_split_ratios(0.7, 0.3, 0.0)


def test_build_splits_requires_relative_path():
    df = pd.DataFrame({"label": ["crack", "non_crack"]})
    with pytest.raises(ValueError):
        build_splits_from_manifest(df)


def test_write_split_files_creates_expected_files(tmp_path):
    df = pd.DataFrame(
        {
            "relative_path": ["a.jpg", "b.jpg", "c.jpg"],
            "label": ["crack", "non_crack", "crack"],
            "split": ["train", "val", "test"],
        }
    )
    write_split_files(df, tmp_path)
    assert (tmp_path / "train.csv").exists()
    assert (tmp_path / "val.csv").exists()
    assert (tmp_path / "test.csv").exists()
    assert (tmp_path / "robustness.csv").exists()