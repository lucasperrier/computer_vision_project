import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2


class CrackDataset(Dataset):
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = [int(x) for x in labels]  # enforce 0/1 ints
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = int(self.labels[idx])

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = np.array(image)  # albumentations expects numpy
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Ensure dtype works with CrossEntropyLoss
        label = torch.tensor(label, dtype=torch.long)
        return image, label


class CrackDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: float = 0.1,
        test_split: float = 0.1,
        robustness_split: float = 0.1,
        sdnet_path: str = "data/raw/sdnet2018",
        ccic_path: str = "data/raw/ccic",
        seed: int = 42,
        verbose: bool = False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = float(val_split)
        self.test_split = float(test_split)
        self.robustness_split = float(robustness_split)
        self.seed = int(seed)
        self.verbose = bool(verbose)

        # Paths
        self.sdnet_path = sdnet_path
        self.ccic_path = ccic_path

        # Transforms
        self.train_transform = A.Compose(
            [
                A.Resize(224, 224),
                A.Rotate(limit=30, p=0.5),
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5
                ),
                A.GaussNoise(std_range=(0.02, 0.10), p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )

        self.val_test_transform = A.Compose(
            [
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.robust_dataset = None

    def prepare_data(self):
        # Optionally validate disk structure here (no downloading implemented)
        pass

    def setup(self, stage: Optional[str] = None):
        # Load all data
        sdnet_paths, sdnet_labels = self._get_paths_labels(self.sdnet_path, label_map=None)
        ccic_paths, ccic_labels = self._get_paths_labels(
            self.ccic_path, label_map={"Positive": 1, "Negative": 0}
        )

        all_paths = sdnet_paths + ccic_paths
        all_labels = sdnet_labels + ccic_labels

        if self.verbose:
            def _count(lbls):
                return {
                    0: int(sum(int(x) == 0 for x in lbls)),
                    1: int(sum(int(x) == 1 for x in lbls)),
                }

            print("[CrackDataModule] SDNET count:", _count(sdnet_labels), "n=", len(sdnet_labels))
            print("[CrackDataModule] CCIC  count:", _count(ccic_labels), "n=", len(ccic_labels))
            print("[CrackDataModule] TOTAL count:", _count(all_labels), "n=", len(all_labels))

        if len(all_paths) == 0:
            raise RuntimeError(
                "No images found. Check that data exists under "
                f"{self.sdnet_path} and/or {self.ccic_path}."
            )

        # Ensure labels are 0/1 ints
        all_labels = [int(x) for x in all_labels]
        unique = sorted(set(all_labels))
        if any(l not in (0, 1) for l in unique):
            raise ValueError(f"Labels must be 0/1. Found labels: {unique}")

        if len(unique) < 2:
            raise ValueError(
                f"Need both classes present for stratified split. Found only: {unique}"
            )

        # Validate split ratios
        total_holdout = self.val_split + self.test_split
        if not (0.0 < self.robustness_split < 1.0):
            raise ValueError("robustness_split must be in (0, 1).")
        if not (0.0 < total_holdout < 1.0):
            raise ValueError("val_split + test_split must be in (0, 1).")
        if total_holdout >= (1.0 - self.robustness_split):
            raise ValueError(
                "val_split + test_split is too large given robustness_split. "
                "Must leave some samples for train."
            )

        # Split: robustness hold-out first
        paths_train_val_test, paths_robust, labels_train_val_test, labels_robust = train_test_split(
            all_paths,
            all_labels,
            test_size=self.robustness_split,
            random_state=self.seed,
            stratify=all_labels,
        )

        # Then train vs (val+test)
        paths_train, paths_temp, labels_train, labels_temp = train_test_split(
            paths_train_val_test,
            labels_train_val_test,
            test_size=total_holdout,
            random_state=self.seed,
            stratify=labels_train_val_test,
        )

        # Then val vs test
        # fraction of temp that should be test:
        test_frac_of_temp = self.test_split / total_holdout
        paths_val, paths_test, labels_val, labels_test = train_test_split(
            paths_temp,
            labels_temp,
            test_size=test_frac_of_temp,
            random_state=self.seed,
            stratify=labels_temp,
        )

        self.train_dataset = CrackDataset(paths_train, labels_train, transform=self.train_transform)
        self.val_dataset = CrackDataset(paths_val, labels_val, transform=self.val_test_transform)
        self.test_dataset = CrackDataset(paths_test, labels_test, transform=self.val_test_transform)
        self.robust_dataset = CrackDataset(paths_robust, labels_robust, transform=self.val_test_transform)

    def _get_paths_labels(
        self, dataset_path: str, label_map: Optional[Dict[str, int]] = None
    ) -> Tuple[List[str], List[int]]:
        paths: List[str] = []
        labels: List[int] = []

        if not os.path.exists(dataset_path):
            return paths, labels

        # normalize label_map to be case-insensitive
        label_map_ci: Optional[Dict[str, int]] = None
        if label_map is not None:
            label_map_ci = {str(k).strip().lower(): int(v) for k, v in label_map.items()}

        for root, _, files in os.walk(dataset_path):
            for file in files:
                if not file.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue

                p = os.path.join(root, file)
                root_l = root.lower()

                if label_map_ci is not None:
                    folder = os.path.basename(root).strip().lower()
                    if folder not in label_map_ci:
                        # also allow label to be in parent folder name
                        parent = os.path.basename(os.path.dirname(root)).strip().lower()
                        if parent in label_map_ci:
                            label = int(label_map_ci[parent])
                        else:
                            continue
                    else:
                        label = int(label_map_ci[folder])
                else:
                    # SDNET-style inference: try to be robust
                    # Common patterns: crack / cracks / noncrack / no_crack / negative / positive
                    if "noncrack" in root_l or "no_crack" in root_l or "negative" in root_l:
                        label = 0
                    elif "crack" in root_l or "positive" in root_l:
                        label = 1
                    else:
                        # Unknown folder naming => skip (or raise). Skipping avoids poisoning labels.
                        continue

                paths.append(p)
                labels.append(int(label))

        return paths, labels
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def robustness_dataloader(self):
        return DataLoader(
            self.robust_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )