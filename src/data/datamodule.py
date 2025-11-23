# Placeholder for CrackDataModule implementation
# Handles loading, splitting, and augmenting SDNET2018 and CCIC datasets
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split

class CrackDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels  # list of 0/1
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = np.array(image)  # albumentations expects numpy
            augmented = self.transform(image=image)
            image = augmented['image']
        return image, label

class CrackDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, num_workers=4, val_split=0.1, test_split=0.1, robustness_split=0.1):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.robustness_split = robustness_split

        # Paths
        self.sdnet_path = 'data/raw/sdnet2018'
        self.ccic_path = 'data/raw/ccic'

        # Transforms
        self.train_transform = A.Compose([
            A.Resize(224, 224),
            A.Rotate(limit=30, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.GaussNoise(var_limit=(10, 50), p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

        self.val_test_transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    def prepare_data(self):
        # Download or prepare data here if needed (e.g., extract archives)
        pass

    def setup(self, stage=None):
        # Load all data
        sdnet_paths, sdnet_labels = self._get_paths_labels(self.sdnet_path)
        ccic_paths, ccic_labels = self._get_paths_labels(self.ccic_path, label_map={'Positive': 1, 'Negative': 0})
        all_paths = sdnet_paths + ccic_paths
        all_labels = sdnet_labels + ccic_labels

        # Map labels to 0/1 if needed
        all_labels = [1 if 'crack' in str(l).lower() else 0 for l in all_labels]

        # Split: robustness hold-out first
        paths_train_val_test, paths_robust, labels_train_val_test, labels_robust = train_test_split(
            all_paths, all_labels, test_size=self.robustness_split, random_state=42, stratify=all_labels
        )

        # Then train/val/test
        paths_train, paths_temp, labels_train, labels_temp = train_test_split(
            paths_train_val_test, labels_train_val_test, test_size=(self.val_split + self.test_split), random_state=42, stratify=labels_train_val_test
        )
        paths_val, paths_test, labels_val, labels_test = train_test_split(
            paths_temp, labels_temp, test_size=(self.test_split / (self.val_split + self.test_split)), random_state=42, stratify=labels_temp
        )

        # Create datasets
        self.train_dataset = CrackDataset(paths_train, labels_train, transform=self.train_transform)
        self.val_dataset = CrackDataset(paths_val, labels_val, transform=self.val_test_transform)
        self.test_dataset = CrackDataset(paths_test, labels_test, transform=self.val_test_transform)
        self.robust_dataset = CrackDataset(paths_robust, labels_robust, transform=self.val_test_transform)

    def _get_paths_labels(self, dataset_path, label_map=None):
        paths = []
        labels = []
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    paths.append(os.path.join(root, file))
                    if label_map:
                        folder = os.path.basename(root)
                        label = label_map.get(folder, 0)  # default to 0
                    else:
                        label = 1 if 'crack' in root.lower() else 0
                    labels.append(label)
        return paths, labels

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def robustness_dataloader(self):
        return DataLoader(self.robust_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)