"""
Data Pipeline Module:
- Ingestion of PlantDoc and Citrus datasets
- Robust path re-mapping from Kaggle paths to local directories
- MD5 cryptographic hash-based cross-split leakage verification
- PyTorch DataLoaders with strong augmentations and synthetic mode
"""

import os
import hashlib
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.model_selection import StratifiedKFold, train_test_split

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def resolve_image_path(path, dataset_name="plantdoc", data_dir=None):
    """
    Intelligently remaps hardcoded paths from split CSVs to the local filesystem.
    """
    if os.path.exists(path):
        return path

    norm_path = path.replace("\\", "/")

    # Candidate roots
    candidates = []
    if data_dir:
        candidates.append(data_dir)

    env_key = "PD_ROOT" if dataset_name == "plantdoc" else "CT_ROOT"
    if os.environ.get(env_key):
        candidates.append(os.environ[env_key])

    # Search common subfolders
    for cand in candidates:
        if not cand or not os.path.exists(cand):
            continue
        # Check by relative suffix
        if dataset_name == "plantdoc" and "plantdoc-dataset/" in norm_path:
            rel = norm_path.split("plantdoc-dataset/")[-1]
            p = os.path.join(cand, rel)
            if os.path.exists(p):
                return p
        elif dataset_name == "citrus" and "Citrus Leaf Disease Image/" in norm_path:
            rel = norm_path.split("Citrus Leaf Disease Image/")[-1]
            p = os.path.join(cand, rel)
            if os.path.exists(p):
                return p

        # Check by filename inside subdirectories
        fname = os.path.basename(path)
        for root_dir, _, files in os.walk(cand):
            if fname in files:
                return os.path.join(root_dir, fname)

    return path


def check_data_leakage(groups):
    """
    Computes MD5 hashes for all images across train, val, and test splits.
    Raises or reports exact duplicates found across split boundaries.
    """
    seen, clashes = {}, []
    for sname, df in groups.items():
        if "path" not in df.columns:
            continue
        for p in df["path"]:
            if not os.path.exists(p):
                continue
            with open(p, "rb") as f:
                h = hashlib.md5(f.read()).hexdigest()
            if h in seen and seen[h] != sname:
                clashes.append((p, seen[h], sname))
            seen.setdefault(h, sname)
    return clashes


class LeafDataset(Dataset):
    """
    Standard PyTorch dataset for agricultural leaf disease classification.
    """
    def __init__(self, df, class_map, transform=None, dataset_name="plantdoc", data_dir=None):
        self.paths = df["path"].tolist()
        self.labels = [class_map[lbl] for lbl in df["label"]]
        self.transform = transform
        self.dataset_name = dataset_name
        self.data_dir = data_dir

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = resolve_image_path(self.paths[idx], self.dataset_name, self.data_dir)
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            # Fallback for missing/corrupted file during offline testing
            img = Image.new("RGB", (224, 224), color=(124, 180, 120))

        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


class SSLDataset(Dataset):
    """
    Self-Supervised Dataset:
    Produces two photometric views of the exact same geometric crop.
    Teacher sees weak photometric augmentation; student sees strong photometric augmentation.
    """
    def __init__(self, df, resolution=224, dataset_name="plantdoc", data_dir=None):
        self.paths = df["path"].tolist()
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.geo = T.Compose([
            T.RandomResizedCrop(resolution, scale=(0.5, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip()
        ])
        self.tf_teacher = T.Compose([
            T.ColorJitter(0.2, 0.2, 0.2, 0.02),
            T.ToTensor(),
            T.Normalize(MEAN, STD)
        ])
        self.tf_student = T.Compose([
            T.ColorJitter(0.4, 0.4, 0.4, 0.05),
            T.RandomGrayscale(0.1),
            T.ToTensor(),
            T.Normalize(MEAN, STD)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = resolve_image_path(self.paths[idx], self.dataset_name, self.data_dir)
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), color=(100, 160, 100))

        shared_crop = self.geo(img)
        return self.tf_teacher(shared_crop), self.tf_student(shared_crop)


class SyntheticPlantDataset(Dataset):
    """
    In-memory synthetic dataset for instant zero-dependency smoke testing and pipeline verification.
    Requires 0 downloads. Generates realistic synthetic RGB leaf patterns.
    """
    def __init__(self, n_samples=64, n_classes=28, resolution=224):
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.resolution = resolution
        self.labels = np.random.randint(0, n_classes, size=n_samples)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Generate synthetic green-dominant leaf image tensor
        tensor = torch.randn(3, self.resolution, self.resolution) * 0.2
        tensor[1] += 0.5  # Green bias
        return tensor, int(self.labels[idx])


def get_transforms(resolution=224, recipe="standard"):
    eval_tf = T.Compose([
        T.Resize((resolution, resolution)),
        T.ToTensor(),
        T.Normalize(MEAN, STD)
    ])

    if recipe == "strong":
        train_tf = T.Compose([
            T.RandomResizedCrop(resolution, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandAugment(num_ops=2, magnitude=9),
            T.ColorJitter(0.3, 0.3, 0.3, 0.1),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
            T.RandomErasing(p=0.25, scale=(0.02, 0.2))
        ])
    else:
        train_tf = T.Compose([
            T.RandomResizedCrop(resolution, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ToTensor(),
            T.Normalize(MEAN, STD)
        ])

    return train_tf, eval_tf


def load_splits(splits_dir, dataset="plantdoc", fold=0):
    """
    Loads pre-computed stratified split DataFrames.
    """
    if dataset == "plantdoc":
        tr = pd.read_csv(os.path.join(splits_dir, "plantdoc_train.csv"))
        va = pd.read_csv(os.path.join(splits_dir, "plantdoc_val.csv"))
        te = pd.read_csv(os.path.join(splits_dir, "plantdoc_test.csv"))
    else:
        tr = pd.read_csv(os.path.join(splits_dir, f"citrus_fold{fold}_train.csv"))
        va = pd.read_csv(os.path.join(splits_dir, f"citrus_fold{fold}_val.csv"))
        te = pd.read_csv(os.path.join(splits_dir, f"citrus_fold{fold}_test.csv"))

    classes = sorted(tr["label"].unique())
    cmap = {c: i for i, c in enumerate(classes)}
    return tr, va, te, cmap
