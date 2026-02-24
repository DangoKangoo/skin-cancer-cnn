from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass(frozen=True)
class DataConfig:
    image_size: int = 224
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)  # ImageNet
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)   # ImageNet


def build_transforms(split: str, cfg: DataConfig) -> transforms.Compose:
    """
    split: 'train' | 'val' | 'test'
    """
    if split == "train":
        return transforms.Compose([
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.mean, std=cfg.std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.mean, std=cfg.std),
        ])


class ISICBinaryDataset(Dataset):
    """
    Expects a CSV with columns:
      - image (e.g., ISIC_0000000)
      - label (0/1)
      - filepath (full/relative path to jpg)

    Our make_dataset.py generates exactly that.
    """
    def __init__(self, csv_path: str | Path, split: str, cfg: DataConfig | None = None):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        self.split = split
        self.cfg = cfg or DataConfig()
        self.tfms = build_transforms(split, self.cfg)

        df = pd.read_csv(self.csv_path)

        required = {"image", "label", "filepath"}
        if not required.issubset(df.columns):
            raise ValueError(f"CSV missing required columns {required}. Found: {list(df.columns)}")

        self.images = df["image"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.paths = [Path(p) for p in df["filepath"].astype(str).tolist()]

        # Fast sanity check (first few only to avoid slow startup)
        for p in self.paths[:20]:
            if not p.exists():
                raise FileNotFoundError(f"Image file not found: {p}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        img_path = self.paths[idx]
        label = self.labels[idx]

        # PIL is robust for jpg; convert to RGB to avoid grayscale/alpha weirdness
        img = Image.open(img_path).convert("RGB")
        x = self.tfms(img)
        y = torch.tensor(label, dtype=torch.long)

        return x, y