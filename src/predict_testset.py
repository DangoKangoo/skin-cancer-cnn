from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.dataset import ISICBinaryDataset  # noqa: E402


def build_model(num_classes: int = 2) -> nn.Module:
    m = models.mobilenet_v2(weights=None)
    in_features = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_features, num_classes)
    return m


@torch.no_grad()
def main():
    ckpt_path = REPO / "models" / "best_mobilenetv2.pt"
    test_csv = REPO / "data" / "splits" / "test.csv"
    out_dir = REPO / "reports" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "test_predictions.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = ISICBinaryDataset(test_csv, split="test")
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    model = build_model(2).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # load original test split for image names/paths
    df = pd.read_csv(test_csv)
    probs_malig = []
    preds_05 = []

    i = 0
    for x, _y in dl:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().tolist()  # P(malignant)
        probs_malig.extend(probs)
        i += len(probs)

    df["prob_malignant"] = probs_malig
    df["pred_0.5"] = (df["prob_malignant"] >= 0.5).astype(int)

    df.to_csv(out_path, index=False)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()