from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
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
def run_eval(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    y_true, y_pred = [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        preds = torch.argmax(logits, dim=1).detach().cpu().tolist()

        y_true.extend(y.tolist())
        y_pred.extend(preds)

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
    }


def save_confusion_matrix(cm, out_path: Path):
    fig = plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix (Test)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], ["Benign(0)", "Malignant(1)"])
    plt.yticks([0, 1], ["Benign(0)", "Malignant(1)"])
    for (i, j), v in zip([(0,0),(0,1),(1,0),(1,1)], [cm[0][0], cm[0][1], cm[1][0], cm[1][1]]):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="models/best_mobilenetv2.pt")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = REPO / args.ckpt
    test_csv = REPO / "data" / "splits" / "test.csv"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not test_csv.exists():
        raise FileNotFoundError(f"Missing test split: {test_csv}")

    # Data
    test_ds = ISICBinaryDataset(test_csv, split="test")
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # Model
    model = build_model(num_classes=2).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    metrics = run_eval(model, test_loader, device)

    out_dir = REPO / "reports" / "metrics"
    fig_dir = REPO / "reports" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "test_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    cm = metrics["confusion_matrix"]
    save_confusion_matrix(cm, fig_dir / "test_confusion_matrix.png")

    print("Saved:", json_path)
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()