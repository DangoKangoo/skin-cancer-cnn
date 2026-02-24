from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.data.dataset import ISICBinaryDataset  # noqa: E402


def build_model(num_classes: int = 2) -> nn.Module:
    m = models.mobilenet_v2(weights=None)
    in_features = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_features, num_classes)
    return m


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Generate test_predictions.csv for the Streamlit dashboard.")
    parser.add_argument("--ckpt", type=str, default="models/best_mobilenetv2.pt", help="Checkpoint path")
    parser.add_argument("--split", type=str, default="data/splits/test.csv", help="Split CSV path")
    parser.add_argument("--out", type=str, default="reports/metrics/test_predictions.csv", help="Output CSV path")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for inference")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (Windows: keep 0)")
    args = parser.parse_args()

    ckpt_path = (REPO / args.ckpt).resolve()
    split_csv = (REPO / args.split).resolve()
    out_path = (REPO / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not split_csv.exists():
        raise FileNotFoundError(f"Split CSV not found: {split_csv}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset + loader must preserve CSV order
    ds = ISICBinaryDataset(split_csv, split="test")
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(2).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)

    # supports either {"model": state_dict} or raw state_dict
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    # Load the same CSV used by dataset so we can append probabilities
    df = pd.read_csv(split_csv)

    probs_malig: list[float] = []

    for x, _y in dl:
        x = x.to(device)
        logits = model(x)
        p = torch.softmax(logits, dim=1)[:, 1]  # P(malignant)
        probs_malig.extend(p.detach().cpu().tolist())

    # Sanity check alignment
    if len(probs_malig) != len(df):
        raise RuntimeError(
            f"Prediction count mismatch: got {len(probs_malig)} probs but CSV has {len(df)} rows.\n"
            "This usually means dataset ordering doesn't match CSV or some images failed to load."
        )

    df["prob_malignant"] = probs_malig
    df["pred_0.5"] = (df["prob_malignant"] >= 0.5).astype(int)  # optional but nice for quick checks

    df.to_csv(out_path, index=False)

    print("Saved:", out_path)
    print(f"Rows: {len(df)} | Device: {device} | Ckpt: {ckpt_path.name}")


if __name__ == "__main__":
    main()