from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from time import time

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models

# Make "import src..." work when running as a script
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.dataset import ISICBinaryDataset  # noqa: E402


@dataclass
class TrainConfig:
    model_name: str = "mobilenet_v2"
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 5
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 0  # keep 0 on Windows for fewer headaches
    seed: int = 42
    log_dir: str = "runs"
    save_path: str = "models/best_mobilenetv2.pt"


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(num_classes: int = 2) -> nn.Module:
    # MobileNetV2 pretrained on ImageNet
    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    in_features = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_features, num_classes)
    return m


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    y_true = []
    y_pred = []
    total_loss = 0.0
    n = 0

    # loss needs to match training; we'll compute outside if needed
    # here we just use plain CE for reporting; training uses weighted CE
    crit = nn.CrossEntropyLoss()

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = crit(logits, y)

        total_loss += loss.item() * x.size(0)
        n += x.size(0)

        preds = torch.argmax(logits, dim=1)
        y_true.extend(y.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    avg_loss = total_loss / max(n, 1)

    return {"loss": avg_loss, "acc": acc, "precision": p, "recall": r, "f1": f1}


def compute_class_weights(train_csv: Path) -> torch.Tensor:
    # CSV has labels 0/1. Weight = N / (2 * count)
    import pandas as pd

    df = pd.read_csv(train_csv)
    counts = df["label"].value_counts().to_dict()
    n0 = counts.get(0, 0)
    n1 = counts.get(1, 0)
    n = n0 + n1
    if n0 == 0 or n1 == 0:
        return torch.tensor([1.0, 1.0], dtype=torch.float)

    w0 = n / (2.0 * n0)
    w1 = n / (2.0 * n1)
    return torch.tensor([w0, w1], dtype=torch.float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    ap.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    ap.add_argument("--lr", type=float, default=TrainConfig.lr)
    ap.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    ap.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    ap.add_argument("--save-path", type=str, default=TrainConfig.save_path)
    ap.add_argument("--log-dir", type=str, default=TrainConfig.log_dir)
    args = ap.parse_args()

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        save_path=args.save_path,
        log_dir=args.log_dir,
    )

    set_seed(cfg.seed)
    device = get_device()
    print("device:", device)

    # Paths
    train_csv = REPO / "data" / "splits" / "train.csv"
    val_csv = REPO / "data" / "splits" / "val.csv"

    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError("Missing split CSVs. Run: python src\\data\\make_dataset.py")

    # Data
    train_ds = ISICBinaryDataset(train_csv, split="train")
    val_ds = ISICBinaryDataset(val_csv, split="val")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Model
    model = build_model(num_classes=2).to(device)

    # Weighted loss (important with 11% positives)
    class_w = compute_class_weights(train_csv).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_w)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # Logging / saving
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
    Path(Path(cfg.save_path).parent).mkdir(parents=True, exist_ok=True)

    run_name = f"mobilenetv2_bs{cfg.batch_size}_lr{cfg.lr}_{int(time())}"
    writer = SummaryWriter(log_dir=str(Path(cfg.log_dir) / run_name))
    print("tensorboard log:", str(Path(cfg.log_dir) / run_name))

    best_f1 = -1.0
    global_step = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n = 0
        t0 = time()

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = x.size(0)
            epoch_loss += loss.item() * bs
            n += bs

            writer.add_scalar("train/loss_step", loss.item(), global_step)
            global_step += 1

        train_loss = epoch_loss / max(n, 1)
        val_metrics = evaluate(model, val_loader, device)

        writer.add_scalar("train/loss_epoch", train_loss, epoch)
        writer.add_scalar("val/loss", val_metrics["loss"], epoch)
        writer.add_scalar("val/acc", val_metrics["acc"], epoch)
        writer.add_scalar("val/precision", val_metrics["precision"], epoch)
        writer.add_scalar("val/recall", val_metrics["recall"], epoch)
        writer.add_scalar("val/f1", val_metrics["f1"], epoch)

        dt = time() - t0
        print(
            f"epoch {epoch:02d}/{cfg.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} acc={val_metrics['acc']:.4f} "
            f"p={val_metrics['precision']:.4f} r={val_metrics['recall']:.4f} f1={val_metrics['f1']:.4f} | "
            f"{dt:.1f}s"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_f1": best_f1,
                    "class_weights": class_w.detach().cpu(),
                    "model_name": cfg.model_name,
                },
                cfg.save_path,
            )
            print(f"  saved best -> {cfg.save_path} (val_f1={best_f1:.4f})")

    writer.close()
    print("done. best val f1:", best_f1)


if __name__ == "__main__":
    main()