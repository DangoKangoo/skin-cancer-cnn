from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

DEFAULT_CKPT = os.getenv("MODEL_CKPT", "models/best_mobilenetv2.pt")
CKPT_PATH = (REPO / DEFAULT_CKPT).resolve()

IMG_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

LABEL_NAME = {0: "Non-melanoma", 1: "Melanoma"}

PRED_PATH = REPO / "reports" / "metrics" / "test_predictions.csv"
METRICS_JSON = REPO / "reports" / "metrics" / "test_metrics.json"
CM_PNG = REPO / "reports" / "figures" / "test_confusion_matrix.png"