from __future__ import annotations

import pandas as pd

from app.core.config import PRED_PATH
from src.utils.paths import portable_image_path


def load_predictions():
    if not PRED_PATH.exists():
        raise FileNotFoundError(
            "Missing reports/metrics/test_predictions.csv\n"
            "Run: python src\\predict_testset.py"
        )
    dfp = pd.read_csv(PRED_PATH)
    if "filepath" in dfp.columns:
        dfp["filepath"] = dfp["filepath"].astype(str).map(lambda p: str(portable_image_path(p)))
    y_true = dfp["label"].astype(int).values
    p_malig = dfp["prob_malignant"].astype(float).values
    return dfp, y_true, p_malig
