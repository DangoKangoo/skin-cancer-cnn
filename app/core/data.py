from __future__ import annotations

import pandas as pd

from app.core.config import PRED_PATH


def load_predictions():
    if not PRED_PATH.exists():
        raise FileNotFoundError(
            "Missing reports/metrics/test_predictions.csv\n"
            "Run: python src\\predict_testset.py"
        )
    dfp = pd.read_csv(PRED_PATH)
    y_true = dfp["label"].astype(int).values
    p_malig = dfp["prob_malignant"].astype(float).values
    return dfp, y_true, p_malig