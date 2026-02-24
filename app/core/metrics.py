from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_curve,
)


def metrics_at_threshold(y_true: np.ndarray, p_malig: np.ndarray, thr: float):
    y_pred = (p_malig >= thr).astype(int)
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return acc, p, r, f1, cm, (tn, fp, fn, tp), y_pred


def plot_confusion_matrix(cm: np.ndarray):
    fig = plt.figure(figsize=(5.2, 4.2), dpi=120)
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], ["Non-mel(0)", "Mel(1)"])
    plt.yticks([0, 1], ["Non-mel(0)", "Mel(1)"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    return fig


def plot_roc_pr(y_true: np.ndarray, p_malig: np.ndarray):
    fpr, tpr, _ = roc_curve(y_true, p_malig)
    roc_auc = auc(fpr, tpr)

    prec, rec, _ = precision_recall_curve(y_true, p_malig)
    pr_auc = auc(rec, prec)

    fig1 = plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1])
    plt.title(f"ROC (AUC={roc_auc:.3f})")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.tight_layout()

    fig2 = plt.figure()
    plt.plot(rec, prec)
    plt.title(f"PR (AUC={pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()

    return fig1, fig2, roc_auc, pr_auc


def plot_calibration(y_true: np.ndarray, p_malig: np.ndarray, bins: int = 10):
    p = np.clip(p_malig, 1e-8, 1 - 1e-8)
    bin_edges = np.linspace(0, 1, bins + 1)
    bin_ids = np.digitize(p, bin_edges) - 1
    bin_ids = np.clip(bin_ids, 0, bins - 1)

    conf, acc, counts = [], [], []
    for b in range(bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        conf.append(p[mask].mean())
        acc.append(y_true[mask].mean())
        counts.append(int(mask.sum()))

    fig = plt.figure()
    plt.plot([0, 1], [0, 1])
    plt.scatter(conf, acc)
    plt.title("Calibration")
    plt.xlabel("Mean predicted P(mel)")
    plt.ylabel("Observed melanoma rate")
    plt.tight_layout()

    brier = brier_score_loss(y_true, p_malig)
    return fig, brier, list(zip(conf, acc, counts))