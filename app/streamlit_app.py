from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    auc,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_curve,
)
from torchvision import models, transforms
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = os.getenv("MODEL_CKPT", "models/best_mobilenetv2.pt")
CKPT_PATH = (REPO / DEFAULT_CKPT).resolve()

IMG_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

TFM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

LABEL_NAME = {0: "Non-melanoma", 1: "Melanoma"}

PRED_PATH = REPO / "reports" / "metrics" / "test_predictions.csv"
METRICS_JSON = REPO / "reports" / "metrics" / "test_metrics.json"
CM_PNG = REPO / "reports" / "figures" / "test_confusion_matrix.png"


@dataclass(frozen=True)
class AppMeta:
    title: str = "Skin Cancer CNN Dashboard"
    subtitle: str = "ISIC 2018 dermoscopy | MobileNetV2 transfer learning"
    disclaimer: str = (
        "⚠️ Trained/validated on ISIC dermoscopy images. Not a medical device; "
        "not validated for general clinical photos."
    )


def build_model(num_classes: int = 2) -> nn.Module:
    m = models.mobilenet_v2(weights=None)
    in_features = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_features, num_classes)
    return m


@st.cache_resource
def load_model():
    if not CKPT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CKPT_PATH}\n\n"
            "Train first: python src\\train.py"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(2).to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, device


def preprocess(image: Image.Image) -> torch.Tensor:
    return TFM(image.convert("RGB")).unsqueeze(0)


@torch.no_grad()
def predict_probs(model: nn.Module, device: torch.device, image: Image.Image) -> np.ndarray:
    x = preprocess(image).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
    return probs


def compute_gradcam_mobilenetv2(model: nn.Module, device: torch.device, image: Image.Image, target_class: int) -> np.ndarray:
    target_layer = model.features[-1]
    activations = None
    gradients = None

    def fwd_hook(_m, _i, o):
        nonlocal activations
        activations = o

    def bwd_hook(_m, _gin, gout):
        nonlocal gradients
        gradients = gout[0]

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    try:
        model.eval()
        x = preprocess(image).to(device)
        x.requires_grad_(True)

        logits = model(x)
        score = logits[0, target_class]

        model.zero_grad(set_to_none=True)
        score.backward()

        w = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (w * activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.astype(np.float32)
    finally:
        h1.remove()
        h2.remove()


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    base = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    base_np = np.array(base).astype(np.float32)

    heat = (heatmap * 255.0).clip(0, 255).astype(np.uint8)
    heat_rgb = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    heat_rgb[..., 0] = heat  # red overlay

    out = (1 - alpha) * base_np + alpha * heat_rgb.astype(np.float32)
    out = out.clip(0, 255).astype(np.uint8)
    return Image.fromarray(out)


def metrics_at_threshold(y_true: np.ndarray, p_malig: np.ndarray, thr: float):
    y_pred = (p_malig >= thr).astype(int)
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)  # [[TN,FP],[FN,TP]]
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


def inject_css():
    st.markdown(
        """
        <style>
        .block-container { padding-top: 0.9rem; padding-bottom: 1.3rem; max-width: 1350px; }
        div[data-testid="stVerticalBlock"] { gap: 0.6rem; }
        div[data-testid="stAlert"] { padding: 0.5rem 0.75rem; border-radius: 14px; }

        .small { font-size: 0.98rem; opacity: 0.86; }
        .muted { font-size: 0.86rem; opacity: 0.75; }

        .kpi-row { display: grid; grid-template-columns: repeat(8, minmax(0, 1fr)); gap: 10px; }
        .kpi {
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 14px;
            padding: 10px 12px;
            background: rgba(255,255,255,0.03);
            line-height: 1.1;
        }
        .kpi .label { font-size: 0.84rem; opacity: 0.78; margin-bottom: 6px; }
        .kpi .value { font-size: 1.18rem; font-weight: 750; }

        .card {
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 16px;
            padding: 12px 14px;
            background: rgba(255,255,255,0.03);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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


# -------------------- APP --------------------
meta = AppMeta()
st.set_page_config(page_title=meta.title, page_icon="🩺", layout="wide")
inject_css()

st.markdown(f"## {meta.title}")
st.markdown(f"<div class='small'>{meta.subtitle}</div>", unsafe_allow_html=True)
st.warning(meta.disclaimer)

with st.sidebar:
    st.header("Controls")
    page = st.radio("Section", ["Test-set Analytics", "Upload + Explainability", "Model Card"], index=0)
    st.divider()
    threshold = st.slider("Decision threshold (Melanoma)", 0.05, 0.95, 0.50, 0.01)
    st.divider()
    show_curves = st.checkbox("Show ROC/PR", value=True)
    show_cal = st.checkbox("Show calibration", value=True)
    st.divider()
    gallery_n = st.slider("Errors per group", 4, 24, 8, 1)
    show_gradcam = st.checkbox("Grad-CAM in error gallery", value=False)
    heat_alpha = st.slider("Heatmap intensity", 0.15, 0.75, 0.45, 0.05, disabled=not show_gradcam)

model, device = load_model()

if page == "Test-set Analytics":
    dfp, y_true, p_malig = load_predictions()
    acc, prec, rec, f1, cm, (tn, fp, fn, tp), y_pred = metrics_at_threshold(y_true, p_malig, threshold)

    # Compact KPIs
    st.markdown("### Overview")
    st.markdown(
        f"""
        <div class="kpi-row">
          <div class="kpi"><div class="label">Accuracy</div><div class="value">{acc:.3f}</div></div>
          <div class="kpi"><div class="label">Precision</div><div class="value">{prec:.3f}</div></div>
          <div class="kpi"><div class="label">Recall</div><div class="value">{rec:.3f}</div></div>
          <div class="kpi"><div class="label">F1</div><div class="value">{f1:.3f}</div></div>
          <div class="kpi"><div class="label">TP</div><div class="value">{tp}</div></div>
          <div class="kpi"><div class="label">FP</div><div class="value">{fp}</div></div>
          <div class="kpi"><div class="label">FN</div><div class="value">{fn}</div></div>
          <div class="kpi"><div class="label">TN</div><div class="value">{tn}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Primary + compact panel layout
    left, right = st.columns([1.6, 0.9], gap="large")

    with left:
        st.markdown("### Core view")
        c1, c2 = st.columns([1.0, 1.0], gap="medium")

        with c1:
            st.pyplot(plot_confusion_matrix(cm), clear_figure=True)

        with c2:
            fig = plt.figure(figsize=(5.2, 4.2), dpi=120)
            plt.hist(p_malig[y_true == 0], bins=30, alpha=0.7, label="True Non-melanoma")
            plt.hist(p_malig[y_true == 1], bins=30, alpha=0.7, label="True Melanoma")
            plt.axvline(threshold, linestyle="--")
            plt.title("P(melanoma) distribution")
            plt.xlabel("P(melanoma)")
            plt.ylabel("Count")
            plt.legend()
            plt.tight_layout()
            st.pyplot(fig, clear_figure=True)

        # Advanced plots hidden behind expanders
        if show_curves:
            with st.expander("Curves (ROC / PR)", expanded=False):
                roc_fig, pr_fig, roc_auc, pr_auc = plot_roc_pr(y_true, p_malig)
                cc1, cc2 = st.columns(2)
                with cc1:
                    st.pyplot(roc_fig, clear_figure=True)
                    st.caption(f"ROC AUC = {roc_auc:.3f}")
                with cc2:
                    st.pyplot(pr_fig, clear_figure=True)
                    st.caption(f"PR AUC = {pr_auc:.3f}")

        if show_cal:
            with st.expander("Calibration (confidence realism)", expanded=False):
                cal_fig, brier, _ = plot_calibration(y_true, p_malig, bins=10)
                st.pyplot(cal_fig, clear_figure=True)
                st.caption(f"Brier score = {brier:.4f} (lower is better)")

        # Error explorer stays below core view
        st.markdown("### Error Explorer")
        dfp = dfp.copy()
        dfp["pred_thr"] = (dfp["prob_malignant"] >= threshold).astype(int)

        fp_df = dfp[(dfp["label"] == 0) & (dfp["pred_thr"] == 1)].sort_values("prob_malignant", ascending=False)
        fn_df = dfp[(dfp["label"] == 1) & (dfp["pred_thr"] == 0)].sort_values("prob_malignant", ascending=True)

        colA, colB = st.columns(2, gap="medium")

        def render_gallery(title: str, subdf: pd.DataFrame, target_class_for_cam: int):
            st.subheader(title)
            shown = subdf.head(gallery_n)
            if len(shown) == 0:
                st.info("None at this threshold.")
                return
            for _, r in shown.iterrows():
                p = Path(r["filepath"])
                if not p.exists():
                    continue
                img = Image.open(p)
                st.image(img, caption=f"{p.name} | P(mel)={r['prob_malignant']:.3f}", use_container_width=True)
                if show_gradcam:
                    heat = compute_gradcam_mobilenetv2(model, device, img, target_class=target_class_for_cam)
                    st.image(overlay_heatmap(img, heat, alpha=heat_alpha), caption="Grad-CAM", use_container_width=True)

        with colA:
            render_gallery("False Positives", fp_df, target_class_for_cam=1)
        with colB:
            render_gallery("False Negatives", fn_df, target_class_for_cam=1)

    with right:
        st.markdown("### Summary")
        st.write({"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)})

        if METRICS_JSON.exists():
            with open(METRICS_JSON, "r", encoding="utf-8") as f:
                base = json.load(f)
            with st.expander("Saved test report (threshold=0.5)", expanded=False):
                st.json(base)

        if CM_PNG.exists():
            with st.expander("Saved confusion matrix image", expanded=False):
                st.image(str(CM_PNG), use_container_width=True)

        st.markdown("### Downloads")
        st.download_button(
            "test_predictions.csv",
            data=dfp.to_csv(index=False).encode("utf-8"),
            file_name="test_predictions.csv",
            mime="text/csv",
        )
        report = {
            "threshold": float(threshold),
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
        }
        st.download_button(
            "threshold_report.json",
            data=json.dumps(report, indent=2).encode("utf-8"),
            file_name="threshold_report.json",
            mime="application/json",
        )

elif page == "Upload + Explainability":
    st.markdown("### Upload + Explainability")
    st.caption("Best used with dermoscopy images. Uploads are for demo; correctness may be unknown.")

    uploaded = st.file_uploader("Upload image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    gc = st.checkbox("Show Grad-CAM", value=True)
    alpha = st.slider("Heatmap intensity", 0.15, 0.75, 0.45, 0.05, disabled=not gc)

    if uploaded is None:
        st.info("Upload an image to run inference.")
        st.stop()

    img = Image.open(uploaded)
    st.image(img, caption="Uploaded image", use_container_width=True)

    probs = predict_probs(model, device, img)
    p_non, p_mel = float(probs[0]), float(probs[1])

    pred = 1 if p_mel >= threshold else 0
    st.markdown(f"#### Prediction: **{LABEL_NAME[pred]}**")
    st.write(f"P(melanoma) = **{p_mel:.3f}**  | threshold = {threshold:.2f}")

    st.bar_chart(
        pd.DataFrame({"class": ["Non-melanoma", "Melanoma"], "prob": [p_non, p_mel]}),
        x="class",
        y="prob",
    )

    if gc:
        heat = compute_gradcam_mobilenetv2(model, device, img, target_class=pred)
        st.image(overlay_heatmap(img, heat, alpha=alpha), caption="Grad-CAM overlay", use_container_width=True)

else:
    st.markdown("### Model Card")
    st.markdown(
        """
        <div class="card">
        <b>Task</b>: Melanoma vs non-melanoma (ISIC 2018 Task 3 labels).<br/>
        <b>Model</b>: MobileNetV2 transfer learning.<br/>
        <b>Evaluation</b>: Held-out test split; threshold adjustable to explore tradeoffs.<br/>
        <b>Known limitation</b>: Domain shift (clinical photos vs dermoscopy).<br/>
        <b>Intended use</b>: Educational/demo only.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write({
        "Checkpoint exists": CKPT_PATH.exists(),
        "Predictions file exists": PRED_PATH.exists(),
        "Test metrics json exists": METRICS_JSON.exists(),
        "Confusion matrix png exists": CM_PNG.exists(),
    })