from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from torchvision import models, transforms

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


def build_model(num_classes: int = 2) -> nn.Module:
    m = models.mobilenet_v2(weights=None)
    in_features = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_features, num_classes)
    return m


@st.cache_resource
def load_model():
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")
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
    heat_rgb[..., 0] = heat

    out = (1 - alpha) * base_np + alpha * heat_rgb.astype(np.float32)
    out = out.clip(0, 255).astype(np.uint8)
    return Image.fromarray(out)


def metrics_from_probs(y_true: np.ndarray, p_malig: np.ndarray, thr: float):
    y_pred = (p_malig >= thr).astype(int)
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)  # [[TN,FP],[FN,TP]]
    return acc, p, r, f1, cm


st.set_page_config(page_title="Skin Cancer CNN Dashboard", page_icon="🩺")
st.title("🩺 Skin Cancer CNN Dashboard (ISIC 2018)")
st.caption("Model trained on dermoscopy images. Uploads outside this domain may be unreliable.")

tab1, tab2 = st.tabs(["📊 Test-set Dashboard", "🧪 Upload + Explain"])

# -------------------- TAB 1: TESTSET DASHBOARD --------------------
with tab1:
    preds_path = REPO / "reports" / "metrics" / "test_predictions.csv"
    metrics_json = REPO / "reports" / "metrics" / "test_metrics.json"
    cm_png = REPO / "reports" / "figures" / "test_confusion_matrix.png"

    if not preds_path.exists():
        st.error("Missing reports/metrics/test_predictions.csv. Run: python src\\predict_testset.py")
        st.stop()

    dfp = pd.read_csv(preds_path)
    y_true = dfp["label"].astype(int).values
    p_malig = dfp["prob_malignant"].astype(float).values

    thr = st.slider("Decision threshold for Melanoma (1)", 0.05, 0.95, 0.50, 0.01)
    acc, prec, rec, f1, cm = metrics_from_probs(y_true, p_malig, thr)
    tn, fp, fn, tp = cm.ravel()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{acc:.3f}")
    c2.metric("Precision", f"{prec:.3f}")
    c3.metric("Recall", f"{rec:.3f}")
    c4.metric("F1", f"{f1:.3f}")

    st.write("Confusion matrix at current threshold:")
    st.write({"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)})
    st.write(cm.tolist())

    if metrics_json.exists():
        with open(metrics_json, "r", encoding="utf-8") as f:
            base = json.load(f)
        st.info(
            f"Saved test metrics (threshold=0.5 at evaluation time): "
            f"acc={base['accuracy']:.3f}, f1={base['f1']:.3f}, recall={base['recall']:.3f}, precision={base['precision']:.3f}"
        )

    if cm_png.exists():
        st.image(str(cm_png), caption="Confusion matrix saved from src/evaluate.py", use_container_width=True)

    st.markdown("### Error gallery")
    st.caption("Inspect false negatives/false positives to understand failure modes.")

    model, device = load_model()

    n_show = st.slider("How many to show per group", 4, 20, 8, 1)

    # compute predictions at selected threshold
    dfp["pred_thr"] = (dfp["prob_malignant"] >= thr).astype(int)

    fp_df = dfp[(dfp["label"] == 0) & (dfp["pred_thr"] == 1)].sort_values("prob_malignant", ascending=False).head(n_show)
    fn_df = dfp[(dfp["label"] == 1) & (dfp["pred_thr"] == 0)].sort_values("prob_malignant", ascending=True).head(n_show)

    colA, colB = st.columns(2)

    with colA:
        st.subheader("False Positives (Benign → predicted Melanoma)")
        for _, r in fp_df.iterrows():
            p = Path(r["filepath"])
            img = Image.open(p)
            st.image(img, caption=f"{p.name} | P(mal)={r['prob_malignant']:.3f}", use_container_width=True)
            if st.checkbox(f"Grad-CAM FP {p.name}", key=f"fp_{p.name}"):
                heat = compute_gradcam_mobilenetv2(model, device, img, target_class=1)
                st.image(overlay_heatmap(img, heat), caption="Grad-CAM", use_container_width=True)

    with colB:
        st.subheader("False Negatives (Melanoma → predicted Benign)")
        for _, r in fn_df.iterrows():
            p = Path(r["filepath"])
            img = Image.open(p)
            st.image(img, caption=f"{p.name} | P(mal)={r['prob_malignant']:.3f}", use_container_width=True)
            if st.checkbox(f"Grad-CAM FN {p.name}", key=f"fn_{p.name}"):
                heat = compute_gradcam_mobilenetv2(model, device, img, target_class=1)
                st.image(overlay_heatmap(img, heat), caption="Grad-CAM", use_container_width=True)

# -------------------- TAB 2: UPLOAD + EXPLAIN --------------------
with tab2:
    uploaded = st.file_uploader("Upload a dermoscopy image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    threshold = st.slider("Decision threshold for Melanoma (1)", 0.05, 0.95, 0.50, 0.01, key="upload_thr")
    show_gradcam = st.checkbox("Show Grad-CAM explanation", value=True, key="upload_gc")
    alpha = st.slider("Heatmap intensity", 0.1, 0.8, 0.45, 0.05, disabled=not show_gradcam, key="upload_alpha")

    if uploaded is None:
        st.info("Upload an image to get a prediction.")
        st.stop()

    img = Image.open(uploaded)
    st.image(img, caption="Uploaded image", use_container_width=True)

    model, device = load_model()
    probs = predict_probs(model, device, img)
    p_benign, p_malig = float(probs[0]), float(probs[1])

    pred = 1 if p_malig >= threshold else 0
    label = LABEL_NAME[pred]
    conf = p_malig if pred == 1 else p_benign

    st.subheader(f"Prediction: **{label}**")
    st.write(f"Confidence: **{conf:.3f}**  |  P(melanoma)=**{p_malig:.3f}**  (threshold={threshold:.2f})")

    st.markdown("### Probabilities")
    st.bar_chart(pd.DataFrame({"class": ["Non-melanoma", "Melanoma"], "prob": [p_benign, p_malig]}), x="class", y="prob")
    st.json({"Non-melanoma (0)": p_benign, "Melanoma (1)": p_malig})

    if show_gradcam:
        heat = compute_gradcam_mobilenetv2(model, device, img, target_class=pred)
        st.image(overlay_heatmap(img, heat, alpha=alpha), caption="Grad-CAM overlay", use_container_width=True)