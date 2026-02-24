from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# ---- Config ----
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


def build_model(num_classes: int = 2) -> nn.Module:
    m = models.mobilenet_v2(weights=None)
    in_features = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_features, num_classes)
    return m


@st.cache_resource
def load_model():
    if not CKPT_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at:\n{CKPT_PATH}\n\n"
            "Train first (python src\\train.py) or set MODEL_CKPT in a .env file."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=2).to(device)

    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, device


@torch.no_grad()
def predict(model: nn.Module, device: torch.device, image: Image.Image):
    x = TFM(image.convert("RGB")).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu()
    pred = int(torch.argmax(probs).item())
    conf = float(probs[pred].item())
    return pred, conf, probs.tolist()


# ---- UI ----
st.set_page_config(page_title="Skin Cancer Detection (ISIC)", page_icon="🩺")

st.title("🩺 Skin Cancer Detection (ISIC 2018)")
st.write("Binary classifier: **Malignant (Melanoma)** vs **Benign (Non-melanoma)**")

with st.expander("Model details"):
    st.write(f"Checkpoint: `{CKPT_PATH}`")
    st.write("Architecture: MobileNetV2 (transfer learning)")

uploaded = st.file_uploader("Upload a skin lesion image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Upload an image to get a prediction.")
    st.stop()

img = Image.open(uploaded)
st.image(img, caption="Uploaded image", use_container_width=True)

try:
    model, device = load_model()
except Exception as e:
    st.error(str(e))
    st.stop()

pred, conf, probs = predict(model, device, img)

label = "Malignant (Melanoma)" if pred == 1 else "Benign (Non-melanoma)"
st.subheader(f"Prediction: **{label}**")
st.write(f"Confidence: **{conf:.3f}**")

st.write("Class probabilities:")
st.json({"Benign (0)": probs[0], "Malignant (1)": probs[1]})