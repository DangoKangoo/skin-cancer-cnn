from __future__ import annotations

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models

from app.core.config import CKPT_PATH


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