from __future__ import annotations

import pandas as pd
import streamlit as st
from PIL import Image

from app.core.config import LABEL_NAME
from app.core.explain import compute_gradcam_mobilenetv2, overlay_heatmap
from app.core.inference import predict_probs


def render_upload_page(model, device, threshold: float):
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