from __future__ import annotations

import streamlit as st
import torch

from app.core.config import CKPT_PATH


def render_sidebar():
    with st.sidebar:
        st.header("Controls")

        page = st.radio(
            "Section",
            ["Test-set Analytics", "Upload + Explainability", "Model Card"],
            index=0,
        )

        # Threshold only matters for Analytics + Upload pages
        if page in ("Test-set Analytics", "Upload + Explainability"):
            st.divider()
            threshold = st.slider("Decision threshold (Melanoma)", 0.05, 0.95, 0.50, 0.01)
        else:
            threshold = 0.50  # default, unused on model card

        # Analytics-only controls
        if page == "Test-set Analytics":
            st.divider()
            st.subheader("Analytics")

            show_curves = st.checkbox("Show ROC/PR", value=True)
            show_cal = st.checkbox("Show calibration", value=True)

            st.divider()
            st.subheader("Error Explorer")

            gallery_n = st.slider("Errors per group", 4, 24, 8, 1)
            show_gradcam = st.checkbox("Grad-CAM in error gallery", value=False)
            heat_alpha = st.slider(
                "Heatmap intensity",
                0.15,
                0.75,
                0.45,
                0.05,
                disabled=not show_gradcam,
            )
        else:
            # Defaults when the controls are hidden
            show_curves = True
            show_cal = True
            gallery_n = 8
            show_gradcam = False
            heat_alpha = 0.45

        st.divider()
        st.subheader("System")
        st.caption(f"Checkpoint: {CKPT_PATH}")
        st.caption(f"CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            st.caption(f"GPU: {torch.cuda.get_device_name(0)}")

    return page, threshold, show_curves, show_cal, gallery_n, show_gradcam, heat_alpha