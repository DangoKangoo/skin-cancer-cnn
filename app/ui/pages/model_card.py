from __future__ import annotations

import streamlit as st

from app.core.config import CKPT_PATH, CM_PNG, METRICS_JSON, PRED_PATH


def render_model_card_page():
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