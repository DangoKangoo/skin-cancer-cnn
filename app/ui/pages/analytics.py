from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from PIL import Image

from app.core.config import CM_PNG, METRICS_JSON
from app.core.data import load_predictions
from app.core.explain import compute_gradcam_mobilenetv2, overlay_heatmap
from app.core.metrics import metrics_at_threshold, plot_calibration, plot_confusion_matrix, plot_roc_pr


def render_analytics_page(model, device, threshold: float, show_curves: bool, show_cal: bool, gallery_n: int, show_gradcam: bool, heat_alpha: float):
    dfp, y_true, p_malig = load_predictions()
    acc, prec, rec, f1, cm, (tn, fp, fn, tp), _ = metrics_at_threshold(y_true, p_malig, threshold)

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
        st.caption("Confusion breakdown at current threshold")
        st.code(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}", language="text")

        if METRICS_JSON.exists():
            with open(METRICS_JSON, "r", encoding="utf-8") as f:
                base = json.load(f)
            with st.expander("Saved test report (threshold=0.5)", expanded=False):
                st.json(base)

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