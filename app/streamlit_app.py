from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from dataclasses import dataclass
import streamlit as st

from app.core.model import load_model
from app.ui.sidebar import render_sidebar
from app.ui.styles import inject_css
from app.ui.pages.analytics import render_analytics_page
from app.ui.pages.upload import render_upload_page
from app.ui.pages.model_card import render_model_card_page

@dataclass(frozen=True)
class AppMeta:
    title: str = "Skin Cancer CNN Dashboard"
    subtitle: str = "ISIC 2018 dermoscopy | MobileNetV2 transfer learning"
    disclaimer: str = (
        "⚠️ Trained/validated on ISIC dermoscopy images. Not a medical device; "
        "not validated for general clinical photos."
    )


def main():
    meta = AppMeta()
    st.set_page_config(page_title=meta.title, page_icon="🩺", layout="wide")
    inject_css()

    st.markdown(f"## {meta.title}")
    st.markdown(f"<div class='small'>{meta.subtitle}</div>", unsafe_allow_html=True)
    st.warning(meta.disclaimer)

    page, threshold, show_curves, show_cal, gallery_n, show_gradcam, heat_alpha = render_sidebar()

    model, device = load_model()

    if page == "Test-set Analytics":
        render_analytics_page(
            model=model,
            device=device,
            threshold=threshold,
            show_curves=show_curves,
            show_cal=show_cal,
            gallery_n=gallery_n,
            show_gradcam=show_gradcam,
            heat_alpha=heat_alpha,
        )
    elif page == "Upload + Explainability":
        render_upload_page(model=model, device=device, threshold=threshold)
    else:
        render_model_card_page()


if __name__ == "__main__":
    main()