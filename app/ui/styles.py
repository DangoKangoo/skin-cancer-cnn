from __future__ import annotations

import streamlit as st


def inject_css():
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.6rem; padding-bottom: 1.3rem; max-width: 1350px; }
        h1, h2, h3 { scroll-margin-top: 90px; }
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