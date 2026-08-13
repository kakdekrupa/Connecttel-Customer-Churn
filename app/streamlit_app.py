"""
ConnectTel Customer Churn Predictor — Streamlit App
Dark corporate theme with an animated prediction pipeline, live risk gauge,
and per-customer SHAP factor breakdown.

Run with:  streamlit run app/streamlit_app.py
"""

import time

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────
# PAGE CONFIG + THEME
# ─────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ConnectTel Churn Predictor",
    page_icon="📡",
    layout="centered",
    initial_sidebar_state="collapsed",
)

DARK_CSS = """
<style>
    :root {
        --bg: #0b1220;
        --panel: #131c2e;
        --panel-border: #223049;
        --text: #e6ebf5;
        --muted: #8a97ab;
        --accent: #38bdf8;
        --danger: #f87171;
        --safe: #34d399;
        --warn: #fbbf24;
    }

    .stApp {
        background: radial-gradient(circle at 20% 0%, #101a2d 0%, #0b1220 55%);
        color: var(--text);
    }

    /* Hide default streamlit chrome for a cleaner "product" feel */
    #MainMenu, footer, header {visibility: hidden;}

    .ct-header {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 4px;
    }
    .ct-logo {
        width: 42px; height: 42px;
        border-radius: 10px;
        background: linear-gradient(135deg, #38bdf8, #0ea5e9);
        display: flex; align-items: center; justify-content: center;
        font-size: 22px;
        box-shadow: 0 0 24px rgba(56,189,248,0.35);
    }
    .ct-title { font-size: 26px; font-weight: 700; margin: 0; color: var(--text); }
    .ct-subtitle { color: var(--muted); font-size: 14px; margin-top: -2px; }

    .ct-card {
        background: var(--panel);
        border: 1px solid var(--panel-border);
        border-radius: 14px;
        padding: 20px 22px;
        margin-bottom: 18px;
    }
    .ct-section-label {
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 12px;
        color: var(--accent);
        font-weight: 600;
        margin-bottom: 10px;
    }

    /* Pipeline stepper */
    .pipeline {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin: 6px 0 4px 0;
    }
    .step {
        flex: 1;
        text-align: center;
        position: relative;
        color: var(--muted);
        font-size: 12.5px;
    }
    .step-circle {
        width: 34px; height: 34px;
        border-radius: 50%;
        margin: 0 auto 8px auto;
        display: flex; align-items: center; justify-content: center;
        font-size: 16px;
        border: 2px solid var(--panel-border);
        background: #0e1626;
        transition: all 0.25s ease;
    }
    .step-pending .step-circle { color: var(--muted); }
    .step-active .step-circle {
        border-color: var(--accent);
        color: var(--accent);
        box-shadow: 0 0 14px rgba(56,189,248,0.45);
    }
    .step-done .step-circle {
        border-color: var(--safe);
        background: rgba(52,211,153,0.12);
        color: var(--safe);
    }
    .step-active .step-label { color: var(--accent); font-weight: 600; }
    .step-done .step-label { color: var(--safe); }
    .step-line {
        position: absolute;
        top: 17px; left: -50%;
        width: 100%; height: 2px;
        background: var(--panel-border);
        z-index: -1;
    }
    .step:first-child .step-line { display: none; }

    .result-high {
        border-left: 4px solid var(--danger);
        background: rgba(248,113,113,0.08);
    }
    .result-low {
        border-left: 4px solid var(--safe);
        background: rgba(52,211,153,0.08);
    }
    .result-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.03em;
    }
    .badge-high { background: rgba(248,113,113,0.15); color: var(--danger); }
    .badge-low { background: rgba(52,211,153,0.15); color: var(--safe); }

    .factor-row { font-size: 13.5px; margin-bottom: 6px; color: var(--text); }
    .factor-bar-bg {
        background: #0e1626;
        border-radius: 6px;
        height: 8px;
        width: 100%;
        overflow: hidden;
        margin-top: 3px;
    }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# LOAD MODEL + EXPLAINER (cached so this only runs once per session)
# ─────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/churn_model.pkl")
    model_columns = joblib.load("models/model_columns.pkl")
    explainer = shap.TreeExplainer(model)
    return model, model_columns, explainer


model, model_columns, explainer = load_artifacts()

# ─────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="ct-header">
        <div class="ct-logo">📡</div>
        <div>
            <p class="ct-title">ConnectTel Churn Predictor</p>
            <p class="ct-subtitle">Live risk scoring for the retention team</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

# ─────────────────────────────────────────────────────────────────────────
# INPUT CARD
# ─────────────────────────────────────────────────────────────────────────
st.markdown('<div class="ct-card">', unsafe_allow_html=True)
st.markdown('<div class="ct-section-label">Customer Profile</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0, step=1.0)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
with col2:
    total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 840.0, step=10.0)
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No"])

predict_clicked = st.button("▶  Run Prediction", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# PIPELINE RENDERER
# ─────────────────────────────────────────────────────────────────────────
STEPS = ["Input Received", "Preprocessing", "Model Scoring", "Result Ready"]


def render_pipeline(active_index, placeholder):
    """active_index: steps before this are 'done', this one is 'active', rest 'pending'."""
    html = '<div class="ct-card"><div class="pipeline">'
    for i, label in enumerate(STEPS):
        if i < active_index:
            state, icon = "step-done", "✓"
        elif i == active_index:
            state, icon = "step-active", "●"
        else:
            state, icon = "step-pending", "○"
        html += f'''
            <div class="step {state}">
                <div class="step-line"></div>
                <div class="step-circle">{icon}</div>
                <div class="step-label">{label}</div>
            </div>
        '''
    html += "</div></div>"
    placeholder.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────
# PREDICTION FLOW
# ─────────────────────────────────────────────────────────────────────────
if predict_clicked:
    pipeline_slot = st.empty()

    # Step 0: Input received
    render_pipeline(0, pipeline_slot)
    time.sleep(0.35)

    # Step 1: Preprocessing — build the aligned feature row
    render_pipeline(1, pipeline_slot)
    input_dict = {
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            f"Contract_{contract}": 1,
            f"InternetService_{internet_service}": 1,
            "TechSupport_Yes": 1 if tech_support == "Yes" else 0,
        }
    input_df = pd.DataFrame([input_dict]).reindex(columns=model_columns, fill_value=0)
    time.sleep(0.45)

    # Step 2: Model scoring
    render_pipeline(2, pipeline_slot)
    proba = model.predict_proba(input_df)[0][1]
    shap_values = explainer.shap_values(input_df)
    row_shap = shap_values[0] if not isinstance(shap_values, list) else shap_values[1][0]
    time.sleep(0.5)

    # Step 3: Done — hold briefly so the user sees all checkmarks, then clear
    render_pipeline(4, pipeline_slot)  # 4 = all steps show as done
    time.sleep(0.5)
    pipeline_slot.empty()  # remove the pipeline entirely — result takes its place, no leftover gap

    is_high_risk = proba > 0.5

    is_high_risk = proba > 0.5
    result_class = "result-high" if is_high_risk else "result-low"
    badge_class = "badge-high" if is_high_risk else "badge-low"
    badge_text = "HIGH RISK" if is_high_risk else "LOW RISK"

    # ── Result card with gauge ──────────────────────────────────────────
    st.markdown(f'<div class="ct-card {result_class}">', unsafe_allow_html=True)
    st.markdown('<div class="ct-section-label">Prediction Result</div>', unsafe_allow_html=True)
    st.markdown(
        f'<span class="result-badge {badge_class}">{badge_text}</span>',
        unsafe_allow_html=True,
    )

    gauge_color = "#f87171" if is_high_risk else "#34d399"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            number={"suffix": "%", "font": {"color": "#e6ebf5", "size": 40}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8a97ab"},
                "bar": {"color": gauge_color, "thickness": 0.3},
                "bgcolor": "#0e1626",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "rgba(52,211,153,0.18)"},
                    {"range": [40, 70], "color": "rgba(251,191,36,0.18)"},
                    {"range": [70, 100], "color": "rgba(248,113,113,0.18)"},
                ],
                "threshold": {
                    "line": {"color": "#e6ebf5", "width": 3},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=260,
        margin=dict(l=20, r=20, t=10, b=10),
        font={"color": "#e6ebf5"},
    )
    st.plotly_chart(fig, use_container_width=True)

    if is_high_risk:
        st.warning("Recommend routing this customer to the retention team.")
    else:
        st.success("Customer appears stable — no action needed.")
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Top contributing factors (per-customer SHAP breakdown) ─────────
    st.markdown('<div class="ct-card">', unsafe_allow_html=True)
    st.markdown('<div class="ct-section-label">Why This Score — Top Contributing Factors</div>', unsafe_allow_html=True)

    factor_df = (
        pd.DataFrame({"feature": model_columns, "impact": row_shap})
        .assign(abs_impact=lambda d: d["impact"].abs())
        .sort_values("abs_impact", ascending=False)
        .head(5)
    )
    max_abs = factor_df["abs_impact"].max() or 1

    for _, row in factor_df.iterrows():
        pct_width = round((row["abs_impact"] / max_abs) * 100)
        bar_color = "#f87171" if row["impact"] > 0 else "#38bdf8"
        direction = "increases risk" if row["impact"] > 0 else "decreases risk"
        st.markdown(
            f"""
            <div class="factor-row">
                {row['feature']} — <span style="color:{bar_color}">{direction}</span>
                <div class="factor-bar-bg">
                    <div style="width:{pct_width}%; background:{bar_color}; height:100%;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown(
        '<p style="color:#8a97ab; font-size:13px; text-align:center;">'
        "Fill in the customer profile above and click Run Prediction."
        "</p>",
        unsafe_allow_html=True,
    )