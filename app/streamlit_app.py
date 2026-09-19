"""
ConnectTel Customer Churn Predictor — Streamlit App
Editorial "ink & paper" data-terminal theme: monospace terminal-style pipeline
status, a hand-built radial tick-ring gauge, and rung-tick SHAP factor bars.
Visual language adapted from an editorial print-data aesthetic (hairline
rules, an ink/paper palette, ladder-based grayscale hierarchy) combined with
a terminal boot-log status pattern for the live scoring pipeline.

Run with:  streamlit run app/streamlit_app.py
"""

import math
import time

import joblib
import pandas as pd
import shap
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ConnectTel Churn Predictor",
    page_icon="◆",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────
# DESIGN TOKENS — ink/paper editorial palette, dark ground
# ─────────────────────────────────────────────────────────────────────────
INK = "#1b1b19"     # background
PAPER = "#f1f0ec"   # primary text
MUTED = "#8b8a84"   # secondary text
HAIR = "#3a3934"    # hairline rule
FAINT = "#4c4b45"   # unfilled tick color
RISK = "#d9695f"    # muted editorial red  — increases risk
INFO = "#6fa8c9"    # muted editorial blue — decreases risk
SAFE = "#5fa87c"    # muted editorial green
WARN = "#c9a227"    # muted editorial amber

EDITORIAL_CSS = f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=IBM+Plex+Serif:wght@500;600&display=swap" rel="stylesheet">
<style>
    :root {{
        --ink: {INK}; --paper: {PAPER}; --muted: {MUTED};
        --hair: {HAIR}; --faint: {FAINT};
        --risk: {RISK}; --info: {INFO}; --safe: {SAFE}; --warn: {WARN};
    }}

    .stApp {{ background: var(--ink); color: var(--paper); }}
    #MainMenu, footer, header {{ visibility: hidden; }}

    html, body, [class*="css"] {{ font-family: 'IBM Plex Mono', monospace; }}

    .eyebrow {{
        font-size: 11px; letter-spacing: 3px; color: var(--muted);
        font-weight: 600; text-transform: uppercase; margin-bottom: 2px;
    }}
    .headline {{
        font-family: 'IBM Plex Serif', serif; font-weight: 600;
        font-size: 30px; color: var(--paper); margin: 4px 0 2px 0;
    }}
    .subtitle {{ color: var(--muted); font-size: 13.5px; margin-bottom: 6px; }}

    .section {{
        border-top: 1px solid var(--hair);
        padding-top: 18px; margin-top: 28px;
    }}
    .section-label {{
        font-size: 10.5px; letter-spacing: 2.5px; color: var(--muted);
        font-weight: 600; text-transform: uppercase; margin-bottom: 16px;
    }}

    /* Terminal-style pipeline status */
    .terminal-steps {{ font-size: 12px; letter-spacing: 0.5px; }}
    .term-step {{ margin-right: 22px; color: var(--faint); }}
    .term-done {{ color: var(--safe); }}
    .term-active {{ color: var(--paper); font-weight: 600; }}
    .term-pending {{ color: var(--faint); }}
    .hairline-track {{ height: 1px; background: var(--hair); margin-top: 12px; }}
    .hairline-fill {{ height: 1px; background: var(--paper); }}

    /* Result */
    .badge {{
        font-size: 13px; letter-spacing: 2px; font-weight: 700;
        text-transform: uppercase; padding-left: 12px; margin-bottom: 4px;
        display: inline-block;
    }}
    .gauge-wrap {{ display: flex; justify-content: center; margin: 8px 0 22px 0; }}
    .verdict-note {{
        text-align: center; color: var(--muted); font-size: 12px;
        margin-top: -8px; margin-bottom: 4px;
    }}

    /* Factor rows */
    .factor-row {{
        display: flex; align-items: center; justify-content: space-between;
        padding: 10px 0; border-bottom: 1px solid var(--hair);
        font-size: 12px; flex-wrap: wrap; gap: 6px;
    }}
    .factor-row:last-child {{ border-bottom: none; }}
    .factor-label {{ color: var(--paper); letter-spacing: 0.2px; flex: 1 1 200px; }}
    .factor-viz {{ display: flex; align-items: center; gap: 12px; }}
    .factor-dir {{
        font-size: 10.5px; letter-spacing: 1px; text-transform: uppercase;
        width: 100px;
    }}
    .footnote {{ color: var(--muted); font-size: 11px; margin-top: 18px; line-height: 1.6; }}

    /* Native widget reskin — flat, hairline-bordered, no rounded gradients */
    div[data-testid="stSlider"] label, div[data-testid="stSelectbox"] label,
    div[data-testid="stNumberInput"] label {{
        color: var(--muted) !important; font-size: 11.5px !important;
        letter-spacing: 1px; text-transform: uppercase;
    }}
    .stButton > button {{
        background: transparent; color: var(--paper);
        border: 1px solid var(--paper); border-radius: 0;
        font-family: 'IBM Plex Mono', monospace; letter-spacing: 2px;
        text-transform: uppercase; font-size: 12.5px; font-weight: 600;
        padding: 10px 0; width: 100%;
    }}
    .stButton > button:hover {{ background: var(--paper); color: var(--ink); }}
</style>
"""
st.markdown(EDITORIAL_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────
# CUSTOM SVG COMPONENTS — radial tick-ring gauge & rung-tick factor bars
# ─────────────────────────────────────────────────────────────────────────
def tick_ring_svg(value: float, size: int = 220, n_ticks: int = 72) -> str:
    """A dial made of radial dashes, one per ~1.4 percentage points, filled
    clockwise from the top. Color reflects the risk band the value falls in."""
    color = RISK if value >= 70 else WARN if value >= 40 else SAFE
    cx = cy = size / 2
    outer_r = size / 2 - 12
    inner_r = outer_r - 16
    filled = round(value / 100 * n_ticks)
    lines = []
    for i in range(n_ticks):
        angle = math.radians(i * (360 / n_ticks) - 90)
        x1, y1 = cx + inner_r * math.cos(angle), cy + inner_r * math.sin(angle)
        x2, y2 = cx + outer_r * math.cos(angle), cy + outer_r * math.sin(angle)
        major = i % (n_ticks // 4) == 0
        sw = 2.6 if major else 1.4
        c = color if i < filled else FAINT
        lines.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{c}" stroke-width="{sw}" stroke-linecap="round"/>'
        )
    return f'''<svg viewBox="0 0 {size} {size}" width="{size}" height="{size}">
        {"".join(lines)}
        <text x="{cx}" y="{cy - 2}" text-anchor="middle" font-family="IBM Plex Mono, monospace"
              font-size="36" font-weight="700" fill="{PAPER}">{value:.0f}%</text>
        <text x="{cx}" y="{cy + 24}" text-anchor="middle" font-family="IBM Plex Mono, monospace"
              font-size="10.5" letter-spacing="2.5" fill="{MUTED}">CHURN RISK</text>
    </svg>'''


def rung_bar_svg(pct_width: float, color: str, width: int = 200, height: int = 22, n: int = 22) -> str:
    """A magnitude bar built from countable ticks rather than a solid fill —
    every 5th tick drawn taller, echoing a ruled scale."""
    filled = round(pct_width / 100 * n)
    tick_w = width / n
    lines = []
    for i in range(n):
        x = i * tick_w + tick_w / 2
        tall = (i + 1) % 5 == 0
        h = height * (0.95 if tall else 0.55)
        y2 = height
        y1 = y2 - h
        c = color if i < filled else FAINT
        sw = 2.4 if tall else 1.5
        lines.append(
            f'<line x1="{x:.1f}" y1="{y1:.1f}" x2="{x:.1f}" y2="{y2:.1f}" '
            f'stroke="{c}" stroke-width="{sw}" stroke-linecap="round"/>'
        )
    return f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}">{"".join(lines)}</svg>'


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
    '<div class="eyebrow">CONNECTTEL ANALYTICS</div>'
    '<div class="headline">Churn Risk Terminal</div>'
    '<div class="subtitle">Live risk scoring for the retention team</div>',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────
# INPUT SECTION
# ─────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section"><div class="section-label">Customer Profile</div></div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0, step=1.0)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
with col2:
    total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 840.0, step=10.0)
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No"])

predict_clicked = st.button("Run Prediction \u2192")

# ─────────────────────────────────────────────────────────────────────────
# TERMINAL-STYLE PIPELINE RENDERER
# ─────────────────────────────────────────────────────────────────────────
STEPS = ["INPUT", "PREPROCESS", "SCORE", "RESULT"]


def render_pipeline(active_index, placeholder):
    """active_index: steps before this are 'done' [\u2713], this one is 'active'
    [\u25cf], the rest are 'pending' [\u00b7]. A hairline rule fills underneath
    proportionally, echoing a terminal boot-log progress bar."""
    parts = []
    for i, label in enumerate(STEPS):
        if i < active_index:
            mark, cls = "\u2713", "term-done"
        elif i == active_index:
            mark, cls = "\u25cf", "term-active"
        else:
            mark, cls = "\u00b7", "term-pending"
        parts.append(f'<span class="term-step {cls}">[{mark}] {label}</span>')
    progress = min(100, active_index / len(STEPS) * 100)
    # NOTE: built as a single line on purpose — a multi-line indented HTML
    # string passed to st.markdown can get misread as an indented code block
    # by the Markdown parser (see the note above render_pipeline's caller).
    html = (
        '<div class="section">'
        '<div class="section-label">Prediction Pipeline</div>'
        f'<div class="terminal-steps">{"".join(parts)}</div>'
        f'<div class="hairline-track"><div class="hairline-fill" style="width:{progress:.0f}%"></div></div>'
        '</div>'
    )
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
    churn_pct = proba * 100
    badge_color = RISK if is_high_risk else SAFE
    badge_text = "\u25b2 High Risk" if is_high_risk else "\u25bc Low Risk"

    # ── Result: badge + tick-ring gauge ─────────────────────────────────
    # Built as a single-line string — see the note in render_pipeline() on
    # why multi-line indented HTML passed to st.markdown is risky here.
    verdict_text = ("Recommend routing this customer to the retention team." if is_high_risk
                     else "Customer appears stable \u2014 no action needed.")
    result_html = (
        '<div class="section">'
        '<div class="section-label">Prediction Result</div>'
        f'<div class="badge" style="border-left:2px solid {badge_color}; color:{badge_color};">{badge_text}</div>'
        f'<div class="gauge-wrap">{tick_ring_svg(churn_pct)}</div>'
        f'<div class="verdict-note">{verdict_text}</div>'
        '</div>'
    )
    st.markdown(result_html, unsafe_allow_html=True)

    # ── Why This Score — rung-tick SHAP factor breakdown ────────────────
    factor_df = (
        pd.DataFrame({"feature": model_columns, "impact": row_shap})
        .assign(abs_impact=lambda d: d["impact"].abs())
        .sort_values("abs_impact", ascending=False)
        .head(5)
    )
    max_abs = factor_df["abs_impact"].max() or 1

    # Each row — and the assembled block below — is built as a single-line
    # string on purpose. st.markdown feeds this through a Markdown parser
    # before it becomes HTML, and a multi-line string with a blank line
    # followed by 4+ spaces of indentation gets misread as an *indented code
    # block* (a plain Markdown rule) instead of being passed through as HTML
    # — which is exactly what happened here during testing: the factor list
    # rendered as literal escaped tags instead of styled rows. Flattening
    # every dynamically-built fragment to one line sidesteps that rule
    # entirely, regardless of how many rows are joined together.
    row_parts = []
    for _, row in factor_df.iterrows():
        pct = (row["abs_impact"] / max_abs) * 100
        color = RISK if row["impact"] > 0 else INFO
        direction = "increases risk" if row["impact"] > 0 else "decreases risk"
        row_parts.append(
            '<div class="factor-row">'
            f'<div class="factor-label">{row["feature"]}</div>'
            '<div class="factor-viz">'
            f'{rung_bar_svg(pct, color)}'
            f'<span class="factor-dir" style="color:{color}">{direction}</span>'
            '</div>'
            '</div>'
        )
    rows_html = "".join(row_parts)

    factors_html = (
        '<div class="section">'
        '<div class="section-label">Why This Score \u2014 Top Contributing Factors</div>'
        f'{rows_html}'
        '<div class="footnote">Red = pushes risk up &nbsp;&nbsp;\u00b7&nbsp;&nbsp; Blue = pulls risk down. '
        "Computed live via shap.TreeExplainer on this customer's inputs.</div>"
        '</div>'
    )
    st.markdown(factors_html, unsafe_allow_html=True)

else:
    st.markdown(
        '<p style="color:#8b8a84; font-size:12.5px; margin-top:24px;">'
        "Fill in the customer profile above and run the prediction."
        "</p>",
        unsafe_allow_html=True,
    )
