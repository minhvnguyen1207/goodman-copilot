import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import google.generativeai as genai
import os
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Goodman Decision Co-Pilot",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main { background-color: #0f1117; }
  .stApp { background-color: #0f1117; color: #e2e8f0; }
  .stTabs [data-baseweb="tab-list"] { background-color: #1a1f2e; border-radius: 10px; padding: 4px; }
  .stTabs [data-baseweb="tab"] { color: #94a3b8; font-weight: 600; border-radius: 8px; }
  .stTabs [aria-selected="true"] { background-color: #f97316 !important; color: white !important; }
  div[data-testid="metric-container"] { background-color: #1a1f2e; border-radius: 10px; padding: 12px; border: 1px solid #2d3748; }
  .stButton > button { background-color: #f97316; color: white; font-weight: 700; border-radius: 8px; border: none; padding: 10px 24px; }
  .stButton > button:hover { background-color: #ea6c0a; }
  .st-chat-message { background-color: #1a1f2e; border-radius: 10px; }
  h1, h2, h3 { color: #f1f5f9 !important; }
  label { color: #94a3b8 !important; }
  .stSlider > div { color: #94a3b8; }
  .stNumberInput > div > div > input { background-color: #1a1f2e; color: #e2e8f0; border: 1px solid #2d3748; }
  .stSelectbox > div > div { background-color: #1a1f2e; color: #e2e8f0; }
  .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
col_logo, col_title, col_badge = st.columns([1, 8, 2])
with col_logo:
    st.markdown("<div style='background:#f97316;width:48px;height:48px;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:24px;font-weight:800;color:white;margin-top:4px'>G</div>", unsafe_allow_html=True)
with col_title:
    st.markdown("<h1 style='margin:0;font-size:26px'>Goodman Decision Co-Pilot</h1>", unsafe_allow_html=True)
    st.markdown("<p style='margin:0;color:#64748b;font-size:13px'>AI-Powered Data Centre Site Intelligence · XGBoost + Google Gemini</p>", unsafe_allow_html=True)
with col_badge:
    st.markdown("<div style='background:#064e3b;border:1px solid #34d399;border-radius:20px;padding:6px 14px;color:#34d399;font-size:12px;font-weight:700;margin-top:8px'>● Live · ML Active</div>", unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# TRAIN XGBOOST MODEL  (runs once, cached)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def train_model():
    """
    Train XGBoost on synthetic Goodman-style site acquisition data.
    Features reflect the real factors used in data centre site selection.
    """
    np.random.seed(42)
    n = 80

    # Simulate realistic distributions for each feature
    power_mw          = np.random.uniform(20, 200, n)          # MW available
    land_cost_m       = np.random.uniform(80, 600, n)          # A$M acquisition cost
    zoning_score      = np.random.uniform(1, 10, n)            # planning/zoning favourability
    fibre_dist_km     = np.random.uniform(0.2, 20, n)          # km to nearest fibre exchange
    hyperscale_demand = np.random.uniform(0, 10, n)            # LOI signals (0=none, 10=multiple)
    labour_score      = np.random.uniform(1, 10, n)            # tech labour availability
    flood_risk        = np.random.uniform(1, 10, n)            # 1=high risk, 10=very low risk
    regulatory_ease   = np.random.uniform(1, 10, n)            # planning/regulatory complexity

    # Ground-truth score formula (domain-informed weights, calibrated to 0-100 range)
    raw = (
          28.0 * (power_mw / 200)
        + 24.0 * (hyperscale_demand / 10)
        - 16.0 * (land_cost_m / 600)
        + 12.0 * (zoning_score / 10)
        -  8.0 * (fibre_dist_km / 20)
        +  8.0 * (labour_score / 10)
        +  5.0 * (flood_risk / 10)
        +  5.0 * (regulatory_ease / 10)
        + np.random.normal(0, 3, n)
    )
    # Scale so average site ~62, excellent site ~88
    score = np.clip(raw * 1.35 + 22, 5, 100)

    X = pd.DataFrame({
        "Power Capacity (MW)":      power_mw,
        "Land Cost (A$M)":          land_cost_m,
        "Zoning Score":             zoning_score,
        "Fibre Distance (km)":      fibre_dist_km,
        "Hyperscale Demand":        hyperscale_demand,
        "Labour Market Score":      labour_score,
        "Flood Risk Score":         flood_risk,
        "Regulatory Ease":          regulatory_ease,
    })
    y = pd.Series(score, name="site_score")

    model = xgb.XGBRegressor(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.15,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
        base_score=0.5,
    )
    model.fit(X, y)

    # Build SHAP explainer once
    explainer = shap.TreeExplainer(model)

    return model, explainer, X, y

model, explainer, X_train, y_train = train_model()
feature_names = list(X_train.columns)

# ══════════════════════════════════════════════════════════════════════════════
# GEMINI SETUP
# ══════════════════════════════════════════════════════════════════════════════
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

def get_gemini_response(user_question: str, site_context: dict, score: float) -> str:
    """Send site data + user question to Gemini and return the response."""
    if not GEMINI_API_KEY:
        return "⚠️ Gemini API key not configured. Add GEMINI_API_KEY to your Streamlit secrets to enable AI chat."

    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-2.0-flash")

    system_context = f"""You are the Goodman Decision Co-Pilot, an AI assistant for Goodman Group's
Investment Committee. You help analysts evaluate data centre site acquisition candidates.

CURRENT SITE DATA:
- Power Capacity: {site_context['Power Capacity (MW)']:.0f} MW
- Land Cost: A${site_context['Land Cost (A$M)']:.0f}M
- Zoning Score: {site_context['Zoning Score']:.1f}/10
- Fibre Distance: {site_context['Fibre Distance (km)']:.1f} km
- Hyperscale Demand Signals: {site_context['Hyperscale Demand']:.1f}/10
- Labour Market Score: {site_context['Labour Market Score']:.1f}/10
- Flood Risk Score: {site_context['Flood Risk Score']:.1f}/10 (10=lowest risk)
- Regulatory Ease: {site_context['Regulatory Ease']:.1f}/10
- XGBoost AI Score: {score:.1f}/100

Provide concise, investment-committee-level analysis. When asked about scenarios
(e.g. 'what if power costs rise?'), reason through the impact on the site score and
IRR. Keep responses under 200 words. Be direct and data-driven."""

    response = gemini_model.generate_content(f"{system_context}\n\nAnalyst question: {user_question}")
    return response.text


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "🏗️  Site Scoring  (XGBoost ML)",
    "⚡  AI Explanation  (SHAP)",
    "💬  Co-Pilot Chat  (Gemini AI)",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: SITE SCORING
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Input Site Parameters")
    st.markdown("<p style='color:#64748b;font-size:13px'>Enter candidate site data below. The XGBoost model will score the site in real time.</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Infrastructure**")
        power_mw      = st.slider("Power Capacity (MW)", 20, 200, 120,
                                   help="Total MW committed by grid operator")
        fibre_dist    = st.slider("Fibre Distance (km)", 0.2, 20.0, 2.1, 0.1,
                                   help="Distance to nearest carrier-grade fibre exchange")
        flood_risk    = st.slider("Flood Risk Score (1=high risk, 10=very low)", 1, 10, 8,
                                   help="Based on climate risk assessment")
        land_cost     = st.number_input("Land Cost (A$M)", 80, 600, 285,
                                         help="Estimated acquisition cost in AUD millions")

    with col2:
        st.markdown("**Market & Regulatory**")
        hyperscale    = st.slider("Hyperscale Demand Signal (0–10)", 0.0, 10.0, 8.5, 0.5,
                                   help="0=no LOIs, 10=multiple confirmed LOIs from AWS/Azure/GCP")
        zoning        = st.slider("Zoning Score (1–10)", 1, 10, 8,
                                   help="Planning/zoning favourability for data centre use")
        labour        = st.slider("Labour Market Score (1–10)", 1, 10, 7,
                                   help="Availability of skilled tech workers in region")
        regulatory    = st.slider("Regulatory Ease (1–10)", 1, 10, 7,
                                   help="Ease of obtaining planning approvals")

    # Build input vector
    site_input = {
        "Power Capacity (MW)":   power_mw,
        "Land Cost (A$M)":       land_cost,
        "Zoning Score":          zoning,
        "Fibre Distance (km)":   fibre_dist,
        "Hyperscale Demand":     hyperscale,
        "Labour Market Score":   labour,
        "Flood Risk Score":      flood_risk,
        "Regulatory Ease":       regulatory,
    }
    X_input = pd.DataFrame([site_input])

    # Store in session state for other tabs
    st.session_state["site_input"] = site_input
    st.session_state["X_input"]    = X_input

    st.markdown("---")

    if st.button("▶  Run AI Site Scoring", use_container_width=False):
        with st.spinner("XGBoost model processing..."):
            pred_score = float(model.predict(X_input)[0])
            pred_score = np.clip(pred_score, 0, 100)
            st.session_state["pred_score"] = pred_score

    # Show results if scored
    if "pred_score" in st.session_state:
        score = st.session_state["pred_score"]

        # Score colour
        if score >= 80:
            color, verdict, emoji = "#34d399", "Recommended", "✅"
        elif score >= 65:
            color, verdict, emoji = "#fbbf24", "Review Required", "⚠️"
        else:
            color, verdict, emoji = "#f87171", "Not Recommended", "❌"

        st.markdown(f"""
        <div style='background:#1a1f2e;border:1px solid {color};border-radius:14px;padding:24px;margin:16px 0;text-align:center'>
          <div style='font-size:13px;color:#94a3b8;font-weight:700;text-transform:uppercase;letter-spacing:1px'>XGBoost AI Site Score</div>
          <div style='font-size:72px;font-weight:900;color:{color};line-height:1.1;margin:8px 0'>{score:.1f}<span style='font-size:28px;color:#64748b'>/100</span></div>
          <div style='font-size:18px;font-weight:700;color:{color}'>{emoji} {verdict}</div>
        </div>
        """, unsafe_allow_html=True)

        # KPI metrics
        m1, m2, m3, m4 = st.columns(4)
        with m1: st.metric("Power Capacity", f"{power_mw} MW", delta="Grid committed" if power_mw >= 100 else None)
        with m2: st.metric("Land Cost",      f"A${land_cost}M", delta=f"{'Below' if land_cost < 300 else 'Above'} median")
        with m3: st.metric("Hyperscale LOIs", f"{hyperscale:.1f}/10", delta="Strong demand" if hyperscale >= 7 else None)
        with m4: st.metric("Forecast IRR",   f"{10 + (score - 60) * 0.08:.1f}%", delta="vs 12% hurdle")

        st.info("💡 Switch to the **AI Explanation** tab to see which features drove this score, or the **Co-Pilot Chat** tab to run scenario analysis.")

    else:
        st.markdown("<div style='background:#1a1f2e;border:1px dashed #2d3748;border-radius:14px;padding:40px;text-align:center;color:#64748b'>Click <strong style='color:#f97316'>▶ Run AI Site Scoring</strong> to generate the XGBoost prediction</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: SHAP EXPLANATION
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### AI Decision Explanation — SHAP Analysis")
    st.markdown("""<p style='color:#64748b;font-size:13px'>
    SHAP (SHapley Additive exPlanations) shows exactly how much each feature
    pushed the AI score up or down from the global average. This makes the
    XGBoost model fully explainable — not a black box.
    </p>""", unsafe_allow_html=True)

    if "X_input" not in st.session_state:
        st.warning("Please go to **Site Scoring** tab first and run a prediction.")
    else:
        X_input     = st.session_state["X_input"]
        site_input  = st.session_state["site_input"]
        score       = st.session_state.get("pred_score", None)

        if score is None:
            st.warning("Run the AI scoring first (Tab 1) to generate SHAP values.")
        else:
            with st.spinner("Calculating SHAP values..."):
                shap_values = explainer(X_input)
                sv          = shap_values[0].values   # shape: (n_features,)
                base_val    = float(shap_values[0].base_values)

            # ── SHAP Waterfall Bar Chart ──────────────────────────────────
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor("#1a1f2e")
            ax.set_facecolor("#1a1f2e")

            colors = ["#34d399" if v >= 0 else "#f87171" for v in sv]
            bars   = ax.barh(feature_names, sv, color=colors, height=0.55, edgecolor="none")

            # Labels on bars
            for bar, val in zip(bars, sv):
                x_pos = bar.get_width() + (0.3 if val >= 0 else -0.3)
                ha    = "left" if val >= 0 else "right"
                ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                        f"{'+' if val >= 0 else ''}{val:.2f}",
                        va="center", ha=ha, color="#e2e8f0", fontsize=10, fontweight="bold")

            ax.axvline(0, color="#4b5563", linewidth=1.2, linestyle="--")
            ax.set_xlabel("SHAP Value (impact on AI score)", color="#94a3b8", fontsize=11)
            ax.tick_params(colors="#94a3b8", labelsize=10)
            ax.spines[["top","right","bottom","left"]].set_visible(False)
            for label in ax.get_yticklabels():
                label.set_color("#e2e8f0")
            ax.set_title(f"Feature Contributions  |  Base: {base_val:.1f}  →  Score: {score:.1f}",
                         color="#f1f5f9", fontsize=13, fontweight="bold", pad=12)

            st.pyplot(fig, use_container_width=True)
            plt.close()

            # ── Feature importance ranking ────────────────────────────────
            st.markdown("#### Feature Impact Summary")
            shap_df = pd.DataFrame({
                "Feature":    feature_names,
                "SHAP Value": sv,
                "Direction":  ["▲ Positive" if v >= 0 else "▼ Negative" for v in sv],
            }).sort_values("SHAP Value", key=abs, ascending=False).reset_index(drop=True)

            shap_df["SHAP Value"] = shap_df["SHAP Value"].map(lambda x: f"{'+' if x >= 0 else ''}{x:.3f}")
            st.dataframe(shap_df, use_container_width=True, hide_index=True)

            # ── Global feature importance ─────────────────────────────────
            st.markdown("#### Global Model Feature Importance (trained on 80 sites)")
            imp = model.feature_importances_
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            fig2.patch.set_facecolor("#1a1f2e")
            ax2.set_facecolor("#1a1f2e")
            sorted_idx = np.argsort(imp)
            ax2.barh([feature_names[i] for i in sorted_idx],
                     [imp[i] for i in sorted_idx],
                     color="#60a5fa", height=0.5, edgecolor="none")
            ax2.set_xlabel("Feature Importance Score", color="#94a3b8", fontsize=11)
            ax2.tick_params(colors="#94a3b8", labelsize=10)
            ax2.spines[["top","right","bottom","left"]].set_visible(False)
            for label in ax2.get_yticklabels():
                label.set_color("#e2e8f0")
            ax2.set_title("XGBoost Global Feature Importance", color="#f1f5f9", fontsize=13,
                          fontweight="bold", pad=12)
            st.pyplot(fig2, use_container_width=True)
            plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: CO-PILOT CHAT (GEMINI)
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### Co-Pilot Chat — Powered by Google Gemini AI")
    st.markdown("""<p style='color:#64748b;font-size:13px'>
    Ask the Co-Pilot to run scenario analyses, explain the AI score, compare sites,
    or draft IC memo language. Each response uses the live site data from Tab 1 as context.
    </p>""", unsafe_allow_html=True)

    # API key input
    with st.expander("🔑 Configure Gemini API Key", expanded=not GEMINI_API_KEY):
        api_key_input = st.text_input(
            "Paste your Google Gemini API key",
            type="password",
            value=GEMINI_API_KEY,
            help="Get a free key at aistudio.google.com → Get API Key"
        )
        if api_key_input:
            os.environ["GEMINI_API_KEY"] = api_key_input
            GEMINI_API_KEY = api_key_input
            st.success("API key saved for this session.")

    if "pred_score" not in st.session_state:
        st.warning("Please score a site in **Tab 1** first — the Co-Pilot uses that site data as context.")
    else:
        site_input = st.session_state["site_input"]
        score      = st.session_state["pred_score"]

        # Display active site context
        st.markdown(f"""
        <div style='background:#1a1f2e;border:1px solid #2d3748;border-radius:10px;padding:12px 16px;margin-bottom:16px;display:flex;gap:20px;flex-wrap:wrap'>
          <div><span style='color:#64748b;font-size:11px'>ACTIVE SITE</span><br><span style='color:#f97316;font-weight:700'>Current Site Analysis</span></div>
          <div><span style='color:#64748b;font-size:11px'>AI SCORE</span><br><span style='color:#34d399;font-weight:700;font-size:18px'>{score:.1f}/100</span></div>
          <div><span style='color:#64748b;font-size:11px'>POWER</span><br><span style='color:#e2e8f0;font-weight:600'>{site_input['Power Capacity (MW)']:.0f} MW</span></div>
          <div><span style='color:#64748b;font-size:11px'>LAND COST</span><br><span style='color:#e2e8f0;font-weight:600'>A${site_input['Land Cost (A$M)']:.0f}M</span></div>
          <div><span style='color:#64748b;font-size:11px'>HYPERSCALE DEMAND</span><br><span style='color:#e2e8f0;font-weight:600'>{site_input['Hyperscale Demand']:.1f}/10</span></div>
          <div><span style='color:#64748b;font-size:11px'>AI MODEL</span><br><span style='color:#60a5fa;font-weight:600'>Google Gemini 1.5 Flash</span></div>
        </div>
        """, unsafe_allow_html=True)

        # Suggested questions
        st.markdown("**Suggested questions:**")
        sq_cols = st.columns(3)
        suggestions = [
            "What if power costs rise by 20%?",
            "What are the top 3 risks for this site?",
            "Summarise this site for the board",
            "Compare this site to a site with score 65",
            "What would improve this site's score most?",
            "Draft a one-paragraph IC recommendation",
        ]
        if "chat_prefill" not in st.session_state:
            st.session_state["chat_prefill"] = ""

        for i, sq in enumerate(suggestions):
            with sq_cols[i % 3]:
                if st.button(sq, key=f"sq_{i}", use_container_width=True):
                    st.session_state["chat_prefill"] = sq

        # Chat history
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        # Display chat
        for msg in st.session_state["chat_history"]:
            with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "👤"):
                st.markdown(msg["content"])

        # Chat input
        user_msg = st.chat_input("Ask the Co-Pilot — e.g. 'What if hyperscale demand drops to 4?'")

        # Handle suggested question prefill
        if st.session_state["chat_prefill"] and not user_msg:
            user_msg = st.session_state["chat_prefill"]
            st.session_state["chat_prefill"] = ""

        if user_msg:
            st.session_state["chat_history"].append({"role": "user", "content": user_msg})
            with st.chat_message("user", avatar="👤"):
                st.markdown(user_msg)

            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Gemini AI analysing..."):
                    response = get_gemini_response(user_msg, site_input, score)
                st.markdown(response)

            st.session_state["chat_history"].append({"role": "assistant", "content": response})

        # Clear chat button
        if st.session_state.get("chat_history"):
            if st.button("🗑 Clear chat history"):
                st.session_state["chat_history"] = []
                st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#374151;font-size:12px'>
  Goodman Decision Co-Pilot · ISYS3443 Assessment 3 · Built with XGBoost, SHAP, Google Gemini AI & Streamlit<br>
  <span style='color:#1f2937'>AI output is for analytical support only. All investment decisions require human IC approval.</span>
</div>
""", unsafe_allow_html=True)
