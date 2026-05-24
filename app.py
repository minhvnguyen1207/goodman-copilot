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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Goodman Decision Co-Pilot",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background-color: #0f1117; color: #e2e8f0; }
  .stTabs [data-baseweb="tab-list"] { background-color: #1a1f2e; border-radius: 10px; padding: 4px; }
  .stTabs [data-baseweb="tab"] { color: #94a3b8; font-weight: 600; border-radius: 8px; }
  .stTabs [aria-selected="true"] { background-color: #f97316 !important; color: white !important; }
  div[data-testid="metric-container"] { background-color: #1a1f2e; border-radius: 10px; padding: 12px; border: 1px solid #2d3748; }
  .stButton > button { background-color: #f97316; color: white; font-weight: 700; border-radius: 8px; border: none; padding: 10px 24px; }
  .stButton > button:hover { background-color: #ea6c0a; }
  h1, h2, h3 { color: #f1f5f9 !important; }
  label { color: #94a3b8 !important; }
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
    st.markdown("<p style='margin:0;color:#64748b;font-size:13px'>AI-Powered Data Centre Site Intelligence · XGBoost + SHAP + Google Gemini</p>", unsafe_allow_html=True)
with col_badge:
    st.markdown("<div style='background:#064e3b;border:1px solid #34d399;border-radius:20px;padding:6px 14px;color:#34d399;font-size:12px;font-weight:700;margin-top:8px'>● Live · ML Active</div>", unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════
# 8 industry-accurate data centre site selection features
FEATURES = [
    "Power Capacity (MW)",
    "Power Cost (A$/MWh)",
    "Renewable Energy (%)",
    "Land Cost (A$M)",
    "Fibre Distance (km)",
    "Hyperscale Demand (0-10)",
    "Labour Market Score (0-10)",
    "Regulatory Ease (0-10)",
]

# ══════════════════════════════════════════════════════════════════════════════
# TRAIN XGBOOST MODEL  (cached — runs once on startup)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def train_model():
    """
    Train XGBoost on synthetic Goodman-style site data.
    Features reflect real data centre investment decision criteria.
    """
    np.random.seed(42)
    n = 80

    power_mw      = np.random.uniform(20, 300, n)     # MW grid capacity available
    power_cost    = np.random.uniform(50, 200, n)     # A$/MWh electricity tariff (lower=better)
    renewable_pct = np.random.uniform(0, 100, n)      # % renewable in local grid
    land_cost     = np.random.uniform(80, 600, n)     # A$M acquisition cost
    fibre_dist    = np.random.uniform(0.2, 20, n)     # km to nearest fibre exchange
    hyperscale    = np.random.uniform(0, 10, n)       # hyperscale LOI demand signals
    labour        = np.random.uniform(1, 10, n)       # tech workforce availability
    regulatory    = np.random.uniform(1, 10, n)       # planning/regulatory ease

    # Domain-informed scoring formula
    # Power cost is negative (cheaper = better for DC operating margins)
    # Renewable % is positive (hyperscalers require green energy commitments)
    raw = (
          25.0 * (power_mw / 300)          # capacity headroom
        + 22.0 * (hyperscale / 10)         # demand anchor
        - 20.0 * (power_cost / 200)        # electricity cost (key OpEx driver)
        + 12.0 * (renewable_pct / 100)     # green energy for ESG requirements
        - 10.0 * (land_cost / 600)         # acquisition cost
        -  8.0 * (fibre_dist / 20)         # connectivity latency
        +  8.0 * (labour / 10)             # workforce depth
        +  5.0 * (regulatory / 10)         # planning risk
        + np.random.normal(0, 3, n)
    )
    score = np.clip(raw * 1.9 + 38, 5, 100)

    X = pd.DataFrame({
        "Power Capacity (MW)":       power_mw,
        "Power Cost (A$/MWh)":       power_cost,
        "Renewable Energy (%)":      renewable_pct,
        "Land Cost (A$M)":           land_cost,
        "Fibre Distance (km)":       fibre_dist,
        "Hyperscale Demand (0-10)":  hyperscale,
        "Labour Market Score (0-10)":labour,
        "Regulatory Ease (0-10)":    regulatory,
    })
    y = pd.Series(score, name="site_score")

    model = xgb.XGBRegressor(
        n_estimators=50, max_depth=3, learning_rate=0.15,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0, base_score=0.5,
    )
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    return model, explainer, X, y

model, explainer, X_train, y_train = train_model()
feature_names = list(X_train.columns)

# ══════════════════════════════════════════════════════════════════════════════
# SITE PRESETS  (realistic values for each market)
# ══════════════════════════════════════════════════════════════════════════════
SITE_PRESETS = {
    "📍 Sydney Olympic Park, AU":     dict(power=120, power_cost=88,  renewable=32, land=285, fibre=2.1, hyperscale=8.5, labour=7, regulatory=7),
    "📍 Osaka Tech Corridor, JP":     dict(power=100, power_cost=162, renewable=22, land=310, fibre=3.0, hyperscale=8.0, labour=8, regulatory=6),
    "📍 Frankfurt West, DE":          dict(power=90,  power_cost=128, renewable=65, land=340, fibre=4.0, hyperscale=7.5, labour=7, regulatory=6),
    "📍 Melbourne Docklands, AU":     dict(power=75,  power_cost=95,  renewable=28, land=260, fibre=5.0, hyperscale=6.0, labour=6, regulatory=5),
    "📍 Singapore Jurong West, SG":   dict(power=60,  power_cost=112, renewable=4,  land=420, fibre=3.5, hyperscale=6.5, labour=8, regulatory=7),
    "📍 Dallas Plano Campus, US":     dict(power=140, power_cost=54,  renewable=35, land=195, fibre=8.0, hyperscale=5.5, labour=6, regulatory=6),
    "📍 Amsterdam Schiphol Zone, NL": dict(power=80,  power_cost=118, renewable=48, land=395, fibre=6.0, hyperscale=6.0, labour=7, regulatory=5),
    "✏️ Custom site (edit manually)": dict(power=100, power_cost=100, renewable=30, land=300, fibre=5.0, hyperscale=5.0, labour=6, regulatory=6),
}

# ══════════════════════════════════════════════════════════════════════════════
# GEMINI SETUP
# ══════════════════════════════════════════════════════════════════════════════
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

def get_gemini_response(question: str, site_ctx: dict, score: float, site_name: str) -> str:
    if not GEMINI_API_KEY:
        return "⚠️ Gemini API key not configured. Add GEMINI_API_KEY to Streamlit secrets."

    genai.configure(api_key=GEMINI_API_KEY)

    ctx = f"""You are the Goodman Decision Co-Pilot — an AI assistant for Goodman Group's
Investment Committee, specialising in data centre site acquisition.

CURRENT SITE: {site_name}
- Power Capacity:       {site_ctx['Power Capacity (MW)']:.0f} MW
- Electricity Cost:     A${site_ctx['Power Cost (A$/MWh)']:.0f}/MWh
- Renewable Energy:     {site_ctx['Renewable Energy (%)']:.0f}%
- Land Acquisition:     A${site_ctx['Land Cost (A$M)']:.0f}M
- Fibre Distance:       {site_ctx['Fibre Distance (km)']:.1f} km
- Hyperscale Demand:    {site_ctx['Hyperscale Demand (0-10)']:.1f}/10
- Labour Market:        {site_ctx['Labour Market Score (0-10)']:.1f}/10
- Regulatory Ease:      {site_ctx['Regulatory Ease (0-10)']:.1f}/10
- XGBoost AI Score:     {score:.1f}/100

Provide concise, IC-level analysis. For scenario questions (e.g. 'what if power costs rise?')
reason through the financial and scoring impact. Under 200 words. Be direct and data-driven."""

    prompt = f"{ctx}\n\nAnalyst question: {question}"

    # Try models in order until one works
    errors = []
    for model_name in ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-flash-latest", "gemini-pro"]:
        try:
            m = genai.GenerativeModel(model_name)
            resp = m.generate_content(prompt)
            return resp.text
        except Exception as e:
            errors.append(f"{model_name}: {str(e)[:80]}")
            continue

    error_detail = " | ".join(errors)
    return f"⚠️ Could not reach Gemini API. Please check your API key in Streamlit secrets.\n\nDebug: {error_detail}"

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
    st.markdown("<p style='color:#64748b;font-size:13px'>Select a candidate site — parameters auto-fill with market-accurate data. Adjust any value, then run the AI scoring.</p>", unsafe_allow_html=True)

    selected  = st.selectbox("🏗️ Select Candidate Site", list(SITE_PRESETS.keys()), index=0)
    preset    = SITE_PRESETS[selected]
    site_name = selected.replace("📍 ", "").replace("✏️ ", "")
    st.session_state["site_name"] = site_name
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**⚡ Power & Energy**")
        power_mw    = st.slider("Power Capacity (MW)", 20, 300, preset["power"],
                                 help="Total MW of grid capacity committed by utility operator")
        power_cost  = st.slider("Electricity Cost (A$/MWh)", 50, 200, preset["power_cost"],
                                 help="Wholesale electricity tariff — key OpEx driver for data centres. Lower = better margins.")
        renewable   = st.slider("Renewable Energy (%)", 0, 100, preset["renewable"],
                                 help="% of local grid powered by renewables. Hyperscalers (AWS, Azure, GCP) require green commitments.")
        land_cost   = st.number_input("Land Acquisition Cost (A$M)", 80, 600, preset["land"],
                                       help="Estimated freehold acquisition cost in AUD millions")

    with col2:
        st.markdown("**📡 Connectivity & Market**")
        fibre_dist  = st.slider("Fibre Distance (km)", 0.2, 20.0, float(preset["fibre"]), 0.1,
                                 help="Distance to nearest carrier-grade fibre exchange. Affects network latency.")
        hyperscale  = st.slider("Hyperscale Demand Signal (0–10)", 0.0, 10.0, float(preset["hyperscale"]), 0.5,
                                 help="0 = no interest, 10 = multiple confirmed LOIs from AWS / Azure / Google Cloud")
        labour      = st.slider("Labour Market Score (0–10)", 0, 10, preset["labour"],
                                 help="Availability of skilled data centre technicians and engineers in region")
        regulatory  = st.slider("Regulatory Ease (0–10)", 0, 10, preset["regulatory"],
                                 help="Speed and ease of obtaining planning approvals and environmental clearances")

    # Build input dict
    site_input = {
        "Power Capacity (MW)":        power_mw,
        "Power Cost (A$/MWh)":        power_cost,
        "Renewable Energy (%)":       renewable,
        "Land Cost (A$M)":            land_cost,
        "Fibre Distance (km)":        fibre_dist,
        "Hyperscale Demand (0-10)":   hyperscale,
        "Labour Market Score (0-10)": labour,
        "Regulatory Ease (0-10)":     regulatory,
    }
    X_input = pd.DataFrame([site_input])
    st.session_state["site_input"] = site_input
    st.session_state["X_input"]    = X_input

    st.markdown("---")
    if st.button("▶  Run AI Site Scoring", use_container_width=False):
        with st.spinner("XGBoost model processing..."):
            pred = float(np.clip(model.predict(X_input)[0], 0, 100))
            st.session_state["pred_score"] = pred

    if "pred_score" in st.session_state:
        score = st.session_state["pred_score"]
        if score >= 80:
            color, verdict, emoji = "#34d399", "Recommended",     "✅"
        elif score >= 65:
            color, verdict, emoji = "#fbbf24", "Review Required", "⚠️"
        else:
            color, verdict, emoji = "#f87171", "Not Recommended", "❌"

        st.markdown(f"""
        <div style='background:#1a1f2e;border:1px solid {color};border-radius:14px;
                    padding:24px;margin:16px 0;text-align:center'>
          <div style='font-size:12px;color:#94a3b8;font-weight:700;
                      text-transform:uppercase;letter-spacing:1px'>XGBoost AI Site Score</div>
          <div style='font-size:72px;font-weight:900;color:{color};
                      line-height:1.1;margin:8px 0'>{score:.1f}
            <span style='font-size:28px;color:#64748b'>/100</span></div>
          <div style='font-size:18px;font-weight:700;color:{color}'>{emoji} {verdict}</div>
          <div style='font-size:13px;color:#64748b;margin-top:6px'>{site_name}</div>
        </div>""", unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        irr = 8 + (score - 50) * 0.12
        with m1: st.metric("Power Capacity",   f"{power_mw} MW")
        with m2: st.metric("Electricity Cost", f"A${power_cost}/MWh",
                            delta="Competitive" if power_cost < 100 else "Above avg")
        with m3: st.metric("Renewable Energy", f"{renewable}%",
                            delta="Green ✓" if renewable >= 40 else "Below target")
        with m4: st.metric("Forecast IRR",     f"{irr:.1f}%",
                            delta="vs 12% hurdle" if irr >= 12 else "Below hurdle")

        st.info("💡 Switch to **AI Explanation** to see which factors drove this score, or **Co-Pilot Chat** to run scenario analysis.")
    else:
        st.markdown("<div style='background:#1a1f2e;border:1px dashed #2d3748;border-radius:14px;"
                    "padding:40px;text-align:center;color:#64748b'>"
                    "Click <strong style='color:#f97316'>▶ Run AI Site Scoring</strong> to generate the XGBoost prediction</div>",
                    unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: SHAP EXPLANATION
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### AI Decision Explanation — SHAP Analysis")
    st.markdown("""<p style='color:#64748b;font-size:13px'>
    SHAP (SHapley Additive exPlanations) shows exactly how much each feature pushed the AI score
    up or down from the global average — making the XGBoost model fully explainable, not a black box.
    </p>""", unsafe_allow_html=True)

    if "pred_score" not in st.session_state:
        st.warning("Run a site scoring in **Tab 1** first.")
    else:
        X_input   = st.session_state["X_input"]
        score     = st.session_state["pred_score"]
        site_name = st.session_state.get("site_name", "Candidate Site")

        with st.spinner("Calculating SHAP values..."):
            shap_values = explainer(X_input)
            sv       = shap_values[0].values
            base_val = float(shap_values[0].base_values)

        # Waterfall chart
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("#1a1f2e")
        ax.set_facecolor("#1a1f2e")
        colors = ["#34d399" if v >= 0 else "#f87171" for v in sv]
        bars = ax.barh(feature_names, sv, color=colors, height=0.55, edgecolor="none")
        for bar, val in zip(bars, sv):
            xp = bar.get_width() + (0.3 if val >= 0 else -0.3)
            ax.text(xp, bar.get_y() + bar.get_height()/2,
                    f"{'+' if val>=0 else ''}{val:.2f}",
                    va="center", ha="left" if val>=0 else "right",
                    color="#e2e8f0", fontsize=10, fontweight="bold")
        ax.axvline(0, color="#4b5563", linewidth=1.2, linestyle="--")
        ax.set_xlabel("SHAP Value (impact on AI score)", color="#94a3b8", fontsize=11)
        ax.tick_params(colors="#94a3b8", labelsize=10)
        ax.spines[["top","right","bottom","left"]].set_visible(False)
        for lbl in ax.get_yticklabels(): lbl.set_color("#e2e8f0")
        ax.set_title(f"{site_name}  |  Base: {base_val:.1f}  →  AI Score: {score:.1f}/100",
                     color="#f1f5f9", fontsize=13, fontweight="bold", pad=12)
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # Feature impact table
        st.markdown("#### Feature Impact Summary")
        shap_df = pd.DataFrame({
            "Feature":    feature_names,
            "SHAP Value": [f"{'+' if v>=0 else ''}{v:.3f}" for v in sv],
            "Direction":  ["▲ Positive" if v>=0 else "▼ Negative" for v in sv],
        }).sort_values("SHAP Value", key=lambda x: x.str.replace("+","").astype(float).abs(), ascending=False).reset_index(drop=True)
        st.dataframe(shap_df, use_container_width=True, hide_index=True)

        # Global feature importance
        st.markdown("#### Global Model Feature Importance (trained on 80 sites)")
        imp = model.feature_importances_
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        fig2.patch.set_facecolor("#1a1f2e")
        ax2.set_facecolor("#1a1f2e")
        idx = np.argsort(imp)
        ax2.barh([feature_names[i] for i in idx], [imp[i] for i in idx],
                 color="#60a5fa", height=0.5, edgecolor="none")
        ax2.set_xlabel("Feature Importance Score", color="#94a3b8", fontsize=11)
        ax2.tick_params(colors="#94a3b8", labelsize=10)
        ax2.spines[["top","right","bottom","left"]].set_visible(False)
        for lbl in ax2.get_yticklabels(): lbl.set_color("#e2e8f0")
        ax2.set_title("XGBoost Global Feature Importance", color="#f1f5f9",
                      fontsize=13, fontweight="bold", pad=12)
        st.pyplot(fig2, use_container_width=True)
        plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: CO-PILOT CHAT
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### Co-Pilot Chat — Powered by Google Gemini AI")
    st.markdown("""<p style='color:#64748b;font-size:13px'>
    Ask the Co-Pilot to analyse scenarios, explain the AI score, compare sites, or draft IC memo language.
    Each response uses live site data from Tab 1 as context.
    </p>""", unsafe_allow_html=True)

    with st.expander("🔑 Configure Gemini API Key", expanded=not GEMINI_API_KEY):
        api_key_input = st.text_input("Paste your Google Gemini API key", type="password",
                                       value=GEMINI_API_KEY,
                                       help="Get free key at aistudio.google.com → Get API Key")
        if api_key_input:
            os.environ["GEMINI_API_KEY"] = api_key_input
            GEMINI_API_KEY = api_key_input
            st.success("API key saved for this session.")

    if "pred_score" not in st.session_state:
        st.warning("Score a site in **Tab 1** first — Co-Pilot uses that data as context.")
    else:
        site_input = st.session_state["site_input"]
        score      = st.session_state["pred_score"]
        site_name  = st.session_state.get("site_name", "Candidate Site")

        st.markdown(f"""
        <div style='background:#1a1f2e;border:1px solid #2d3748;border-radius:10px;
                    padding:12px 16px;margin-bottom:16px;display:flex;gap:20px;flex-wrap:wrap'>
          <div><span style='color:#64748b;font-size:11px'>ACTIVE SITE</span><br>
               <span style='color:#f97316;font-weight:700'>{site_name}</span></div>
          <div><span style='color:#64748b;font-size:11px'>AI SCORE</span><br>
               <span style='color:#34d399;font-weight:700;font-size:18px'>{score:.1f}/100</span></div>
          <div><span style='color:#64748b;font-size:11px'>POWER COST</span><br>
               <span style='color:#e2e8f0;font-weight:600'>A${site_input['Power Cost (A$/MWh)']:.0f}/MWh</span></div>
          <div><span style='color:#64748b;font-size:11px'>RENEWABLE</span><br>
               <span style='color:#e2e8f0;font-weight:600'>{site_input['Renewable Energy (%)']:.0f}%</span></div>
          <div><span style='color:#64748b;font-size:11px'>HYPERSCALE</span><br>
               <span style='color:#e2e8f0;font-weight:600'>{site_input['Hyperscale Demand (0-10)']:.1f}/10</span></div>
          <div><span style='color:#64748b;font-size:11px'>AI MODEL</span><br>
               <span style='color:#60a5fa;font-weight:600'>Google Gemini 2.0 Flash</span></div>
        </div>""", unsafe_allow_html=True)

        # Suggested questions
        st.markdown("**Suggested questions:**")
        suggestions = [
            "What if electricity costs rise by 20%?",
            "How does the renewable % affect our ESG commitments?",
            "What are the top 3 risks for this site?",
            "Compare this site to one with 60% renewable energy",
            "What would most improve this site's score?",
            "Draft a one-paragraph IC recommendation",
        ]
        if "chat_prefill" not in st.session_state:
            st.session_state["chat_prefill"] = ""
        sq_cols = st.columns(3)
        for i, sq in enumerate(suggestions):
            with sq_cols[i % 3]:
                if st.button(sq, key=f"sq_{i}", use_container_width=True):
                    st.session_state["chat_prefill"] = sq

        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        for msg in st.session_state["chat_history"]:
            with st.chat_message(msg["role"], avatar="🤖" if msg["role"]=="assistant" else "👤"):
                st.markdown(msg["content"])

        user_msg = st.chat_input("Ask the Co-Pilot — e.g. 'What if power costs drop to A$60/MWh?'")
        if st.session_state["chat_prefill"] and not user_msg:
            user_msg = st.session_state["chat_prefill"]
            st.session_state["chat_prefill"] = ""

        if user_msg:
            st.session_state["chat_history"].append({"role":"user","content":user_msg})
            with st.chat_message("user", avatar="👤"):
                st.markdown(user_msg)
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Gemini AI analysing..."):
                    response = get_gemini_response(user_msg, site_input, score, site_name)
                st.markdown(response)
            st.session_state["chat_history"].append({"role":"assistant","content":response})

        if st.session_state.get("chat_history"):
            if st.button("🗑 Clear chat history"):
                st.session_state["chat_history"] = []
                st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""<div style='text-align:center;color:#374151;font-size:12px'>
  Goodman Decision Co-Pilot · ISYS3443 Assessment 3 · XGBoost · SHAP · Google Gemini AI · Streamlit<br>
  <span style='color:#1f2937'>AI output is for analytical support only. All investment decisions require human IC approval.</span>
</div>""", unsafe_allow_html=True)
