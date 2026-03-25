import streamlit as st
import pickle
import pandas as pd

# --- Page Config ---
st.set_page_config(
    page_title="F1 DNF Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Model ---
with open("classifier.pkl", "rb") as file:
    model = pickle.load(file)

# --- Custom CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;800&family=Barlow:wght@300;400;500&display=swap');

    /* ── Global ── */
    html, body, [class*="css"] {
        font-family: 'Barlow', sans-serif;
    }
    .stApp {
        background-color: #0d0d0d;
        background-image: radial-gradient(ellipse at 80% 10%, #1a0505 0%, transparent 60%),
                          radial-gradient(ellipse at 20% 90%, #0a0a1a 0%, transparent 60%);
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background-color: #111111;
        border-right: 1px solid #2a2a2a;
    }
    section[data-testid="stSidebar"] .stNumberInput label,
    section[data-testid="stSidebar"] .stSlider label {
        color: #aaaaaa !important;
        font-family: 'Barlow', sans-serif;
        font-size: 0.78rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    section[data-testid="stSidebar"] input {
        background-color: #1c1c1c !important;
        border: 1px solid #333 !important;
        color: #f0f0f0 !important;
        border-radius: 4px !important;
    }
    section[data-testid="stSidebar"] input:focus {
        border-color: #e10600 !important;
        box-shadow: 0 0 0 2px rgba(225, 6, 0, 0.2) !important;
    }
    .sidebar-section-label {
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #e10600;
        padding: 12px 0 4px 0;
        border-top: 1px solid #2a2a2a;
        margin-top: 8px;
    }

    /* ── Hero Header ── */
    .hero {
        padding: 2.5rem 0 1.5rem 0;
        border-bottom: 1px solid #1e1e1e;
        margin-bottom: 2rem;
    }
    .hero-eyebrow {
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.25em;
        text-transform: uppercase;
        color: #e10600;
        margin-bottom: 0.4rem;
    }
    .hero-title {
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 3.8rem;
        font-weight: 800;
        line-height: 1;
        color: #f5f5f5;
        letter-spacing: -0.01em;
        margin: 0;
    }
    .hero-title span {
        color: #e10600;
    }
    .hero-subtitle {
        font-family: 'Barlow', sans-serif;
        font-size: 0.95rem;
        font-weight: 300;
        color: #666;
        margin-top: 0.75rem;
        letter-spacing: 0.02em;
    }

    /* ── Feature Cards ── */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1px;
        background: #1e1e1e;
        border: 1px solid #1e1e1e;
        border-radius: 8px;
        overflow: hidden;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: #111;
        padding: 1.25rem 1.5rem;
        text-align: center;
    }
    .feature-card .value {
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 2rem;
        font-weight: 800;
        color: #e10600;
        line-height: 1;
    }
    .feature-card .label {
        font-size: 0.7rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #555;
        margin-top: 0.3rem;
    }

    /* ── Input Summary Table ── */
    .summary-header {
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: #555;
        margin-bottom: 0.75rem;
    }
    .stDataFrame {
        border: 1px solid #1e1e1e !important;
        border-radius: 6px !important;
    }
    .stDataFrame thead tr th {
        background-color: #161616 !important;
        color: #aaa !important;
        font-size: 0.72rem !important;
        letter-spacing: 0.08em !important;
    }
    .stDataFrame tbody tr td {
        background-color: #111 !important;
        color: #e0e0e0 !important;
        font-family: 'Barlow Condensed', sans-serif !important;
        font-size: 0.95rem !important;
    }

    /* ── Predict Button ── */
    .stButton > button {
        background-color: #e10600 !important;
        color: white !important;
        font-family: 'Barlow Condensed', sans-serif !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.15em !important;
        text-transform: uppercase !important;
        border: none !important;
        border-radius: 4px !important;
        padding: 0.75rem 2.5rem !important;
        width: 100% !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background-color: #ff1a10 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(225, 6, 0, 0.4) !important;
    }
    .stButton > button:active {
        transform: translateY(0) !important;
    }

    /* ── Result Cards ── */
    .result-card {
        border-radius: 8px;
        padding: 2.5rem 2rem;
        text-align: center;
        margin-top: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .result-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
    }
    .result-dnf {
        background: linear-gradient(135deg, #1a0505 0%, #110000 100%);
        border: 1px solid #3a0a0a;
    }
    .result-dnf::before { background: #e10600; }
    .result-finish {
        background: linear-gradient(135deg, #021a06 0%, #001108 100%);
        border: 1px solid #0a3a12;
    }
    .result-finish::before { background: #00c851; }
    .result-icon {
        font-size: 3.5rem;
        line-height: 1;
        margin-bottom: 0.75rem;
    }
    .result-label {
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.3em;
        text-transform: uppercase;
        color: #555;
        margin-bottom: 0.5rem;
    }
    .result-title-dnf {
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        color: #e10600;
        letter-spacing: -0.01em;
        line-height: 1;
    }
    .result-title-finish {
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        color: #00c851;
        letter-spacing: -0.01em;
        line-height: 1;
    }
    .result-sub {
        font-size: 0.8rem;
        color: #555;
        margin-top: 0.75rem;
        letter-spacing: 0.04em;
    }

    /* ── Divider ── */
    hr { border-color: #1e1e1e !important; }

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 1rem !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════
#  SIDEBAR — Inputs
# ══════════════════════════════════════════
with st.sidebar:
    st.markdown("""
        <div style='padding: 1.2rem 0 1rem 0;'>
            <div style='font-family: Barlow Condensed, sans-serif; font-size: 1.4rem;
                        font-weight: 800; color: #f5f5f5; letter-spacing: 0.02em;'>
                🏎️ RACE INPUTS
            </div>
            <div style='font-size: 0.72rem; color: #444; margin-top: 0.25rem;
                        letter-spacing: 0.05em; text-transform: uppercase;'>
                Enter details to predict
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-label">📅 Season</div>', unsafe_allow_html=True)
    year   = st.number_input("Year",  min_value=1950, max_value=2030, value=2023)
    round_ = st.number_input("Round", min_value=1,    max_value=25,   value=1)

    st.markdown('<div class="sidebar-section-label">🏁 Race Details</div>', unsafe_allow_html=True)
    grid          = st.number_input("Grid Position",           min_value=0,   max_value=100,  value=1)
    positionOrder = st.number_input("Finishing Position Order",min_value=1,   max_value=100,  value=1)
    points        = st.number_input("Points Scored",           min_value=0.0, max_value=100.0,value=0.0)
    laps          = st.number_input("Laps Completed",          min_value=0,   max_value=1000, value=50)

    st.markdown('<div class="sidebar-section-label">🗺️ Circuit</div>', unsafe_allow_html=True)
    circuitId = st.number_input("Circuit ID",         min_value=1,   max_value=100, value=1)
    lat       = st.number_input("Latitude",           value=0.0,     format="%.4f")
    lng       = st.number_input("Longitude",          value=0.0,     format="%.4f")
    alt       = st.number_input("Altitude (m)",       value=0.0,     format="%.1f")

    st.markdown('<div class="sidebar-section-label">👤 Driver</div>', unsafe_allow_html=True)
    age_at_race = st.number_input("Age at Race", min_value=15.0, max_value=60.0, value=28.0, format="%.1f")

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("⚡ PREDICT", use_container_width=True)


# ══════════════════════════════════════════
#  MAIN — Hero + Content
# ══════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Formula 1 · Machine Learning</div>
    <div class="hero-title">DNF <span>PREDICTOR</span></div>
    <div class="hero-subtitle">Will the driver see the chequered flag — or retire from the race?</div>
</div>
""", unsafe_allow_html=True)

# Stats strip
st.markdown("""
<div class="feature-grid">
    <div class="feature-card">
        <div class="value">11</div>
        <div class="label">Input Features</div>
    </div>
    <div class="feature-card">
        <div class="value">2</div>
        <div class="label">Outcomes</div>
    </div>
    <div class="feature-card">
        <div class="value">1950–</div>
        <div class="label">Data Range</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Two-column layout
col1, col2 = st.columns([1.1, 1], gap="large")

with col1:
    st.markdown('<div class="summary-header">📋 Input Summary</div>', unsafe_allow_html=True)
    input_data = pd.DataFrame({
        "year": [year], "round": [round_], "grid": [grid],
        "positionOrder": [positionOrder], "points": [points], "laps": [laps],
        "circuitId": [circuitId], "lat": [lat], "lng": [lng],
        "alt": [alt], "age_at_race": [age_at_race]
    })
    st.dataframe(input_data.T.rename(columns={0: "Value"}), use_container_width=True, height=420)

with col2:
    st.markdown('<div class="summary-header">🔮 Prediction</div>', unsafe_allow_html=True)

    if not predict_btn:
        st.markdown("""
        <div style='border: 1px dashed #222; border-radius: 8px; padding: 3rem 2rem;
                    text-align: center; margin-top: 0;'>
            <div style='font-size: 2.5rem; margin-bottom: 1rem;'>🏁</div>
            <div style='font-family: Barlow Condensed, sans-serif; font-size: 0.7rem;
                        letter-spacing: 0.2em; text-transform: uppercase; color: #333;'>
                Awaiting race data
            </div>
            <div style='font-size: 0.8rem; color: #2a2a2a; margin-top: 0.5rem;'>
                Fill in inputs and hit PREDICT
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        try:
            prediction = model.predict(input_data)[0]
            if prediction == 1:
                st.markdown("""
                <div class="result-card result-dnf">
                    <div class="result-icon">🚩</div>
                    <div class="result-label">Prediction Result</div>
                    <div class="result-title-dnf">DNF</div>
                    <div style='font-family: Barlow Condensed, sans-serif; font-size: 1.1rem;
                                color: #c0392b; letter-spacing: 0.1em; margin-top: 0.3rem;'>
                        DID NOT FINISH
                    </div>
                    <div class="result-sub">The driver is predicted to retire from this race.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-card result-finish">
                    <div class="result-icon">🏆</div>
                    <div class="result-label">Prediction Result</div>
                    <div class="result-title-finish">FINISH</div>
                    <div style='font-family: Barlow Condensed, sans-serif; font-size: 1.1rem;
                                color: #27ae60; letter-spacing: 0.1em; margin-top: 0.3rem;'>
                        RACE COMPLETE
                    </div>
                    <div class="result-sub">The driver is predicted to finish the race.</div>
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"⚠️ Prediction error: {e}")

# Footer
st.markdown("""
<div style='margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid #1a1a1a;
            text-align: center; font-size: 0.7rem; color: #2d2d2d;
            letter-spacing: 0.1em; text-transform: uppercase;'>
    F1 DNF Predictor · Built with Streamlit & scikit-learn · Krish Kubadia
</div>
""", unsafe_allow_html=True)
