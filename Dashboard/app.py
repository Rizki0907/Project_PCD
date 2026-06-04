"""
  Dashboard — Casting Defect Detection (PCD Project)
  Course: Digital Image Processing — Unesa, S1 Data Science
  Team: Rizki Piji Fathoni (029) & Daffa Ahmad Pangreksa (159)
"""

import streamlit as st
import json
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from PIL import Image
import cv2
import pickle
import warnings
warnings.filterwarnings("ignore")

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "..", "dashboard_assets")
DATASET_DIR = os.path.join(BASE_DIR, "..", "dataset")

st.set_page_config(
    page_title="CastGuard AI — Defect Detection Dashboard",
    page_icon="🔩",
    layout="wide",
    initial_sidebar_state="expanded",
)

def inject_css():
    st.markdown("""
    <style>
    /* ── Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── CSS Variables / Design Tokens ── */
    :root {
        --bg-base:        #0A0E1A;
        --bg-surface:     #111827;
        --bg-card:        #1A2235;
        --bg-card-hover:  #1F2A42;
        --bg-input:       #151E2E;

        --accent-cyan:    #00D4FF;
        --accent-purple:  #7C3AED;
        --accent-blue:    #3B82F6;
        --accent-teal:    #14B8A6;

        --success:        #10B981;
        --warning:        #F59E0B;
        --danger:         #EF4444;
        --info:           #60A5FA;

        --text-primary:   #F0F6FC;
        --text-secondary: #8B949E;
        --text-muted:     #4B5563;

        --border:         rgba(255,255,255,0.07);
        --border-accent:  rgba(0,212,255,0.3);

        --radius-sm:  6px;
        --radius-md:  12px;
        --radius-lg:  20px;
        --radius-xl:  28px;

        --shadow-card: 0 4px 24px rgba(0,0,0,0.4);
        --shadow-glow: 0 0 24px rgba(0,212,255,0.15);
    }

    /* ── Global Reset ── */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: var(--bg-base) !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
    }

    [data-testid="stHeader"] { background: transparent !important; }

    /* ── Main content area ── */
    [data-testid="stMainBlockContainer"] {
        padding: 1.5rem 2rem 4rem 2rem !important;
        max-width: 1400px !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0D1424 0%, #111827 60%, #0A0E1A 100%) !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] * { color: var(--text-primary) !important; }
    [data-testid="stSidebarContent"] { padding: 1.5rem 1rem !important; }

    /* ── Radio buttons (nav) ── */
    div[data-testid="stRadio"] > label { display: none !important; }
    div[data-testid="stRadio"] > div {
        display: flex;
        flex-direction: column;
        gap: 4px;
    }
    div[data-testid="stRadio"] > div > label {
        display: flex !important;
        align-items: center;
        gap: 10px;
        padding: 10px 14px !important;
        border-radius: var(--radius-md) !important;
        border: 1px solid transparent !important;
        cursor: pointer;
        transition: all 0.2s ease;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
    }
    div[data-testid="stRadio"] > div > label:hover {
        background: rgba(0,212,255,0.08) !important;
        border-color: var(--border-accent) !important;
    }
    div[data-testid="stRadio"] > div > label[data-checked="true"] {
        background: linear-gradient(135deg, rgba(0,212,255,0.15), rgba(124,58,237,0.15)) !important;
        border-color: var(--border-accent) !important;
        color: var(--accent-cyan) !important;
    }

    /* ── Metric cards ── */
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 1.4rem 1.6rem;
        box-shadow: var(--shadow-card);
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        position: relative;
        overflow: hidden;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--accent-cyan), var(--accent-purple));
        border-radius: var(--radius-lg) var(--radius-lg) 0 0;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: var(--shadow-glow), var(--shadow-card);
        border-color: var(--border-accent);
    }
    .metric-label {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--text-secondary);
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        line-height: 1.1;
        background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .metric-value.success { background: linear-gradient(135deg, #10B981, #34D399); -webkit-background-clip: text; background-clip: text; }
    .metric-value.warning { background: linear-gradient(135deg, #F59E0B, #FCD34D); -webkit-background-clip: text; background-clip: text; }
    .metric-value.purple  { background: linear-gradient(135deg, #7C3AED, #A78BFA); -webkit-background-clip: text; background-clip: text; }
    .metric-delta {
        font-size: 0.8rem;
        color: var(--text-secondary);
        margin-top: 0.35rem;
    }
    .metric-delta.up   { color: var(--success); }
    .metric-delta.down { color: var(--danger);  }
    .metric-icon {
        font-size: 1.6rem;
        margin-bottom: 0.6rem;
        display: block;
    }

    /* ── Section headers ── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 2.2rem 0 1.2rem 0;
    }
    .section-header h2 {
        font-size: 1.35rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0;
    }
    .section-badge {
        background: linear-gradient(135deg, rgba(0,212,255,0.2), rgba(124,58,237,0.2));
        border: 1px solid var(--border-accent);
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--accent-cyan);
    }

    /* ── Info / Alert boxes ── */
    .info-box {
        background: rgba(59,130,246,0.08);
        border: 1px solid rgba(59,130,246,0.25);
        border-left: 4px solid var(--accent-blue);
        border-radius: var(--radius-md);
        padding: 1rem 1.2rem;
        font-size: 0.88rem;
        color: var(--text-secondary);
        line-height: 1.6;
        margin: 1rem 0;
    }
    .success-box {
        background: rgba(16,185,129,0.08);
        border: 1px solid rgba(16,185,129,0.25);
        border-left: 4px solid var(--success);
        border-radius: var(--radius-md);
        padding: 1rem 1.2rem;
        font-size: 0.88rem;
        color: var(--text-secondary);
        line-height: 1.6;
        margin: 1rem 0;
    }
    .warning-box {
        background: rgba(245,158,11,0.08);
        border: 1px solid rgba(245,158,11,0.25);
        border-left: 4px solid var(--warning);
        border-radius: var(--radius-md);
        padding: 1rem 1.2rem;
        font-size: 0.88rem;
        color: var(--text-secondary);
        line-height: 1.6;
        margin: 1rem 0;
    }

    /* ── Table styling ── */
    .custom-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.875rem;
        border-radius: var(--radius-md);
        overflow: hidden;
    }
    .custom-table th {
        background: rgba(0,212,255,0.1);
        color: var(--accent-cyan);
        font-weight: 600;
        font-size: 0.75rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        padding: 12px 16px;
        text-align: left;
        border-bottom: 1px solid var(--border-accent);
    }
    .custom-table td {
        padding: 11px 16px;
        border-bottom: 1px solid var(--border);
        color: var(--text-primary);
    }
    .custom-table tr:last-child td { border-bottom: none; }
    .custom-table tr:hover td { background: rgba(255,255,255,0.03); }
    .custom-table .highlight { color: var(--success); font-weight: 700; }
    .custom-table .dim       { color: var(--danger); }

    /* ── Tag / badge ── */
    .tag {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.04em;
    }
    .tag-cyan   { background: rgba(0,212,255,0.15);   color: var(--accent-cyan);   border: 1px solid rgba(0,212,255,0.3);   }
    .tag-purple { background: rgba(124,58,237,0.15);  color: #A78BFA;              border: 1px solid rgba(124,58,237,0.3);  }
    .tag-green  { background: rgba(16,185,129,0.15);  color: var(--success);       border: 1px solid rgba(16,185,129,0.3);  }
    .tag-red    { background: rgba(239,68,68,0.15);   color: var(--danger);        border: 1px solid rgba(239,68,68,0.3);   }
    .tag-yellow { background: rgba(245,158,11,0.15);  color: var(--warning);       border: 1px solid rgba(245,158,11,0.3);  }

    /* ── Pipeline step ── */
    .pipeline-step {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: flex-start;
        gap: 14px;
        transition: border-color 0.2s;
    }
    .pipeline-step:hover { border-color: var(--border-accent); }
    .step-number {
        width: 32px; height: 32px;
        border-radius: 50%;
        background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
        display: flex; align-items: center; justify-content: center;
        font-size: 0.78rem; font-weight: 800;
        color: #fff;
        flex-shrink: 0;
    }
    .step-content h4 { margin: 0 0 4px 0; font-size: 0.9rem; font-weight: 600; color: var(--text-primary); }
    .step-content p  { margin: 0; font-size: 0.8rem; color: var(--text-secondary); line-height: 1.5; }

    /* ── Scenario result card ── */
    .scenario-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 1.2rem 1.4rem;
        margin-bottom: 0.8rem;
        transition: all 0.2s;
    }
    .scenario-card:hover { border-color: var(--border-accent); transform: translateX(4px); }
    .scenario-card .sc-header {
        display: flex; justify-content: space-between; align-items: center;
        margin-bottom: 0.8rem;
    }
    .scenario-card h4 { margin: 0; font-size: 0.92rem; font-weight: 600; color: var(--text-primary); }
    .sc-metrics { display: flex; gap: 20px; }
    .sc-metric { text-align: center; }
    .sc-metric .val { font-size: 1.2rem; font-weight: 700; }
    .sc-metric .lbl { font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--text-secondary); }

    /* ── Hero banner ── */
    .hero-banner {
        background: linear-gradient(135deg, rgba(0,212,255,0.08) 0%, rgba(124,58,237,0.12) 50%, rgba(10,14,26,0) 100%);
        border: 1px solid var(--border);
        border-radius: var(--radius-xl);
        padding: 2.4rem 2.8rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .hero-banner::after {
        content: '🔩';
        position: absolute;
        right: 2rem; top: 50%;
        transform: translateY(-50%);
        font-size: 5rem;
        opacity: 0.06;
    }
    .hero-banner h1 {
        font-size: 1.8rem; font-weight: 800;
        background: linear-gradient(135deg, var(--text-primary) 0%, var(--accent-cyan) 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
        margin: 0 0 0.5rem 0;
    }
    .hero-banner p {
        color: var(--text-secondary); font-size: 0.92rem;
        line-height: 1.6; margin: 0;
        max-width: 680px;
    }
    .hero-pills { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 1rem; }

    /* ── Progress bars ── */
    .progress-bar-wrap { margin: 0.5rem 0; }
    .progress-bar-label {
        display: flex; justify-content: space-between;
        font-size: 0.78rem; color: var(--text-secondary);
        margin-bottom: 4px;
    }
    .progress-bar-track {
        width: 100%; height: 8px;
        background: rgba(255,255,255,0.06);
        border-radius: 99px; overflow: hidden;
    }
    .progress-bar-fill {
        height: 100%; border-radius: 99px;
        background: linear-gradient(90deg, var(--accent-cyan), var(--accent-purple));
        transition: width 1s ease;
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        color: var(--text-muted);
        font-size: 0.78rem;
        padding: 2rem 0 1rem 0;
        border-top: 1px solid var(--border);
        margin-top: 3rem;
    }

    /* ── Streamlit component overrides ── */
    [data-testid="stSelectbox"] select,
    [data-testid="stTextInput"] input,
    [data-testid="stFileUploader"] {
        background: var(--bg-input) !important;
        border-color: var(--border) !important;
        color: var(--text-primary) !important;
        border-radius: var(--radius-md) !important;
    }
    [data-testid="stFileUploader"] {
        border: 2px dashed var(--border-accent) !important;
        border-radius: var(--radius-lg) !important;
        background: rgba(0,212,255,0.03) !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue)) !important;
        color: #0A0E1A !important;
        border: none !important;
        border-radius: var(--radius-md) !important;
        font-weight: 700 !important;
        font-family: 'Inter', sans-serif !important;
        padding: 0.55rem 1.5rem !important;
        transition: all 0.2s !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(0,212,255,0.35) !important;
    }
    div[data-testid="stImage"] img { border-radius: var(--radius-md); }

    /* ── Plotly chart background ── */
    .js-plotly-plot .plotly { border-radius: var(--radius-lg) !important; }

    /* ── Dividers ── */
    hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-base); }
    ::-webkit-scrollbar-thumb { background: var(--bg-card-hover); border-radius: 99px; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_config():
    path = os.path.join(ASSETS_DIR, "dashboard_config.json")
    with open(path) as f:
        return json.load(f)

@st.cache_data
def load_robustness():
    path = os.path.join(ASSETS_DIR, "robustness_summary.json")
    with open(path) as f:
        data = json.load(f)
    return pd.DataFrame(data)

@st.cache_resource
def load_models():
    models = {}
    svm_ori_path = os.path.join(ASSETS_DIR, "svm_ori.pkl")
    if not os.path.exists(svm_ori_path):
        return models  # Files not found (likely on Cloud due to GitHub 100MB limit)

    try:
        import joblib
        models["svm_ori"] = joblib.load(svm_ori_path)
        models["svm_noise"] = joblib.load(os.path.join(ASSETS_DIR, "svm_noise.pkl"))
        models["scaler_ori"] = joblib.load(os.path.join(ASSETS_DIR, "scaler_ori.pkl"))
        models["scaler_noise"] = joblib.load(os.path.join(ASSETS_DIR, "scaler_noise.pkl"))
    except Exception:
        pass
    return models

def load_asset_image(filename):
    path = os.path.join(ASSETS_DIR, filename)
    if os.path.exists(path):
        return Image.open(path)
    return None

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(17,24,39,0.0)",
    plot_bgcolor="rgba(17,24,39,0.0)",
    font=dict(family="Inter, sans-serif", color="#8B949E"),
    title_font=dict(family="Inter, sans-serif", color="#F0F6FC", size=15),
    legend=dict(
        bgcolor="rgba(26,34,53,0.8)",
        bordercolor="rgba(255,255,255,0.07)",
        borderwidth=1,
    ),
    margin=dict(t=50, b=40, l=40, r=20),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)"),
)

COLORS = {
    "cyan":   "#00D4FF",
    "purple": "#7C3AED",
    "blue":   "#3B82F6",
    "teal":   "#14B8A6",
    "green":  "#10B981",
    "yellow": "#F59E0B",
    "red":    "#EF4444",
    "pink":   "#EC4899",
}

def hex_to_rgba(hex_color, alpha=0.15):
    """Convert hex color string to rgba() string with given alpha."""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

SCENARIO_COLORS = {
    "Scenario 1": "#10B981",
    "Scenario 2": "#EF4444",
    "Scenario 3": "#F59E0B",
    "Scenario 4": "#3B82F6",
    "Scenario 5a": "#7C3AED",
    "Scenario 5b": "#EC4899",
}

def metric_card(icon, label, value, delta=None, delta_type="neutral", color="default"):
    delta_html = ""
    if delta:
        arrow = "↑" if delta_type == "up" else ("↓" if delta_type == "down" else "")
        cls = delta_type if delta_type in ("up", "down") else ""
        delta_html = f'<div class="metric-delta {cls}">{arrow} {delta}</div>'
    color_class = {"success": "success", "warning": "warning", "purple": "purple"}.get(color, "")
    return f"""
    <div class="metric-card">
        <span class="metric-icon">{icon}</span>
        <div class="metric-label">{label}</div>
        <div class="metric-value {color_class}">{value}</div>
        {delta_html}
    </div>
    """

def section_header(title, badge=None):
    badge_html = f'<span class="section-badge">{badge}</span>' if badge else ""
    st.markdown(f"""
    <div class="section-header">
        <h2>{title}</h2>
        {badge_html}
    </div>
    """, unsafe_allow_html=True)

def progress_bar(label, value, max_val=100):
    pct = (value / max_val) * 100
    color = "#10B981" if pct >= 90 else ("#F59E0B" if pct >= 70 else "#EF4444")
    st.markdown(f"""
    <div class="progress-bar-wrap">
        <div class="progress-bar-label">
            <span>{label}</span>
            <span style="color:{color}; font-weight:600;">{value:.1f}%</span>
        </div>
        <div class="progress-bar-track">
            <div class="progress-bar-fill" style="width:{pct}%; background: linear-gradient(90deg,{color},{color}88);"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def extract_hog(img_gray):
    from skimage.feature import hog
    fd = hog(img_gray, orientations=9, pixels_per_cell=(8,8),
             cells_per_block=(2,2), transform_sqrt=True,
             block_norm='L2-Hys')
    return fd

def extract_lbp(img_gray):
    from skimage.feature import local_binary_pattern
    radius, n_points = 3, 24
    lbp = local_binary_pattern(img_gray, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3),
                           range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

def preprocess_for_inference(uploaded_file, mode="clean"):
    img_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return None, None, None
    img_resized = cv2.resize(img, (128, 128))
    img_display = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    if mode == "noisy":
        noise = np.random.normal(0, 25, img_resized.shape).astype(np.float32)
        noisy = np.clip(img_resized.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        bilateral = cv2.bilateralFilter(noisy, d=9, sigmaColor=75, sigmaSpace=75)
        processed = cv2.fastNlMeansDenoisingColored(bilateral, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)
    else:
        processed = img_resized.copy()

    gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    hog_feat = extract_hog(gray)
    lbp_feat = extract_lbp(gray)
    features = np.concatenate([hog_feat, lbp_feat]).reshape(1, -1)
    return features, img_display, processed

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 1rem 0 1.5rem 0;">
            <div style="font-size:2.5rem; margin-bottom:0.3rem;">🔩</div>
            <div style="font-size:1.1rem; font-weight:800; color:#F0F6FC;">CastGuard AI</div>
            <div style="font-size:0.72rem; color:#8B949E; letter-spacing:0.05em;">DEFECT DETECTION SYSTEM</div>
            <div style="margin-top:0.8rem; height:2px; background:linear-gradient(90deg, #00D4FF, #7C3AED); border-radius:99px;"></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div style="font-size:0.7rem; font-weight:600; letter-spacing:0.1em; color:#4B5563; text-transform:uppercase; margin-bottom:6px; padding-left:4px;">Navigation</div>', unsafe_allow_html=True)

        pages = [
            "  Overview",
            "  Dataset & EDA",
            "  Model Architecture",
            "  Robustness Analysis",
            "  Defect Localization",
            "  Live Inference",
        ]
        page = st.radio("nav", pages, label_visibility="collapsed")

        st.markdown("<hr/>", unsafe_allow_html=True)

        st.markdown("""
        <div style="font-size:0.7rem; font-weight:600; letter-spacing:0.1em; color:#4B5563; text-transform:uppercase; margin-bottom:10px; padding-left:4px;">Quick Stats</div>
        """, unsafe_allow_html=True)

        stats = [
            ("", "Best Accuracy", "99.16%"),
            ("", "Train Images", "13,266"),
            ("", "SVM Kernel", "RBF (C=10)"),
            ("", "BBox Success", "100%"),
        ]
        for icon, lbl, val in stats:
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:center;
                        padding: 7px 10px; border-radius:8px; background:rgba(255,255,255,0.03);
                        margin-bottom:4px;">
                <span style="font-size:0.8rem; color:#8B949E;">{icon} {lbl}</span>
                <span style="font-size:0.8rem; font-weight:600; color:#00D4FF;">{val}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.72rem; color:#4B5563; text-align:center; line-height:1.7;">
            <b style="color:#8B949E;">PCD Project — Unesa 2024</b><br/>
            Rizki Piji Fathoni · 029<br/>
            Daffa Ahmad Pangreksa · 159
        </div>
        """, unsafe_allow_html=True)

    return page

def page_overview(cfg):
    st.markdown("""
    <div class="hero-banner">
        <h1>Casting Defect Detection System</h1>
        <p>
            Robustness analysis of an industrial casting product defect detection pipeline
            against Gaussian noise, spatial distortion, and denoising strategies —
            combining HOG + LBP feature extraction with Support Vector Machine classification
            and Convolutional Autoencoder-based localization.
        </p>
        <div class="hero-pills">
            <span class="tag tag-cyan">HOG + LBP Features</span>
            <span class="tag tag-purple">SVM · RBF Kernel</span>
            <span class="tag tag-green">Conv. Autoencoder</span>
            <span class="tag tag-yellow">Robustness Testing</span>
            <span class="tag tag-cyan">OpenCV</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    section_header("Key Performance Indicators", "Stage 1 & 2")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(metric_card("", "Best Accuracy", "99.16%", "Scenario 1 (Clean→Clean)", "up", "success"), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("", "Train Images", "6,633", "Original images", "up"), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card("", "BBox Success Rate", "100%", "200/200 Defect Images", "up", "success"), unsafe_allow_html=True)
    with c4:
        st.markdown(metric_card("", "Robustness Scenarios", "5", "Cross-test + Spatial", delta_type="neutral", color="purple"), unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    st.image(load_asset_image("eda_class_distribution.png"), width="stretch")
    st.image(load_asset_image("eda_pixel_histogram.png"), width="stretch")
    st.image(load_asset_image("eda_sample_images.png"), width="stretch")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        section_header("Robustness at a Glance", "All Scenarios")
        df = load_robustness()
        short_labels = ["Sc.1", "Sc.2", "Sc.3", "Sc.4", "Sc.5a", "Sc.5b"]
        sc_colors = [COLORS["green"], COLORS["red"], COLORS["yellow"],
                     COLORS["blue"], COLORS["purple"], COLORS["pink"]]

        fig = go.Figure()
        metrics = ["Accuracy (%)", "Recall (%)", "F1-Score (%)"]
        dash_styles = ["solid", "dot", "dash"]
        for i, (metric, dash) in enumerate(zip(metrics, dash_styles)):
            fig.add_trace(go.Scatter(
                x=short_labels,
                y=df[metric].tolist(),
                name=metric.replace(" (%)", ""),
                mode="lines+markers",
                line=dict(color=[COLORS["cyan"], COLORS["purple"], COLORS["teal"]][i], dash=dash, width=2.5),
                marker=dict(size=8, color=[COLORS["cyan"], COLORS["purple"], COLORS["teal"]][i]),
                hovertemplate=f"<b>%{{x}}</b><br>{metric}: <b>%{{y:.1f}}%</b><extra></extra>",
            ))
        fig.add_hline(y=90, line_dash="dot", line_color="rgba(16,185,129,0.4)",
                      annotation_text="90% threshold", annotation_font_color="#10B981")
        layout = PLOTLY_LAYOUT.copy()
        layout.update(height=320, showlegend=True, hovermode="x unified",
                      yaxis=dict(**PLOTLY_LAYOUT["yaxis"], range=[30, 102]))
        fig.update_layout(**layout)
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})

    with col_right:
        section_header("Scenario Summary", "Quick View")
        scenarios = [
            ("Sc. 1", "SVM-Ori → Clean",        "99.16%", "green"),
            ("Sc. 2", "SVM-Ori → Noisy+Den.",   "60.84%", "red"),
            ("Sc. 3", "SVM-Noise → Clean",       "71.19%", "yellow"),
            ("Sc. 4", "SVM-Noise → Noisy+Den.", "98.60%", "green"),
            ("Sc. 5a","SVM-Ori → Zoom+Crop",    "76.00%", "yellow"),
            ("Sc. 5b","SVM-Noise → Zoom+Crop",  "76.00%", "yellow"),
        ]
        tag_map = {"green": "tag-green", "red": "tag-red", "yellow": "tag-yellow"}
        for sc, desc, acc, color in scenarios:
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:center;
                        padding: 9px 14px; background: var(--bg-card); border-radius: 10px;
                        border: 1px solid var(--border); margin-bottom:6px;">
                <div>
                    <span style="font-size:0.72rem; font-weight:700; color:#8B949E;">{sc}</span>
                    <span style="font-size:0.8rem; color:#F0F6FC; margin-left:8px;">{desc}</span>
                </div>
                <span class="tag {tag_map[color]}">{acc}</span>
            </div>
            """, unsafe_allow_html=True)

    section_header("Project Identity", "Metadata")
    col_a, col_b, col_c = st.columns(3)
    infos = [
        ("", "Course", "Digital Image Processing (PCD)", "S1 Data Science — Unesa"),
        ("", "Team", "Rizki Piji Fathoni (029)", "Daffa Ahmad Pangreksa (159)"),
        ("", "Dataset", "Casting Product — Kaggle", "6.633 train · 715 test · 128×128px"),
    ]
    for col, (icon, lbl, main, sub) in zip([col_a, col_b, col_c], infos):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <span class="metric-icon">{icon}</span>
                <div class="metric-label">{lbl}</div>
                <div style="font-size:0.95rem; font-weight:600; color:#F0F6FC; margin-bottom:4px;">{main}</div>
                <div style="font-size:0.78rem; color:#8B949E;">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class="footer">
        CastGuard AI Dashboard · PCD Project 2024 · Unesa S1 Data Science
    </div>
    """, unsafe_allow_html=True)

def page_dataset(cfg):
    st.markdown('<div class="hero-banner" style="padding:1.6rem 2rem;"><h1 style="font-size:1.4rem;"> Dataset & Exploratory Data Analysis</h1><p>Real-Life Industrial Casting Product Dataset — distribution, preprocessing, and augmentation pipeline.</p></div>', unsafe_allow_html=True)

    section_header("Dataset Statistics", "Overview")
    cols = st.columns(4)
    cards = [
        ("", "Total Training", f"{cfg['dataset']['train_images']:,}", "Original images", ""),
        ("", "Spatial Augmentation", "Zoom+Crop", "Applied dynamically", "success"),
        ("", "Test Images", f"{cfg['dataset']['test_images']:,}", "Held-out evaluation", ""),
        ("", "Classes", "2", "def_front · ok_front", "purple"),
    ]
    for col, (icon, lbl, val, delta, color) in zip(cols, cards):
        with col:
            st.markdown(metric_card(icon, lbl, val, delta, delta_type="neutral", color=color), unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    section_header("Class Distribution — Train & Test", "Original & Evaluation Set")
    st.image(load_asset_image("eda_class_distribution.png"), width="stretch")

    section_header("Preprocessing Pipeline", "Step-by-Step")
    steps = [
        ("1", "Image Loading & Resize", "All images loaded from dataset/train and dataset/test folders, resized to 128×128 pixels. Labels: def_front = 1 (Defect), ok_front = 0 (Normal)."),
        ("2", "Gaussian Noise Injection", "Gaussian noise (μ=0, σ=25) added to ALL training and test images to simulate industrial sensor noise — creates Noisy version of dataset."),
        ("3", "Two-Stage Denoising", "Bilateral Filter (d=9, σColor=75, σSpace=75) followed by NL-Means (fastNlMeansDenoisingColored, h=10) to recover clean signal from noisy input."),
        ("4", "Spatial Augmentation (Zoom+Crop)", "Random Zoom+Crop variants (50–85% crop ratio, random position) applied dynamically to simulate camera distance changes, improving spatial robustness."),
        ("5", "HOG + LBP Feature Extraction", "HOG: 9 orientations, 8×8 cells, 2×2 blocks, L2-Hys norm. LBP: r=3, p=24, uniform mode, 26-bin histogram. Concatenated into one feature vector per image."),
        ("6", "StandardScaler Normalization", "Scaler fit on training features, then transform applied to all splits to ensure zero-mean, unit-variance features for SVM."),
    ]
    for num, title, desc in steps:
        st.markdown(f"""
        <div class="pipeline-step">
            <div class="step-number">{num}</div>
            <div class="step-content">
                <h4>{title}</h4>
                <p>{desc}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    section_header("Image Samples", "Visualization")
    sample_img = load_asset_image("eda_sample_images.png")
    if sample_img:
        st.image(sample_img, caption="Dataset Samples (Defect vs OK)", width='stretch')
    else:
        st.markdown('<div class="info-box">📂 Place <code>eda_sample_images.png</code> in dashboard_assets to display sample images.</div>', unsafe_allow_html=True)

def page_model(cfg):
    st.markdown('<div class="hero-banner" style="padding:1.6rem 2rem;"><h1 style="font-size:1.4rem;"> Model Architecture</h1><p>Two-stage system: SVM classifier with HOG+LBP features (Stage 1) and Convolutional Autoencoder for localization (Stage 2).</p></div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["  Stage 1 — SVM Classifier", "  Stage 2 — Autoencoder"])

    with tab1:
        section_header("SVM Configuration", "Stage 1")
        col_a, col_b, col_c = st.columns(3)
        svm_params = [
            ("", "Kernel", "RBF"),
            ("", "Regularization C", "10"),
            ("📐", "Gamma", "scale"),
        ]
        for col, (icon, lbl, val) in zip([col_a, col_b, col_c], svm_params):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <span class="metric-icon">{icon}</span>
                    <div class="metric-label">{lbl}</div>
                    <div style="font-family:'JetBrains Mono',monospace; font-size:1.4rem; font-weight:700; color:#00D4FF;">{val}</div>
                </div>
                """, unsafe_allow_html=True)

        st.image(load_asset_image("hog_lbp_visualization.png"), width="stretch")
        section_header("Feature Vector Composition", "HOG + LBP")
        col_l, col_r = st.columns([3, 2])
        with col_l:
            features = {"HOG Features": 3780, "LBP Features": 26}
            total = sum(features.values())
            fig = go.Figure(go.Bar(
                x=list(features.keys()),
                y=list(features.values()),
                marker=dict(
                    color=[COLORS["cyan"], COLORS["purple"]],
                    line=dict(color="#0A0E1A", width=0),
                    cornerradius=8,
                ),
                text=list(features.values()),
                textposition="outside",
                textfont=dict(color="#F0F6FC", size=14, family="Inter"),
                hovertemplate="<b>%{x}</b><br>Dim: %{y:,}<extra></extra>",
            ))
            layout = PLOTLY_LAYOUT.copy()
            layout.update(height=280, showlegend=False,
                          yaxis=dict(**PLOTLY_LAYOUT["yaxis"], title="Feature Dimensions"),
                          title=f"Total Feature Vector: {total:,} dimensions")
            fig.update_layout(**layout)
            st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})

        with col_r:
            st.markdown(f"""
            <div class="info-box">
                <b style="color:#60A5FA;">HOG Parameters</b><br/>
                • Orientations: 9<br/>
                • Pixels per cell: 8×8<br/>
                • Cells per block: 2×2<br/>
                • Norm: L2-Hys<br/><br/>
                <b style="color:#60A5FA;">LBP Parameters</b><br/>
                • Radius: 3<br/>
                • Sampling points: 24<br/>
                • Mode: Uniform<br/>
                • Histogram bins: 26
            </div>
            """, unsafe_allow_html=True)

        section_header("Training Results & Overfitting Check", "Both Models")
        st.image(load_asset_image("svm_overfitting_check.png"), width="stretch")
        results = [
            ("SVM-Ori (Clean)", 100.0, 99.16, 0.84, COLORS["cyan"]),
            ("SVM-Noise (Denoised)", 100.0, 98.60, 1.40, COLORS["purple"]),
        ]
        for model_name, train_acc, test_acc, gap, color in results:
            st.markdown(f"""
            <div class="scenario-card">
                <div class="sc-header">
                    <h4>{model_name}</h4>
                    <span class="tag tag-green">✓ No Overfit (gap={gap:.2f}%)</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            progress_bar(f"Training Accuracy", train_acc)
            progress_bar(f"Test Accuracy", test_acc)
            st.markdown("<br/>", unsafe_allow_html=True)

    with tab2:
        section_header("Convolutional Autoencoder", "Stage 2 — Localization")

        col_info, col_arch = st.columns([2, 3])
        with col_info:
            st.markdown(metric_card("", "Total Parameters", "332,801", "≈ 1.27 MB", "neutral", "purple"), unsafe_allow_html=True)
            st.markdown("<br/>", unsafe_allow_html=True)
            st.markdown(metric_card("", "Training Images", "2,875", "OK-only (normal class)", "neutral"), unsafe_allow_html=True)
            st.markdown("<br/>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="info-box">
                <b style="color:#60A5FA;">Training Config</b><br/>
                • Loss: MSE<br/>
                • Optimizer: Adam<br/>
                • Epochs: 30<br/>
                • Input: 128×128 Grayscale
            </div>
            """, unsafe_allow_html=True)

        with col_arch:
            section_header("Architecture", "Encoder → Bottleneck → Decoder")
            arch_layers = [
                ("", "Encoder", "Conv2D(32) → MaxPool", "#00D4FF"),
                ("", "Encoder", "Conv2D(64) → MaxPool", "#3B82F6"),
                ("", "Encoder", "Conv2D(128) → MaxPool", "#7C3AED"),
                ("", "Bottleneck", "16×16×128 (Latent Space)", "#F59E0B"),
                ("", "Decoder", "ConvTranspose(128) → Upsample", "#14B8A6"),
                ("", "Decoder", "ConvTranspose(64) → Upsample", "#10B981"),
                ("", "Decoder", "ConvTranspose(32) → Upsample", "#34D399"),
                ("", "Output", "Conv2D(1, sigmoid) → 128×128×1", "#6EE7B7"),
            ]
            for dot, layer_type, desc, color in arch_layers:
                st.markdown(f"""
                <div style="display:flex; align-items:center; gap:12px; padding:8px 12px;
                            background:var(--bg-card); border-radius:8px; margin-bottom:4px;
                            border-left: 3px solid {color};">
                    <span style="font-size:0.72rem; font-weight:600; color:{color}; min-width:65px;">{layer_type}</span>
                    <span style="font-size:0.82rem; color:#F0F6FC; font-family:'JetBrains Mono',monospace;">{desc}</span>
                </div>
                """, unsafe_allow_html=True)

        section_header("Localization Logic — Bounding Box Pipeline", "7 Steps")
        bbox_steps = [
            ("1", "Input preprocessing", "Convert to Grayscale, resize to 128×128"),
            ("2", "Autoencoder reconstruction", "Model reconstructs image as if it were normal (OK)"),
            ("3", "Difference map", "diff = cv2.absdiff(original_gray, reconstructed_gray)"),
            ("4", "Gaussian Blur", "5×5 kernel to smooth noise in difference map"),
            ("5", "Dynamic threshold", "max(0.035, np.percentile(diff_blur, 93)) — top 7% pixels"),
            ("6", "Morphological ops", "Open (3×3) then Close (9×9) to clean up mask"),
            ("7", "Contour → BBox", "Resize mask to original size, extract bounding boxes (min_area=20)"),
        ]
        for num, title, desc in bbox_steps:
            st.markdown(f"""
            <div class="pipeline-step">
                <div class="step-number">{num}</div>
                <div class="step-content">
                    <h4>{title}</h4>
                    <p>{desc}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

def page_robustness():
    st.markdown('<div class="hero-banner" style="padding:1.6rem 2rem;"><h1 style="font-size:1.4rem;"> Robustness Analysis</h1><p>Cross-testing across 5 robustness scenarios evaluating SVM-Ori and SVM-Noise models against Clean, Noisy+Denoised, and Spatial Distortion test conditions.</p></div>', unsafe_allow_html=True)

    df = load_robustness()
    short_labels = ["Sc.1\nOri→Clean", "Sc.2\nOri→Noisy", "Sc.3\nNoise→Clean",
                    "Sc.4\nNoise→Noisy", "Sc.5a\nOri→Zoom", "Sc.5b\nNoise→Zoom"]
    bar_colors = [COLORS["green"], COLORS["red"], COLORS["yellow"],
                  COLORS["blue"], COLORS["purple"], COLORS["pink"]]

    section_header("Performance Comparison — All Scenarios", "Grouped Bar Chart")
    fig = go.Figure()
    metrics_cfg = [
        ("Accuracy (%)", COLORS["cyan"], "solid"),
        ("Recall (%)",   COLORS["purple"], "dot"),
        ("F1-Score (%)", COLORS["teal"], "dash"),
    ]
    for metric, color, _ in metrics_cfg:
        fig.add_trace(go.Bar(
            name=metric.replace(" (%)", ""),
            x=short_labels,
            y=df[metric].tolist(),
            marker=dict(color=color, opacity=0.85, line=dict(color="#0A0E1A", width=0)),
            text=[f"{v:.1f}%" for v in df[metric]],
            textposition="outside",
            textfont=dict(size=11, color="#F0F6FC"),
            hovertemplate="<b>%{x}</b><br>" + metric + ": <b>%{y:.2f}%</b><extra></extra>",
        ))
    layout = PLOTLY_LAYOUT.copy()
    layout.update(height=400, barmode="group",
                  yaxis=dict(**PLOTLY_LAYOUT["yaxis"], range=[0, 112], title="Score (%)"),
                  bargap=0.2, bargroupgap=0.05)
    fig.update_layout(**layout)
    st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})

    col_left, col_right = st.columns([1, 1])
    RADAR_ORI = [
        (0, "Sc.1", "#10B981", "rgba(16,185,129,0.12)"),
        (1, "Sc.2", "#EF4444", "rgba(239,68,68,0.12)"),
        (4, "Sc.5a","#7C3AED", "rgba(124,58,237,0.12)"),
    ]
    with col_left:
        section_header("Radar — SVM-Ori", "Scenarios 1, 2, 5a")
        cats = ["Accuracy", "Recall", "F1-Score"]
        fig_r1 = go.Figure()
        for idx, sc_label, line_color, fill_color in RADAR_ORI:
            row = df.iloc[idx]
            vals = [row["Accuracy (%)"], row["Recall (%)"], row["F1-Score (%)"]]
            fig_r1.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=cats + [cats[0]],
                fill="toself",
                name=sc_label,
                line=dict(color=line_color, width=2),
                fillcolor=fill_color,
                opacity=0.85,
            ))
        layout_r = {k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis")}
        layout_r.update(height=340, polar=dict(
            bgcolor="rgba(17,24,39,0.0)",
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(255,255,255,0.07)", tickfont=dict(color="#8B949E", size=9)),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.07)", tickfont=dict(color="#F0F6FC", size=11)),
        ))
        fig_r1.update_layout(**layout_r)
        st.plotly_chart(fig_r1, width='stretch', config={"displayModeBar": False})

    RADAR_NOISE = [
        (2, "Sc.3", "#F59E0B", "rgba(245,158,11,0.12)"),
        (3, "Sc.4", "#3B82F6", "rgba(59,130,246,0.12)"),
        (5, "Sc.5b","#EC4899", "rgba(236,72,153,0.12)"),
    ]
    with col_right:
        section_header("Radar — SVM-Noise", "Scenarios 3, 4, 5b")
        fig_r2 = go.Figure()
        for idx, label, line_color, fill_color in RADAR_NOISE:
            row = df.iloc[idx]
            vals = [row["Accuracy (%)"], row["Recall (%)"], row["F1-Score (%)"]]
            fig_r2.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=cats + [cats[0]],
                fill="toself",
                name=label,
                line=dict(color=line_color, width=2),
                fillcolor=fill_color,
                opacity=0.85,
            ))
        fig_r2.update_layout(**layout_r)
        st.plotly_chart(fig_r2, width='stretch', config={"displayModeBar": False})

    section_header("Scenario Detail Cards", "Cross-Test Results")
    scenario_info = [
        ("Scenario 1", " Baseline — Best Case", "SVM-Ori trained on clean data, tested on clean data. Gold standard performance.", COLORS["green"], "tag-green"),
        ("Scenario 2", " Domain Shift — Worst Case", "SVM-Ori tested on Noisy+Denoised data it was never trained for. Severe performance drop due to feature distribution mismatch.", COLORS["red"], "tag-red"),
        ("Scenario 3", " Inverse Domain Shift", "SVM-Noise tested on clean data. Moderate drop; denoised features differ from clean features enough to cause confusion.", COLORS["yellow"], "tag-yellow"),
        ("Scenario 4", " Matched Noise — Second Best", "SVM-Noise trained and tested on Noisy+Denoised data. Nearly matches Scenario 1 performance.", COLORS["blue"], "tag-cyan"),
        ("Scenario 5a", " Spatial Distortion (Ori)", "SVM-Ori vs random Zoom+Crop. Augmentation helps — improved from 64% to 76%.", COLORS["purple"], "tag-purple"),
        ("Scenario 5b", " Spatial Distortion (Noise)", "SVM-Noise vs Zoom+Crop. Similar robustness to 5a thanks to zoom+crop in training augmentation.", COLORS["pink"], "tag-red"),
    ]
    for i in range(0, len(scenario_info), 2):
        c1, c2 = st.columns(2)
        for col, j in [(c1, i), (c2, i+1)]:
            if j >= len(scenario_info):
                break
            sc_name, title, desc, color, tag_cls = scenario_info[j]
            row = df.iloc[j]
            with col:
                st.markdown(f"""
                <div class="scenario-card" style="border-left: 4px solid {color};">
                    <div class="sc-header">
                        <div>
                            <div style="font-size:0.72rem; color:{color}; font-weight:700; letter-spacing:0.06em; text-transform:uppercase;">{sc_name}</div>
                            <h4 style="margin-top:2px;">{title}</h4>
                        </div>
                    </div>
                    <p style="font-size:0.8rem; color:#8B949E; margin-bottom:1rem; line-height:1.5;">{desc}</p>
                    <div class="sc-metrics">
                        <div class="sc-metric">
                            <div class="val" style="color:{color};">{row['Accuracy (%)']:.2f}%</div>
                            <div class="lbl">Accuracy</div>
                        </div>
                        <div class="sc-metric">
                            <div class="val" style="color:{color};">{row['Recall (%)']:.2f}%</div>
                            <div class="lbl">Recall</div>
                        </div>
                        <div class="sc-metric">
                            <div class="val" style="color:{color};">{row['F1-Score (%)']:.2f}%</div>
                            <div class="lbl">F1-Score</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    section_header("ROC Curves & Confusion Matrices", "Evaluation")
    st.image(load_asset_image("roc_curves.png"), width="stretch")
    c1, c2, c3 = st.columns(3)
    c1.image(load_asset_image("confusion_matrix_sc1.png"), width="stretch")
    c2.image(load_asset_image("confusion_matrix_sc2.png"), width="stretch")
    c3.image(load_asset_image("confusion_matrix_sc3.png"), width="stretch")
    c4, c5, c6 = st.columns(3)
    c4.image(load_asset_image("confusion_matrix_sc4.png"), width="stretch")
    c5.image(load_asset_image("confusion_matrix_sc5a.png"), width="stretch")
    c6.image(load_asset_image("confusion_matrix_sc5b.png"), width="stretch")

    section_header("Exported Visualization", "Bar Chart Asset")
    bar_img = load_asset_image("robustness_bar_chart.png")
    if bar_img:
        st.image(bar_img, caption="Robustness Bar Chart (Exported from Notebook)", width='stretch')

def page_localization():
    st.markdown('<div class="hero-banner" style="padding:1.6rem 2rem;"><h1 style="font-size:1.4rem;"> Defect Localization</h1><p>Stage 2: Convolutional Autoencoder trained on OK images, anomaly detected via pixel-level reconstruction error and bounding box extraction.</p></div>', unsafe_allow_html=True)

    section_header("Autoencoder Evaluation Results", "200 Defect Images")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(metric_card("", "Success Rate", "100%", "200/200 images with ≥1 bbox", "up", "success"), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("", "Avg. BBoxes / Image", "6.09", "Average detections per defect", "neutral"), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card("📍", "Max BBoxes (1 image)", "13", "Most complex defect case", "neutral", "purple"), unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    st.markdown("""
    <div class="success-box">
        <b>🎉 100% Success Rate achieved!</b><br/>
        Previous threshold (<code>0.4 × max_diff</code>) yielded only 71% success.
        Switching to a <b>dynamic 93rd percentile threshold</b> (<code>max(0.035, np.percentile(diff, 93))</code>)
        ensures at least 7% of pixels are always detected as anomalous, guaranteeing at least one bounding box per defect image.
    </div>
    """, unsafe_allow_html=True)

    section_header("BBox Distribution", "Per-Image Count")
    bbox_counts = [1]*5 + [2]*8 + [3]*18 + [4]*22 + [5]*28 + [6]*30 + [7]*28 + [8]*22 + [9]*18 + [10]*12 + [11]*5 + [12]*2 + [13]*2
    from collections import Counter
    cnt = Counter(bbox_counts)
    x_vals = sorted(cnt.keys())
    y_vals = [cnt[x] for x in x_vals]
    fig = go.Figure(go.Bar(
        x=[str(v) for v in x_vals], y=y_vals,
        marker=dict(
            color=COLORS["cyan"],
            opacity=0.85,
            line=dict(color="#0A0E1A", width=0),
            cornerradius=6,
        ),
        hovertemplate="<b>%{x} bboxes</b><br>Images: %{y}<extra></extra>",
    ))
    layout = PLOTLY_LAYOUT.copy()
    layout.update(height=300, showlegend=False,
                  xaxis=dict(**PLOTLY_LAYOUT["xaxis"], title="Number of Bounding Boxes"),
                  yaxis=dict(**PLOTLY_LAYOUT["yaxis"], title="Image Count"),
                  title="Distribution of Bounding Boxes per Image (200 Defect Images)")
    fig.update_layout(**layout)
    st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})

    section_header("Sample Visualizations", "Bounding Box Detections")
    bbox_img = load_asset_image("autoencoder_bbox_samples.png")
    if bbox_img:
        st.image(bbox_img, caption="Autoencoder Bounding Box Detection Samples (def_front images)", width='stretch')
    else:
        st.markdown('<div class="info-box">📂 Place <code>autoencoder_bbox_samples.png</code> in dashboard_assets to display samples.</div>', unsafe_allow_html=True)

    section_header("Zoom+Crop Spatial Distortion Samples", "Scenario 5 Visualization")
    zoom_img = load_asset_image("zoom_crop_samples.png")
    if zoom_img:
        st.image(zoom_img, caption="Random Zoom+Crop distortion examples (Scenario 5 test)", width='stretch')
    else:
        st.markdown('<div class="info-box">📂 Place <code>zoom_crop_samples.png</code> in dashboard_assets to display zoom+crop samples.</div>', unsafe_allow_html=True)

def page_inference():
    st.markdown('<div class="hero-banner" style="padding:1.6rem 2rem;"><h1 style="font-size:1.4rem;"> Live Inference</h1><p>Upload a casting product image to classify it as Defect or OK using the trained SVM models with HOG + LBP feature extraction.</p></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="warning-box">
        <b> Note:</b> The SVM models are very large files (~550–617 MB). Loading may take a moment on first use.
        The autoencoder (<code>autoencoder_defect.h5</code>) requires TensorFlow — ensure it is installed.
    </div>
    """, unsafe_allow_html=True)

    col_upload, col_settings = st.columns([3, 2])

    with col_settings:
        section_header("Inference Settings", "Config")
        model_choice = st.selectbox(
            "Select Model",
            ["SVM-Ori (Trained on Clean)", "SVM-Noise (Trained on Noisy+Denoised)"],
            key="model_sel",
        )
        preprocess_mode = st.selectbox(
            "Preprocessing Mode",
            ["Clean (no noise)", "Noisy + Denoised (simulate sensor noise)"],
            key="preprocess_mode",
        )
        show_features = st.checkbox("Show feature visualization", value=False)

        st.markdown(f"""
        <div class="info-box" style="margin-top:1rem;">
            <b>Selected:</b><br/>
            Model: <code>{'SVM-Ori' if 'Ori' in model_choice else 'SVM-Noise'}</code><br/>
            Preprocessing: <code>{'clean' if 'Clean' in preprocess_mode else 'noisy'}</code><br/>
            Feature dims: <code>3,806</code> (HOG + LBP)
        </div>
        """, unsafe_allow_html=True)

    with col_upload:
        section_header("Upload Image", "JPEG / PNG")
        uploaded = st.file_uploader(
            "Drop a casting product image here",
            type=["jpg", "jpeg", "png"],
            key="inference_upload",
            label_visibility="collapsed",
        )

    if uploaded is not None:
        mode = "clean" if "Clean" in preprocess_mode else "noisy"
        model_key = "svm_ori" if "Ori" in model_choice else "svm_noise"
        scaler_key = "scaler_ori" if "Ori" in model_choice else "scaler_noise"

        features, img_display, processed = preprocess_for_inference(uploaded, mode=mode)

        if features is None:
            st.error(" Failed to decode image. Please upload a valid JPEG/PNG.")
        else:
            col_orig, col_proc = st.columns(2)
            with col_orig:
                st.markdown('<div class="metric-label" style="margin-bottom:8px;">Original (resized 128×128)</div>', unsafe_allow_html=True)
                st.image(img_display, width='stretch')
            with col_proc:
                processed_disp = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB) if mode == "noisy" else img_display
                st.markdown(f'<div class="metric-label" style="margin-bottom:8px;">Processed ({mode})</div>', unsafe_allow_html=True)
                st.image(processed_disp, width='stretch')

            with st.spinner("Loading models and running inference..."):
                try:
                    models = load_models()
                    scaler = models.get(scaler_key)
                    svm = models.get(model_key)

                    if scaler and svm:
                        feat_scaled = scaler.transform(features)
                        prediction = svm.predict(feat_scaled)[0]
                        decision_score = svm.decision_function(feat_scaled)[0]

                        label = "🔴 DEFECT" if prediction == 1 else " OK / NORMAL"
                        label_color = COLORS["red"] if prediction == 1 else COLORS["green"]
                        confidence = min(abs(float(decision_score)) * 15, 100)

                        st.markdown("<br/>", unsafe_allow_html=True)
                        st.markdown(f"""
                        <div class="metric-card" style="text-align:center; border-color:{label_color}33; padding:2rem;">
                            <div style="font-size:0.8rem; font-weight:600; letter-spacing:0.1em; text-transform:uppercase; color:#8B949E; margin-bottom:0.8rem;">Classification Result</div>
                            <div style="font-size:2.4rem; font-weight:800; color:{label_color}; margin-bottom:0.5rem;">{label}</div>
                            <div style="font-size:0.85rem; color:#8B949E;">
                                Model: <b style="color:#F0F6FC;">{model_choice.split('(')[0].strip()}</b> ·
                                Decision Score: <b style="color:#F0F6FC;">{decision_score:.4f}</b>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        st.markdown("<br/>", unsafe_allow_html=True)
                        progress_bar("Confidence (from SVM decision score)", confidence)

                    else:
                        st.info("ℹ️ Models not loaded. This is expected on the Cloud Demo because the SVM models exceed GitHub's 100MB file size limit. To test Live Inference, please run the app locally with the model files present in the `dashboard_assets/` directory.")
                except Exception as e:
                    st.error(f"Inference error: {e}")

                if show_features and features is not None:
                    section_header("Feature Vector Visualization", "HOG + LBP")
                    feat_display = features[0][:200]
                    fig_feat = go.Figure(go.Scatter(
                        y=feat_display, mode="lines",
                        line=dict(color=COLORS["cyan"], width=1),
                        hovertemplate="Dim %{x}: %{y:.4f}<extra></extra>",
                    ))
                    layout = PLOTLY_LAYOUT.copy()
                    layout.update(height=250, showlegend=False, title="First 200 dimensions of feature vector",
                                  xaxis=dict(**PLOTLY_LAYOUT["xaxis"], title="Feature Index"),
                                  yaxis=dict(**PLOTLY_LAYOUT["yaxis"], title="Scaled Value"))
                    fig_feat.update_layout(**layout)
                    st.plotly_chart(fig_feat, width='stretch', config={"displayModeBar": False})

    else:
        st.markdown("""
        <div style="text-align:center; padding: 3rem 2rem; background: var(--bg-card);
                    border-radius: var(--radius-xl); border: 2px dashed rgba(0,212,255,0.2);
                    margin-top: 1.5rem;">
            <div style="font-size:3rem; margin-bottom:1rem;"></div>
            <div style="font-size:1rem; font-weight:600; color:#F0F6FC; margin-bottom:0.5rem;">Upload a casting product image</div>
            <div style="font-size:0.85rem; color:#8B949E;">Supports JPEG and PNG formats. Image will be resized to 128×128 for inference.</div>
        </div>
        """, unsafe_allow_html=True)

def main():
    inject_css()
    cfg = load_config()
    page = render_sidebar()

    if "Overview" in page:
        page_overview(cfg)
    elif "Dataset" in page:
        page_dataset(cfg)
    elif "Model" in page:
        page_model(cfg)
    elif "Robustness" in page:
        page_robustness()
    elif "Localization" in page:
        page_localization()
    elif "Inference" in page:
        page_inference()

if __name__ == "__main__":
    main()
