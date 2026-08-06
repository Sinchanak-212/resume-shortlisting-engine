import streamlit as st
import os
import shutil
import pandas as pd
import json
from pathlib import Path
from dotenv import load_dotenv

# Set page config
st.set_page_config(
    page_title="AI Resume Shortlisting Engine",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environmental variables
load_dotenv()

# Import pipeline elements
import importlib
import main
from models.schemas import ParsedJD
from utils.llm import LLMClient
from agents.jd_parser import JDParserAgent
from config import REPORTS_DIR, SCRATCH_DIR


@st.cache_resource(show_spinner="Loading AI models (first run only - this can take a minute)...")
def get_cached_agents():
    """
    BUGFIX (perf): process_resumes() used to rebuild every agent - including EasyOCR,
    SentenceTransformer, and spaCy - from scratch on every single 'Process & Rank' click.
    st.cache_resource ensures this heavy initialization happens once per server process
    and is reused across reruns/clicks.
    """
    return main.build_pipeline_agents()


# ──────────────────────────────────────────────────────────────────────────────
# PREMIUM CSS THEME — Glassmorphism Dark Mode with Animations
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
<style>
    /* ── Global Reset & Theme ──────────────────────────────────────────── */
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0f1629 40%, #111827 100%);
        color: #e2e8f0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hide default Streamlit branding */
    #MainMenu, footer, header {visibility: hidden;}

    /* ── Sidebar ───────────────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #111827 100%);
        border-right: 1px solid rgba(99, 102, 241, 0.15);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        font-family: 'Inter', sans-serif;
        color: #f1f5f9;
    }

    /* ── Typography ────────────────────────────────────────────────────── */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif !important;
        color: #f8fafc !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    p, span, div, label {
        font-family: 'Inter', sans-serif;
    }

    /* ── Animations ────────────────────────────────────────────────────── */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(24px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes shimmer {
        0%   { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    @keyframes pulseGlow {
        0%, 100% { box-shadow: 0 0 15px rgba(99, 102, 241, 0.15); }
        50%      { box-shadow: 0 0 30px rgba(99, 102, 241, 0.3); }
    }
    @keyframes gradientFlow {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* ── Hero Section ──────────────────────────────────────────────────── */
    .hero-container {
        text-align: center;
        padding: 60px 20px 40px;
        animation: fadeInUp 0.8s ease-out;
    }
    .hero-title {
        font-size: 3.2rem;
        font-weight: 900;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #818cf8, #6366f1, #a78bfa, #818cf8);
        background-size: 300% 300%;
        animation: gradientFlow 6s ease infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 12px;
        line-height: 1.15;
    }
    .hero-subtitle {
        font-size: 1.15rem;
        color: #94a3b8;
        font-weight: 400;
        max-width: 700px;
        margin: 0 auto 48px;
        line-height: 1.65;
    }

    /* ── Feature Cards Grid ────────────────────────────────────────────── */
    .features-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 24px;
        max-width: 960px;
        margin: 0 auto 40px;
    }
    .feature-card {
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(99, 102, 241, 0.12);
        border-radius: 16px;
        padding: 32px 24px;
        text-align: center;
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        animation: fadeInUp 0.8s ease-out backwards;
    }
    .feature-card:nth-child(2) { animation-delay: 0.1s; }
    .feature-card:nth-child(3) { animation-delay: 0.2s; }
    .feature-card:hover {
        transform: translateY(-6px);
        border-color: rgba(99, 102, 241, 0.35);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3), 0 0 30px rgba(99, 102, 241, 0.1);
    }
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 16px;
        display: block;
    }
    .feature-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 8px;
    }
    .feature-desc {
        font-size: 0.88rem;
        color: #64748b;
        line-height: 1.55;
    }

    /* ── Workflow Steps ─────────────────────────────────────────────────── */
    .workflow-bar {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        margin: 10px auto 40px;
        max-width: 700px;
        animation: fadeInUp 0.8s ease-out 0.3s backwards;
    }
    .wf-step {
        display: flex;
        align-items: center;
        gap: 8px;
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(99, 102, 241, 0.1);
        border-radius: 999px;
        padding: 8px 18px;
        font-size: 0.82rem;
        color: #94a3b8;
        font-weight: 500;
    }
    .wf-step-num {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        width: 22px; height: 22px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.72rem;
        font-weight: 700;
        flex-shrink: 0;
    }
    .wf-arrow {
        color: #334155;
        font-size: 1.1rem;
    }

    /* ── Glassmorphism KPI Metric Cards ────────────────────────────────── */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 20px;
        margin-bottom: 36px;
    }
    .kpi-card {
        background: rgba(15, 23, 42, 0.5);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        transition: all 0.3s ease;
        animation: fadeInUp 0.6s ease-out backwards;
    }
    .kpi-card:nth-child(2) { animation-delay: 0.08s; }
    .kpi-card:nth-child(3) { animation-delay: 0.16s; }
    .kpi-card:nth-child(4) { animation-delay: 0.24s; }
    .kpi-card:hover {
        transform: translateY(-3px);
        border-color: rgba(99, 102, 241, 0.25);
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.2);
    }
    .kpi-icon { font-size: 1.6rem; margin-bottom: 8px; }
    .kpi-value {
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1.1;
        margin-bottom: 6px;
    }
    .kpi-label {
        font-size: 0.82rem;
        color: #64748b;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .kpi-accent-blue   { color: #818cf8; }
    .kpi-accent-green  { color: #34d399; }
    .kpi-accent-amber  { color: #fbbf24; }
    .kpi-accent-red    { color: #f87171; }

    /* ── Section Headers ───────────────────────────────────────────────── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 32px 0 20px;
        animation: fadeInUp 0.5s ease-out;
    }
    .section-header-icon {
        font-size: 1.5rem;
    }
    .section-header-text {
        font-size: 1.5rem;
        font-weight: 800;
        color: #f1f5f9;
        letter-spacing: -0.02em;
    }
    .section-header-line {
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, rgba(99, 102, 241, 0.3), transparent);
    }

    /* ── Candidate Inspector Card ──────────────────────────────────────── */
    .inspector-card {
        background: rgba(15, 23, 42, 0.5);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 28px;
        margin-bottom: 16px;
        animation: fadeInUp 0.5s ease-out;
    }
    .inspector-card:hover {
        border-color: rgba(99, 102, 241, 0.2);
    }

    /* ── Profile Header ────────────────────────────────────────────────── */
    .profile-header {
        display: flex;
        align-items: center;
        gap: 20px;
        margin-bottom: 24px;
    }
    .profile-avatar {
        width: 56px; height: 56px;
        border-radius: 16px;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        font-weight: 800;
        color: white;
        flex-shrink: 0;
    }
    .profile-name {
        font-size: 1.4rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 4px;
    }
    .profile-file {
        font-size: 0.82rem;
        color: #64748b;
        font-family: 'SF Mono', 'Fira Code', monospace;
    }

    /* ── Status & Confidence Badges ────────────────────────────────────── */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 4px 14px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .badge-shortlisted {
        background: rgba(52, 211, 153, 0.12);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.25);
    }
    .badge-reserve {
        background: rgba(251, 191, 36, 0.12);
        color: #fbbf24;
        border: 1px solid rgba(251, 191, 36, 0.25);
    }
    .badge-failed {
        background: rgba(248, 113, 113, 0.12);
        color: #f87171;
        border: 1px solid rgba(248, 113, 113, 0.25);
    }
    .badge-conf-high {
        background: rgba(52, 211, 153, 0.1);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.2);
    }
    .badge-conf-medium {
        background: rgba(96, 165, 250, 0.1);
        color: #60a5fa;
        border: 1px solid rgba(96, 165, 250, 0.2);
    }
    .badge-conf-low {
        background: rgba(167, 139, 250, 0.1);
        color: #a78bfa;
        border: 1px solid rgba(167, 139, 250, 0.2);
    }

    /* ── Score Ring ─────────────────────────────────────────────────────── */
    .score-ring-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 8px;
    }
    .score-ring {
        position: relative;
        width: 110px; height: 110px;
    }
    .score-ring svg {
        transform: rotate(-90deg);
    }
    .score-ring-bg {
        fill: none;
        stroke: rgba(255, 255, 255, 0.06);
        stroke-width: 8;
    }
    .score-ring-fill {
        fill: none;
        stroke-width: 8;
        stroke-linecap: round;
        transition: stroke-dashoffset 1s ease-out;
    }
    .score-ring-text {
        position: absolute;
        top: 50%; left: 50%;
        transform: translate(-50%, -50%);
        font-size: 1.5rem;
        font-weight: 800;
        color: #f1f5f9;
    }
    .score-ring-label {
        font-size: 0.78rem;
        color: #64748b;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    /* ── Stat Row ──────────────────────────────────────────────────────── */
    .stat-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    }
    .stat-row:last-child { border-bottom: none; }
    .stat-label {
        font-size: 0.88rem;
        color: #94a3b8;
        font-weight: 500;
    }
    .stat-value {
        font-size: 0.92rem;
        color: #e2e8f0;
        font-weight: 600;
    }

    /* ── Skill Chips ───────────────────────────────────────────────────── */
    .skill-chip {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 6px 14px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 4px;
        transition: all 0.2s ease;
    }
    .skill-chip:hover {
        transform: translateY(-1px);
    }
    .chip-exact {
        background: rgba(52, 211, 153, 0.1);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.25);
    }
    .chip-synonym {
        background: rgba(96, 165, 250, 0.1);
        color: #60a5fa;
        border: 1px solid rgba(96, 165, 250, 0.25);
    }
    .chip-partial, .chip-implicit {
        background: rgba(251, 191, 36, 0.1);
        color: #fbbf24;
        border: 1px solid rgba(251, 191, 36, 0.25);
    }
    .chip-none {
        background: rgba(248, 113, 113, 0.08);
        color: #f87171;
        border: 1px solid rgba(248, 113, 113, 0.2);
    }
    .chip-extracted {
        background: rgba(99, 102, 241, 0.08);
        color: #a5b4fc;
        border: 1px solid rgba(99, 102, 241, 0.18);
    }

    /* ── Explanation Bullets ────────────────────────────────────────────── */
    .explanation-list {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    .explanation-item {
        display: flex;
        align-items: flex-start;
        gap: 12px;
        padding: 14px 16px;
        background: rgba(15, 23, 42, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.04);
        border-radius: 12px;
        margin-bottom: 10px;
        font-size: 0.9rem;
        color: #cbd5e1;
        line-height: 1.55;
        transition: all 0.2s ease;
    }
    .explanation-item:hover {
        border-color: rgba(99, 102, 241, 0.15);
        background: rgba(15, 23, 42, 0.6);
    }
    .explanation-bullet {
        color: #818cf8;
        font-size: 1.1rem;
        flex-shrink: 0;
        margin-top: 1px;
    }

    /* ── Download Buttons ──────────────────────────────────────────────── */
    .stDownloadButton > button {
        background: rgba(99, 102, 241, 0.1) !important;
        border: 1px solid rgba(99, 102, 241, 0.25) !important;
        color: #a5b4fc !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.3s ease !important;
    }
    .stDownloadButton > button:hover {
        background: rgba(99, 102, 241, 0.2) !important;
        border-color: rgba(99, 102, 241, 0.5) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 16px rgba(99, 102, 241, 0.15) !important;
    }

    /* ── Primary Button ────────────────────────────────────────────────── */
    .stButton > button[kind="primary"],
    button[data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        border: none !important;
        color: white !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
        padding: 12px 24px !important;
        transition: all 0.3s ease !important;
        letter-spacing: 0.01em !important;
    }
    button[data-testid="stBaseButton-primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 24px rgba(99, 102, 241, 0.35) !important;
    }

    /* ── Tabs ──────────────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(15, 23, 42, 0.4);
        border-radius: 12px;
        padding: 4px;
        border: 1px solid rgba(255, 255, 255, 0.04);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        color: #64748b;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(99, 102, 241, 0.15) !important;
        color: #a5b4fc !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: transparent !important;
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }

    /* ── Dataframe ─────────────────────────────────────────────────────── */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }

    /* ── Divider ───────────────────────────────────────────────────────── */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.2), transparent);
        margin: 36px 0;
        border: none;
    }

    /* ── Sidebar Step Indicators ────────────────────────────────────────── */
    .sidebar-step {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 16px;
    }
    .sidebar-step-num {
        width: 28px; height: 28px;
        border-radius: 50%;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.78rem;
        font-weight: 700;
        flex-shrink: 0;
    }
    .sidebar-step-text {
        font-size: 0.95rem;
        font-weight: 600;
        color: #e2e8f0;
    }

    /* ── Responsive ────────────────────────────────────────────────────── */
    @media (max-width: 768px) {
        .features-grid { grid-template-columns: 1fr; }
        .kpi-grid { grid-template-columns: repeat(2, 1fr); }
        .hero-title { font-size: 2rem; }
        .workflow-bar { flex-wrap: wrap; }
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────────────────────────────────────
if "ranked_candidates" not in st.session_state:
    st.session_state.ranked_candidates = None
if "report_paths" not in st.session_state:
    st.session_state.report_paths = None
if "current_jd" not in st.session_state:
    st.session_state.current_jd = None


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Polished Steps
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 16px 0 8px;">
        <span style="font-size: 2rem;">💼</span>
        <div style="font-size: 1.1rem; font-weight: 800; color: #f1f5f9; margin-top: 4px; letter-spacing: -0.02em;">
            Resume Engine
        </div>
        <div style="font-size: 0.75rem; color: #64748b; font-weight: 500;">AI-Powered Shortlisting</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # Step 1
    st.markdown("""
    <div class="sidebar-step">
        <div class="sidebar-step-num">1</div>
        <div class="sidebar-step-text">Upload Resumes</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Choose PDF Resumes",
        type="pdf",
        accept_multiple_files=True,
        help="Upload single or multiple resumes (handles multi-column and scanned PDFs)",
        label_visibility="collapsed"
    )

    if uploaded_files:
        st.success(f"📄 {len(uploaded_files)} resume{'s' if len(uploaded_files) > 1 else ''} uploaded")

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # Step 2
    st.markdown("""
    <div class="sidebar-step">
        <div class="sidebar-step-num">2</div>
        <div class="sidebar-step-text">Job Description</div>
    </div>
    """, unsafe_allow_html=True)

    jd_source = st.radio(
        "JD Input Mode",
        ["Select Predefined Role", "Paste Unstructured JD Text"],
        label_visibility="collapsed"
    )

    parsed_jd = None

    if jd_source == "Select Predefined Role":
        selected_role_slug = st.selectbox(
            "Choose Role",
            options=list(main.DEFAULT_JDS.keys()),
            format_func=lambda x: main.DEFAULT_JDS[x].role_name
        )
        parsed_jd = main.DEFAULT_JDS[selected_role_slug]

        st.markdown(f"""
        <div style="background: rgba(99, 102, 241, 0.08); border: 1px solid rgba(99, 102, 241, 0.15);
                    border-radius: 12px; padding: 14px; margin-top: 8px; font-size: 0.82rem;">
            <div style="color: #a5b4fc; font-weight: 600; margin-bottom: 6px;">📋 {parsed_jd.role_name}</div>
            <div style="color: #94a3b8;">
                <b>Required:</b> {', '.join(parsed_jd.required_skills[:4])}{'...' if len(parsed_jd.required_skills) > 4 else ''}<br/>
                <b>CGPA ≥</b> {parsed_jd.min_cgpa:.1f} &nbsp;|&nbsp; <b>Slots:</b> {parsed_jd.slots}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        raw_jd_text = st.text_area(
            "Paste Job Description Here",
            height=180,
            placeholder="Paste a LinkedIn post, text file content, or raw list of specifications..."
        )
        st.caption("⚡ Custom JDs are parsed dynamically by the AI agent during processing.")

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # Step 3
    st.markdown("""
    <div class="sidebar-step">
        <div class="sidebar-step-num">3</div>
        <div class="sidebar-step-text">Configure & Run</div>
    </div>
    """, unsafe_allow_html=True)

    testing_limit = st.slider(
        "Limit processing count",
        -1, 30, -1,
        help="-1 processes all uploaded documents"
    )

    min_required_skill_match_ratio = st.slider(
        "Required skill match threshold",
        0.0, 1.0, 1.0, 0.1,
        help="Fraction of required JD skills a candidate must match to be eligible for shortlist."
    )

    min_shortlist_score = st.slider(
        "Minimum shortlist score",
        0.0, 100.0, 50.0, 5.0,
        help="Minimum overall score a candidate must achieve to consume a shortlist slot."
    )

    st.caption("Real-world baseline: candidates must meet required skills and a minimum fit score before being shortlisted.")

    st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)

    process_btn = st.button(
        "⚡ Process & Rank Candidates",
        type="primary",
        disabled=not uploaded_files,
        use_container_width=True
    )


# ──────────────────────────────────────────────────────────────────────────────
# PROCESSING LOGIC
# ──────────────────────────────────────────────────────────────────────────────
if process_btn:
    # 1. Clean temp directory
    temp_dir = SCRATCH_DIR / "temp_uploads"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    # 2. Save uploaded files
    with st.spinner("Saving uploaded PDF resumes to workspace..."):
        for f in uploaded_files:
            file_path = temp_dir / f.name
            with open(file_path, "wb") as buffer:
                buffer.write(f.getbuffer())

    # 3. Parse Custom JD if required
    if jd_source == "Paste Unstructured JD Text":
        if not raw_jd_text.strip():
            st.error("Please paste job description text first!")
            st.stop()
        with st.spinner("🧠 Parsing Job Description using LLM Agent..."):
            try:
                llm = LLMClient()
                jd_agent = JDParserAgent(llm)
                parsed_jd = jd_agent.parse_jd(raw_jd_text)
            except Exception as e:
                st.error(f"Failed to parse Job Description: {e}")
                st.stop()

    st.session_state.current_jd = parsed_jd

    # 4. Run pipeline with progress
    pipeline_stages = [
        "📄 Parsing PDFs",
        "🔍 OCR Fallback",
        "🧠 Extracting Info",
        "📊 Normalizing Grades",
        "🛠️ Extracting Skills",
        "🎯 Matching Skills",
        "📈 Scoring Candidates",
        "💡 Generating Explanations",
        "🏆 Ranking & Allocating Slots"
    ]

    progress_bar = st.progress(0)
    status_container = st.empty()

    # Show animated pipeline stages
    import time
    for i, stage in enumerate(pipeline_stages[:3]):
        progress_bar.progress((i + 1) / len(pipeline_stages))
        status_container.markdown(f"""
        <div style="display:flex; align-items:center; gap:12px; padding:12px 16px;
                    background:rgba(99,102,241,0.08); border:1px solid rgba(99,102,241,0.15);
                    border-radius:12px; margin:8px 0;">
            <div style="font-size:1.2rem;">{stage.split(' ')[0]}</div>
            <div style="color:#a5b4fc; font-weight:600; font-size:0.9rem;">{' '.join(stage.split(' ')[1:])}</div>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(0.3)

    with st.spinner("Running multi-agent pipeline..."):
        try:
            importlib.reload(main)
            candidates, report_paths = main.process_resumes(
                resumes_dir=str(temp_dir),
                parsed_jd=parsed_jd,
                limit=testing_limit,
                agents=get_cached_agents(),
                min_required_skill_match_ratio=min_required_skill_match_ratio,
                min_shortlist_score=min_shortlist_score,
            )
            st.session_state.ranked_candidates = candidates
            st.session_state.report_paths = report_paths
        except Exception as e:
            st.error(f"Pipeline crashed: {e}")
            st.exception(e)

    progress_bar.progress(1.0)
    time.sleep(0.3)
    progress_bar.empty()
    status_container.empty()
    st.success("✅ Evaluation pipeline complete!")
    st.balloons()


# ──────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────
def get_status_badge(cand):
    """Returns HTML badge for candidate status."""
    if cand.is_shortlisted:
        return '<span class="badge badge-shortlisted">✓ Shortlisted</span>'
    elif cand.is_reserve:
        return '<span class="badge badge-reserve">◉ Reserve</span>'
    else:
        return '<span class="badge badge-failed">✗ Review Needed</span>'

def get_confidence_badge(confidence):
    """Returns HTML badge for confidence level."""
    return f'<span class="badge badge-conf-{confidence.lower()}">{confidence} Confidence</span>'

def get_score_color(score):
    """Returns hex color based on score value."""
    if score >= 70:
        return "#34d399"
    elif score >= 45:
        return "#fbbf24"
    else:
        return "#f87171"

def render_score_ring(score):
    """Returns HTML for a circular score indicator."""
    color = get_score_color(score)
    circumference = 2 * 3.14159 * 45  # radius=45
    offset = circumference - (score / 100) * circumference
    return f"""
    <div class="score-ring-container">
        <div class="score-ring">
            <svg width="110" height="110" viewBox="0 0 110 110">
                <circle class="score-ring-bg" cx="55" cy="55" r="45"/>
                <circle class="score-ring-fill" cx="55" cy="55" r="45"
                    stroke="{color}"
                    stroke-dasharray="{circumference}"
                    stroke-dashoffset="{offset}"/>
            </svg>
            <div class="score-ring-text">{score:.0f}</div>
        </div>
        <div class="score-ring-label">Match Score</div>
    </div>
    """

def render_section_header(icon, text):
    """Returns HTML for a styled section header."""
    return f"""
    <div class="section-header">
        <span class="section-header-icon">{icon}</span>
        <span class="section-header-text">{text}</span>
        <div class="section-header-line"></div>
    </div>
    """


# ──────────────────────────────────────────────────────────────────────────────
# DASHBOARD VIEW — After Processing
# ──────────────────────────────────────────────────────────────────────────────
if st.session_state.ranked_candidates:
    candidates = st.session_state.ranked_candidates
    jd = st.session_state.current_jd
    reports = st.session_state.report_paths

    # ── KPI Metrics Bar ────────────────────────────────────────────────
    shortlisted_cnt = len([c for c in candidates if c.is_shortlisted])
    reserve_cnt = len([c for c in candidates if c.is_reserve])
    fail_cnt = len([c for c in candidates if c.parse_quality == "Failed"])

    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-icon">👥</div>
            <div class="kpi-value kpi-accent-blue">{len(candidates)}</div>
            <div class="kpi-label">Candidates Evaluated</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon">✅</div>
            <div class="kpi-value kpi-accent-green">{shortlisted_cnt}</div>
            <div class="kpi-label">Shortlisted (Slots: {jd.slots})</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon">⏳</div>
            <div class="kpi-value kpi-accent-amber">{reserve_cnt}</div>
            <div class="kpi-label">Reserve List</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon">⚠️</div>
            <div class="kpi-value kpi-accent-red">{fail_cnt}</div>
            <div class="kpi-label">Failed Parses</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Leaderboard ────────────────────────────────────────────────────
    st.markdown(render_section_header("🏆", "Shortlisting Leaderboard"), unsafe_allow_html=True)

    with st.expander("🔎 Filter Rankings", expanded=False):
        f_col1, f_col2 = st.columns([1, 1])
        with f_col1:
            confidence_filter = st.multiselect(
                "Confidence Level",
                options=["High", "Medium", "Low"],
                default=["High", "Medium", "Low"]
            )
        with f_col2:
            show_only_shortlisted = st.checkbox("Show shortlisted only", value=False)

    # Build leaderboard rows
    rows = []
    for rank, c in enumerate(candidates, 1):
        status = "Shortlisted" if c.is_shortlisted else ("Reserve" if c.is_reserve else "Failed Parse")

        if c.confidence not in confidence_filter:
            continue
        if show_only_shortlisted and not c.is_shortlisted:
            continue

        rows.append({
            "Rank": str(rank) if status != "Failed Parse" else "—",
            "Candidate": c.candidate_name,
            "Score": c.score,
            "CGPA": c.normalized_cgpa,
            "Status": status,
            "Confidence": c.confidence,
            "Parse Quality": c.parse_quality,
            "Resume File": c.resume_file
        })

    df_leaderboard = pd.DataFrame(rows)

    if not df_leaderboard.empty:
        st.dataframe(
            df_leaderboard,
            use_container_width=True,
            height=min(400, 45 + len(rows) * 38),
            column_config={
                "Score": st.column_config.ProgressColumn(
                    "Match Score",
                    min_value=0.0,
                    max_value=100.0,
                    format="%.1f"
                ),
                "CGPA": st.column_config.NumberColumn(
                    "Norm. CGPA",
                    format="%.2f"
                ),
            }
        )
    else:
        st.info("No candidates match the selected filters.")

    # ── Downloads ──────────────────────────────────────────────────────
    st.markdown(render_section_header("📥", "Download Reports"), unsafe_allow_html=True)

    dl1, dl2, dl3, dl4 = st.columns(4)
    report_configs = [
        (dl1, "csv", "📊 Leaderboard CSV", "text/csv"),
        (dl2, "json", "📋 Full JSON Data", "application/json"),
        (dl3, "markdown", "📝 Markdown Report", "text/markdown"),
        (dl4, "quality_report", "🔍 Parse Quality CSV", "text/csv"),
    ]
    for col, key, label, mime in report_configs:
        with col:
            if os.path.exists(reports[key]):
                with open(reports[key], "r", encoding="utf-8") as f:
                    st.download_button(
                        label,
                        data=f.read(),
                        file_name=os.path.basename(reports[key]),
                        mime=mime,
                        use_container_width=True
                    )

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ── Candidate Deep-Dive Inspector ──────────────────────────────────
    st.markdown(render_section_header("🔍", "Candidate Deep-Dive Inspector"), unsafe_allow_html=True)

    candidate_names = [c.candidate_name for c in candidates]
    selected_candidate_name = st.selectbox(
        "Select Candidate to Inspect",
        candidate_names,
        label_visibility="collapsed"
    )

    c_idx = candidate_names.index(selected_candidate_name)
    cand = candidates[c_idx]

    # Profile header
    initials = "".join([w[0].upper() for w in cand.candidate_name.split()[:2]]) if cand.candidate_name.split() else "?"
    status_badge = get_status_badge(cand)
    conf_badge = get_confidence_badge(cand.confidence)

    st.markdown(f"""
    <div class="inspector-card">
        <div class="profile-header">
            <div class="profile-avatar">{initials}</div>
            <div>
                <div class="profile-name">{cand.candidate_name}</div>
                <div class="profile-file">{cand.resume_file}</div>
            </div>
            <div style="margin-left: auto; display:flex; gap:8px; flex-wrap:wrap;">
                {status_badge}
                {conf_badge}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Tabbed Inspector
    tab_profile, tab_skills, tab_explanation = st.tabs(["📊 Profile & Score", "🛠️ Skill Matching", "💡 AI Explanation"])

    with tab_profile:
        p_col1, p_col2 = st.columns([1, 2])

        with p_col1:
            st.markdown(render_score_ring(cand.score), unsafe_allow_html=True)

        with p_col2:
            st.markdown(f"""
            <div class="inspector-card" style="padding:20px;">
                <div class="stat-row">
                    <span class="stat-label">Match Score</span>
                    <span class="stat-value" style="color:{get_score_color(cand.score)}">{cand.score:.2f} / 100</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Normalized CGPA</span>
                    <span class="stat-value">{cand.normalized_cgpa:.2f} / 10.0</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">ATS Score</span>
                    <span class="stat-value">{cand.ats_score:.1f} / 100</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Parse Quality</span>
                    <span class="stat-value">{cand.parse_quality}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Confidence</span>
                    <span class="stat-value">{cand.confidence}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Resume File</span>
                    <span class="stat-value" style="font-family:'SF Mono','Fira Code',monospace; font-size:0.82rem;">{cand.resume_file}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with tab_skills:
        # Skill Match Chips
        if cand.skills_matched:
            st.markdown("##### Skill-to-JD Match Results")

            skill_chips_html = ""
            for match in cand.skills_matched:
                chip_class = f"chip-{match.match_type}"
                icon_map = {"exact": "✓", "synonym": "≈", "partial": "◐", "implicit": "◐", "none": "✗"}
                icon = icon_map.get(match.match_type, "•")
                skill_chips_html += f'<span class="skill-chip {chip_class}">{icon} {match.skill}</span>'

            st.markdown(f'<div style="margin-bottom:20px;">{skill_chips_html}</div>', unsafe_allow_html=True)

            # Legend
            st.markdown("""
            <div style="display:flex; gap:16px; flex-wrap:wrap; margin-bottom:16px; font-size:0.78rem; color:#64748b;">
                <span><span style="color:#34d399">✓</span> Exact</span>
                <span><span style="color:#60a5fa">≈</span> Synonym</span>
                <span><span style="color:#fbbf24">◐</span> Partial/Implicit</span>
                <span><span style="color:#f87171">✗</span> Not Found</span>
            </div>
            """, unsafe_allow_html=True)

            # Detailed table
            rows_match = []
            for match in cand.skills_matched:
                rows_match.append({
                    "Skill": match.skill,
                    "Match": match.match_type.upper(),
                    "Similarity": match.score,
                    "Matched Term": match.matched_term or "—",
                    "Reason": match.reason
                })

            df_skills = pd.DataFrame(rows_match)
            st.dataframe(
                df_skills,
                use_container_width=True,
                column_config={
                    "Similarity": st.column_config.ProgressColumn(
                        "Similarity",
                        min_value=0.0,
                        max_value=1.0,
                        format="%.3f"
                    ),
                }
            )
        else:
            st.info("No skill matches computed (failed or empty parse).")

        # Extracted skills
        if cand.skills_extracted:
            st.markdown("##### All Extracted Skills")
            extracted_html = "".join(
                [f'<span class="skill-chip chip-extracted">{s}</span>' for s in cand.skills_extracted]
            )
            st.markdown(extracted_html, unsafe_allow_html=True)

    with tab_explanation:
        st.markdown("##### 🧠 AI-Generated Match Justification")

        if cand.explanation:
            bullets_html = '<ul class="explanation-list">'
            for bullet in cand.explanation:
                bullets_html += f"""
                <li class="explanation-item">
                    <span class="explanation-bullet">▸</span>
                    <span>{bullet}</span>
                </li>
                """
            bullets_html += "</ul>"
            st.markdown(bullets_html, unsafe_allow_html=True)
        else:
            st.info("No explanation generated for this candidate.")

# ──────────────────────────────────────────────────────────────────────────────
# LANDING PAGE — Before Processing (Hero + Features)
# ──────────────────────────────────────────────────────────────────────────────
else:
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">AI Resume Shortlisting Engine</div>
        <div class="hero-subtitle">
            Multi-agent pipeline that parses, normalizes, semantically matches,
            and explainably ranks candidates against your job description — all powered by AI.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature cards
    st.markdown("""
    <div class="features-grid">
        <div class="feature-card">
            <span class="feature-icon">📄</span>
            <div class="feature-title">Upload PDFs</div>
            <div class="feature-desc">Drag-and-drop single or batch resumes. Handles multi-column layouts, scanned images, and OCR fallback automatically.</div>
        </div>
        <div class="feature-card">
            <span class="feature-icon">🧠</span>
            <div class="feature-title">AI Multi-Agent Pipeline</div>
            <div class="feature-desc">9 specialized agents — parsing, OCR, extraction, grade normalization, skill matching, scoring, confidence, and explanation.</div>
        </div>
        <div class="feature-card">
            <span class="feature-icon">🏆</span>
            <div class="feature-title">Ranked Results</div>
            <div class="feature-desc">Explainable leaderboard with match scores, confidence levels, skill breakdowns, and downloadable CSV/JSON/Markdown reports.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Workflow bar
    st.markdown("""
    <div class="workflow-bar">
        <div class="wf-step">
            <div class="wf-step-num">1</div>
            Upload Resumes
        </div>
        <span class="wf-arrow">→</span>
        <div class="wf-step">
            <div class="wf-step-num">2</div>
            Set Job Description
        </div>
        <span class="wf-arrow">→</span>
        <div class="wf-step">
            <div class="wf-step-num">3</div>
            Process & Rank
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; color:#475569; font-size:0.88rem; padding-bottom:40px;">
        👈 Use the sidebar to upload resumes, select a role, and click <b>Process & Rank</b> to begin.
    </div>
    """, unsafe_allow_html=True)
