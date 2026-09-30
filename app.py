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
from main import process_resumes, build_pipeline_agents, DEFAULT_JDS
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
    return build_pipeline_agents()


# ──────────────────────────────────────────────────────────────────────────────
# PREMIUM CSS THEME — Glassmorphism Dark Mode with Animations
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
<style>
    :root { --bg:#f6f8fb; --surface:#fff; --line:#e2e8f0; --ink:#0f172a; --muted:#475569; --accent:#1d4ed8; --accent-soft:#eff4ff; --ok:#047857; --warn:#b45309; --bad:#b91c1c; }
    .stApp { background: var(--bg); color: var(--ink); font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif; }
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { max-width: 1180px; padding-top: 2rem; }
    h1, h2, h3, h4, h5, h6 { color: var(--ink); font-family: 'Inter', sans-serif; letter-spacing: -0.01em; }
    p, span, div, label { font-family: 'Inter', sans-serif; }
    section[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--line); }
    section[data-testid="stSidebar"] .stMarkdown h3 { font-size: .8rem; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); }
    .custom-divider { height: 1px; background: var(--line); margin: 20px 0; }

    /* Hero */
    .hero-container { background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 40px 44px; margin-bottom: 24px; }
    .hero-title { font-size: 2.1rem; font-weight: 700; color: var(--ink); line-height: 1.2; margin: 0 0 10px; }
    .hero-subtitle { font-size: 1.05rem; color: var(--muted); max-width: 680px; line-height: 1.6; margin: 0; }

    /* Cards */
    .features-grid, .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 24px; }
    .feature-card, .kpi-card, .inspector-card { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 20px; box-shadow: 0 1px 2px rgba(15,23,42,.04); }
    .feature-card:hover, .kpi-card:hover { border-color: #c7d2fe; box-shadow: 0 4px 12px rgba(15,23,42,.06); }
    .feature-icon, .kpi-icon { width: 36px; height: 36px; display: flex; align-items: center; justify-content: center; background: var(--accent-soft); border-radius: 8px; font-size: 1.1rem; margin-bottom: 12px; }
    .feature-title { font-weight: 600; color: var(--ink); margin-bottom: 4px; }
    .feature-desc, .kpi-label { font-size: .86rem; color: var(--muted); line-height: 1.5; }
    .kpi-value { font-size: 1.9rem; font-weight: 700; color: var(--ink); }
    .kpi-accent-blue { border-top: 3px solid var(--accent); } .kpi-accent-green { border-top: 3px solid var(--ok); }
    .kpi-accent-amber { border-top: 3px solid var(--warn); } .kpi-accent-red { border-top: 3px solid var(--bad); }

    /* Workflow */
    .workflow-bar { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 14px 18px; margin-bottom: 24px; }
    .wf-step { display: flex; align-items: center; gap: 8px; font-size: .86rem; font-weight: 500; color: var(--ink); }
    .wf-step-num, .sidebar-step-num { width: 22px; height: 22px; border-radius: 50%; background: var(--accent); color: #fff; font-size: .72rem; font-weight: 600; display: inline-flex; align-items: center; justify-content: center; }
    .wf-arrow { color: #475569; }
    .sidebar-step { display: flex; gap: 10px; align-items: center; margin-bottom: 8px; }
    .sidebar-step-text { font-size: .85rem; color: var(--muted); }

    /* Section header */
    .section-header { display: flex; align-items: center; gap: 10px; margin: 28px 0 14px; }
    .section-header-icon { font-size: 1rem; }
    .section-header-text { font-size: 1.15rem; font-weight: 600; color: var(--ink); }
    .section-header-line { flex: 1; height: 1px; background: var(--line); }

    /* Candidate profile */
    .profile-header { display: flex; align-items: center; gap: 14px; margin-bottom: 14px; }
    .profile-avatar { width: 44px; height: 44px; border-radius: 50%; background: var(--accent-soft); color: var(--accent); font-weight: 600; display: flex; align-items: center; justify-content: center; }
    .profile-name { font-size: 1.15rem; font-weight: 600; color: var(--ink); }
    .profile-file { font-size: .8rem; color: var(--muted); }
    .badge { display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: .75rem; font-weight: 600; border: 1px solid transparent; margin-right: 6px; }
    .badge-shortlisted, .badge-conf-high { background: #ecfdf5; color: var(--ok); border-color: #a7f3d0; }
    .badge-reserve, .badge-conf-medium { background: #fffbeb; color: var(--warn); border-color: #fde68a; }
    .badge-failed, .badge-conf-low { background: #fef2f2; color: var(--bad); border-color: #fecaca; }
    .score-ring-container { display: flex; flex-direction: column; align-items: center; gap: 8px; }
    .score-ring { position: relative; width: 110px; height: 110px; }
    .score-ring svg { transform: rotate(-90deg); }
    .score-ring-bg { fill: none; stroke: var(--line); stroke-width: 8; }
    .score-ring-fill { fill: none; stroke-width: 8; stroke-linecap: round; }
    .score-ring-text { position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%); font-size: 1.5rem; font-weight: 700; color: var(--ink); }
    .score-ring-label { font-size: .75rem; color: var(--muted); font-weight: 500; text-transform: uppercase; letter-spacing: .06em; }
    .stat-row { display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid var(--line); }
    .stat-row:last-child { border-bottom: none; }
    .stat-label { color: var(--muted); font-size: .86rem; } .stat-value { color: var(--ink); font-weight: 500; font-size: .9rem; }

    /* Skill chips and explanation */
    .skill-chip { display: inline-block; padding: 4px 11px; margin: 0 6px 6px 0; border-radius: 6px; font-size: .8rem; font-weight: 500; border: 1px solid var(--line); background: #0f172a; color: var(--ink); }
    .chip-exact { background: #ecfdf5; color: var(--ok); border-color: #a7f3d0; }
    .chip-synonym { background: var(--accent-soft); color: var(--accent); border-color: #c7d2fe; }
    .chip-partial, .chip-implicit { background: #fffbeb; color: var(--warn); border-color: #fde68a; }
    .chip-none { background: #fef2f2; color: var(--bad); border-color: #fecaca; }
    .chip-extracted { background: #0f172a; color: var(--muted); }
    .explanation-list { display: flex; flex-direction: column; gap: 10px; }
    .explanation-item { display: flex; gap: 12px; background: var(--surface); border: 1px solid var(--line); border-left: 3px solid var(--accent); border-radius: 8px; padding: 14px 16px; color: var(--ink); line-height: 1.55; }
    .explanation-bullet { color: var(--accent); font-weight: 600; }

    /* Streamlit widgets */
    button[data-testid="stBaseButton-primary"] { background: var(--accent) !important; color: #fff !important; border: none !important; border-radius: 8px !important; font-weight: 600 !important; }
    button[data-testid="stBaseButton-primary"]:hover { background: #1e40af !important; }
    .stDownloadButton > button { background: var(--surface); color: var(--ink); border: 1px solid var(--line); border-radius: 8px; font-weight: 500; }
    .stDownloadButton > button:hover { border-color: var(--accent); color: var(--accent); }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--line); }
    .stTabs [data-baseweb="tab"] { color: var(--muted); font-weight: 500; }
    .stTabs [aria-selected="true"] { color: var(--accent); }
    .stTabs [data-baseweb="tab-highlight"] { background: var(--accent); }
    .stDataFrame { border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
    @media (max-width: 768px) { .hero-container { padding: 24px; } .hero-title { font-size: 1.6rem; } }
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
        <div style="font-size: 1.1rem; font-weight: 800; color: #0f172a; margin-top: 4px; letter-spacing: -0.02em;">
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
            options=list(DEFAULT_JDS.keys()),
            format_func=lambda x: DEFAULT_JDS[x].role_name
        )
        parsed_jd = DEFAULT_JDS[selected_role_slug]

        st.markdown(f"""
        <div style="background: rgba(99, 102, 241, 0.08); border: 1px solid rgba(99, 102, 241, 0.15);
                    border-radius: 12px; padding: 14px; margin-top: 8px; font-size: 0.82rem;">
            <div style="color: #1d4ed8; font-weight: 600; margin-bottom: 6px;">📋 {parsed_jd.role_name}</div>
            <div style="color: #475569;">
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
                    background:#eff4ff; border:1px solid #c7d2fe;
                    border-radius:12px; margin:8px 0;">
            <div style="font-size:1.2rem;">{stage.split(' ')[0]}</div>
            <div style="color:#1d4ed8; font-weight:600; font-size:0.9rem;">{' '.join(stage.split(' ')[1:])}</div>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(0.3)

    pipeline_ok = False
    with st.spinner("Running multi-agent pipeline..."):
        try:
            candidates, report_paths = process_resumes(
                resumes_dir=str(temp_dir),
                parsed_jd=parsed_jd,
                limit=testing_limit,
                agents=get_cached_agents()
            )
            st.session_state.ranked_candidates = candidates
            st.session_state.report_paths = report_paths
            pipeline_ok = True
        except Exception as e:
            st.error(f"Pipeline crashed: {e}")
            st.exception(e)

    progress_bar.progress(1.0)
    time.sleep(0.3)
    progress_bar.empty()
    status_container.empty()
    if pipeline_ok:
        st.success("Evaluation pipeline complete.")


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
        return "#047857"
    elif score >= 45:
        return "#b45309"
    else:
        return "#b91c1c"

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
                <span><span style="color:#047857">✓</span> Exact</span>
                <span><span style="color:#1d4ed8">≈</span> Synonym</span>
                <span><span style="color:#b45309">◐</span> Partial/Implicit</span>
                <span><span style="color:#b91c1c">✗</span> Not Found</span>
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
