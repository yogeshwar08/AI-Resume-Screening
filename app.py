import html
import os
import tempfile
 
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
 
from src.pdf_parser import extract_text_from_pdf
from src.text_preprocessor import clean_text
from src.keyword_matcher import calculate_bm25_scores
from src.semantic_matcher import calculate_semantic_scores
from src.skill_matcher import calculate_skill_match
from src.profile_matcher import calculate_profile_match
from src.hybrid_scorer import calculate_hybrid_score
 
 
# ============================================================
# PAGE CONFIG
# ============================================================
 
st.set_page_config(
    page_title="AI Resume Screening | Talent Intelligence",
    page_icon="🏆",
    layout="wide",
)
 
 
# ============================================================
# PREMIUM CSS
# ============================================================
 
st.markdown(
    """
    <style>
 
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
 
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
 
    .stApp {
        background:
            radial-gradient(circle at 10% 0%, rgba(99,102,241,0.20), transparent 40%),
            radial-gradient(circle at 90% 10%, rgba(236,72,153,0.16), transparent 40%),
            radial-gradient(circle at 50% 100%, rgba(34,211,238,0.10), transparent 45%),
            #060912;
        color: #eef1fb;
    }
 
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1020 0%, #0d1326 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
 
    /* Header */
    .main-title {
        font-size: 48px;
        font-weight: 900;
        text-align: center;
        letter-spacing: -0.02em;
        margin-bottom: 6px;
        background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6, #fbbf24);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
 
    .subtitle {
        text-align: center;
        color: #9aa4c2;
        font-size: 17px;
        font-weight: 500;
        margin-bottom: 34px;
    }
 
    .badge-row {
        display: flex;
        justify-content: center;
        gap: 10px;
        flex-wrap: wrap;
        margin-bottom: 28px;
    }
 
    .badge {
        padding: 6px 14px;
        border-radius: 999px;
        font-size: 12.5px;
        font-weight: 600;
        color: #cdd6f4;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10);
    }
 
    /* Section titles */
    .section-title {
        font-size: 23px;
        font-weight: 800;
        letter-spacing: -0.01em;
        margin-top: 34px;
        margin-bottom: 14px;
        color: #f4f6ff;
    }
 
    /* Panels */
    .panel {
        background: linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0.015));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 22px 24px;
    }
 
    /* Winner card */
    .winner-card {
        padding: 34px;
        border-radius: 26px;
        background: linear-gradient(135deg, rgba(99,102,241,0.30), rgba(236,72,153,0.22) 60%, rgba(251,191,36,0.10));
        border: 1px solid rgba(255,255,255,0.14);
        box-shadow: 0 20px 60px rgba(0,0,0,0.45), inset 0 0 0 1px rgba(255,255,255,0.03);
        text-align: center;
        margin: 22px 0 30px 0;
        position: relative;
        overflow: hidden;
    }
 
    .winner-title {
        font-size: 13px;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #fbbf24;
        font-weight: 800;
    }
 
    .winner-name {
        font-size: 34px;
        font-weight: 900;
        margin: 10px 0 4px 0;
        color: #ffffff;
    }
 
    .winner-score {
        font-size: 56px;
        font-weight: 900;
        background: linear-gradient(90deg, #60a5fa, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
    }
 
    .winner-sub {
        color: #b7bfdd;
        font-size: 14px;
        font-weight: 500;
        margin-top: 2px;
    }
 
    /* ========================================================
       MULTI-GRADIENT ANIMATED SLIDEBARS (skill bars)
       ======================================================== */
 
    .skillbar-row {
        margin-bottom: 16px;
    }
 
    .skillbar-label {
        display: flex;
        justify-content: space-between;
        font-size: 13.5px;
        font-weight: 600;
        color: #d7deee;
        margin-bottom: 6px;
    }
 
    .skillbar-value {
        font-weight: 800;
        color: #ffffff;
    }
 
    .skillbar-track {
        width: 100%;
        height: 11px;
        border-radius: 999px;
        background: rgba(255,255,255,0.07);
        overflow: hidden;
        position: relative;
    }
 
    .skillbar-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #60a5fa, #818cf8, #a78bfa, #e879f9, #f472b6);
        background-size: 220% 100%;
        box-shadow: 0 0 14px rgba(167,139,250,0.55);
        position: relative;
        overflow: hidden;
        animation: slidebarFlow 4s linear infinite;
        transition: width 0.7s ease;
    }
 
    .skillbar-fill::after {
        content: "";
        position: absolute;
        top: 0;
        left: -60%;
        width: 40%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.55), transparent);
        animation: slidebarShine 2.6s ease-in-out infinite;
    }
 
    @keyframes slidebarFlow {
        0%   { background-position: 0% 50%; }
        100% { background-position: 220% 50%; }
    }
 
    @keyframes slidebarShine {
        0%   { left: -60%; }
        100% { left: 130%; }
    }
 
    /* Rank chip */
    .rank-chip {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 34px;
        height: 34px;
        padding: 0 8px;
        border-radius: 10px;
        font-weight: 800;
        font-size: 14px;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.12);
        margin-right: 8px;
    }
 
    .rank-gold   { background: linear-gradient(135deg,#fbbf24,#f59e0b); color:#1a1200; }
    .rank-silver { background: linear-gradient(135deg,#e5e7eb,#9ca3af); color:#111; }
    .rank-bronze { background: linear-gradient(135deg,#f0b27a,#c07f4a); color:#1a1200; }
 
    /* Buttons */
    .stButton > button, .stDownloadButton > button {
        border-radius: 12px !important;
        font-weight: 700 !important;
        border: none !important;
        background: linear-gradient(90deg, #6366f1, #a855f7) !important;
        box-shadow: 0 8px 24px rgba(99,102,241,0.35);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
 
    .stButton > button:hover, .stDownloadButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 12px 30px rgba(168,85,247,0.45);
    }
 
    /* Metrics */
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.045);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 14px 16px;
    }
 
    /* Dataframe */
    div[data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.08);
    }
 
    /* Expander */
    details {
        border-radius: 14px !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        background: rgba(255,255,255,0.03) !important;
    }
 
    /* Hero stat strip */
    .hero-stats {
        display: flex;
        gap: 14px;
        flex-wrap: wrap;
        justify-content: center;
        margin: 10px 0 8px 0;
    }
 
    .hero-stat {
        flex: 1;
        min-width: 150px;
        text-align: center;
        padding: 16px 10px;
        border-radius: 16px;
        background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.015));
        border: 1px solid rgba(255,255,255,0.08);
    }
 
    .hero-stat-value {
        font-size: 28px;
        font-weight: 900;
        background: linear-gradient(90deg, #60a5fa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
 
    .hero-stat-label {
        font-size: 12.5px;
        color: #9aa4c2;
        font-weight: 600;
        margin-top: 2px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
 
    /* ========================================================
       MULTI-GRADIENT TOP-3 SUMMARY CARDS
       ======================================================== */
 
    .top3-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 18px;
        width: 100%;
        margin: 18px 0 28px 0;
    }
 
    .top3-card {
        position: relative;
        border-radius: 18px;
        padding: 2px;
        background: var(--card-border-gradient);
        box-shadow: 0 18px 45px rgba(0,0,0,0.40);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
 
    .top3-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 26px 60px rgba(0,0,0,0.5), 0 0 30px var(--card-glow);
    }
 
    .top3-card-inner {
        background: linear-gradient(160deg, rgba(10,12,24,0.94), rgba(10,12,24,0.86));
        border-radius: 16px;
        padding: 20px;
        display: flex;
        flex-direction: column;
        gap: 12px;
        height: 100%;
    }
 
    .top3-card.rank-1 {
        --card-border-gradient: linear-gradient(135deg, #fde68a, #fbbf24, #f59e0b, #f472b6);
        --card-glow: rgba(251,191,36,0.35);
    }
    .top3-card.rank-2 {
        --card-border-gradient: linear-gradient(135deg, #e5e7eb, #a5b4fc, #818cf8, #60a5fa);
        --card-glow: rgba(129,140,248,0.30);
    }
    .top3-card.rank-3 {
        --card-border-gradient: linear-gradient(135deg, #fbcfa0, #f0b27a, #c07f4a, #f472b6);
        --card-glow: rgba(240,178,122,0.30);
    }
 
    .top3-rank-label {
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 0.10em;
        text-transform: uppercase;
        background: var(--card-border-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
 
    .top3-header {
        display: flex;
        align-items: center;
        gap: 12px;
    }
 
    .top3-avatar {
        width: 44px;
        height: 44px;
        min-width: 44px;
        border-radius: 50%;
        background: var(--card-border-gradient);
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 14px;
        color: #0a0c18;
        box-shadow: 0 0 16px var(--card-glow);
    }
 
    .top3-name {
        font-weight: 700;
        font-size: 14.5px;
        color: #ffffff;
        overflow-wrap: anywhere;
        word-break: break-word;
    }
 
    .top3-score-row {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin-top: 4px;
    }
 
    .top3-score-label {
        color: #9aa4c2;
        font-size: 12.5px;
        font-weight: 600;
    }
 
    .top3-score-value {
        font-size: 27px;
        font-weight: 900;
        background: var(--card-border-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
 
    .top3-score-track {
        height: 8px;
        border-radius: 999px;
        background: rgba(255,255,255,0.08);
        overflow: hidden;
        position: relative;
    }
 
    .top3-score-fill {
        height: 100%;
        border-radius: 999px;
        background: var(--card-border-gradient);
        background-size: 220% 100%;
        box-shadow: 0 0 14px var(--card-glow);
        position: relative;
        overflow: hidden;
        animation: slidebarFlow 4s linear infinite;
        transition: width 0.7s ease;
    }
 
    .top3-score-fill::after {
        content: "";
        position: absolute;
        top: 0;
        left: -60%;
        width: 40%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.65), transparent);
        animation: slidebarShine 2.6s ease-in-out infinite;
    }
 
    @media (max-width: 800px) {
        .top3-grid {
            grid-template-columns: 1fr;
        }
    }
 
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        background: rgba(255,255,255,0.04);
        padding: 8px 16px;
    }
 
    </style>
    """,
    unsafe_allow_html=True,
)
 
 
# ============================================================
# HELPERS
# ============================================================
 
def clamp_pct(value: float) -> float:
    """Clamp a score into the 0-100 range, keeping one decimal of precision."""
    try:
        return max(0.0, min(round(float(value), 1), 100.0))
    except (TypeError, ValueError):
        return 0.0
 
 
def render_skillbar(label: str, value: float) -> None:
    pct = clamp_pct(value)
    safe_label = html.escape(str(label))
    st.markdown(
        f"""
        <div class="skillbar-row">
            <div class="skillbar-label">
                <span>{safe_label}</span>
                <span class="skillbar-value">{pct}%</span>
            </div>
            <div class="skillbar-track">
                <div class="skillbar-fill" style="width:{pct}%;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
 
 
def get_initials(name: str) -> str:
    """Derive 1-2 letter initials from a resume filename for the avatar chip."""
    base = os.path.splitext(str(name))[0]
    tokens = [t for t in base.replace("_", " ").replace("-", " ").replace(".", " ").split() if t]
    if not tokens:
        return "?"
    if len(tokens) == 1:
        return tokens[0][:2].upper()
    return (tokens[0][0] + tokens[1][0]).upper()
 
 
def render_podium(results: list[dict]) -> None:
    """Render a multi-gradient Top-3 summary row from the sorted results."""
 
    if not results:
        return
 
    top3 = results[:3]
    blocks = []
 
    for i, candidate in enumerate(top3):
        rank = i + 1
        safe_name = html.escape(str(candidate.get("Candidate", "Unknown Candidate")))
        initials = html.escape(get_initials(candidate.get("Candidate", "")))
        score = clamp_pct(candidate.get("Final Score", 0))
 
        # Built as single-line HTML with no leading indentation so
        # Streamlit's Markdown parser doesn't treat it as a code block.
        block = (
            f'<div class="top3-card rank-{rank}">'
            f'<div class="top3-card-inner">'
            f'<div class="top3-rank-label">Rank {rank}</div>'
            f'<div class="top3-header">'
            f'<div class="top3-avatar">{initials}</div>'
            f'<div class="top3-name">{safe_name}</div>'
            f'</div>'
            f'<div class="top3-score-row">'
            f'<span class="top3-score-label">Final Score</span>'
            f'<span class="top3-score-value">{score}%</span>'
            f'</div>'
            f'<div class="top3-score-track">'
            f'<div class="top3-score-fill" style="width:{score}%;"></div>'
            f'</div>'
            f'</div>'
            f'</div>'
        )
        blocks.append(block)
 
    st.markdown(f'<div class="top3-grid">{"".join(blocks)}</div>', unsafe_allow_html=True)
 
 
def render_radar_chart(results: list[dict], max_candidates: int = 5) -> None:
    """Overlaid radar chart comparing top candidates across all 7 scoring signals."""
    dimensions = ["BM25", "Semantic", "Skills", "Profile", "Education", "Role", "Experience"]
    top_n = results[:max_candidates]
 
    fig = go.Figure()
 
    palette = ["#60a5fa", "#f472b6", "#fbbf24", "#34d399", "#a78bfa"]
 
    for i, candidate in enumerate(top_n):
        values = [candidate[dim] for dim in dimensions]
        fig.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=dimensions + [dimensions[0]],
                fill="toself",
                name=candidate["Candidate"],
                line=dict(color=palette[i % len(palette)], width=2),
                fillcolor=palette[i % len(palette)],
                opacity=0.35,
            )
        )
 
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(255,255,255,0.12)",
                             tickfont=dict(color="#9aa4c2", size=10)),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.12)", tickfont=dict(color="#e6e9f5", size=12)),
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, x=0.5, xanchor="center",
                    font=dict(color="#d7deee")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=20, b=40),
        height=440,
    )
 
    st.plotly_chart(fig, use_container_width=True)
 
 
def rank_chip_html(rank: int) -> str:
    if rank == 1:
        return '<span class="rank-chip rank-gold">🥇</span>'
    if rank == 2:
        return '<span class="rank-chip rank-silver">🥈</span>'
    if rank == 3:
        return '<span class="rank-chip rank-bronze">🥉</span>'
    return f'<span class="rank-chip">#{rank}</span>'
 
 
@st.cache_data(show_spinner=False)
def extract_and_clean_resume(file_bytes: bytes, filename: str) -> tuple[str, str | None]:
    """
    Extract + clean text from a single uploaded PDF.
    Returns (cleaned_text, error_message). error_message is None on success.
    Cached on file content, so re-runs of the Streamlit script (e.g. from
    expanding a card) don't re-parse PDFs that haven't changed.
    """
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file_bytes)
            temp_path = temp_file.name
 
        raw_text = extract_text_from_pdf(temp_path)
 
        if not raw_text or not raw_text.strip():
            return "", f"No extractable text found in '{filename}' (scanned image or empty PDF?)."
 
        return clean_text(raw_text), None
 
    except Exception as exc:  # noqa: BLE001 - isolate per-file failures
        return "", f"Failed to read '{filename}': {exc}"
 
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
 
 
@st.cache_data(show_spinner=False)
def score_candidates(job_description: str, resumes: tuple[str, ...]) -> dict:
    """Run all scoring engines once per (JD, resume-set) combination."""
    bm25_scores = calculate_bm25_scores(job_description, list(resumes))
    semantic_scores = calculate_semantic_scores(job_description, list(resumes))
 
    skill_scores = []
    profile_results = []
 
    for resume in resumes:
        skill_result = calculate_skill_match(job_description, resume)
        skill_scores.append(skill_result["score"])
 
        profile_result = calculate_profile_match(job_description, resume)
        profile_results.append(profile_result)
 
    return {
        "bm25": bm25_scores,
        "semantic": semantic_scores,
        "skills": skill_scores,
        "profiles": profile_results,
    }
 
 
# ============================================================
# HEADER
# ============================================================
 
st.markdown('<div class="main-title">AI Resume Screening</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Intelligent Resume Ranking &bull; Hybrid AI Matching &bull; Candidate Comparison</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="badge-row">
        <span class="badge">🔎 BM25 Keyword Match</span>
        <span class="badge">🧠 Semantic Similarity</span>
        <span class="badge">🛠️ Skill Matching</span>
        <span class="badge">👤 Profile Matching</span>
        <span class="badge">🎓 Education</span>
        <span class="badge">💼 Role Fit</span>
        <span class="badge">⏳ Experience</span>
    </div>
    """,
    unsafe_allow_html=True,
)
 
 
# ============================================================
# SIDEBAR
# ============================================================
 
with st.sidebar:
    st.markdown("## ⚙️ Screening Settings")
    st.info(
        "Upload up to **10 resumes**.\n\n"
        "Every resume is compared against the same Job Description "
        "using a weighted hybrid of seven scoring signals."
    )
    st.markdown("---")
    st.markdown("### 📊 Scoring Engine")
    for line in [
        "🔎 BM25 Keyword Matching",
        "🧠 Semantic Similarity",
        "🛠️ Skill Matching",
        "👤 Profile Matching",
        "🎓 Education Matching",
        "💼 Role Matching",
        "⏳ Experience Matching",
    ]:
        st.write(line)
 
 
# ============================================================
# JOB DESCRIPTION
# ============================================================
 
st.markdown('<div class="section-title">📋 Job Description</div>', unsafe_allow_html=True)
 
job_description_input = st.text_area(
    "Paste the job description here",
    height=250,
    placeholder=(
        "Example:\n\n"
        "We are looking for an AI/ML Engineer with "
        "Python, Machine Learning, TensorFlow, "
        "LangChain and NLP experience..."
    ),
)
 
 
# ============================================================
# RESUME UPLOAD
# ============================================================
 
st.markdown('<div class="section-title">📄 Upload Resumes</div>', unsafe_allow_html=True)
 
uploaded_files = st.file_uploader(
    "Upload candidate resumes",
    type=["pdf"],
    accept_multiple_files=True,
)
 
if uploaded_files:
    if len(uploaded_files) > 10:
        st.error("❌ Maximum 10 resumes can be uploaded.")
        st.stop()
 
    st.success(f"✅ {len(uploaded_files)} resume(s) uploaded")
    for file in uploaded_files:
        st.write(f"📄 {html.escape(file.name)}")
 
 
# ============================================================
# ANALYZE BUTTON
# ============================================================
 
analyze = st.button("🚀 Analyze & Rank Candidates", use_container_width=True, type="primary")
 
 
# ============================================================
# ANALYSIS
# ============================================================
 
if analyze:
 
    if not job_description_input.strip():
        st.error("Please enter a Job Description.")
        st.stop()
 
    if not uploaded_files:
        st.error("Please upload at least one resume.")
        st.stop()
 
    if len(uploaded_files) > 10:
        st.error("Maximum 10 resumes are allowed.")
        st.stop()
 
    job_description = clean_text(job_description_input)
 
    # --------------------------------------------------------
    # EXTRACT RESUMES (cached, isolated per file)
    # --------------------------------------------------------
 
    resumes = []
    resume_names = []
    extraction_errors = []
 
    progress = st.progress(0)
    status = st.empty()
 
    for index, uploaded_file in enumerate(uploaded_files):
        status.write(f"📖 Reading {html.escape(uploaded_file.name)}...")
 
        file_bytes = uploaded_file.getvalue()
        cleaned_text, error = extract_and_clean_resume(file_bytes, uploaded_file.name)
 
        if error:
            extraction_errors.append(error)
        else:
            resumes.append(cleaned_text)
            resume_names.append(uploaded_file.name)
 
        progress.progress((index + 1) / len(uploaded_files))
 
    status.write("✅ Resume extraction completed.")
 
    if extraction_errors:
        with st.expander(f"⚠️ {len(extraction_errors)} resume(s) skipped", expanded=True):
            for err in extraction_errors:
                st.warning(html.escape(err))
 
    if not resumes:
        st.error("No resumes could be processed. Please check the uploaded files and try again.")
        st.stop()
 
    # --------------------------------------------------------
    # SCORING (cached on JD + resume content)
    # --------------------------------------------------------
 
    with st.spinner("🧠 Running hybrid scoring engine..."):
        scores = score_candidates(job_description, tuple(resumes))
 
    bm25_scores = scores["bm25"]
    semantic_scores = scores["semantic"]
    skill_scores = scores["skills"]
    profile_results = scores["profiles"]
 
    # --------------------------------------------------------
    # FINAL HYBRID SCORE
    # --------------------------------------------------------
 
    results = []
 
    for i, name in enumerate(resume_names):
        profile_score = profile_results[i]["score"]
 
        final_score = calculate_hybrid_score(
            bm25_scores[i],
            semantic_scores[i],
            skill_scores[i],
            profile_score,
        )
 
        results.append(
            {
                "Candidate": name,
                "BM25": clamp_pct(bm25_scores[i]),
                "Semantic": clamp_pct(semantic_scores[i]),
                "Skills": clamp_pct(skill_scores[i]),
                "Profile": clamp_pct(profile_score),
                "Education": clamp_pct(profile_results[i]["education_score"]),
                "Role": clamp_pct(profile_results[i]["role_score"]),
                "Experience": clamp_pct(profile_results[i]["experience_score"]),
                "Required Experience": profile_results[i]["required_years"],
                "Candidate Experience": profile_results[i]["candidate_years"],
                "Final Score": clamp_pct(final_score),
            }
        )
 
    results.sort(key=lambda x: x["Final Score"], reverse=True)
 
    for index, result in enumerate(results, start=1):
        result["Rank"] = index
 
    winner = results[0]
    safe_winner_name = html.escape(winner["Candidate"])
 
    # --------------------------------------------------------
    # WINNER
    # --------------------------------------------------------
 
    st.markdown('<div class="section-title">🏆 Recommendation</div>', unsafe_allow_html=True)
 
    st.markdown(
        f"""
        <div class="winner-card">
            <div class="winner-title">Top Candidate</div>
            <div class="winner-name">{safe_winner_name}</div>
            <div class="winner-score">{winner["Final Score"]}%</div>
            <div class="winner-sub">Overall Hybrid Recommendation Score</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
 
    avg_score = round(sum(r["Final Score"] for r in results) / len(results), 1)
    top_gap = round(winner["Final Score"] - (results[1]["Final Score"] if len(results) > 1 else 0), 1)
 
    st.markdown(
        f"""
        <div class="hero-stats">
            <div class="hero-stat">
                <div class="hero-stat-value">{len(results)}</div>
                <div class="hero-stat-label">Candidates Screened</div>
            </div>
            <div class="hero-stat">
                <div class="hero-stat-value">{winner['Final Score']}%</div>
                <div class="hero-stat-label">Top Score</div>
            </div>
            <div class="hero-stat">
                <div class="hero-stat-value">{avg_score}%</div>
                <div class="hero-stat-label">Pool Average</div>
            </div>
            <div class="hero-stat">
                <div class="hero-stat-value">+{top_gap}</div>
                <div class="hero-stat-label">Lead Over #2</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
 
    # --------------------------------------------------------
    # TOP 3 SUMMARY CARDS
    # --------------------------------------------------------
 
    st.markdown('<div class="section-title">🏅 Top 3 Candidates</div>', unsafe_allow_html=True)
    render_podium(results)
 
    # --------------------------------------------------------
    # WINNER BREAKDOWN + RADAR (tabbed)
    # --------------------------------------------------------
 
    st.markdown('<div class="section-title">📊 Score Breakdown & Comparison</div>', unsafe_allow_html=True)
 
    tab_winner, tab_radar = st.tabs(["🏆 Winner Breakdown", "🕸️ Multi-Candidate Radar"])
 
    with tab_winner:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        bcol1, bcol2 = st.columns(2)
        winner_metrics = [
            ("BM25 Keyword Match", winner["BM25"]),
            ("Semantic Similarity", winner["Semantic"]),
            ("Skill Match", winner["Skills"]),
            ("Profile Match", winner["Profile"]),
            ("Education", winner["Education"]),
            ("Role Fit", winner["Role"]),
            ("Experience", winner["Experience"]),
        ]
        for idx, (label, value) in enumerate(winner_metrics):
            with (bcol1 if idx % 2 == 0 else bcol2):
                render_skillbar(label, value)
        st.markdown("</div>", unsafe_allow_html=True)
 
    with tab_radar:
        st.caption("Overlaying up to 5 top candidates across all seven scoring dimensions.")
        render_radar_chart(results, max_candidates=5)
 
    # --------------------------------------------------------
    # CANDIDATE COMPARISON TABLE
    # --------------------------------------------------------
 
    st.markdown('<div class="section-title">📈 Candidate Comparison</div>', unsafe_allow_html=True)
 
    dataframe = pd.DataFrame(results)
 
    display_columns = [
        "Rank",
        "Candidate",
        "Final Score",
        "BM25",
        "Semantic",
        "Skills",
        "Profile",
        "Education",
        "Role",
        "Experience",
        "Required Experience",
        "Candidate Experience",
    ]
 
    st.dataframe(
        dataframe[display_columns],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Final Score": st.column_config.ProgressColumn(
                "Final Score", min_value=0, max_value=100, format="%.1f%%"
            ),
            "BM25": st.column_config.ProgressColumn("BM25", min_value=0, max_value=100, format="%.1f%%"),
            "Semantic": st.column_config.ProgressColumn("Semantic", min_value=0, max_value=100, format="%.1f%%"),
            "Skills": st.column_config.ProgressColumn("Skills", min_value=0, max_value=100, format="%.1f%%"),
            "Profile": st.column_config.ProgressColumn("Profile", min_value=0, max_value=100, format="%.1f%%"),
        },
    )
 
    # --------------------------------------------------------
    # RANKING CARDS
    # --------------------------------------------------------
 
    st.markdown('<div class="section-title">🥇 Candidate Ranking</div>', unsafe_allow_html=True)
 
    for result in results:
        rank = result["Rank"]
        safe_name = html.escape(result["Candidate"])
 
        with st.expander(f"{'🥇' if rank == 1 else '🥈' if rank == 2 else '🥉' if rank == 3 else f'#{rank}'}  {safe_name} — {result['Final Score']}%"):
 
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Final Score", f'{result["Final Score"]}%')
            with m2:
                st.metric("Skills", f'{result["Skills"]}%')
            with m3:
                st.metric("Profile", f'{result["Profile"]}%')
            with m4:
                st.metric("Experience", f'{result["Candidate Experience"]} yrs')
 
            st.markdown("<br/>", unsafe_allow_html=True)
 
            render_skillbar("BM25 Keyword Match", result["BM25"])
            render_skillbar("Semantic Similarity", result["Semantic"])
            render_skillbar("Skill Match", result["Skills"])
            render_skillbar("Profile Match", result["Profile"])
 
            st.write(f"**Education:** {result['Education']}%")
            st.write(f"**Role:** {result['Role']}%")
            st.write(f"**Experience Match:** {result['Experience']}%")
            st.write(f"**Required Experience:** {result['Required Experience']} years")
            st.write(f"**Candidate Experience:** {result['Candidate Experience']} years")
 
    # --------------------------------------------------------
    # EXPORT
    # --------------------------------------------------------
 
    st.markdown('<div class="section-title">📥 Export Results</div>', unsafe_allow_html=True)
 
    csv_data = dataframe[display_columns].to_csv(index=False)
 
    st.download_button(
        label="⬇️ Download Candidate Ranking CSV",
        data=csv_data,
        file_name="resume_screening_results.csv",
        mime="text/csv",
        use_container_width=True,
    )
 
    st.success(
        f"🎉 Analysis completed! {len(results)} candidate(s) compared. "
        f"Winner: {winner['Candidate']} with {winner['Final Score']}%."
    )
 
    if extraction_errors:
        st.caption(f"Note: {len(extraction_errors)} uploaded file(s) were excluded due to read errors — see warning above.")
 
    st.balloons()
 
else:
    st.markdown(
        """
        <div class="panel" style="text-align:center; margin-top: 20px;">
            <div style="font-size:15px; color:#9aa4c2;">
                Paste a job description, upload up to 10 resumes, then hit
                <strong>Analyze &amp; Rank Candidates</strong> to generate a
                full hybrid scoring report — Top 3 summary, radar comparison,
                candidate-by-candidate breakdown, and a downloadable CSV.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
 






