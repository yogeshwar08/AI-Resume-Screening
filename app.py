import gc
import html
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import warnings

# ============================================================
# RUNTIME LOGGING CONFIGURATION
# ============================================================

os.environ["TRANSFORMERS_VERBOSITY"] = "critical"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings("ignore")

logging.getLogger("transformers").setLevel(logging.CRITICAL)
logging.getLogger("transformers.utils").setLevel(logging.CRITICAL)
logging.getLogger("huggingface_hub").setLevel(logging.CRITICAL)
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

    /* ========================================================
       FILE UPLOADER — dark background, white text
       Uses [data-testid*="FileUploader"] (contains-match) so it
       catches every element Streamlit tags with "FileUploader"
       anywhere in the testid, regardless of exact naming across
       versions — plus a class-based fallback with maximum
       specificity so it wins over Streamlit's own inline styles.
       ======================================================== */

    [data-testid*="FileUploader"] {
        background-color: #0a0c18 !important;
        color: #ffffff !important;
        border-color: rgba(255,255,255,0.18) !important;
    }

    [data-testid*="FileUploader"] * {
        color: #ffffff !important;
        background-color: transparent !important;
    }

    [data-testid*="FileUploader"] section {
        background-color: #0a0c18 !important;
        border: 1px dashed rgba(255,255,255,0.25) !important;
        border-radius: 14px !important;
    }

    [data-testid*="FileUploader"] button {
        background: linear-gradient(90deg, #6366f1, #a855f7) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
    }

    [data-testid*="FileUploader"] svg {
        fill: #ffffff !important;
        stroke: #ffffff !important;
    }

    [data-testid*="FileUploader"] small,
    [data-testid*="FileUploader"] span {
        color: #d7deee !important;
    }

    /* ========================================================
       SIDEBAR — make label/body text readable on the dark panel
       ======================================================== */

    section[data-testid="stSidebar"] * {
        color: #e6e9f5 !important;
    }

    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p {
        color: #e6e9f5 !important;
    }

    /* ========================================================
       CUSTOM COMPARISON TABLE — dark, multi-gradient, replaces
       the default st.dataframe grid (which can't be restyled
       with CSS since it renders on a canvas, not real HTML).
       ======================================================== */

    .cmp-table-wrap {
        overflow-x: auto;
        border-radius: 18px;
        border: 1px solid rgba(255,255,255,0.08);
        background: linear-gradient(160deg, rgba(10,12,24,0.96), rgba(10,12,24,0.88));
        box-shadow: 0 18px 45px rgba(0,0,0,0.35);
    }

    table.cmp-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13.5px;
        color: #eef1fb;
    }

    table.cmp-table thead th {
        text-align: left;
        padding: 14px 16px;
        font-weight: 800;
        font-size: 12px;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #9aa4c2;
        background: rgba(255,255,255,0.04);
        border-bottom: 1px solid rgba(255,255,255,0.08);
        white-space: nowrap;
    }

    table.cmp-table tbody td {
        padding: 12px 16px;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        white-space: nowrap;
        vertical-align: middle;
    }

    table.cmp-table tbody tr:last-child td {
        border-bottom: none;
    }

    table.cmp-table tbody tr:hover {
        background: rgba(255,255,255,0.03);
    }

    .cmp-rank {
        font-weight: 800;
        color: #cdd6f4;
    }

    .cmp-candidate {
        font-weight: 700;
        color: #ffffff;
    }

    .cmp-bar-cell {
        min-width: 150px;
    }

    .cmp-bar-row {
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .cmp-bar-track {
        flex: 1;
        height: 9px;
        border-radius: 999px;
        background: rgba(255,255,255,0.08);
        overflow: hidden;
        position: relative;
        min-width: 80px;
    }

    .cmp-bar-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #60a5fa, #818cf8, #a78bfa, #e879f9, #f472b6);
        background-size: 220% 100%;
        box-shadow: 0 0 10px rgba(167,139,250,0.5);
        animation: slidebarFlow 4s linear infinite;
    }

    .cmp-bar-fill.final {
        background: linear-gradient(90deg, #fde68a, #fbbf24, #f59e0b, #f472b6);
        box-shadow: 0 0 10px rgba(251,191,36,0.5);
    }

    .cmp-bar-value {
        min-width: 46px;
        text-align: right;
        font-weight: 700;
        color: #eef1fb;
    }

    .cmp-plain {
        color: #cdd6f4;
        font-weight: 600;
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


def _cmp_bar_cell(value: float, is_final: bool = False) -> str:
    """Build one gradient progress-bar cell for the comparison table."""
    pct = clamp_pct(value)
    fill_class = "cmp-bar-fill final" if is_final else "cmp-bar-fill"
    return (
        f'<td class="cmp-bar-cell">'
        f'<div class="cmp-bar-row">'
        f'<div class="cmp-bar-track"><div class="{fill_class}" style="width:{pct}%;"></div></div>'
        f'<span class="cmp-bar-value">{pct}%</span>'
        f'</div>'
        f'</td>'
    )


def render_comparison_table(results: list[dict]) -> None:
    """
    Render the candidate comparison table as custom dark, multi-gradient
    HTML instead of st.dataframe. st.dataframe renders via a canvas-based
    grid component internally, so CSS can't restyle its cell colors —
    building real HTML here gives full control and matches the rest of
    the app's dark gradient theme.
    """
    rows_html = []

    for r in results:
        safe_name = html.escape(str(r["Candidate"]))

        row = (
            "<tr>"
            f'<td class="cmp-rank">#{r["Rank"]}</td>'
            f'<td class="cmp-candidate">{safe_name}</td>'
            + _cmp_bar_cell(r["Final Score"], is_final=True)
            + _cmp_bar_cell(r["BM25"])
            + _cmp_bar_cell(r["Semantic"])
            + _cmp_bar_cell(r["Skills"])
            + _cmp_bar_cell(r["Profile"])
            + f'<td class="cmp-plain">{clamp_pct(r["Education"])}%</td>'
            + f'<td class="cmp-plain">{clamp_pct(r["Role"])}%</td>'
            + f'<td class="cmp-plain">{clamp_pct(r["Experience"])}%</td>'
            + f'<td class="cmp-plain">{r["Required Experience"]} yrs</td>'
            + f'<td class="cmp-plain">{r["Candidate Experience"]} yrs</td>'
            + "</tr>"
        )
        rows_html.append(row)

    table_html = (
        '<div class="cmp-table-wrap">'
        '<table class="cmp-table">'
        "<thead><tr>"
        "<th>Rank</th><th>Candidate</th><th>Final Score</th><th>BM25</th>"
        "<th>Semantic</th><th>Skills</th><th>Profile</th><th>Education</th>"
        "<th>Role</th><th>Experience</th><th>Required Exp.</th><th>Candidate Exp.</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody>"
        "</table>"
        "</div>"
    )

    st.markdown(table_html, unsafe_allow_html=True)


def rank_chip_html(rank: int) -> str:
    if rank == 1:
        return '<span class="rank-chip rank-gold">🥇</span>'
    if rank == 2:
        return '<span class="rank-chip rank-silver">🥈</span>'
    if rank == 3:
        return '<span class="rank-chip rank-bronze">🥉</span>'
    return f'<span class="rank-chip">#{rank}</span>'


# NOTE (memory): max_entries + ttl added so this cache can't grow forever.
# On a 512MB instance, an unbounded @st.cache_data will eventually consume
# all available memory across a long-running session with many different
# uploads.
@st.cache_data(show_spinner=False, max_entries=10, ttl=3600)
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


def extract_all_resumes(files: list) -> tuple[list[str], list[str], list[str]]:
    """
    Extract text from every uploaded file concurrently instead of one-by-one.
    PDF parsing is dominated by I/O + per-file CPU work that releases the GIL
    during library calls, so a thread pool lets files overlap instead of
    queueing behind each other.

    NOTE (memory): worker count capped at 3 (down from 8). On a 512MB
    instance, running many PDF parses concurrently multiplies peak memory
    right at the moment the semantic model may also be loading — a lower
    cap trades a little speed for a much safer memory ceiling.

    Returns (resume_texts, resume_names, error_messages), in a stable order
    matching the original upload order.
    """
    n = len(files)
    ordered_text: list[str | None] = [None] * n
    ordered_error: list[str | None] = [None] * n

    progress = st.progress(0)
    status = st.empty()
    completed = 0

    with ThreadPoolExecutor(max_workers=min(3, max(1, n))) as pool:
        future_to_index = {
            pool.submit(extract_and_clean_resume, f.getvalue(), f.name): idx
            for idx, f in enumerate(files)
        }
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            text, error = future.result()
            ordered_text[idx] = text
            ordered_error[idx] = error
            completed += 1
            status.write(f"📖 Read {completed}/{n} resume(s)...")
            progress.progress(completed / n)

    status.write("✅ Resume extraction completed.")

    resumes, resume_names, extraction_errors = [], [], []
    for idx, f in enumerate(files):
        if ordered_error[idx]:
            extraction_errors.append(ordered_error[idx])
        else:
            resumes.append(ordered_text[idx])
            resume_names.append(f.name)

    return resumes, resume_names, extraction_errors


def _score_one_resume(job_description: str, resume: str) -> dict:
    """Run the per-resume scoring engines (skill + profile match) for one candidate."""
    skill_result = calculate_skill_match(job_description, resume)
    profile_result = calculate_profile_match(job_description, resume)
    return {"skill_score": skill_result["score"], "profile": profile_result}


# NOTE (memory): max_entries + ttl added, same reasoning as extract_and_clean_resume above.
@st.cache_data(show_spinner=False, max_entries=10, ttl=3600)
def score_candidates(job_description: str, resumes: tuple[str, ...]) -> dict:
    """
    Run all scoring engines once per (JD, resume-set) combination.
    BM25 and semantic scoring already operate on the whole batch in one call.
    Skill + profile matching are per-resume, so they're fanned out across a
    thread pool instead of looping sequentially.

    NOTE (memory): worker count capped at 3 (down from 8), same reasoning
    as extract_all_resumes above — fewer concurrent workers means a lower
    peak memory footprint on a constrained instance.
    """
    bm25_scores = calculate_bm25_scores(job_description, list(resumes))
    semantic_scores = calculate_semantic_scores(job_description, list(resumes))

    skill_scores: list = [None] * len(resumes)
    profile_results: list = [None] * len(resumes)

    with ThreadPoolExecutor(max_workers=min(3, max(1, len(resumes)))) as pool:
        future_to_index = {
            pool.submit(_score_one_resume, job_description, resume): idx
            for idx, resume in enumerate(resumes)
        }
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            result = future.result()
            skill_scores[idx] = result["skill_score"]
            profile_results[idx] = result["profile"]

    return {
        "bm25": bm25_scores,
        "semantic": semantic_scores,
        "skills": skill_scores,
        "profiles": profile_results,
    }


def run_analysis(job_description_input: str, uploaded_files: list) -> None:
    """Run the full pipeline once and stash everything needed to render in session_state."""

    job_description = clean_text(job_description_input)

    resumes, resume_names, extraction_errors = extract_all_resumes(uploaded_files)

    if not resumes:
        st.session_state["analysis_error"] = (
            "No resumes could be processed. Please check the uploaded files and try again."
        )
        st.session_state.pop("results", None)
        return

    with st.spinner("🧠 Running hybrid scoring engine..."):
        scores = score_candidates(job_description, tuple(resumes))

    bm25_scores = scores["bm25"]
    semantic_scores = scores["semantic"]
    skill_scores = scores["skills"]
    profile_results = scores["profiles"]

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

    # Persist everything needed to render, so future reruns (expanding a
    # card, switching a tab, etc.) redraw from state instead of vanishing.
    st.session_state["results"] = results
    st.session_state["extraction_errors"] = extraction_errors
    st.session_state.pop("analysis_error", None)

    # NOTE (memory): explicitly drop large intermediates and force a
    # collection pass right after the heavy scoring work, instead of
    # waiting for Python's garbage collector to get around to it. On a
    # 512MB instance this helps release PDF text buffers and embedding
    # arrays promptly rather than letting them linger until the next GC
    # cycle.
    del resumes, scores, bm25_scores, semantic_scores, skill_scores, profile_results
    gc.collect()


def render_results(results: list[dict], extraction_errors: list[str]) -> None:
    """Render the full results screen from already-computed data (no recompute)."""

    if extraction_errors:
        with st.expander(f"⚠️ {len(extraction_errors)} resume(s) skipped", expanded=False):
            for err in extraction_errors:
                st.warning(html.escape(err))

    winner = results[0]
    safe_winner_name = html.escape(winner["Candidate"])

    # ----------------------------------------------------
    # WINNER
    # ----------------------------------------------------

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

    # ----------------------------------------------------
    # TOP 3 SUMMARY CARDS
    # ----------------------------------------------------

    st.markdown('<div class="section-title">🏅 Top 3 Candidates</div>', unsafe_allow_html=True)
    render_podium(results)

    # ----------------------------------------------------
    # WINNER BREAKDOWN + RADAR (tabbed)
    # ----------------------------------------------------

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

    # ----------------------------------------------------
    # CANDIDATE COMPARISON TABLE
    # ----------------------------------------------------

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

    render_comparison_table(results)



    # ----------------------------------------------------
    # RANKING CARDS
    # ----------------------------------------------------

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

    # ----------------------------------------------------
    # EXPORT
    # ----------------------------------------------------

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

    st.markdown("<br/>", unsafe_allow_html=True)
    if st.button("🔁 Start a New Search", use_container_width=True):
        st.session_state.pop("results", None)
        st.session_state.pop("extraction_errors", None)
        st.session_state.pop("analysis_error", None)
        st.session_state["balloons_shown"] = False
        st.rerun()

    if not st.session_state.get("balloons_shown", False):
        st.balloons()
        st.session_state["balloons_shown"] = True


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
# If we already have results in session_state, show them and stop
# here — this is what keeps the report on screen when the user
# interacts with a card, tab, or the dataframe instead of forcing
# a full re-analysis.
# ============================================================

if st.session_state.get("results"):
    render_results(st.session_state["results"], st.session_state.get("extraction_errors", []))
    st.stop()


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

# Job Description text area styling — must be wrapped in st.markdown with
# unsafe_allow_html=True. Raw CSS placed directly in a .py file (outside a
# string) is invalid Python and will crash the app with a SyntaxError.
st.markdown(
    """
    <style>
    div[data-testid="stTextArea"] textarea {
        background-color: #000000 !important;
        color: #ffffff !important;
        caret-color: #ffffff !important;
    }

    div[data-testid="stTextArea"] textarea::placeholder {
        color: #9ca3af !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
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

if st.session_state.get("analysis_error"):
    st.error(st.session_state["analysis_error"])


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

    run_analysis(job_description_input, uploaded_files)
    st.rerun()

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