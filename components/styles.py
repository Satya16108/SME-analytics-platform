import streamlit as st


def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── GLOBAL ───────────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    .main .block-container {
        padding: 1.2rem 2rem 2rem 2rem;
        max-width: 1450px;
    }
    h1,h2,h3,h4 { font-family: 'Inter', sans-serif !important; }

    /* ── SIDEBAR ──────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(175deg, #1B3A6B 0%, #0d2040 100%) !important;
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    [data-testid="stSidebar"] * { color: #C8D8EC !important; }
    [data-testid="stSidebar"] .stRadio label { font-size:0.83rem !important; padding:0.35rem 0 !important; }
    [data-testid="stSidebar"] .stRadio > div { gap:0 !important; }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.1) !important; }

    .sb-logo  { font-size:2.8rem; text-align:center; padding-top:0.5rem; }
    .sb-title { font-size:0.92rem; font-weight:700; color:#FFFFFF !important;
                text-align:center; letter-spacing:0.3px; line-height:1.4; }
    .sb-sub   { font-size:0.68rem; color:#7A9CC4 !important; text-align:center; margin-top:0.15rem; }
    .sb-label { font-size:0.6rem; font-weight:700; letter-spacing:1.8px;
                color:#4A6FA5 !important; text-transform:uppercase; padding:0.2rem 0.3rem; }
    .sb-ver   { font-size:0.62rem; color:#3A5578 !important; text-align:center; padding-top:0.4rem; }

    .sb-stats { display:flex; justify-content:space-around; padding:0.6rem 0; }
    .sb-stat  { text-align:center; }
    .sb-val   { display:block; font-size:1.25rem; font-weight:800; color:#E07B39 !important; }
    .sb-lbl   { font-size:0.62rem; color:#7A9CC4 !important; }

    /* ── PAGE HEADER ──────────────────────────────────────── */
    .page-hdr {
        background: linear-gradient(135deg, #1B3A6B 0%, #0E7C86 100%);
        border-radius: 14px;
        padding: 1.4rem 1.8rem 1.2rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 20px rgba(27,58,107,0.25);
    }
    .page-hdr-title {
        font-size: 1.55rem; font-weight: 800; color: #FFFFFF;
        margin: 0 0 0.15rem 0; letter-spacing: -0.3px;
    }
    .page-hdr-sub  { font-size:0.88rem; color:rgba(255,255,255,0.75); margin:0; }
    .page-hdr-badge {
        display:inline-block; background:rgba(255,255,255,0.18);
        color:#fff; padding:0.18rem 0.7rem; border-radius:20px;
        font-size:0.72rem; font-weight:600; margin-top:0.5rem;
        border:1px solid rgba(255,255,255,0.25);
    }
    .prob-banner {
        background: rgba(0,0,0,0.18); border-left:3px solid #E07B39;
        padding:0.45rem 0.9rem; border-radius:0 8px 8px 0;
        margin-top:0.8rem; font-size:0.8rem; color:rgba(255,255,255,0.88);
    }

    /* ── KPI CARDS ────────────────────────────────────────── */
    .kpi-card {
        background: #FFFFFF; border-radius: 12px;
        padding: 1.1rem 1.2rem 0.9rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.07);
        border: 1px solid #E8ECF0;
        position: relative; overflow: hidden; min-height: 108px;
    }
    .kpi-card::before {
        content:''; position:absolute; top:0;left:0;right:0;
        height:4px; background:var(--kpi-accent,#1B3A6B);
        border-radius:12px 12px 0 0;
    }
    .kpi-icon  { font-size:1.35rem; display:block; margin-bottom:0.35rem; }
    .kpi-lbl   { font-size:0.67rem; font-weight:700; text-transform:uppercase;
                 letter-spacing:0.9px; color:#8A9BB0; margin-bottom:0.3rem; }
    .kpi-val   { font-size:1.65rem; font-weight:800; color:#1E2A3B; line-height:1.1; }
    .kpi-delta { font-size:0.72rem; font-weight:500; display:block; margin-top:0.25rem; }
    .kpi-delta.pos  { color:#27AE60; }
    .kpi-delta.neg  { color:#C0392B; }
    .kpi-delta.neu  { color:#F39C12; }
    .kpi-sub   { font-size:0.67rem; color:#A0B0C0; margin-top:0.15rem; }

    /* ── HOME CARDS ───────────────────────────────────────── */
    .home-stat {
        background:#fff; border-radius:12px; padding:1.1rem;
        text-align:center; box-shadow:0 2px 8px rgba(0,0,0,0.06);
        border:1px solid #E8ECF0;
    }
    .home-stat-icon { font-size:1.6rem; }
    .home-stat-val  { font-size:1.75rem; font-weight:800; color:#1B3A6B; display:block; margin:0.2rem 0; }
    .home-stat-lbl  { font-size:0.75rem; color:#7A8B9A; }

    .sector-card {
        background:#fff; border-radius:12px; padding:1.1rem 1.3rem;
        margin-bottom:0.9rem;
        box-shadow:0 2px 8px rgba(0,0,0,0.06); border:1px solid #E8ECF0;
        transition: box-shadow 0.2s, transform 0.15s;
        cursor: default;
    }
    .sector-card:hover { box-shadow:0 6px 20px rgba(0,0,0,0.11); transform:translateY(-1px); }
    .sc-head   { display:flex; justify-content:space-between; align-items:center; margin-bottom:0.55rem; }
    .sc-name   { font-size:0.97rem; font-weight:700; color:#1B3A6B; }
    .sc-tag    { background:#EBF4FB; color:#0E7C86; font-size:0.68rem;
                 font-weight:600; padding:0.18rem 0.6rem; border-radius:12px; }
    .sc-prob   { font-size:0.79rem; color:#5D6D7E; margin-bottom:0.3rem; }
    .sc-model  { font-size:0.75rem; color:#8A9BB0; }

    /* ── SECTION HEADERS ──────────────────────────────────── */
    .sec-hdr {
        font-size:0.95rem; font-weight:700; color:#1B3A6B;
        margin:1rem 0 0.6rem; padding-bottom:0.35rem;
        border-bottom:2px solid #E8ECF0;
    }

    /* ── RISK BADGES ──────────────────────────────────────── */
    .badge-high   { background:#FDEDEC; color:#C0392B; padding:0.12rem 0.55rem;
                    border-radius:12px; font-size:0.7rem; font-weight:700; }
    .badge-medium { background:#FEF9E7; color:#D68910; padding:0.12rem 0.55rem;
                    border-radius:12px; font-size:0.7rem; font-weight:700; }
    .badge-low    { background:#EAFAF1; color:#1E8449; padding:0.12rem 0.55rem;
                    border-radius:12px; font-size:0.7rem; font-weight:700; }

    /* ── INSIGHT CARDS ────────────────────────────────────── */
    .i-card {
        background:#fff; border-radius:10px; padding:0.9rem 1.1rem;
        margin-bottom:0.65rem; border-left:4px solid #1B3A6B;
        box-shadow:0 1px 5px rgba(0,0,0,0.06);
    }
    .i-card.warn     { border-left-color:#E07B39; }
    .i-card.critical { border-left-color:#C0392B; }
    .i-card.ok       { border-left-color:#27AE60; }
    .i-title { font-size:0.87rem; font-weight:700; color:#2C3E50; margin-bottom:0.3rem; }
    .i-body  { font-size:0.78rem; color:#5D6D7E; line-height:1.55; }

    /* ── TABS ─────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] { gap:0.4rem; background:transparent; }
    .stTabs [data-baseweb="tab"] {
        background:#fff; border-radius:8px 8px 0 0;
        border:1px solid #E8ECF0; padding:0.45rem 1.1rem;
        font-size:0.82rem; font-weight:500; color:#5D6D7E;
    }
    .stTabs [aria-selected="true"] {
        background:#1B3A6B !important; color:#fff !important;
        border-color:#1B3A6B !important;
    }

    /* ── MISC ─────────────────────────────────────────────── */
    [data-testid="stMetricValue"] { font-size:1.4rem !important; font-weight:700 !important; }
    div[data-testid="stExpander"] { border:1px solid #E8ECF0; border-radius:10px; }
    </style>
    """, unsafe_allow_html=True)
