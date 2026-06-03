"""Entry point — Mon Petit Prono WC2026 Streamlit app."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from datetime import datetime, timezone
import streamlit as st

st.set_page_config(
    page_title="Mon Petit Prono — WC 2026",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
)

_CSS = """
<style>
/* ── Sidebar gradient ─────────────────────────────────────────── */
[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(180deg, #0A2342 0%, #1565C0 55%, #1976D2 100%);
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] .stCaption { color: rgba(255,255,255,0.85) !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #ffffff !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.25) !important; }
[data-testid="stSidebar"] [data-testid="stNavLink"] {
    border-radius: 8px;
    margin: 2px 0;
    color: rgba(255,255,255,0.85) !important;
}
[data-testid="stSidebar"] [data-testid="stNavLink"]:hover {
    background: rgba(255,255,255,0.15) !important;
}
[data-testid="stSidebar"] [data-testid="stNavLink"][aria-selected="true"] {
    background: rgba(255,255,255,0.2) !important;
    color: #ffffff !important;
    font-weight: 600;
}

/* ── Metric cards ──────────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #DDE3F0;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 2px 8px rgba(21,101,192,0.07);
}
[data-testid="stMetricValue"] { font-weight: 800; }

/* ── Buttons ───────────────────────────────────────────────────── */
.stButton > button, .stDownloadButton > button {
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.15s ease;
}

/* ── Dividers ──────────────────────────────────────────────────── */
hr { border-color: #E8ECF4 !important; }

/* ── Tabs ──────────────────────────────────────────────────────── */
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-weight: 600;
}
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)

pages = st.navigation([
    st.Page("pages/1_dashboard.py",    title="Dashboard",          icon="🏠"),
    st.Page("pages/2_calendrier.py",   title="Calendrier & Pronos", icon="📅"),
    st.Page("pages/3_detail_match.py", title="Détail Match",       icon="🔍"),
    st.Page("pages/4_bracket.py",      title="Bracket",            icon="🏆"),
    st.Page("pages/5_methodologie.py", title="Méthodologie",       icon="📐"),
])

with st.sidebar:
    st.markdown("""
<div style="text-align:center;padding:8px 0 4px">
  <div style="font-size:1.3rem;font-weight:800;color:#fff;letter-spacing:0.5px">🏆 Mon Petit Prono</div>
  <div style="font-size:0.7rem;color:rgba(255,255,255,0.55);margin-top:2px">
    WC 2026 · Elo + Poisson · Cotes MPP
  </div>
</div>
""", unsafe_allow_html=True)

    wc_start = datetime(2026, 6, 11, 18, 0, 0, tzinfo=timezone.utc)
    now_utc  = datetime.now(tz=timezone.utc)
    delta    = wc_start - now_utc
    if delta.total_seconds() > 0:
        days  = delta.days
        hours = delta.seconds // 3600
        mins  = (delta.seconds % 3600) // 60
        st.markdown(f"""
<div style="background:rgba(255,255,255,0.1);border-radius:10px;padding:12px;
            text-align:center;margin:10px 0 4px;border:1px solid rgba(255,255,255,0.15)">
  <div style="font-size:0.6rem;color:rgba(255,255,255,0.55);letter-spacing:1.5px;margin-bottom:6px">
    COUP D'ENVOI DANS
  </div>
  <div style="font-size:1.7rem;font-weight:800;color:#fff;line-height:1;letter-spacing:1px">
    {days}j {hours:02d}h {mins:02d}m
  </div>
  <div style="font-size:0.62rem;color:rgba(255,255,255,0.45);margin-top:6px">
    11 juin 2026 · Mexico City
  </div>
</div>
""", unsafe_allow_html=True)
    else:
        st.markdown("""
<div style="background:rgba(46,125,50,0.35);border-radius:10px;padding:10px;
            text-align:center;margin:10px 0 4px">
  <div style="color:#A5D6A7;font-weight:700;font-size:0.85rem">🏆 Tournoi en cours !</div>
</div>
""", unsafe_allow_html=True)

    st.divider()

pages.run()
