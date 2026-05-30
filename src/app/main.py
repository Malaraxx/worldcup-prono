"""Entry point — Mon Petit Prono WC2026 Streamlit app."""
import sys
from pathlib import Path

# Ajoute le projet root au path pour que les imports src.* fonctionnent
sys.path.insert(0, str(Path(__file__).parents[2]))

import streamlit as st

st.set_page_config(
    page_title="Mon Petit Prono — WC 2026",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = st.navigation([
    st.Page("pages/1_dashboard.py",  title="Dashboard",         icon="🏠"),
    st.Page("pages/2_calendrier.py", title="Calendrier & Pronos", icon="📅"),
    st.Page("pages/3_detail_match.py", title="Détail Match",    icon="🔍"),
    st.Page("pages/4_bracket.py",    title="Bracket",           icon="🏆"),
])

with st.sidebar:
    st.markdown("### 🏆 Mon Petit Prono")
    st.caption("WC 2026 · Modèle Elo + Poisson · Cotes MPP")
    st.divider()

pages.run()
