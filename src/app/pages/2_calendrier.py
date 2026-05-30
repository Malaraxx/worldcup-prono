"""Page 2 — Calendrier & Pronos."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[3]))

import pandas as pd
import streamlit as st

from src.app.utils import (
    load_fixtures, load_predictions, load_picks, load_teams,
    STAGE_LABELS, STAGE_ORDER, flag, format_pct, format_ev, mpp_implied,
)

st.title("📅 Calendrier & Pronos")

# ── Données ───────────────────────────────────────────────────────────────────
fix   = load_fixtures()
pred  = load_predictions()
picks = load_picks()
teams = load_teams()

# Fusion principale
base = fix.merge(
    pred[["match_id", "elo_diff", "lambda_home", "lambda_away",
          "p_home_win_cal", "p_draw_cal", "p_away_win_cal"]],
    on="match_id", how="left",
).merge(
    picks[["match_id", "cote_home", "cote_draw", "cote_away",
           "safe_score", "safe_ev", "safe_wr",
           "value_score", "value_ev", "value_wr",
           "lottery_score", "lottery_ev", "lottery_wr",
           "mode_recommended", "edge_value_vs_safe_pct"]],
    on="match_id", how="left",
).sort_values("kickoff_dt")

# ── Filtres ───────────────────────────────────────────────────────────────────
col_g, col_s, col_d1, col_d2 = st.columns([1, 1, 1, 1])

all_groups = sorted(fix["group"].dropna().unique())
groups_options = ["Tous"] + all_groups
sel_group = col_g.selectbox("Groupe", groups_options)

stage_labels_ordered = [STAGE_LABELS.get(s, s) for s in STAGE_ORDER if s in fix["stage"].unique()]
all_stages = ["Tous"] + stage_labels_ordered
sel_stage = col_s.selectbox("Phase", all_stages)

min_date = fix["date_local"].dt.date.min()
max_date = fix["date_local"].dt.date.max()
sel_d1 = col_d1.date_input("Du", value=min_date, min_value=min_date, max_value=max_date)
sel_d2 = col_d2.date_input("Au", value=max_date, min_value=min_date, max_value=max_date)

# Appliquer filtres
filtered = base.copy()
if sel_group != "Tous":
    filtered = filtered[filtered["group"] == sel_group]
if sel_stage != "Tous":
    inv_labels = {v: k for k, v in STAGE_LABELS.items()}
    stage_key = inv_labels.get(sel_stage, sel_stage)
    filtered = filtered[filtered["stage"] == stage_key]
filtered = filtered[
    (filtered["date_local"].dt.date >= sel_d1) &
    (filtered["date_local"].dt.date <= sel_d2)
]

st.caption(f"{len(filtered)} match(s) affiché(s)")

# ── Tableau principal ─────────────────────────────────────────────────────────
rows = []
for _, r in filtered.iterrows():
    home = r["home_slot"]
    away = r["away_slot"]
    dt_str = r["date_local"].strftime("%d/%m %H:%M") if pd.notna(r["date_local"]) else "—"

    # Probas modèle
    ph  = r.get("p_home_win_cal")
    pd_ = r.get("p_draw_cal")
    pa  = r.get("p_away_win_cal")
    probas_str = "—"
    if pd.notna(ph):
        probas_str = f"{ph:.0%} / {pd_:.0%} / {pa:.0%}"

    # Cotes MPP + implicites
    ch = r.get("cote_home")
    cd = r.get("cote_draw")
    ca = r.get("cote_away")
    cotes_str = "—"
    if pd.notna(ch):
        cotes_str = f"{int(ch)} / {int(cd)} / {int(ca)}"

    # Picks
    mode = r.get("mode_recommended", "")
    safe_str    = r.get("safe_score", "—") if pd.notna(r.get("safe_score")) else "—"
    value_str   = r.get("value_score", "—") if pd.notna(r.get("value_score")) else "—"
    lottery_str = r.get("lottery_score", "—") if pd.notna(r.get("lottery_score")) else "—"
    ev_str      = format_ev(r.get("value_ev", 0)) if pd.notna(r.get("value_ev")) else "—"

    rows.append({
        "ID":        int(r["match_id"]),
        "Date":      dt_str,
        "Match":     f"{flag(home)} {home}  vs  {flag(away)} {away}",
        "Gr.":       r.get("group", "—") or "—",
        "Phase":     STAGE_LABELS.get(r["stage"], r["stage"]),
        "Cotes MPP": cotes_str,
        "Probas (1-N-2)": probas_str,
        "SAFE":      safe_str,
        "VALUE":     value_str,
        "LOTTERY":   lottery_str,
        "Mode":      mode.upper() if mode else "—",
        "EV value":  ev_str,
    })

if rows:
    df_table = pd.DataFrame(rows).set_index("ID")
    st.dataframe(df_table, use_container_width=True, height=520)

    # Lien vers détail
    st.markdown("---")
    match_options = {
        f"#{r['ID']} — {r['Match']}": r["ID"] for r in rows
    }
    sel_match_label = st.selectbox(
        "Voir le détail d'un match :", list(match_options.keys())
    )
    if st.button("🔍 Ouvrir le détail"):
        st.session_state["selected_match_id"] = match_options[sel_match_label]
        st.switch_page("pages/3_detail_match.py")
else:
    st.info("Aucun match ne correspond aux filtres sélectionnés.")

# ── Légende ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
**Légende des modes :**
- 🛡️ **SAFE** — Score le plus probable dans l'issue la plus probable (WR maximisé)
- 💎 **VALUE** — Meilleur EV parmi les issues à WR ≥ 35% (proba résultat correct)
- 🎰 **LOTTERY** — Meilleur EV absolu (sans contrainte WR) — score rare à forte prime MPP
- **EV** = espérance de points si prono exact · **WR** = proba que le résultat (1/N/2) soit correct
""")
