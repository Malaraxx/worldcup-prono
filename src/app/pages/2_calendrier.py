"""Page 2 — Calendrier & Pronos."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[3]))

import numpy as np
import pandas as pd
import streamlit as st

from src.app.utils import (
    load_fixtures, load_predictions, load_picks, load_teams,
    STAGE_LABELS, STAGE_ORDER, flag, format_pct, format_ev, mpp_implied,
)

_STAGE_ORDER = {s: i for i, s in enumerate(STAGE_ORDER)}

# Couleur par pot matchup (1vs1=rouge, 1vs2=orange, 1vs3=jaune, 1vs4=vert, autres=gris)
_POT_CLASH_COLOR = {
    (1, 1): "#FFCDD2", (2, 2): "#FFE0B2", (3, 3): "#FFF9C4",
    (4, 4): "#F5F5F5", (1, 2): "#FFE0B2", (1, 3): "#FFF9C4",
    (1, 4): "#E8F5E9", (2, 3): "#FFF9C4", (2, 4): "#E8F5E9",
    (3, 4): "#E8F5E9",
}

st.markdown("""
<div style="background:linear-gradient(135deg,#0A2342 0%,#1565C0 55%,#1976D2 100%);
            border-radius:14px;padding:22px 28px;margin-bottom:20px;
            box-shadow:0 4px 20px rgba(21,101,192,0.25)">
  <div style="font-size:1.9rem;font-weight:800;color:#fff;letter-spacing:0.5px">
    📅 Calendrier & Pronos
  </div>
  <div style="font-size:0.9rem;color:rgba(255,255,255,0.65);margin-top:4px">
    104 matchs · Phase de groupes &amp; KO · Filtres par groupe, phase et date
  </div>
</div>
""", unsafe_allow_html=True)

# ── Données ───────────────────────────────────────────────────────────────────
fix   = load_fixtures()
pred  = load_predictions()
picks = load_picks()
teams = load_teams()

# Fusion principale (left join → 80 matchs KO auront NaN sur cols picks)
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

# Enrichir avec Pot des équipes
pot_map = dict(zip(teams["team"], teams["pot"]))

# ── Bandeau couverture picks ──────────────────────────────────────────────────
n_with_picks = base["mode_recommended"].notna().sum()
n_total      = len(base)
n_missing    = n_total - n_with_picks
st.info(
    f"📊 Picks recommandés disponibles pour **{n_with_picks} matchs sur {n_total}** "
    f"(phase de groupes avec cotes MPP saisies). "
    f"Les {n_missing} matchs KO n'ont pas encore de picks ni de probas affichées "
    f"(équipes non qualifiées — voir la page Bracket pour les probabilités de qualification)."
)

# ── Filtres ───────────────────────────────────────────────────────────────────
col_g, col_s, col_f, col_d1, col_d2 = st.columns([1, 1, 1, 1, 1])

all_groups = sorted(fix["group"].dropna().unique())
sel_group = col_g.selectbox("Groupe", ["Tous"] + all_groups)

stage_labels_ordered = [STAGE_LABELS.get(s, s) for s in STAGE_ORDER if s in fix["stage"].unique()]
sel_stage = col_s.selectbox("Phase", ["Tous"] + stage_labels_ordered)

sel_picks_only = col_f.selectbox("Afficher", ["Tous les matchs", "Picks seulement", "KO seulement"])

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
if sel_picks_only == "Picks seulement":
    filtered = filtered[filtered["mode_recommended"].notna()]
elif sel_picks_only == "KO seulement":
    filtered = filtered[filtered["mode_recommended"].isna()]
filtered = filtered[
    (filtered["date_local"].dt.date >= sel_d1) &
    (filtered["date_local"].dt.date <= sel_d2)
]

# Tri : picks en premier, puis groupes avant KO, puis date ASC
filtered = filtered.copy()
filtered["_has_pick"]    = filtered["mode_recommended"].notna()
filtered["_stage_order"] = filtered["stage"].map(_STAGE_ORDER).fillna(99)
filtered = filtered.sort_values(
    ["_has_pick", "_stage_order", "kickoff_dt"],
    ascending=[False, True, True],
)

st.caption(f"{len(filtered)} match(s) affiché(s)")

# ── Tableau principal ─────────────────────────────────────────────────────────
rows = []
row_colors = []

for _, r in filtered.iterrows():
    home = r["home_slot"]
    away = r["away_slot"]
    dt_str = r["date_local"].strftime("%d/%m %H:%M") if pd.notna(r["date_local"]) else "—"

    # Elo diff (favori)
    elo_diff = r.get("elo_diff")
    if pd.notna(elo_diff):
        if elo_diff > 0:
            elo_str = f"+{elo_diff:.0f} {flag(home)}"
        elif elo_diff < 0:
            elo_str = f"{elo_diff:.0f} {flag(away)}"
        else:
            elo_str = "Égal"
    else:
        elo_str = "—"

    # Pot matchup color
    pot_h = pot_map.get(home)
    pot_a = pot_map.get(away)
    if pot_h and pot_a:
        key = (min(pot_h, pot_a), max(pot_h, pot_a))
        row_colors.append(_POT_CLASH_COLOR.get(key, ""))
        pot_str = f"P{pot_h} vs P{pot_a}"
    else:
        row_colors.append("")
        pot_str = "—"

    # Probas modèle — uniquement pour les matchs de poule avec équipes nominales
    ph  = r.get("p_home_win_cal")
    pd_ = r.get("p_draw_cal")
    pa  = r.get("p_away_win_cal")
    probas_str = (
        f"{ph:.0%} / {pd_:.0%} / {pa:.0%}"
        if pd.notna(ph) and r["stage"] == "group"
        else "—"
    )

    # Cotes MPP
    ch = r.get("cote_home")
    cd = r.get("cote_draw")
    ca = r.get("cote_away")
    cotes_str = f"{int(ch)} / {int(cd)} / {int(ca)}" if pd.notna(ch) else "—"

    # Picks
    mode        = r.get("mode_recommended")
    safe_str    = str(r["safe_score"])    if pd.notna(r.get("safe_score"))    else "—"
    value_str   = str(r["value_score"])   if pd.notna(r.get("value_score"))   else "—"
    lottery_str = str(r["lottery_score"]) if pd.notna(r.get("lottery_score")) else "—"
    mode_str    = str(mode).upper()       if pd.notna(mode)                   else "—"
    ev_float    = float(r["value_ev"])    if pd.notna(r.get("value_ev"))      else np.nan

    # WR du mode recommandé
    if pd.notna(mode) and mode == "value":
        wr_float = float(r["value_wr"]) if pd.notna(r.get("value_wr")) else np.nan
    elif pd.notna(mode) and mode == "lottery":
        wr_float = float(r["lottery_wr"]) if pd.notna(r.get("lottery_wr")) else np.nan
    elif pd.notna(mode) and mode == "safe":
        wr_float = float(r["safe_wr"]) if pd.notna(r.get("safe_wr")) else np.nan
    else:
        wr_float = np.nan

    rows.append({
        "ID":             int(r["match_id"]),
        "Date":           dt_str,
        "Match":          f"{flag(home)} {home}  vs  {flag(away)} {away}",
        "Gr.":            r.get("group") or "—",
        "Pots":           pot_str,
        "Elo diff":       elo_str,
        "Cotes (1/N/2)":  cotes_str,
        "Probas grp.":    probas_str,
        "SAFE":           safe_str,
        "VALUE":          value_str,
        "LOTTERY":        lottery_str,
        "Mode":           mode_str,
        "WR":             wr_float,
        "EV":             ev_float,
    })

if rows:
    df_table = pd.DataFrame(rows).set_index("ID")

    def _ev_bg(val):
        if pd.isna(val):
            return ""
        if val >= 15:
            return "background-color: #C8E6C9; color: #1B5E20; font-weight:600"
        if val >= 8:
            return "background-color: #FFF9C4; color: #E65100; font-weight:600"
        return "background-color: #F5F5F5; color: #9E9E9E"

    def _mode_bg(val):
        if val == "SAFE":
            return "background-color:#E3F2FD;color:#1565C0;font-weight:700"
        if val == "VALUE":
            return "background-color:#E8F5E9;color:#2E7D32;font-weight:700"
        if val == "LOTTERY":
            return "background-color:#EDE7F6;color:#6A1B9A;font-weight:700"
        return ""

    styled = (
        df_table.style
        .map(_ev_bg, subset=["EV"])
        .map(_mode_bg, subset=["Mode"])
        .format({
            "EV": lambda x: f"{x:.1f}" if pd.notna(x) else "—",
            "WR": lambda x: f"{x:.0%}" if pd.notna(x) else "—",
        })
    )
    st.dataframe(styled, use_container_width=True, height=520)

    # ── Navigation vers détail ────────────────────────────────────────────────
    st.markdown("---")
    col_sel, col_btn = st.columns([4, 1])
    match_options = {f"#{r['ID']} — {r['Match']}": r["ID"] for r in rows}
    with col_sel:
        sel_match_label = st.selectbox(
            "Voir le détail d'un match :",
            list(match_options.keys()),
            label_visibility="collapsed",
        )
    with col_btn:
        if st.button("🔍 Détail", use_container_width=True):
            st.session_state["selected_match_id"] = match_options[sel_match_label]
            st.switch_page("pages/3_detail_match.py")

    # ── Export CSV ────────────────────────────────────────────────────────────
    picks_rows = [r for r in rows if r["Mode"] != "—"]
    if picks_rows:
        export_df = pd.DataFrame(picks_rows)[["ID", "Date", "Match", "Mode", "SAFE", "VALUE", "LOTTERY", "Cotes (1/N/2)", "WR", "EV"]]
        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Télécharger les picks (CSV)",
            data=csv_bytes,
            file_name="picks_mpp.csv",
            mime="text/csv",
        )

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
- **EV couleur** : 🟢 ≥ 15 pts (très bon) · 🟡 8–15 pts (correct) · ⬜ < 8 pts (faible)
- **Pots** : P1 vs P4 🟢 (déséquilibre attendu) · P1 vs P1 🔴 (choc de favoris)
""")
