"""Page 1 — Dashboard principal."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[3]))

from datetime import datetime, timezone
import numpy as np
import pandas as pd
import streamlit as st

from src.app.utils import (
    load_fixtures, load_picks, load_tournament_probabilities, load_teams,
    STAGE_LABELS, flag, format_pct, model_update_time, load_my_score,
)

st.title("🏆 Mon Petit Prono — WC 2026")

# ── Données ───────────────────────────────────────────────────────────────────
fix   = load_fixtures()
picks = load_picks()
tp    = load_tournament_probabilities()
teams = load_teams()

now_utc = datetime.now(tz=timezone.utc)
total_matches  = len(fix)
played_matches = int((fix["kickoff_dt"] < now_utc).sum())
n_picks        = len(picks)
my_score       = load_my_score()

# ── Métriques ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Matchs total", total_matches)
c2.metric("Matchs joués", played_matches)
c3.metric("Pronos disponibles", n_picks, help="Matchs de poule avec cotes MPP")
c4.metric("Mes points", my_score if my_score is not None else "—")

st.divider()

# ── Prochains matchs ──────────────────────────────────────────────────────────
st.subheader("Prochains matchs")

upcoming = fix[fix["kickoff_dt"] >= now_utc].copy()

if upcoming.empty:
    st.info("Tous les matchs sont terminés.")
else:
    upcoming_with_picks = upcoming.merge(
        picks[["match_id", "mode_recommended",
               "value_score", "value_ev",
               "safe_score", "safe_ev",
               "lottery_score", "lottery_ev"]],
        on="match_id", how="left",
    )

    # AMÉLIORATION 2 — picks en premier, puis date asc
    upcoming_with_picks["_has_pick"] = upcoming_with_picks["mode_recommended"].notna()
    upcoming_with_picks = (
        upcoming_with_picks
        .sort_values(["_has_pick", "kickoff_dt"], ascending=[False, True])
        .head(12)
    )

    rows = []
    for _, r in upcoming_with_picks.iterrows():
        home = r["home_slot"]
        away = r["away_slot"]
        dt_str = r["date_local"].strftime("%d/%m %H:%M") if pd.notna(r["date_local"]) else "—"

        mode = r.get("mode_recommended")  # NaN pour les KO sans picks

        # Sélectionner le bon score/EV selon le mode
        if pd.notna(mode) and mode == "value":
            pick_score = str(r["value_score"])   if pd.notna(r.get("value_score"))   else "—"
            pick_ev    = float(r["value_ev"])     if pd.notna(r.get("value_ev"))      else np.nan
        elif pd.notna(mode) and mode == "lottery":
            pick_score = str(r["lottery_score"]) if pd.notna(r.get("lottery_score")) else "—"
            pick_ev    = float(r["lottery_ev"])   if pd.notna(r.get("lottery_ev"))    else np.nan
        elif pd.notna(mode) and mode == "safe":
            pick_score = str(r["safe_score"])    if pd.notna(r.get("safe_score"))    else "—"
            pick_ev    = float(r["safe_ev"])      if pd.notna(r.get("safe_ev"))       else np.nan
        else:
            pick_score, pick_ev = "—", np.nan

        mode_str = str(mode).upper() if pd.notna(mode) else "—"

        rows.append({
            "ID":    int(r["match_id"]),
            "Date":  dt_str,
            "Match": f"{flag(home)} {home}  vs  {flag(away)} {away}",
            "Stade": r.get("venue", ""),
            "Phase": STAGE_LABELS.get(r["stage"], r["stage"]),
            "Pick":  pick_score,
            "Mode":  mode_str,
            "EV":    pick_ev,   # float/NaN pour coloration
        })

    df_up = pd.DataFrame(rows).set_index("ID")

    def _ev_bg(val):
        if pd.isna(val):
            return ""
        if val >= 15:
            return "background-color: #C8E6C9; color: #1B5E20"
        if val >= 8:
            return "background-color: #FFF9C4; color: #F57F17"
        return "background-color: #F5F5F5; color: #757575"

    styled = (
        df_up.style
        .map(_ev_bg, subset=["EV"])
        .format({"EV": lambda x: f"{x:.1f}" if pd.notna(x) else "—"})
    )
    st.dataframe(styled, use_container_width=True, height=400)

    st.caption("💡 Cliquez sur **Calendrier & Pronos** pour filtrer par groupe/stage, ou **Détail Match** pour la heatmap score-par-score.")

st.divider()

# ── Top 10 candidats vainqueur ────────────────────────────────────────────────
st.subheader("Top 10 — Candidats vainqueur")

tp_teams = tp.merge(teams[["team", "group", "confederation"]], on="team", how="left")
top10 = tp_teams.sort_values("proba_winner", ascending=False).head(10).copy()

rows_tp = []
for _, r in top10.iterrows():
    rows_tp.append({
        "Équipe":       f"{flag(r['team'])} {r['team']}",
        "Groupe":       r.get("group", "—"),
        "Conf.":        r.get("confederation", "—"),
        "Elo":          f"{r['elo_rating']:.0f}",
        "P(Vainqueur)": format_pct(r["proba_winner"]),
        "P(Finale)":    format_pct(r["proba_final"]),
        "P(Demi)":      format_pct(r["proba_sf"]),
        "P(R32)":       format_pct(r["proba_r32"]),
    })

st.dataframe(pd.DataFrame(rows_tp), use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    f"Dernière mise à jour modèle : {model_update_time()} · "
    "Simulation Monte-Carlo 10 000 itérations · Elo + Poisson + Platt scaling"
)
