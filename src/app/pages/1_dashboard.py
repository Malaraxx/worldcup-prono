"""Page 1 — Dashboard principal."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[3]))

from datetime import datetime, timezone, timedelta
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
total_matches   = len(fix)
played_matches  = int((fix["kickoff_dt"] < now_utc).sum())
n_picks         = len(picks)
my_score        = load_my_score()

# ── Métriques ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Matchs total", total_matches)
c2.metric("Matchs joués", played_matches)
c3.metric("Pronos disponibles", n_picks, help="Matchs de poule avec cotes MPP")
c4.metric("Mes points", my_score if my_score is not None else "—")

st.divider()

# ── Prochains matchs ──────────────────────────────────────────────────────────
st.subheader("Prochains matchs")

upcoming = fix[fix["kickoff_dt"] >= now_utc].sort_values("kickoff_dt").head(12).copy()

if upcoming.empty:
    st.info("Tous les matchs sont terminés.")
else:
    # Joindre les picks
    upcoming_with_picks = upcoming.merge(
        picks[["match_id", "mode_recommended", "value_score", "value_ev", "value_wr",
               "safe_score", "safe_ev", "lottery_score", "lottery_ev"]],
        on="match_id", how="left",
    )

    rows = []
    for _, r in upcoming_with_picks.iterrows():
        home = r["home_slot"]
        away = r["away_slot"]
        dt   = r["date_local"]
        dt_str = dt.strftime("%d/%m %H:%M") if pd.notna(dt) else "—"
        mode = r.get("mode_recommended", "")
        if mode == "value":
            pick_score = r.get("value_score", "—")
            pick_ev    = r.get("value_ev", None)
        elif mode == "lottery":
            pick_score = r.get("lottery_score", "—")
            pick_ev    = r.get("lottery_ev", None)
        elif mode == "safe":
            pick_score = r.get("safe_score", "—")
            pick_ev    = r.get("safe_ev", None)
        else:
            pick_score, pick_ev = "—", None

        rows.append({
            "ID":    int(r["match_id"]),
            "Date":  dt_str,
            "Match": f"{flag(home)} {home}  vs  {flag(away)} {away}",
            "Stade": r.get("venue", ""),
            "Stage": STAGE_LABELS.get(r["stage"], r["stage"]),
            "Pick":  pick_score if mode else "—",
            "Mode":  mode.upper() if mode else "—",
            "EV":    f"{pick_ev:.1f}" if pick_ev is not None else "—",
        })

    df_up = pd.DataFrame(rows).set_index("ID")
    st.dataframe(df_up, use_container_width=True, height=400)

    st.caption("💡 Cliquez sur **Calendrier & Pronos** pour filtrer par groupe/stage, ou **Détail Match** pour la heatmap score-par-score.")

st.divider()

# ── Top 10 candidats vainqueur ────────────────────────────────────────────────
st.subheader("Top 10 — Candidats vainqueur")

tp_teams = tp.merge(teams[["team", "group", "confederation"]], on="team", how="left")
top10 = tp_teams.sort_values("proba_winner", ascending=False).head(10).copy()

rows_tp = []
for _, r in top10.iterrows():
    rows_tp.append({
        "Équipe":      f"{flag(r['team'])} {r['team']}",
        "Groupe":      r.get("group", "—"),
        "Conf.":       r.get("confederation", "—"),
        "Elo":         f"{r['elo_rating']:.0f}",
        "P(Vainqueur)": format_pct(r["proba_winner"]),
        "P(Finale)":   format_pct(r["proba_final"]),
        "P(Demi)":     format_pct(r["proba_sf"]),
        "P(R32)":      format_pct(r["proba_r32"]),
    })

st.dataframe(pd.DataFrame(rows_tp), use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(f"Dernière mise à jour modèle : {model_update_time()} · Simulation Monte-Carlo 10 000 itérations · Elo + Poisson + Platt scaling")
