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
    STAGE_LABELS, flag, format_pct, model_update_time,
    load_mv_baseline,
)

st.markdown("""
<div style="background:linear-gradient(135deg,#0A2342 0%,#1565C0 55%,#1976D2 100%);
            border-radius:14px;padding:22px 28px;margin-bottom:20px;
            box-shadow:0 4px 20px rgba(21,101,192,0.25)">
  <div style="font-size:1.9rem;font-weight:800;color:#fff;letter-spacing:0.5px">
    🏆 Mon Petit Prono
  </div>
  <div style="font-size:0.9rem;color:rgba(255,255,255,0.65);margin-top:4px">
    FIFA World Cup 2026 &nbsp;·&nbsp; Modèle Elo + Poisson + Platt Scaling &nbsp;·&nbsp; Monte-Carlo 10 000 simulations
  </div>
</div>
""", unsafe_allow_html=True)

# ── Données ───────────────────────────────────────────────────────────────────
fix   = load_fixtures()
picks = load_picks()
tp    = load_tournament_probabilities()
teams = load_teams()
mv    = load_mv_baseline()

now_utc = datetime.now(tz=timezone.utc)
total_matches  = len(fix)
played_matches = int((fix["kickoff_dt"] < now_utc).sum())
n_picks        = len(picks)

# ── Métriques ─────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
c1.metric("Matchs total", total_matches)
c2.metric("Matchs joués", played_matches)
c3.metric("Pronos disponibles", n_picks, help="Matchs de poule avec cotes MPP")

st.divider()

# ── Prochains matchs ──────────────────────────────────────────────────────────
st.markdown("### 📅 Prochains matchs")

upcoming = fix[fix["kickoff_dt"] >= now_utc].copy()

if upcoming.empty:
    st.info("Tous les matchs sont terminés.")
else:
    picks_cols = ["match_id", "mode_recommended",
                  "cote_home", "cote_draw", "cote_away",
                  "safe_score", "safe_ev", "safe_wr",
                  "value_score", "value_ev", "value_wr",
                  "lottery_score", "lottery_ev", "lottery_wr"]
    upcoming_with_picks = upcoming.merge(picks[picks_cols], on="match_id", how="left")
    upcoming_with_picks["_has_pick"] = upcoming_with_picks["mode_recommended"].notna()
    upcoming_with_picks = upcoming_with_picks.sort_values(
        ["_has_pick", "kickoff_dt"], ascending=[False, True]
    )

    def _build_rows(df):
        rows = []
        for _, r in df.iterrows():
            home = r["home_slot"]
            away = r["away_slot"]
            dt_str = r["date_local"].strftime("%d/%m %H:%M") if pd.notna(r["date_local"]) else "—"
            mode = r.get("mode_recommended")

            if pd.notna(mode) and mode == "value":
                pick_score = str(r["value_score"]) if pd.notna(r.get("value_score")) else "—"
                pick_ev    = float(r["value_ev"])   if pd.notna(r.get("value_ev"))    else np.nan
                pick_wr    = float(r["value_wr"])   if pd.notna(r.get("value_wr"))    else np.nan
            elif pd.notna(mode) and mode == "lottery":
                pick_score = str(r["lottery_score"]) if pd.notna(r.get("lottery_score")) else "—"
                pick_ev    = float(r["lottery_ev"])   if pd.notna(r.get("lottery_ev"))    else np.nan
                pick_wr    = float(r["lottery_wr"])   if pd.notna(r.get("lottery_wr"))    else np.nan
            elif pd.notna(mode) and mode == "safe":
                pick_score = str(r["safe_score"]) if pd.notna(r.get("safe_score")) else "—"
                pick_ev    = float(r["safe_ev"])   if pd.notna(r.get("safe_ev"))    else np.nan
                pick_wr    = float(r["safe_wr"])   if pd.notna(r.get("safe_wr"))    else np.nan
            else:
                pick_score, pick_ev, pick_wr = "—", np.nan, np.nan

            ch = r.get("cote_home")
            cd = r.get("cote_draw")
            ca = r.get("cote_away")
            cotes_str = f"{int(ch)}/{int(cd)}/{int(ca)}" if pd.notna(ch) else "—"
            mode_str  = str(mode).upper() if pd.notna(mode) else "—"

            rows.append({
                "ID":    int(r["match_id"]),
                "Date":  dt_str,
                "Match": f"{flag(home)} {home}  vs  {flag(away)} {away}",
                "Phase": STAGE_LABELS.get(r["stage"], r["stage"]),
                "Cotes (1/N/2)": cotes_str,
                "Pick":  pick_score,
                "Mode":  mode_str,
                "WR":    pick_wr,
                "EV":    pick_ev,
            })
        return rows

    def _ev_bg(val):
        if pd.isna(val):
            return ""
        if val >= 15:
            return "background-color: #C8E6C9; color: #1B5E20"
        if val >= 8:
            return "background-color: #FFF9C4; color: #F57F17"
        return "background-color: #F5F5F5; color: #757575"

    # Section picks
    with_picks = upcoming_with_picks[upcoming_with_picks["_has_pick"]]
    without_picks = upcoming_with_picks[~upcoming_with_picks["_has_pick"]]

    if not with_picks.empty:
        rows_picks = _build_rows(with_picks)
        df_picks = pd.DataFrame(rows_picks).set_index("ID")
        styled_picks = (
            df_picks.style
            .map(_ev_bg, subset=["EV"])
            .format({
                "EV": lambda x: f"{x:.1f}" if pd.notna(x) else "—",
                "WR": lambda x: f"{x:.0%}" if pd.notna(x) else "—",
            })
        )
        st.markdown("**📊 Matchs avec picks recommandés**")
        h = min(36 + len(rows_picks) * 35, 520)
        st.dataframe(styled_picks, use_container_width=True, height=h)

    # Section KO sans picks
    if not without_picks.empty:
        st.markdown(f"**🏆 Matchs KO à venir** *(équipes non encore qualifiées — {len(without_picks)} matchs)*")
        rows_ko = _build_rows(without_picks.head(16))
        df_ko = pd.DataFrame(rows_ko).set_index("ID")
        st.dataframe(df_ko.style.format({
            "EV": lambda x: "—",
            "WR": lambda x: "—",
        }), use_container_width=True, height=min(36 + len(rows_ko) * 35, 380))

    st.caption("💡 **Calendrier & Pronos** pour filtrer par groupe/stage · **Détail Match** pour la heatmap score-par-score")

st.divider()

# ── Top 10 candidats vainqueur ────────────────────────────────────────────────
st.markdown("### 🥇 Top 10 — Candidats vainqueur")

tp_teams = (
    tp
    .merge(teams[["team", "group", "confederation", "pot"]], on="team", how="left")
    .merge(mv[["team", "mv_current_eur"]], on="team", how="left")
)
top10 = tp_teams.sort_values("proba_winner", ascending=False).head(10).copy()

rows_tp = []
for _, r in top10.iterrows():
    mv_eur = r.get("mv_current_eur")
    mv_str = f"{mv_eur/1e6:.0f} M€" if pd.notna(mv_eur) and mv_eur > 0 else "—"
    rows_tp.append({
        "Équipe":       f"{flag(r['team'])} {r['team']}",
        "Pot":          int(r["pot"]) if pd.notna(r.get("pot")) else "—",
        "Groupe":       r.get("group", "—"),
        "Conf.":        r.get("confederation", "—"),
        "Elo":          f"{r['elo_rating']:.0f}",
        "Squad MV":     mv_str,
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
