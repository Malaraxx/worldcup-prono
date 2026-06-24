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
    STAGE_LABELS, flag, flag_html, format_pct, model_update_time,
    load_mv_baseline, load_results,
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
fix     = load_fixtures()
picks   = load_picks()
tp      = load_tournament_probabilities()
teams   = load_teams()
mv      = load_mv_baseline()
results = load_results()

now_utc        = datetime.now(tz=timezone.utc)
total_matches  = len(fix)
played_matches = len(results)   # basé sur les résultats saisis, pas sur l'heure
n_picks        = len(picks)

# ── Picks performance ─────────────────────────────────────────────────────────
n_correct     = 0
n_played_with_picks = 0
pts_gagnes    = 0.0

if not results.empty and not picks.empty:
    played_picks = picks.merge(
        results[["match_id", "home_score", "away_score"]],
        on="match_id", how="inner",
    )
    for _, r in played_picks.iterrows():
        mode = r.get("mode_recommended")
        if pd.isna(mode):
            continue
        n_played_with_picks += 1
        real_score = f"{int(r['home_score'])}-{int(r['away_score'])}"
        if mode == "safe":
            pick, ev, wr = r["safe_score"],    r["safe_ev"],    r["safe_wr"]
        elif mode == "value":
            pick, ev, wr = r["value_score"],   r["value_ev"],   r["value_wr"]
        else:
            pick, ev, wr = r["lottery_score"], r["lottery_ev"], r["lottery_wr"]
        if str(pick) == real_score and float(wr) > 0:
            n_correct += 1
            pts_gagnes += float(ev) / float(wr)

# ── Métriques ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Matchs total", total_matches)
c2.metric("Matchs joués", played_matches, help="Basé sur les résultats saisis dans wc2026_results.csv")
c3.metric("Pronos disponibles", n_picks, help="Matchs de poule avec cotes MPP")
if n_played_with_picks > 0:
    c4.metric(
        "Picks corrects",
        f"{n_correct}/{n_played_with_picks}",
        help=f"Scores exacts · Points MPP gagnés : {pts_gagnes:.0f} pts",
    )
else:
    c4.metric("Picks corrects", "—", help="En attente des premiers résultats")

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

    _BADGE = {
        "safe":    ("#2E7D32", "#E8F5E9", "SAFE"),
        "value":   ("#E65100", "#FFF3E0", "VALUE"),
        "lottery": ("#6A1B9A", "#F3E5F5", "LOTTERY"),
    }

    def _pick_card(r) -> str:
        home = r["home_slot"]
        away = r["away_slot"]
        dl = r.get("date_local")
        dt_str = dl.strftime("%d/%m %H:%M") if pd.notna(dl) else "—"
        mode = str(r.get("mode_recommended", "")).lower()
        grp = r.get("group", "")
        stg = STAGE_LABELS.get(r.get("stage", ""), r.get("stage", ""))
        bcolor, bbg, blabel = _BADGE.get(mode, ("#607D8B", "#ECEFF1", mode.upper()))
        ctx = f"Gr. {grp}" if stg == "Groupes" else stg

        if mode == "safe":
            score_pick = r.get("safe_score", "—")
            wr, ev = r.get("safe_wr"), r.get("safe_ev")
        elif mode == "value":
            score_pick = r.get("value_score", "—")
            wr, ev = r.get("value_wr"), r.get("value_ev")
        else:
            score_pick = r.get("lottery_score", "—")
            wr, ev = r.get("lottery_wr"), r.get("lottery_ev")

        wr_pct = f"{float(wr):.0%}" if pd.notna(wr) else "—"
        ev_str = f"{float(ev):.1f}" if pd.notna(ev) else "—"
        ch, cd, ca = r.get("cote_home"), r.get("cote_draw"), r.get("cote_away")
        cotes = f"{int(ch)} · {int(cd)} · {int(ca)}" if pd.notna(ch) else "—"
        hfl = flag_html(home, size=28)
        afl = flag_html(away, size=28)

        return f"""
<div style="background:#fff;border:1px solid #DDE3F0;border-radius:14px;
            box-shadow:0 2px 10px rgba(21,101,192,0.07);overflow:hidden;height:100%">
  <div style="background:{bbg};padding:10px 16px 9px;border-bottom:1px solid #F0F2F7;
              display:flex;align-items:center;justify-content:space-between">
    <span style="background:{bcolor};color:#fff;font-size:0.63rem;font-weight:800;
                 letter-spacing:1.5px;padding:3px 10px;border-radius:20px">{blabel}</span>
    <span style="font-size:0.68rem;color:#9AA5B8;font-weight:500">{ctx} · {dt_str}</span>
  </div>
  <div style="padding:16px 14px 10px">
    <div style="display:flex;align-items:center;justify-content:space-between;gap:6px">
      <div style="flex:1;text-align:center">
        <div style="font-size:1.5rem;margin-bottom:5px">{hfl}</div>
        <div style="font-size:0.78rem;font-weight:700;color:#1A1A2E;
                    white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{home}</div>
      </div>
      <div style="text-align:center;min-width:52px">
        <div style="background:#0A2342;color:#fff;font-size:0.9rem;font-weight:800;
                    padding:6px 10px;border-radius:8px;letter-spacing:1px">{score_pick}</div>
        <div style="font-size:0.58rem;color:#B0BAC8;margin-top:3px;font-weight:700;letter-spacing:0.8px">PICK</div>
      </div>
      <div style="flex:1;text-align:center">
        <div style="font-size:1.5rem;margin-bottom:5px">{afl}</div>
        <div style="font-size:0.78rem;font-weight:700;color:#1A1A2E;
                    white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{away}</div>
      </div>
    </div>
  </div>
  <div style="padding:8px 16px 12px;border-top:1px solid #F5F7FA;
              display:flex;justify-content:space-between;font-size:0.71rem;color:#9AA5B8">
    <span>WR <strong style="color:#1A1A2E">{wr_pct}</strong></span>
    <span>EV <strong style="color:#1565C0">{ev_str}</strong></span>
    <span>Cotes <span style="color:#666">{cotes}</span></span>
  </div>
</div>"""

    # Section picks
    with_picks = upcoming_with_picks[upcoming_with_picks["_has_pick"]]
    without_picks = upcoming_with_picks[~upcoming_with_picks["_has_pick"]]

    if not with_picks.empty:
        st.markdown("**📊 Matchs avec picks recommandés**")
        cards = [_pick_card(r) for _, r in with_picks.iterrows()]
        # Grille responsive : 3 cards par ligne max
        grid = '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:14px;margin-bottom:8px">'
        grid += "".join(cards)
        grid += "</div>"
        st.markdown(grid, unsafe_allow_html=True)

    # Section KO sans picks
    if not without_picks.empty:
        st.markdown(f"**🏆 Matchs KO à venir** *(équipes non encore qualifiées — {len(without_picks)} matchs)*")
        ko_rows = []
        for _, r in without_picks.head(16).iterrows():
            home, away = r["home_slot"], r["away_slot"]
            dl = r.get("date_local")
            ko_rows.append({
                "ID":    int(r["match_id"]),
                "Date":  dl.strftime("%d/%m %H:%M") if pd.notna(dl) else "—",
                "Match": f"{flag(home)} {home}  vs  {flag(away)} {away}",
                "Phase": STAGE_LABELS.get(r["stage"], r["stage"]),
            })
        st.dataframe(pd.DataFrame(ko_rows).set_index("ID"), use_container_width=True,
                     height=min(36 + len(ko_rows) * 35, 380))

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

max_pw = top10["proba_winner"].max() or 1.0

winner_html = '<div style="display:flex;flex-direction:column;gap:8px;margin-bottom:8px">'
for rank, (_, r) in enumerate(top10.iterrows(), 1):
    team = r["team"]
    pw   = float(r["proba_winner"])
    pf   = float(r["proba_final"])
    psf  = float(r["proba_sf"])
    elo  = f"{r['elo_rating']:.0f}"
    mv_eur = r.get("mv_current_eur")
    mv_str = f"{mv_eur/1e6:.0f}M€" if pd.notna(mv_eur) and mv_eur > 0 else "—"
    bar_w = pw / max_pw * 100

    medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"{rank}.")
    hfl = flag_html(team, size=22)

    winner_html += f"""
<div style="background:#fff;border:1px solid #E8ECF4;border-radius:10px;padding:10px 14px;
            display:flex;align-items:center;gap:14px;box-shadow:0 1px 4px rgba(21,101,192,0.05)">
  <div style="font-size:1.1rem;min-width:28px;text-align:center">{medal}</div>
  <div style="font-size:1.2rem">{hfl}</div>
  <div style="flex:2;min-width:0">
    <div style="font-size:0.82rem;font-weight:700;color:#1A1A2E;
                white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{team}</div>
    <div style="margin-top:5px;background:#EEF4FF;border-radius:4px;height:6px;overflow:hidden">
      <div style="background:linear-gradient(90deg,#1565C0,#42A5F5);height:100%;
                  width:{bar_w:.1f}%;border-radius:4px"></div>
    </div>
  </div>
  <div style="text-align:right;white-space:nowrap;flex-shrink:0">
    <div style="font-size:1rem;font-weight:800;color:#1565C0">{pw:.1%}</div>
    <div style="font-size:0.65rem;color:#9AA5B8;margin-top:1px">Finale {pf:.0%} · Demi {psf:.0%}</div>
  </div>
  <div style="text-align:right;white-space:nowrap;flex-shrink:0;min-width:58px">
    <div style="font-size:0.72rem;color:#555;font-weight:600">Elo {elo}</div>
    <div style="font-size:0.65rem;color:#9AA5B8">{mv_str}</div>
  </div>
</div>"""

winner_html += "</div>"
st.markdown(winner_html, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    f"Dernière mise à jour modèle : {model_update_time()} · "
    "Simulation Monte-Carlo 10 000 itérations · Elo + Poisson + Platt scaling"
)
