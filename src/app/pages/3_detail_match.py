"""Page 3 — Détail d'un match."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[3]))

import pandas as pd
import streamlit as st

from src.app.utils import (
    load_fixtures, get_match, STAGE_LABELS, flag,
    format_pct, format_ev, mpp_implied,
)
from src.app.components.score_heatmap import score_heatmap
from src.app.components.probas_bar import probas_bar
from src.app.components.pick_card import pick_card

# ── Sélection du match ────────────────────────────────────────────────────────
fix = load_fixtures()

# Priorité : session_state → query_params → sélecteur
match_id = st.session_state.get("selected_match_id")
if match_id is None and "match_id" in st.query_params:
    try:
        match_id = int(st.query_params["match_id"])
    except ValueError:
        match_id = None

group_fix = fix[fix["stage"] == "group"].sort_values("kickoff_dt")
match_options = {
    f"#{int(r['match_id'])} — {flag(r['home_slot'])} {r['home_slot']}  vs  {flag(r['away_slot'])} {r['away_slot']}": int(r["match_id"])
    for _, r in group_fix.iterrows()
}
all_ids = fix.sort_values("kickoff_dt")["match_id"].tolist()
default_id = match_id if match_id in all_ids else all_ids[0]

# Sélecteur en haut (couvre group + autres stages)
all_options = {
    f"#{int(r['match_id'])} — {flag(r['home_slot'])} {r['home_slot']}  vs  {flag(r['away_slot'])} {r['away_slot']} ({STAGE_LABELS.get(r['stage'],r['stage'])})": int(r["match_id"])
    for _, r in fix.sort_values("kickoff_dt").iterrows()
}
default_label = next((k for k, v in all_options.items() if v == default_id), list(all_options.keys())[0])
sel_label = st.selectbox("Match", list(all_options.keys()), index=list(all_options.keys()).index(default_label))
match_id = all_options[sel_label]

m = get_match(match_id)
if m is None:
    st.error(f"Match {match_id} introuvable.")
    st.stop()

home, away = m["home"], m["away"]
pred  = m["pred"]
picks = m["picks"]
dist  = m["dist"]

# ── Header ────────────────────────────────────────────────────────────────────
dt_obj = m.get("date_local")
dt_str = dt_obj.strftime("%A %d %B %Y · %H:%M (Paris)") if pd.notna(dt_obj) and dt_obj else "—"

st.markdown(f"""
<div style="text-align:center;padding:16px 0 8px">
  <div style="font-size:2.4rem;font-weight:800;letter-spacing:2px">
    {m['home_flag']} {home} &nbsp;vs&nbsp; {away} {m['away_flag']}
  </div>
  <div style="font-size:1rem;color:#666;margin-top:4px">
    {dt_str} &nbsp;·&nbsp; {m['venue']}, {m['city']} &nbsp;·&nbsp; {STAGE_LABELS.get(m['stage'], m['stage'])} {('Gr. ' + m['group']) if m['group'] else ''}
  </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── 3 colonnes : probas | elo/lambda | picks ───────────────────────────────
col_p, col_e, col_k = st.columns([1.4, 1.2, 1.4])

ph  = pred.get("p_home_win_cal", 0.0)
pd_ = pred.get("p_draw_cal", 0.0)
pa  = pred.get("p_away_win_cal", 0.0)

has_picks = bool(picks)

with col_p:
    st.markdown("##### Probabilités & Cotes")
    if has_picks:
        ch, cd, ca = int(picks["cote_home"]), int(picks["cote_draw"]), int(picks["cote_away"])
        ip_h, ip_d, ip_a = mpp_implied(ch, cd, ca)

        # Tableau modèle vs MPP
        data_cmp = {
            "":               [home, "Nul", away],
            "Cote MPP":       [ch, cd, ca],
            "MPP implicite":  [format_pct(ip_h), format_pct(ip_d), format_pct(ip_a)],
            "Modèle calibré": [format_pct(ph), format_pct(pd_), format_pct(pa)],
            "Écart":          [
                f"{(ph-ip_h):+.1%}", f"{(pd_-ip_d):+.1%}", f"{(pa-ip_a):+.1%}"
            ],
        }
        st.dataframe(pd.DataFrame(data_cmp).set_index(""), use_container_width=True)

        max_edge = max(abs(ph-ip_h), abs(pd_-ip_d), abs(pa-ip_a))
        if max_edge > 0.15:
            st.warning(f"**VALUE BET ⚠️** — Écart modèle/MPP > 15% ({max_edge:.0%})")
    else:
        st.markdown(f"""
| Issue | Proba modèle |
|-------|-------------|
| {home} | {format_pct(ph)} |
| Nul | {format_pct(pd_)} |
| {away} | {format_pct(pa)} |
""")

with col_e:
    st.markdown("##### Elo & Poisson")
    elo_h     = pred.get("elo_home", "—")
    elo_a     = pred.get("elo_away", "—")
    elo_h_adj = pred.get("elo_home_adj", "—")
    elo_a_adj = pred.get("elo_away_adj", "—")
    lam_h     = pred.get("lambda_home", "—")
    lam_a     = pred.get("lambda_away", "—")
    diff      = pred.get("elo_diff", "—")

    st.markdown(f"""
| | {home} | {away} |
|--|--:|--:|
| **Elo brut** | {elo_h:.0f} | {elo_a:.0f} |
| **Elo ajusté** | {elo_h_adj:.0f} | {elo_a_adj:.0f} |
| **Conf.** | {m['home_conf']} | {m['away_conf']} |
| **λ Poisson** | {lam_h:.3f} | {lam_a:.3f} |
""")
    if isinstance(diff, float):
        direction = home if diff > 0 else away
        st.caption(f"Elo diff ajusté : {diff:+.1f} en faveur de **{direction}**")

with col_k:
    st.markdown("##### Picks recommandés")
    if has_picks:
        mode_rec = picks.get("mode_recommended", "safe")
        edge     = picks.get("edge_value_vs_safe_pct", 0.0)
        for m_name in ["safe", "value", "lottery"]:
            pick_card(
                mode=m_name,
                score=picks.get(f"{m_name}_score", "—"),
                ev=picks.get(f"{m_name}_ev", 0.0),
                wr=picks.get(f"{m_name}_wr", 0.0),
                recommended=(m_name == mode_rec),
                edge_pct=edge if m_name == "value" else None,
            )
    else:
        st.info("Picks disponibles uniquement pour la phase de groupes (cotes MPP requises).")

st.divider()

# ── Graphique modèle vs MPP ───────────────────────────────────────────────────
st.subheader("Modèle vs MPP")

if has_picks:
    ch, cd, ca = int(picks["cote_home"]), int(picks["cote_draw"]), int(picks["cote_away"])
    ip_h, ip_d, ip_a = mpp_implied(ch, cd, ca)
    fig_bar = probas_bar(ph, pd_, pa, home, away, ip_h, ip_d, ip_a)
else:
    fig_bar = probas_bar(ph, pd_, pa, home, away)

st.plotly_chart(fig_bar, use_container_width=True)

# ── Heatmap scores ────────────────────────────────────────────────────────────
if not dist.empty:
    st.divider()
    st.subheader("Distribution des scores — Heatmap")
    fig_heat = score_heatmap(dist, home, away)
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Top 10 scores ─────────────────────────────────────────────────────────
    st.subheader("Top 10 scores les plus probables")

    if has_picks:
        ch, cd, ca = int(picks["cote_home"]), int(picks["cote_draw"]), int(picks["cote_away"])
    else:
        ch = cd = ca = 100

    from src.strategy.optimal_pick import estimate_rarity

    top_scores = dist.sort_values("proba", ascending=False).head(10).copy()
    rows_sc = []
    for _, r in top_scores.iterrows():
        i, j   = int(r["i"]), int(r["j"])
        proba  = float(r["proba"])
        rarity = estimate_rarity(i, j, ch, cd, ca)
        ev_val = proba * (ch if i > j else (ca if j > i else cd) + rarity)
        rows_sc.append({
            "Score":        f"{i} — {j}",
            "Proba":        format_pct(proba),
            "Issue":        "Victoire " + home if i > j else ("Nul" if i == j else "Victoire " + away),
            "Bonus rareté": rarity,
            "EV":           format_ev(ev_val),
        })

    st.dataframe(pd.DataFrame(rows_sc), use_container_width=True, hide_index=True)
else:
    st.info("Heatmap et top scores disponibles uniquement pour la phase de groupes.")
