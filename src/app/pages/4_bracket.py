"""Page 4 — Bracket KO + pick équipe vainqueur."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[3]))

import pandas as pd
import streamlit as st

from src.app.utils import (
    load_ko_predictions, load_tournament_probabilities, load_teams,
    load_group_simulations, STAGE_LABELS, flag, format_pct,
    load_my_winner_pick, save_my_winner_pick,
)

st.title("🏆 Bracket WC 2026")

ko   = load_ko_predictions()
tp   = load_tournament_probabilities()
teams = load_teams()
gsim = load_group_simulations()

# ── Phase de groupes — Standings ──────────────────────────────────────────────
st.subheader("Standings de poule (MC 10 000 simulations)")

all_groups = sorted(gsim["group"].unique())
tabs_groups = st.tabs([f"Groupe {g}" for g in all_groups])

for tab, grp in zip(tabs_groups, all_groups):
    with tab:
        g_df = gsim[gsim["group"] == grp].sort_values("proba_1st", ascending=False).copy()
        rows = []
        for _, r in g_df.iterrows():
            team = r["team"]
            rows.append({
                "Équipe":     f"{flag(team)} {team}",
                "Elo":        f"{r['elo_rating']:.0f}",
                "P(1er)":    format_pct(r["proba_1st"]),
                "P(2e)":     format_pct(r["proba_2nd"]),
                "P(3e)":     format_pct(r["proba_3rd"]),
                "P(Élim.)":  format_pct(r["proba_elim"]),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.divider()

# ── Bracket KO ────────────────────────────────────────────────────────────────
st.subheader("Phase éliminatoire")
st.caption("Top 3 équipes les plus probables pour chaque slot · proba conditionnelle d'atteindre ce slot")

STAGE_SEQ = ["r32", "r16", "qf", "sf", "final"]
STAGE_NAMES = {"r32": "R32 (32→16)", "r16": "R16 (16→8)", "qf": "Quarts (8→4)", "sf": "Demies (4→2)", "final": "Finale"}

for stage_key in STAGE_SEQ:
    stage_df = ko[ko["stage"] == stage_key]
    if stage_df.empty:
        continue

    st.markdown(f"#### {STAGE_NAMES.get(stage_key, stage_key)}")
    n_matches = len(stage_df)
    cols = st.columns(min(n_matches, 4))

    for col_idx, (_, row) in enumerate(stage_df.iterrows()):
        col = cols[col_idx % len(cols)]
        with col:
            # Home slot
            h_teams = [
                (row.get(f"home_team_{k}"), row.get(f"home_team_{k}_prob"))
                for k in range(1, 4)
                if pd.notna(row.get(f"home_team_{k}"))
            ]
            a_teams = [
                (row.get(f"away_team_{k}"), row.get(f"away_team_{k}_prob"))
                for k in range(1, 4)
                if pd.notna(row.get(f"away_team_{k}"))
            ]

            def fmt_team(name: str, prob: float) -> str:
                tag = ("strong", "strong") if prob >= 0.50 else ("span", "span")
                return f"<{tag[0]}>{flag(name)} {name}</{tag[1]}> ({prob:.0%})"

            home_lines = "\n".join(f"  {fmt_team(n, p)}" for n, p in h_teams if n)
            away_lines = "\n".join(f"  {fmt_team(n, p)}" for n, p in a_teams if n)

            slot_label = row["home_slot"].replace("Runner-up", "2e").replace("Winner", "1er")
            slot_a_label = row["away_slot"].replace("Runner-up", "2e").replace("Winner", "1er")

            st.markdown(f"""
<div style="border:1px solid #ddd;border-radius:6px;padding:10px;font-size:0.8rem;margin-bottom:8px">
<div style="font-weight:600;color:#1565C0;margin-bottom:4px">{slot_label}</div>
{home_lines}
<hr style="margin:6px 0;border-color:#eee">
<div style="font-weight:600;color:#BF360C;margin-bottom:4px">{slot_a_label}</div>
{away_lines}
</div>
""", unsafe_allow_html=True)

    # Séparation entre stages
    if stage_key != "final":
        st.markdown("")

st.divider()

# ── Top candidats vainqueur + pick Flo ───────────────────────────────────────
st.subheader("🏅 Équipe vainqueur")

left, right = st.columns([1.2, 1])

with left:
    st.markdown("**Top 10 — Probabilité de remporter le titre**")
    top_tp = tp.sort_values("proba_winner", ascending=False).head(10).copy()
    rows_tp = []
    for _, r in top_tp.iterrows():
        team = r["team"]
        rows_tp.append({
            "Équipe":      f"{flag(team)} {team}",
            "P(Vainqueur)": format_pct(r["proba_winner"]),
            "P(Finale)":   format_pct(r["proba_final"]),
            "P(Demi)":     format_pct(r["proba_sf"]),
        })
    st.dataframe(pd.DataFrame(rows_tp), use_container_width=True, hide_index=True)

with right:
    st.markdown("**Mon pick équipe vainqueur**")
    current_pick = load_my_winner_pick()

    all_teams_sorted = tp.sort_values("proba_winner", ascending=False)["team"].tolist()
    team_options = [f"{flag(t)} {t}" for t in all_teams_sorted]
    team_map = {f"{flag(t)} {t}": t for t in all_teams_sorted}

    default_idx = 0
    if current_pick and current_pick in all_teams_sorted:
        default_idx = all_teams_sorted.index(current_pick)

    sel = st.selectbox("Sélectionner l'équipe vainqueur", team_options, index=default_idx)
    sel_team = team_map[sel]

    if current_pick:
        pick_proba = tp.loc[tp["team"] == current_pick, "proba_winner"]
        pwin_str = format_pct(pick_proba.values[0]) if not pick_proba.empty else "—"
        st.info(f"Pick actuel : {flag(current_pick)} **{current_pick}** — P(win) = {pwin_str}")

    if st.button("💾 Sauvegarder mon pick"):
        save_my_winner_pick(sel_team)
        st.success(f"Pick sauvegardé : {flag(sel_team)} **{sel_team}**")
        st.rerun()

# ── 3e place ─────────────────────────────────────────────────────────────────
third_df = ko[ko["stage"] == "3rd"]
if not third_df.empty:
    st.divider()
    st.subheader("Match pour la 3e place")
    r = third_df.iloc[0]
    h_teams = [(r.get(f"home_team_{k}"), r.get(f"home_team_{k}_prob")) for k in range(1,4) if pd.notna(r.get(f"home_team_{k}"))]
    a_teams = [(r.get(f"away_team_{k}"), r.get(f"away_team_{k}_prob")) for k in range(1,4) if pd.notna(r.get(f"away_team_{k}"))]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Équipe 1**")
        for nm, pb in h_teams:
            if nm:
                st.write(f"{flag(nm)} {nm} ({pb:.0%})")
    with c2:
        st.markdown("**Équipe 2**")
        for nm, pb in a_teams:
            if nm:
                st.write(f"{flag(nm)} {nm} ({pb:.0%})")
