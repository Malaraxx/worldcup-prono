"""Page 4 — Bracket KO + pick équipe vainqueur."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[3]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.app.utils import (
    load_ko_predictions, load_tournament_probabilities, load_teams,
    load_group_simulations, STAGE_LABELS, flag, flag_html, format_pct,
)

st.markdown("""
<div style="background:linear-gradient(135deg,#0A2342 0%,#1565C0 55%,#1976D2 100%);
            border-radius:14px;padding:22px 28px;margin-bottom:20px;
            box-shadow:0 4px 20px rgba(21,101,192,0.25)">
  <div style="font-size:1.9rem;font-weight:800;color:#fff;letter-spacing:0.5px">
    🏆 Bracket WC 2026
  </div>
  <div style="font-size:0.9rem;color:rgba(255,255,255,0.65);margin-top:4px">
    Standings de poule · Phase éliminatoire · Pick vainqueur
  </div>
</div>
""", unsafe_allow_html=True)

ko   = load_ko_predictions()
tp   = load_tournament_probabilities()
teams = load_teams()
gsim = load_group_simulations()

# ── Phase de groupes — Standings ──────────────────────────────────────────────
st.markdown("### 📊 Standings de poule *(Monte-Carlo · 10 000 simulations)*")

all_groups = sorted(gsim["group"].unique())
tabs_groups = st.tabs([f"Groupe {g}" for g in all_groups])

pts_cols = [c for c in gsim.columns if c.startswith("pts_")]

for tab, grp in zip(tabs_groups, all_groups):
    with tab:
        g_df = gsim[gsim["group"] == grp].sort_values("proba_1st", ascending=False).copy()

        col_table, col_pts = st.columns([1.2, 1.8])

        with col_table:
            rows = []
            for _, r in g_df.iterrows():
                team = r["team"]
                rows.append({
                    "Équipe":    f"{flag(team)} {team}",
                    "Elo":       f"{r['elo_rating']:.0f}",
                    "P(1er)":   format_pct(r["proba_1st"]),
                    "P(2e)":    format_pct(r["proba_2nd"]),
                    "P(3e)":    format_pct(r["proba_3rd"]),
                    "P(Élim.)": format_pct(r["proba_elim"]),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        with col_pts:
            if pts_cols:
                # Distribution des points simulés par équipe
                fig = go.Figure()
                colors = ["#1565C0", "#2E7D32", "#E65100", "#6A1B9A"]
                for idx, (_, r) in enumerate(g_df.iterrows()):
                    team = r["team"]
                    pts_vals = [int(c.split("_")[1]) for c in pts_cols]
                    probas   = [float(r[c]) for c in pts_cols]
                    fig.add_trace(go.Bar(
                        name=f"{flag(team)} {team}",
                        x=pts_vals,
                        y=probas,
                        marker_color=colors[idx % len(colors)],
                        opacity=0.85,
                    ))
                fig.update_layout(
                    barmode="group",
                    title_text="Distribution des points en phase de groupes",
                    xaxis_title="Points",
                    yaxis_title="Probabilité",
                    yaxis_tickformat=".0%",
                    height=280,
                    margin=dict(l=10, r=10, t=40, b=30),
                    legend=dict(orientation="h", y=-0.25, font_size=10),
                    xaxis=dict(tickmode="array", tickvals=pts_vals),
                )
                st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Bracket KO ────────────────────────────────────────────────────────────────
st.markdown("### ⚔️ Phase éliminatoire")
st.caption("Top 3 équipes les plus probables pour chaque slot · proba conditionnelle d'atteindre ce slot")

STAGE_SEQ   = ["r32", "r16", "qf", "sf", "final"]
STAGE_NAMES = {
    "r32": "R32 (32→16)", "r16": "R16 (16→8)",
    "qf": "Quarts (8→4)", "sf": "Demies (4→2)", "final": "Finale",
}

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
                return f"<{tag[0]}>{flag_html(name)} {name}</{tag[1]}> ({prob:.0%})"

            home_lines = "\n".join(f"  {fmt_team(n, p)}" for n, p in h_teams if n)
            away_lines = "\n".join(f"  {fmt_team(n, p)}" for n, p in a_teams if n)

            raw_h = row["home_slot"]
            raw_a = row["away_slot"]

            # Winner = bleu, Runner-up = orange
            if "Winner" in raw_h or "1er" in raw_h:
                h_color, h_label = "#1565C0", raw_h.replace("Winner", "🥇 1er")
            else:
                h_color, h_label = "#E65100", raw_h.replace("Runner-up", "🥈 2e")

            if "Winner" in raw_a or "1er" in raw_a:
                a_color, a_label = "#1565C0", raw_a.replace("Winner", "🥇 1er")
            else:
                a_color, a_label = "#E65100", raw_a.replace("Runner-up", "🥈 2e")

            h_bg = "rgba(21,101,192,0.06)" if "#1565C0" in h_color else "rgba(230,81,0,0.06)"
            a_bg = "rgba(21,101,192,0.06)" if "#1565C0" in a_color else "rgba(230,81,0,0.06)"
            st.markdown(f"""
<div style="border:1px solid #DDE3F0;border-radius:10px;overflow:hidden;
            margin-bottom:10px;box-shadow:0 2px 8px rgba(0,0,0,0.06)">
  <div style="background:{h_bg};padding:10px 14px;border-bottom:1px solid #E8ECF4">
    <div style="font-size:0.68rem;font-weight:700;color:{h_color};
                letter-spacing:1px;margin-bottom:5px">{h_label}</div>
    <div style="font-size:0.8rem;line-height:1.6">{home_lines}</div>
  </div>
  <div style="background:{a_bg};padding:10px 14px">
    <div style="font-size:0.68rem;font-weight:700;color:{a_color};
                letter-spacing:1px;margin-bottom:5px">{a_label}</div>
    <div style="font-size:0.8rem;line-height:1.6">{away_lines}</div>
  </div>
</div>
""", unsafe_allow_html=True)

    if stage_key != "final":
        st.markdown("")

st.divider()

# ── Top candidats vainqueur ──────────────────────────────────────────────────
st.markdown("### 🏅 Équipe vainqueur")

top_tp = tp.merge(teams[["team", "pot", "confederation"]], on="team", how="left")
top_tp = top_tp.sort_values("proba_winner", ascending=False).head(10).copy()
rows_tp = []
for _, r in top_tp.iterrows():
    team = r["team"]
    rows_tp.append({
        "Équipe":       f"{flag(team)} {team}",
        "Pot":          int(r["pot"]) if pd.notna(r.get("pot")) else "—",
        "P(Vainqueur)": format_pct(r["proba_winner"]),
        "P(Finale)":    format_pct(r["proba_final"]),
        "P(Demi)":      format_pct(r["proba_sf"]),
        "P(R32)":       format_pct(r["proba_r32"]),
    })
st.dataframe(pd.DataFrame(rows_tp), use_container_width=True, hide_index=True)

# ── 3e place ─────────────────────────────────────────────────────────────────
third_df = ko[ko["stage"] == "3rd"]
if not third_df.empty:
    st.divider()
    st.subheader("Match pour la 3e place")
    st.caption("Les 2 perdants des demi-finales s'affrontent pour la 3e place")
    r = third_df.iloc[0]
    h_teams = [(r.get(f"home_team_{k}"), r.get(f"home_team_{k}_prob")) for k in range(1, 4) if pd.notna(r.get(f"home_team_{k}"))]
    a_teams = [(r.get(f"away_team_{k}"), r.get(f"away_team_{k}_prob")) for k in range(1, 4) if pd.notna(r.get(f"away_team_{k}"))]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Équipe 1 (perdant Demi 1)**")
        for nm, pb in h_teams:
            if nm:
                st.write(f"{flag(nm)} {nm} ({pb:.0%})")
    with c2:
        st.markdown("**Équipe 2 (perdant Demi 2)**")
        for nm, pb in a_teams:
            if nm:
                st.write(f"{flag(nm)} {nm} ({pb:.0%})")
