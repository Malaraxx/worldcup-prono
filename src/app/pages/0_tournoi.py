"""Page 0 — Tournoi : classements par groupe + scores en temps réel."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[3]))

from datetime import timezone
import pandas as pd
import streamlit as st

from src.app.utils import load_fixtures, load_results, flag

st.markdown("""
<div style="background:linear-gradient(135deg,#0A2342 0%,#1565C0 55%,#1976D2 100%);
            border-radius:14px;padding:22px 28px;margin-bottom:20px;
            box-shadow:0 4px 20px rgba(21,101,192,0.25)">
  <div style="font-size:1.9rem;font-weight:800;color:#fff;letter-spacing:0.5px">
    🌍 Tournoi — Phase de groupes
  </div>
  <div style="font-size:0.9rem;color:rgba(255,255,255,0.65);margin-top:4px">
    Classements et scores en direct · FIFA World Cup 2026
  </div>
</div>
""", unsafe_allow_html=True)

# ── Données ───────────────────────────────────────────────────────────────────
fix     = load_fixtures()
results = load_results()

group_fix = fix[fix["stage"] == "group"].copy()
results_map = (
    results.set_index("match_id")[["home_score", "away_score"]].to_dict("index")
    if not results.empty else {}
)

GROUPS = sorted(group_fix["group"].dropna().unique())


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_standings(grp: str) -> pd.DataFrame:
    gf = group_fix[group_fix["group"] == grp]
    teams_in_group = sorted(set(gf["home_slot"]) | set(gf["away_slot"]))
    rows = {t: {"J": 0, "G": 0, "N": 0, "P": 0, "bp": 0, "bc": 0, "diff": 0, "Pts": 0}
            for t in teams_in_group}

    for _, m in gf.iterrows():
        mid  = int(m["match_id"])
        home = m["home_slot"]
        away = m["away_slot"]
        res  = results_map.get(mid)
        if res is None:
            continue
        hs, as_ = int(res["home_score"]), int(res["away_score"])
        rows[home]["J"] += 1; rows[away]["J"] += 1
        rows[home]["bp"] += hs; rows[home]["bc"] += as_
        rows[away]["bp"] += as_; rows[away]["bc"] += hs
        if hs > as_:
            rows[home]["G"] += 1; rows[home]["Pts"] += 3
            rows[away]["P"] += 1
        elif hs == as_:
            rows[home]["N"] += 1; rows[home]["Pts"] += 1
            rows[away]["N"] += 1; rows[away]["Pts"] += 1
        else:
            rows[away]["G"] += 1; rows[away]["Pts"] += 3
            rows[home]["P"] += 1

    for t in rows:
        rows[t]["diff"] = rows[t]["bp"] - rows[t]["bc"]

    df = pd.DataFrame(rows).T.reset_index().rename(columns={"index": "team"})
    df = df.sort_values(["Pts", "diff", "bp"], ascending=[False, False, False]).reset_index(drop=True)
    df.index = df.index + 1
    return df


def render_group(grp: str) -> None:
    st.markdown(f"### Groupe {grp}")

    # Classement
    stand = compute_standings(grp)
    n_teams = len(stand)
    n_played_max = stand["J"].max()

    def row_style(i: int) -> str:
        if n_played_max == 3:
            if i <= 2:
                return "background:#e8f5e9"
            return "background:#ffebee"
        if i <= 2:
            return "background:rgba(46,125,50,0.12)"
        return ""

    rows_html = ""
    for pos, row in stand.iterrows():
        style = row_style(pos)
        t = row["team"]
        f = flag(t)
        rows_html += (
            f"<tr style='{style}'>"
            f"<td style='font-weight:700;color:#555;width:24px'>{pos}</td>"
            f"<td style='white-space:nowrap'>{f} {t}</td>"
            f"<td style='text-align:center'>{int(row['J'])}</td>"
            f"<td style='text-align:center'>{int(row['G'])}</td>"
            f"<td style='text-align:center'>{int(row['N'])}</td>"
            f"<td style='text-align:center'>{int(row['P'])}</td>"
            f"<td style='text-align:center'>{int(row['bp'])}-{int(row['bc'])}</td>"
            f"<td style='text-align:center'>{'+' if row['diff']>=0 else ''}{int(row['diff'])}</td>"
            f"<td style='text-align:center;font-weight:800;color:#1565C0'>{int(row['Pts'])}</td>"
            "</tr>"
        )

    st.markdown(f"""
<table style="width:100%;border-collapse:collapse;font-size:0.82rem;margin-bottom:10px">
  <thead>
    <tr style="border-bottom:2px solid #1565C0;color:#1565C0;font-weight:700">
      <th></th><th style="text-align:left">Équipe</th>
      <th style="text-align:center">J</th><th style="text-align:center">G</th>
      <th style="text-align:center">N</th><th style="text-align:center">P</th>
      <th style="text-align:center">Buts</th><th style="text-align:center">Diff</th>
      <th style="text-align:center">Pts</th>
    </tr>
  </thead>
  <tbody>{rows_html}</tbody>
</table>
""", unsafe_allow_html=True)

    # Matchs du groupe
    gf = group_fix[group_fix["group"] == grp].sort_values("kickoff_utc")
    for _, m in gf.iterrows():
        mid  = int(m["match_id"])
        home = m["home_slot"]
        away = m["away_slot"]
        hf   = flag(home)
        af   = flag(away)
        res  = results_map.get(mid)

        if res is not None:
            hs, as_ = int(res["home_score"]), int(res["away_score"])
            score_html = (
                f"<span style='font-size:1.05rem;font-weight:800;color:#1565C0'>"
                f"{hs} – {as_}</span>"
            )
            date_str = ""
        else:
            score_html = (
                "<span style='font-size:0.85rem;color:#aaa;font-weight:600'>vs</span>"
            )
            dt = m.get("date_local")
            if hasattr(dt, "strftime"):
                date_str = dt.strftime("%d/%m %H:%M")
            else:
                date_str = ""

        st.markdown(
            f"<div style='display:flex;align-items:center;gap:6px;"
            f"padding:4px 0;border-bottom:1px solid #f0f0f0;font-size:0.85rem'>"
            f"<span style='color:#999;font-size:0.75rem;min-width:80px'>{date_str}</span>"
            f"<span style='flex:1;text-align:right'>{hf} {home}</span>"
            f"<span style='min-width:60px;text-align:center'>{score_html}</span>"
            f"<span style='flex:1'>{af} {away}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)


# ── Layout : 3 colonnes × 4 lignes ───────────────────────────────────────────
st.markdown(
    "<p style='font-size:0.78rem;color:#888;margin-bottom:12px'>"
    "Fond vert = qualifié (top 2) après 3 matchs joués · Fond rouge = éliminé</p>",
    unsafe_allow_html=True,
)

for row_idx in range(0, len(GROUPS), 3):
    cols = st.columns(3)
    for col_idx, grp in enumerate(GROUPS[row_idx: row_idx + 3]):
        with cols[col_idx]:
            with st.container(border=True):
                render_group(grp)
