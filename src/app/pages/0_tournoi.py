"""Page 0 — Tournoi : classements par groupe + scores en temps réel."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[3]))

import pandas as pd
import streamlit as st

from src.app.utils import load_fixtures, load_results, flag_html as flag

# ── CSS global ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.group-card {
    background: #fff;
    border-radius: 14px;
    border: 1px solid #E8ECF4;
    box-shadow: 0 2px 12px rgba(21,101,192,0.07);
    overflow: hidden;
    margin-bottom: 16px;
}
.group-header {
    background: linear-gradient(135deg, #0A2342 0%, #1565C0 100%);
    padding: 10px 16px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.group-title {
    font-size: 1rem;
    font-weight: 800;
    color: #fff;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.group-body { padding: 0 0 8px 0; }

/* Standings */
.standings-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.78rem;
}
.standings-table thead tr {
    background: #F5F7FA;
    border-bottom: 2px solid #E8ECF4;
}
.standings-table thead th {
    padding: 6px 8px;
    color: #7A8BA6;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    font-size: 0.68rem;
}
.standings-table thead th:first-child { padding-left: 14px; }
.standings-table tbody tr {
    border-bottom: 1px solid #F0F2F7;
    transition: background 0.1s;
}
.standings-table tbody td { padding: 7px 8px; }
.standings-table tbody td:first-child { padding-left: 14px; }
.pos-badge {
    display: inline-flex; align-items: center; justify-content: center;
    width: 20px; height: 20px; border-radius: 6px;
    font-weight: 800; font-size: 0.72rem; color: #fff;
}
.team-name { font-weight: 600; font-size: 0.8rem; color: #1A1A2E; white-space: nowrap; }
.stat-cell { text-align: center; color: #555; }
.pts-cell  { text-align: center; font-weight: 800; font-size: 0.88rem; color: #1565C0; }
.diff-pos  { color: #2E7D32; font-weight: 700; }
.diff-neg  { color: #C62828; font-weight: 700; }
.diff-zero { color: #888; font-weight: 600; }

/* Match rows */
.matches-section { padding: 8px 12px 4px 12px; }
.match-row {
    display: flex;
    align-items: center;
    padding: 6px 4px;
    border-bottom: 1px solid #F5F7FA;
    gap: 4px;
    font-size: 0.78rem;
}
.match-row:last-child { border-bottom: none; }
.match-date {
    min-width: 68px;
    color: #9AA5B8;
    font-size: 0.7rem;
    font-weight: 500;
    flex-shrink: 0;
}
.match-team {
    flex: 1;
    font-weight: 600;
    color: #1A1A2E;
    font-size: 0.78rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.match-team-home { text-align: right; }
.match-score-box {
    min-width: 52px;
    text-align: center;
    flex-shrink: 0;
}
.score-played {
    background: #1565C0;
    color: #fff;
    font-weight: 800;
    font-size: 0.82rem;
    padding: 3px 8px;
    border-radius: 6px;
    white-space: nowrap;
}
.score-future {
    color: #C0C8D8;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 1px;
}
.matches-label {
    font-size: 0.67rem;
    font-weight: 700;
    color: #9AA5B8;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 8px 16px 4px 16px;
    border-top: 1px solid #F0F2F7;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background:linear-gradient(135deg,#0A2342 0%,#1565C0 55%,#1976D2 100%);
            border-radius:14px;padding:22px 28px;margin-bottom:24px;
            box-shadow:0 4px 20px rgba(21,101,192,0.25)">
  <div style="font-size:1.9rem;font-weight:800;color:#fff;letter-spacing:0.5px">
    🌍 Phase de groupes
  </div>
  <div style="font-size:0.9rem;color:rgba(255,255,255,0.65);margin-top:4px">
    Classements en temps réel · FIFA World Cup 2026
  </div>
</div>
""", unsafe_allow_html=True)

# ── Données ───────────────────────────────────────────────────────────────────
fix     = load_fixtures()
results = load_results()

group_fix   = fix[fix["stage"] == "group"].copy()
results_map = (
    results.set_index("match_id")[["home_score", "away_score"]].to_dict("index")
    if not results.empty else {}
)

GROUPS = sorted(group_fix["group"].dropna().unique())

# ── Standings ────────────────────────────────────────────────────────────────

def compute_standings(grp: str) -> pd.DataFrame:
    gf = group_fix[group_fix["group"] == grp]
    teams_in = sorted(set(gf["home_slot"]) | set(gf["away_slot"]))
    rows = {t: {"J": 0, "G": 0, "N": 0, "P": 0, "bp": 0, "bc": 0, "Pts": 0}
            for t in teams_in}

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

    df = pd.DataFrame(rows).T.reset_index().rename(columns={"index": "team"})
    df["diff"] = df["bp"] - df["bc"]
    df = df.sort_values(["Pts", "diff", "bp"], ascending=[False, False, False]).reset_index(drop=True)
    df.index = df.index + 1
    return df


def pos_badge(pos: int, max_played: int) -> str:
    if max_played == 3:
        colors = {1: "#F4C542", 2: "#1565C0", 3: "#C62828", 4: "#C62828"}
    else:
        colors = {1: "#1565C0", 2: "#1565C0", 3: "#9AA5B8", 4: "#9AA5B8"}
    bg = colors.get(pos, "#9AA5B8")
    return f'<span class="pos-badge" style="background:{bg}">{pos}</span>'


def render_group(grp: str) -> None:
    stand     = compute_standings(grp)
    max_played = int(stand["J"].max()) if not stand.empty else 0

    # ── Header ────────────────────────────────────────────────────────────────
    matches_in_group = group_fix[group_fix["group"] == grp]
    played_count = sum(1 for mid in matches_in_group["match_id"].astype(int) if mid in results_map)
    total_count  = len(matches_in_group)

    html = f"""
<div class="group-card">
  <div class="group-header">
    <div class="group-title">Groupe {grp}</div>
    <div style="margin-left:auto;font-size:0.7rem;color:rgba(255,255,255,0.55);font-weight:600">
      {played_count}/{total_count} matchs
    </div>
  </div>
  <div class="group-body">
"""

    # ── Standings table ───────────────────────────────────────────────────────
    rows_html = ""
    for pos, row in stand.iterrows():
        t  = row["team"]
        f  = flag(t)
        diff = int(row["diff"])
        diff_html = (
            f'<span class="diff-pos">+{diff}</span>' if diff > 0 else
            f'<span class="diff-neg">{diff}</span>'  if diff < 0 else
            f'<span class="diff-zero">0</span>'
        )

        # Row background
        if max_played == 3:
            bg = "#F0FBF0" if pos <= 2 else "#FFF5F5"
        else:
            bg = "#EEF4FF" if pos <= 2 else "transparent"

        rows_html += f"""
<tr style="background:{bg}">
  <td>{pos_badge(pos, max_played)}</td>
  <td><span class="team-name">{f} {t}</span></td>
  <td class="stat-cell">{int(row['J'])}</td>
  <td class="stat-cell">{int(row['G'])}</td>
  <td class="stat-cell">{int(row['N'])}</td>
  <td class="stat-cell">{int(row['P'])}</td>
  <td class="stat-cell">{int(row['bp'])}:{int(row['bc'])}</td>
  <td class="stat-cell">{diff_html}</td>
  <td class="pts-cell">{int(row['Pts'])}</td>
</tr>"""

    html += f"""
<table class="standings-table">
  <thead>
    <tr>
      <th style="width:28px"></th>
      <th style="text-align:left">Équipe</th>
      <th>J</th><th>G</th><th>N</th><th>P</th>
      <th>Buts</th><th>Diff</th><th>Pts</th>
    </tr>
  </thead>
  <tbody>{rows_html}</tbody>
</table>
"""

    # ── Match rows ────────────────────────────────────────────────────────────
    html += '<div class="matches-label">Matchs</div><div class="matches-section">'

    gf_sorted = matches_in_group.sort_values("kickoff_utc")
    for _, m in gf_sorted.iterrows():
        mid  = int(m["match_id"])
        home = m["home_slot"]
        away = m["away_slot"]
        hf   = flag(home)
        af   = flag(away)
        res  = results_map.get(mid)

        if res is not None:
            hs, as_ = int(res["home_score"]), int(res["away_score"])
            score_html = f'<span class="score-played">{hs} – {as_}</span>'
            date_str   = ""
        else:
            score_html = '<span class="score-future">· · ·</span>'
            dt = m.get("date_local")
            date_str = dt.strftime("%d/%m %Hh%M") if hasattr(dt, "strftime") else ""

        html += f"""
<div class="match-row">
  <span class="match-date">{date_str}</span>
  <span class="match-team match-team-home">{hf} {home}</span>
  <span class="match-score-box">{score_html}</span>
  <span class="match-team">{af} {away}</span>
</div>"""

    html += "</div></div></div>"
    st.markdown(html, unsafe_allow_html=True)


# ── Légende ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;gap:20px;margin-bottom:16px;font-size:0.75rem;color:#7A8BA6;align-items:center">
  <span><span style="display:inline-block;width:12px;height:12px;background:#EEF4FF;border-radius:3px;border:1px solid #B8C8E8;margin-right:5px"></span>Zone qualification (en cours)</span>
  <span><span style="display:inline-block;width:12px;height:12px;background:#F0FBF0;border-radius:3px;border:1px solid #A5D6A7;margin-right:5px"></span>Qualifié</span>
  <span><span style="display:inline-block;width:12px;height:12px;background:#FFF5F5;border-radius:3px;border:1px solid #FFCDD2;margin-right:5px"></span>Éliminé</span>
</div>
""", unsafe_allow_html=True)

# ── Layout 3 colonnes ─────────────────────────────────────────────────────────
for row_idx in range(0, len(GROUPS), 3):
    cols = st.columns(3, gap="medium")
    for col_idx, grp in enumerate(GROUPS[row_idx: row_idx + 3]):
        with cols[col_idx]:
            render_group(grp)
