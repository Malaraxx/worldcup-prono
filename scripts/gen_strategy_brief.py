"""Genere strategy_brief.md — recommandations MPP par match."""
import sys, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
logging.disable(logging.CRITICAL)

import pandas as pd

from src.strategy.optimal_pick import load_merged, recommend_bonus_x2, recommend_winner_pick

PROCESSED = Path(__file__).parents[1] / "data" / "processed"

picks_df = pd.read_csv(PROCESSED / "optimal_picks.csv")
tp_df    = pd.read_csv(PROCESSED / "tournament_probabilities.csv")
merged   = load_merged()

lines = []
def h(t=""): lines.append(str(t))


def mpp_implied(cote_h, cote_d, cote_a):
    """Probabilites implicites MPP via 1/cote normalise."""
    s = 1 / cote_h + 1 / cote_d + 1 / cote_a
    return 1 / cote_h / s, 1 / cote_d / s, 1 / cote_a / s


def value_bet_note(ph, pd_, pa, ip_h, ip_d, ip_a, home, away):
    """Retourne une note explicite si le modele diverge de MPP >15% sur un resultat."""
    diffs = [
        (abs(ph - ip_h), "home", home,  ph,  ip_h),
        (abs(pd_ - ip_d), "draw", "Nul", pd_, ip_d),
        (abs(pa - ip_a), "away", away,  pa,  ip_a),
    ]
    best = max(diffs, key=lambda x: x[0])
    diff, side, label, model_p, mpp_p = best
    if diff <= 0.15:
        return ""
    if model_p > mpp_p:
        return (f"*Modele plus confiant sur **{label}** que MPP "
                f"({model_p:.1%} vs {mpp_p:.1%} implicite) — "
                f"MPP sous-estime cette issue.*")
    else:
        return (f"*MPP surestime **{label}** vs modele "
                f"({mpp_p:.1%} vs {model_p:.1%}) — "
                f"edge potentiel sur les autres issues.*")


h("# Strategy Brief — Mon Petit Prono | WC2026")
h()
h("> Cotes MPP natives (16 = faible = favori). Probas modele Elo+Poisson calibre (Platt).")
h("> EV = esperance de points si prono exact. WR = proba result (H/N/A) correct.")
h("> **VALUE BET** = ecart >15% entre proba modele et proba implicite MPP.")
h()
h("---")
h()

for _, pick in picks_df.iterrows():
    ch = int(pick["cote_home"])
    cd = int(pick["cote_draw"])
    ca = int(pick["cote_away"])
    ph = float(pick["p_home"])
    pd_ = float(pick["p_draw"])
    pa = float(pick["p_away"])

    ip_h, ip_d, ip_a = mpp_implied(ch, cd, ca)
    max_diff = max(abs(ph - ip_h), abs(pd_ - ip_d), abs(pa - ip_a))
    vb_flag = " — **VALUE BET ⚠️**" if max_diff > 0.15 else ""

    mode = pick["mode_recommended"]
    if mode == "lottery":
        rec_score = pick["lottery_score"]
        rec_ev    = pick["lottery_ev"]
        rec_wr    = pick["lottery_wr"]
    elif mode == "value":
        rec_score = pick["value_score"]
        rec_ev    = pick["value_ev"]
        rec_wr    = pick["value_wr"]
    else:
        rec_score = pick["safe_score"]
        rec_ev    = pick["safe_ev"]
        rec_wr    = pick["safe_wr"]

    h(f"## {pick['home']} vs {pick['away']}{vb_flag}")
    h()
    h(f"**Cotes MPP :** {pick['home']} {ch} — Nul {cd} — {pick['away']} {ca}")
    h()
    h(f"**Probas modele :** {ph:.1%} / {pd_:.1%} / {pa:.1%} "
      f"*(implicite MPP : {ip_h:.1%} / {ip_d:.1%} / {ip_a:.1%})*")
    h()
    h("| Mode | Score | EV | WR |")
    h("|------|-------|----|----|")
    h(f"| SAFE    | {pick['safe_score']}    | {pick['safe_ev']:.2f}  | {pick['safe_wr']:.1%}  |")
    h(f"| VALUE   | {pick['value_score']}   | {pick['value_ev']:.2f}  | {pick['value_wr']:.1%}  |")
    h(f"| LOTTERY | {pick['lottery_score']} | {pick['lottery_ev']:.2f} | {pick['lottery_wr']:.1%} |")
    h()
    h(f"**→ {mode.upper()} — Prono : `{rec_score}` | EV : {rec_ev:.2f} | WR : {rec_wr:.1%}**")
    h()

    # Justification
    edge = pick["edge_value_vs_safe_pct"]
    if mode == "lottery":
        ratio = pick["lottery_ev"] / pick["value_ev"] if pick["value_ev"] > 0 else 0
        h(f"*Justification : EV lottery ({pick['lottery_ev']:.1f}) = {ratio:.1f}x EV value "
          f"({pick['value_ev']:.1f}) — score rare a forte prime de cote MPP.*")
    elif mode == "safe":
        h(f"*Justification : Edge value vs safe = {edge:.1f}% < 10% — "
          f"modele d'accord avec MPP, score le plus probable suffit.*")
    else:
        h(f"*Justification : Edge value vs safe = {edge:.1f}% — "
          f"score alternatif capture la prime de rarete MPP (EV +{edge:.0f}% vs SAFE).*")

    note = value_bet_note(ph, pd_, pa, ip_h, ip_d, ip_a, pick["home"], pick["away"])
    if note:
        h()
        h(note)

    h()
    h("---")
    h()

# ── Bonus x2 ─────────────────────────────────────────────────────────────────
h("## Bonus x2 — Top 3 matchs recommandes")
h()
h("*Critere : value_wr >= 40%, trie par value_ev decroissant.*")
h()
bonus = recommend_bonus_x2(picks_df)
h("| # | Match | Prono | EV | WR |")
h("|---|-------|-------|----|----|")
for rank, (_, row) in enumerate(bonus.head(3).iterrows(), 1):
    h(f"| {rank} | **{row['home']} vs {row['away']}** "
      f"| {row['value_score']} | {row['value_ev']:.2f} | {row['value_wr']:.1%} |")
h()
h("*Le bonus x2 double les points si le prono est correct — privilegier fort EV et WR >= 40%.*")
h()

# ── Vainqueur ─────────────────────────────────────────────────────────────────
h("---")
h()
h("## Equipe vainqueur — Top 5 candidats Monte-Carlo")
h()
h("> Cotes MPP vainqueur non disponibles. A recalculer quand cotes MPP dispo.")
h()
winner = recommend_winner_pick(tp_df)
h("| Rang | Equipe | P(Vainqueur) | P(R32) |")
h("|------|--------|-------------|--------|")
for rank, (_, row) in enumerate(winner.head(5).iterrows(), 1):
    h(f"| {rank} | **{row['team']}** | {row['proba_winner']:.1%} | {row['proba_r32']:.1%} |")
h()
h("*Simulation Monte-Carlo 10k iterations, modele Elo+Poisson, tirage WC2026 officiel.*")

out = Path(__file__).parents[1] / "strategy_brief.md"
out.write_text("\n".join(lines), encoding="utf-8")
print(f"OK strategy_brief.md ecrit ({len(lines)} lignes)")
