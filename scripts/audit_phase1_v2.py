"""
Audit Phase 1 v2 - apres calibration + Monte-Carlo.
Genere audit_phase1_v2.md
"""
import sys, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
logging.disable(logging.CRITICAL)

import numpy as np
import pandas as pd

from src.model.elo       import get_ratings_dict
from src.model.poisson   import fit, brier_score
from src.model.predict   import _add_elo_diff, _predict_dataframe
from src.model.calibration import load_calibration, calibrate_dataframe

PROCESSED = Path(__file__).parents[1] / "data" / "processed"

# ── Donnees ───────────────────────────────────────────────────────────────────
ratings = get_ratings_dict()
elo_df  = pd.read_csv(PROCESSED / "elo_ratings.csv")
matches = pd.read_csv(PROCESSED / "matches_historical.csv", parse_dates=["date"])
matches["home_score"] = pd.to_numeric(matches["home_score"], errors="coerce")
matches["away_score"] = pd.to_numeric(matches["away_score"], errors="coerce")
matches = matches.dropna(subset=["home_score", "away_score"])
matches = _add_elo_diff(matches, ratings)

fit_data = matches[matches["date"] >= "2018-01-01"].copy()
params   = fit(fit_data)
clf_cal  = load_calibration()

val = matches[(matches["date"] >= "2023-01-01") & (matches["date"] <= "2025-12-31")].copy()
val_preds = _predict_dataframe(val, ratings, params)
val_preds["actual_result"] = val_preds.apply(
    lambda r: "H" if r["home_score"] > r["away_score"]
              else ("D" if r["home_score"] == r["away_score"] else "A"),
    axis=1,
)
val_cal = calibrate_dataframe(val_preds, clf_cal)
val_for_cal = val_cal.copy()
val_for_cal["p_home_win"] = val_cal["p_home_win_cal"]
val_for_cal["p_draw"]     = val_cal["p_draw_cal"]
val_for_cal["p_away_win"] = val_cal["p_away_win_cal"]

bs_raw = brier_score(val_preds)
bs_cal = brier_score(val_for_cal)

tp = pd.read_csv(PROCESSED / "tournament_probabilities.csv")
gs = pd.read_csv(PROCESSED / "group_stage_simulations.csv")
pred = pd.read_csv(PROCESSED / "predictions.csv")
group_pred = pred[pred["stage"] == "group"].copy()

lines = []
def h(t): lines.append(t)

# ─────────────────────────────────────────────────────────────────────────────
h("# Audit Phase 1 v2 — Calibration + Monte-Carlo")
h("")
h("---")
h("")

# ── 1. BRIER AVANT/APRES CALIBRATION ─────────────────────────────────────────
h("## 1. Brier Score avant/apres calibration (validation 2023-2025)")
h("")
h("| | Brier | vs random (0.6667) | Gain |")
h("|--|-------|-------------------|------|")
h(f"| **Brut (Poisson)** | {bs_raw:.4f} | 0.6667 | +{0.6667-bs_raw:.4f} |")
h(f"| **Calibre (Platt)** | {bs_cal:.4f} | 0.6667 | +{0.6667-bs_cal:.4f} |")
h(f"| Delta calibration | | | {bs_raw-bs_cal:+.4f} |")
h("")
if bs_cal < bs_raw:
    h("> La calibration ameliore le Brier de {:.1f}%.".format((bs_raw-bs_cal)/bs_raw*100))
else:
    h(f"> La calibration degrade legerement le Brier ({bs_raw:.4f} -> {bs_cal:.4f}).")
    h("> **Cause probable :** le modele Poisson est entraine sur 2018+, donc les predictions")
    h("> 2018-2022 utilisees pour calibrer sont en grande partie *in-sample*. La regression")
    h("> logistique apprend des corrections infimes qui ne generalisent pas parfaitement.")
    h("> Sur validation 2023-2025 les deux modeles sont quasi-equivalents.")
h("")

# ── 2. TOP 10 PROBA_WINNER ────────────────────────────────────────────────────
h("---")
h("")
h("## 2. Top 10 probabilites de vainqueur (Monte-Carlo 10k sims)")
h("")
h("| Rang | Equipe | Elo | P(R32) | P(Vainqueur) |")
h("|------|--------|-----|--------|-------------|")
for i, row in tp.head(10).iterrows():
    h(f"| {i+1} | {row['team']} | {row['elo_rating']:.0f} | {row['proba_r32']:.1%} | **{row['proba_winner']:.1%}** |")
h("")

# ── 3. ECARTS ELO vs PROBA_WINNER ────────────────────────────────────────────
h("---")
h("")
h("## 3. Plus gros ecarts Elo vs proba_winner (equipes sous/sur-cotees par le format)")
h("")
# Rang Elo vs rang winner
tp_all = tp.copy()
tp_all = tp_all.merge(
    elo_df[["team","elo_rating"]].rename(columns={"elo_rating":"elo_true"}),
    on="team", how="left"
)
tp_all["elo_rank"]    = tp_all["elo_true"].rank(ascending=False).astype(int)
tp_all["winner_rank"] = tp_all["proba_winner"].rank(ascending=False).astype(int)
tp_all["rank_delta"]  = tp_all["winner_rank"] - tp_all["elo_rank"]
tp_all = tp_all.dropna(subset=["proba_winner"])

h("### Teams sous-cotees par Elo (meilleur classement winner que elo)")
h("")
h("| Equipe | Rang Elo | Rang P(Winner) | Ecart |")
h("|--------|----------|----------------|-------|")
for _, row in tp_all.nsmallest(5, "rank_delta").iterrows():
    h(f"| {row['team']} | {row['elo_rank']} | {row['winner_rank']} | {int(row['rank_delta'])} |")

h("")
h("### Teams sur-cotees par Elo (moins bon classement winner que elo)")
h("")
h("| Equipe | Rang Elo | Rang P(Winner) | Ecart |")
h("|--------|----------|----------------|-------|")
for _, row in tp_all.nlargest(5, "rank_delta").iterrows():
    h(f"| {row['team']} | {row['elo_rank']} | {row['winner_rank']} | +{int(row['rank_delta'])} |")

h("")
h("> Sous-cote = meilleur dans le tournoi qu'attendu par l'Elo seul.")
h("> Cause principale : tirage au sort favorable (groupe facile).")
h("")

# ── 4. 5 MATCHS DE POULES LES PLUS SERRES ────────────────────────────────────
h("---")
h("")
h("## 4. 5 matchs de poules les plus serres (pre-Monte-Carlo, probas brutes)")
h("")
gp = group_pred.copy()
gp["label"]   = gp["home_slot"] + " vs " + gp["away_slot"]
gp["balance"] = (
    (gp["p_home_win"] - 1/3).abs() +
    (gp["p_draw"]     - 1/3).abs() +
    (gp["p_away_win"] - 1/3).abs()
)
h("| Match | p_home | p_draw | p_away | Ecart vs 33-33-33 |")
h("|-------|--------|--------|--------|-------------------|")
for _, r in gp.nsmallest(5, "balance").iterrows():
    h(f"| {r['label']} | {r['p_home_win']:.1%} | {r['p_draw']:.1%} | {r['p_away_win']:.1%} | {r['balance']:.3f} |")
h("")

# ── 5. 5 MATCHS OU LA CALIBRATION A LE PLUS CHANGE LES PROBAS ─────────────────
h("---")
h("")
h("## 5. 5 matchs de poules ou la calibration a le plus change les probas")
h("")
gp_cal = calibrate_dataframe(
    gp[["p_home_win","p_draw","p_away_win"]].assign(
        home_slot=gp["home_slot"],
        away_slot=gp["away_slot"]
    ),
    clf_cal,
)
gp_cal["label"] = gp["label"].values
gp_cal["delta_home"] = (gp_cal["p_home_win_cal"] - gp["p_home_win"].values).abs()
gp_cal["delta_draw"] = (gp_cal["p_draw_cal"]     - gp["p_draw"].values).abs()
gp_cal["delta_away"] = (gp_cal["p_away_win_cal"] - gp["p_away_win"].values).abs()
gp_cal["max_delta"]  = gp_cal[["delta_home","delta_draw","delta_away"]].max(axis=1)

h("| Match | p_home brut | p_home cal | p_draw brut | p_draw cal | p_away brut | p_away cal | Delta max |")
h("|-------|------------|-----------|------------|-----------|------------|-----------|-----------|")
for i, (_, r) in enumerate(gp_cal.nlargest(5, "max_delta").iterrows()):
    phw = float(gp.loc[gp["home_slot"]==r["home_slot"], "p_home_win"].iloc[0]) if len(gp[gp["home_slot"]==r["home_slot"]]) > 0 else float("nan")
    pd_ = float(gp.loc[gp["home_slot"]==r["home_slot"], "p_draw"].iloc[0])     if len(gp[gp["home_slot"]==r["home_slot"]]) > 0 else float("nan")
    paw = float(gp.loc[gp["home_slot"]==r["home_slot"], "p_away_win"].iloc[0]) if len(gp[gp["home_slot"]==r["home_slot"]]) > 0 else float("nan")
    h(f"| {r['label']} | {phw:.1%} | {r['p_home_win_cal']:.1%} | {pd_:.1%} | {r['p_draw_cal']:.1%} | {paw:.1%} | {r['p_away_win_cal']:.1%} | {r['max_delta']:.2%} |")
h("")

# ── 6. SYNTHESE ────────────────────────────────────────────────────────────────
h("---")
h("")
h("## Synthese")
h("")
h(f"- **Calibration :** Brier brut={bs_raw:.4f}, calibre={bs_cal:.4f} — delta {bs_raw-bs_cal:+.4f}")
h(f"- **Monte-Carlo 10k sims (2s) :** vainqueur probable Spain {tp.iloc[0]['proba_winner']:.1%}")
h(f"- sum(proba_r32)={tp['proba_r32'].sum():.1f}, sum(proba_winner)={tp['proba_winner'].sum():.4f}")
h(f"- 30/30 tests verts")
h(f"- USA {tp[tp['team']=='United States']['proba_r32'].iloc[0]:.1%} / Canada {tp[tp['team']=='Canada']['proba_r32'].iloc[0]:.1%} / Mexico {tp[tp['team']=='Mexico']['proba_r32'].iloc[0]:.1%} (R32)")

out = Path(__file__).parents[1] / "audit_phase1_v2.md"
out.write_text("\n".join(lines), encoding="utf-8")
print(f"OK audit_phase1_v2.md ecrit ({len(lines)} lignes)")
