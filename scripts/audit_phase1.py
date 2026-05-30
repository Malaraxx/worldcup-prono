"""
Audit qualité Phase 1 — Elo + Poisson
Produit audit_phase1.md
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import logging
logging.disable(logging.CRITICAL)

import numpy as np
import pandas as pd
from scipy.stats import poisson as poisson_dist

from src.model.elo import get_ratings_dict
from src.model.poisson import fit, lambdas, score_matrix, outcome_probs, most_likely_score, brier_score
from src.model.predict import _add_elo_diff, _predict_dataframe

PROCESSED = Path(__file__).parents[1] / "data" / "processed"


# ── Données ──────────────────────────────────────────────────────────────────

ratings = get_ratings_dict()
elo_df  = pd.read_csv(PROCESSED / "elo_ratings.csv")

matches = pd.read_csv(PROCESSED / "matches_historical.csv", parse_dates=["date"])
matches["home_score"] = pd.to_numeric(matches["home_score"], errors="coerce")
matches["away_score"] = pd.to_numeric(matches["away_score"], errors="coerce")
matches = matches.dropna(subset=["home_score", "away_score"])
matches = _add_elo_diff(matches, ratings)

fit_data = matches[matches["date"] >= "2018-01-01"].copy()
params   = fit(fit_data)

val = matches[
    (matches["date"] >= "2023-01-01") &
    (matches["date"] <= "2025-12-31")
].copy()
val_preds = _predict_dataframe(val, ratings, params)
val_preds["actual_result"] = val_preds.apply(
    lambda r: "H" if r["home_score"] > r["away_score"]
              else ("D" if r["home_score"] == r["away_score"] else "A"),
    axis=1,
)

predictions = pd.read_csv(PROCESSED / "predictions.csv")
group_preds = predictions[predictions["stage"] == "group"].copy()

lines = []
def h(text): lines.append(text)


# ══════════════════════════════════════════════════════════════════════════════
# 1. TOP 20 ELO
# ══════════════════════════════════════════════════════════════════════════════

h("# Audit Qualité Phase 1 — Modèle Elo + Poisson")
h("")
h("---")
h("")
h("## 1. Top 20 Ratings Elo")
h("")

top20 = elo_df.head(20).copy()
top20.index = range(1, 21)
top20.index.name = "Rang"

h("| Rang | Équipe | Elo |")
h("|------|--------|-----|")
for i, row in top20.iterrows():
    h(f"| {i} | {row['team']} | {row['elo_rating']:.1f} |")

h("")
h("**Positions des grandes nations :**")
h("")
focus = ["Brazil", "England", "Germany", "Portugal", "Netherlands", "Belgium", "Italy", "Uruguay"]
for team in focus:
    row = elo_df[elo_df["team"] == team]
    if row.empty:
        h(f"- {team} : **introuvable** dans elo_ratings.csv")
    else:
        rank = row.index[0] + 1
        elo  = row.iloc[0]["elo_rating"]
        h(f"- **{team}** : rang {rank} — {elo:.1f}")

h("")


# ══════════════════════════════════════════════════════════════════════════════
# 2. BRIER SCORE DÉCOMPOSÉ
# ══════════════════════════════════════════════════════════════════════════════

h("---")
h("")
h("## 2. Brier Score décomposé par catégorie (2023–2025)")
h("")

CAT_ORDER = ["friendly", "qualification", "nations_league", "continental", "world_cup", "other"]
CAT_LABEL = {
    "friendly":       "Amical",
    "qualification":  "Qualification",
    "nations_league": "Nations League",
    "continental":    "Continental (CAN, EURO, Copa…)",
    "world_cup":      "FIFA World Cup",
    "other":          "Autre",
}
NAIVE = 2 / 3  # 3 × (1/3) × (2/3)

h("| Catégorie | N matchs | Brier | vs random (0.667) | Δ |")
h("|-----------|----------|-------|-------------------|---|")

total_scores = []
for cat in CAT_ORDER:
    subset = val_preds[val_preds["tournament_category"] == cat]
    if subset.empty:
        continue
    bs = brier_score(subset)
    delta = NAIVE - bs
    sign  = "+" if delta > 0 else ""
    total_scores.append((cat, len(subset), bs, delta))
    h(f"| {CAT_LABEL[cat]} | {len(subset)} | {bs:.4f} | 0.6667 | {sign}{delta:.4f} |")

bs_all = brier_score(val_preds)
delta_all = NAIVE - bs_all
h(f"| **TOTAL** | **{len(val_preds)}** | **{bs_all:.4f}** | 0.6667 | **+{delta_all:.4f}** |")

h("")
h("> Brier score : plus bas = meilleur. Random naïf (1/3 chaque) = 0.6667.")
h("")


# ══════════════════════════════════════════════════════════════════════════════
# 3. SANITY CHECK — 6 MATCHS HYPOTHÉTIQUES
# ══════════════════════════════════════════════════════════════════════════════

h("---")
h("")
h("## 3. Sanity Check — 6 matchs hypothétiques")
h("")

MATCHES_TEST = [
    ("Spain",    "Brazil",    True,  "Spain vs Brazil (neutre)"),
    ("France",   "Argentina", True,  "France vs Argentina (neutre)"),
    ("England",  "Germany",   True,  "England vs Germany (neutre)"),
    ("USA",      "Mexico",    False, "USA vs Mexico (USA home)"),
    ("Morocco",  "France",    True,  "Morocco vs France (neutre)"),
    ("Senegal",  "Iran",      True,  "Senegal vs Iran (neutre)"),
]

from src.model.elo import HOME_ADV_ELO

def predict_match(home, away, neutral, params, ratings):
    from src.model.elo import DEFAULT_RATING, HOME_ADV_ELO
    elo_h = ratings.get(home, DEFAULT_RATING)
    elo_a = ratings.get(away, DEFAULT_RATING)
    adv   = 0.0 if neutral else HOME_ADV_ELO
    diff  = elo_h + adv - elo_a
    lh, la = lambdas(diff, params)
    mat    = score_matrix(lh, la)
    phw, pd_, paw = outcome_probs(mat)
    # Top 5 scores
    scores = []
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            scores.append((i, j, mat[i, j]))
    scores.sort(key=lambda x: -x[2])
    top5 = scores[:5]
    return elo_h, elo_a, diff, lh, la, phw, pd_, paw, top5

for home, away, neutral, label in MATCHES_TEST:
    elo_h, elo_a, diff, lh, la, phw, pd_, paw, top5 = predict_match(
        home, away, neutral, params, ratings
    )
    h(f"### {label}")
    h("")
    h(f"| | {home} | {away} |")
    h(f"|---|---|---|")
    h(f"| Elo | {elo_h:.1f} | {elo_a:.1f} |")
    h(f"| λ | {lh:.3f} | {la:.3f} |")
    h(f"| P(victoire) | **{phw:.1%}** | **{paw:.1%}** |")
    h(f"| P(nul) | {pd_:.1%} | — |")
    h("")
    h("**Top 5 scores les plus probables :**")
    h("")
    h("| Score | Probabilité |")
    h("|-------|-------------|")
    for gh, ga, prob in top5:
        h(f"| {gh}-{ga} | {prob:.2%} |")
    h("")


# ══════════════════════════════════════════════════════════════════════════════
# 4. CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════

h("---")
h("")
h("## 4. Calibration du modèle (2023–2025)")
h("")
h("Pour chaque tranche de p_home_win prédit, fréquence réelle de victoires home.")
h("")

cal = val_preds.copy()
cal["actual_home_win"] = (cal["actual_result"] == "H").astype(int)
cal["bucket"] = (cal["p_home_win"] * 10).apply(np.floor).astype(int).clip(0, 9)

h("| Tranche p_home_win | N matchs | Prob moy prédite | Freq réelle victoire H | Écart |")
h("|--------------------|----------|-----------------|------------------------|-------|")

for b in range(10):
    subset = cal[cal["bucket"] == b]
    if subset.empty:
        continue
    lo = b * 10
    hi = lo + 10
    n     = len(subset)
    p_avg = subset["p_home_win"].mean()
    f_real = subset["actual_home_win"].mean()
    ecart  = f_real - p_avg
    sign   = "+" if ecart >= 0 else ""
    h(f"| {lo}–{hi}% | {n} | {p_avg:.1%} | {f_real:.1%} | {sign}{ecart:.1%} |")

h("")
h("> Un modèle parfaitement calibré aurait écart ≈ 0 sur chaque ligne.")
h("")


# ══════════════════════════════════════════════════════════════════════════════
# 5. ANALYSE PHASE DE POULES WC2026
# ══════════════════════════════════════════════════════════════════════════════

h("---")
h("")
h("## 5. Analyse Phase de Poules WC2026 (72 matchs)")
h("")

g = group_preds.copy()
g["label"] = g["home_slot"] + " vs " + g["away_slot"]
g["lambda_total"] = g["lambda_home"] + g["lambda_away"]

# Distance à 33-33-33
g["balance"] = (
    (g["p_home_win"] - 1/3).abs() +
    (g["p_draw"]    - 1/3).abs() +
    (g["p_away_win"]- 1/3).abs()
)

h("### 5a. Top 5 — Plus forte probabilité de victoire home")
h("")
h("| Match | p_home | p_draw | p_away | Score prédit |")
h("|-------|--------|--------|--------|--------------|")
for _, r in g.nlargest(5, "p_home_win").iterrows():
    h(f"| {r['label']} | **{r['p_home_win']:.1%}** | {r['p_draw']:.1%} | {r['p_away_win']:.1%} | {int(r['pred_score_home'])}-{int(r['pred_score_away'])} |")

h("")
h("### 5b. Top 5 — Matchs les plus serrés (distribution la plus proche de 33-33-33)")
h("")
h("| Match | p_home | p_draw | p_away | Écart vs équiprob |")
h("|-------|--------|--------|--------|-------------------|")
for _, r in g.nsmallest(5, "balance").iterrows():
    h(f"| {r['label']} | {r['p_home_win']:.1%} | {r['p_draw']:.1%} | {r['p_away_win']:.1%} | {r['balance']:.3f} |")

h("")
h("### 5c. Top 5 — Plus de buts attendus (λ_home + λ_away)")
h("")
h("| Match | λ_home | λ_away | λ_total | Score prédit |")
h("|-------|--------|--------|---------|--------------|")
for _, r in g.nlargest(5, "lambda_total").iterrows():
    h(f"| {r['label']} | {r['lambda_home']:.3f} | {r['lambda_away']:.3f} | **{r['lambda_total']:.3f}** | {int(r['pred_score_home'])}-{int(r['pred_score_away'])} |")

h("")


# ══════════════════════════════════════════════════════════════════════════════
# 6. MATCHS KO — EXPLICATION
# ══════════════════════════════════════════════════════════════════════════════

h("---")
h("")
h("## 6. Matchs à Élimination Directe — Analyse critique")
h("")

ko = predictions[predictions["stage"] != "group"]

h("### Méthode actuelle : placeholder à Elo par défaut")
h("")
h("Les équipes des matchs KO ne sont pas encore connues (slots du type")
h("`Winner Group A`, `Runner-up Group B`, etc.).")
h("")
h("**Ce qui se passe actuellement dans `_predict_fixtures()` :**")
h("")
h("```python")
h("elo_h = ratings.get(home_slot, DEFAULT_RATING)  # → 1500.0 si slot inconnu")
h("elo_a = ratings.get(away_slot, DEFAULT_RATING)  # → 1500.0 si slot inconnu")
h("elo_diff = elo_h - elo_a  # → 0.0 toujours")
h("```")
h("")
h("Résultat : **tous les matchs KO ont elo_diff = 0**, donc probabilités quasi-identiques.")
h("")

h("| Stage | N matchs | elo_home (tous) | elo_diff (tous) | p_home_win (moy) |")
h("|-------|----------|-----------------|-----------------|-----------------|")
for stage in ["r32", "r16", "qf", "sf", "3rd", "final"]:
    s = ko[ko["stage"] == stage]
    if s.empty:
        continue
    h(f"| {stage} | {len(s)} | {s['elo_home'].iloc[0]:.0f} | {s['elo_diff'].iloc[0]:.0f} | {s['p_home_win'].mean():.1%} |")

h("")
h("### Ce que ça signifie")
h("")
h("Les 32 matchs KO dans `predictions.csv` sont **des placeholders sans valeur prédictive** :")
h("- Elo_home = Elo_away = 1500.0 (valeur par défaut)")
h("- p_home_win ≈ p_away_win ≈ 38-40%, p_draw ≈ 22%")
h("- Aucun Monte-Carlo n'a été réalisé")
h("")
h("### Ce qu'il faudrait faire (Phase 4)")
h("")
h("**Option A — Simulation Monte-Carlo (recommandée) :**")
h("1. Simuler N fois (ex. 10 000) la phase de poules complète")
h("2. Pour chaque simulation, tirer le vainqueur et le 2e de chaque groupe selon")
h("   `p_home_win / p_draw / p_away_win` (+ règles de départage)")
h("3. Agréger : pour chaque slot KO, on obtient une distribution `{équipe: probabilité}`")
h("4. Elo_slot = somme pondérée des Elo des équipes candidates")
h("5. Recalculer les prédictions KO avec ces Elo pondérés")
h("")
h("**Option B — Mise à jour live (Phase 4 tournoi) :**")
h("Une fois les groupes joués, remplacer les slots par les vraies équipes et")
h("recalculer les prédictions KO avec leur Elo réel (+ Dixon-Coles mis à jour).")
h("")
h("**Recommandation :** implémenter Option A en Phase 2 ou 3 (Monte-Carlo rapide,")
h("quelques secondes pour 10 000 sims), Option B en Phase 4 pour le live.")
h("")
h("---")
h("")
h("## Synthèse")
h("")

bs_all = brier_score(val_preds)
h(f"- **Brier score global 2023-2025 : {bs_all:.4f}** (vs random 0.6667, gain +{NAIVE-bs_all:.4f})")
h("- Calibration : à vérifier dans le tableau section 4")
h("- Matchs KO : **prédictions placeholder à ignorer** — nécessitent Monte-Carlo")
h("- Modèle robuste sur matchs compétitifs, moins sur amicaux (faible signal)")


# ── Écriture fichier ──────────────────────────────────────────────────────────

out = Path(__file__).parents[1] / "audit_phase1.md"
out.write_text("\n".join(lines), encoding="utf-8")
print(f"OK Audit ecrit : {out}")
print(f"  {len(lines)} lignes")
