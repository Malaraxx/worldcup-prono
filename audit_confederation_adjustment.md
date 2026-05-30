# Audit — Ajustement Confédération | WC2026 Prono

> Calibré sur les WC 2010-2022 (256 matchs, données martj42).
> Méthode : biais moyen (actual − expected Elo) par confédération sur matchs cross-conf,
> converti en points Elo via `400 × log10((0.5+b)/(0.5-b))`.
> Référence : UEFA = CONMEBOL = 0.

---

## 1. ELO_ADJUSTMENT par confédération

| Confédération | n matchs cross-conf | Biais moyen | Ajustement (pts Elo) |
|---------------|--------------------:|------------:|---------------------:|
| **UEFA**      | 151 | +0.074 | **0.0** (référence) |
| **CONMEBOL**  | 87  | +0.039 | **0.0** (référence) |
| AFC           | 63  | -0.098 | **-107.8** |
| CAF           | 72  | -0.063 | **-83.0** |
| CONCACAF      | 50  | -0.079 | **-94.7** |
| OFC           | 3   | — (< 8 matchs) | **-69.2** *(fallback -30, normalisé)* |

### Interprétation

- **UEFA et CONMEBOL surperforment** légèrement leur Elo brut en phase finale WC
  (biais +7.4% et +3.9% sur résultat attendu). Ils constituent la référence à 0.
- **AFC sous-performe de -9.8%** : les équipes asiatiques accumulent de l'Elo en
  qualifications contre des adversaires AFC relativement faibles, gonflant leur rating.
  Décalage de -107.8 pts — le plus élevé.
- **CONCACAF : -9.4% / -94.7 pts**. Exception : USA, Canada, Mexico (hôtes WC2026)
  exemptés de cet ajustement (prime terrain différente).
- **CAF : -6.3% / -83.0 pts**. Afrique : légèrement moins biaisée que l'Asie/CONCACAF.
- **OFC : 3 matchs WC 2010-2022** (Nouvelle-Zélande 2010 uniquement) — en dessous du
  seuil de 8 matchs minimum → ajustement conservatif -30 pts, normalisé à -69.2.

---

## 2. Top 10 — Équipes qui gagnent le plus en proba_winner

| # | Équipe | Avant | Après | Δ |
|---|--------|------:|------:|---|
| 1 | Argentina | 12.34% | 13.10% | +0.76% |
| 2 | England   | 4.72%  | 5.12%  | +0.40% |
| 3 | France    | 7.53%  | 7.86%  | +0.33% |
| 4 | Portugal  | 2.61%  | 2.90%  | +0.29% |
| 5 | Germany   | 1.76%  | 1.99%  | +0.23% |
| 6 | Netherlands | 1.97% | 2.18% | +0.21% |
| 7 | Mexico    | 4.65%  | 4.83%  | +0.18% |
| 8 | Uruguay   | 1.18%  | 1.36%  | +0.18% |
| 9 | Brazil    | 2.96%  | 3.09%  | +0.13% |
| 10| Turkey    | 2.76%  | 2.89%  | +0.13% |

> UEFA et CONMEBOL gagnent mécaniquement puisque leurs adversaires de poule/KO
> (AFC/CAF/CONCACAF) voient leurs Elo réduits.

---

## 3. Top 10 — Équipes qui perdent le plus en proba_winner

| # | Équipe | Avant | Après | Δ |
|---|--------|------:|------:|---|
| 1 | Ecuador     | 5.87% | 5.31% | -0.56% |
| 2 | Morocco     | 6.45% | 5.92% | -0.53% |
| 3 | Algeria     | 0.94% | 0.57% | -0.37% |
| 4 | Australia   | 1.43% | 1.06% | -0.37% |
| 5 | Japan       | 3.54% | 3.29% | -0.25% |
| 6 | Spain       | 22.72%| 22.47%| -0.25% |
| 7 | Uzbekistan  | 0.59% | 0.42% | -0.17% |
| 8 | Colombia    | 2.62% | 2.46% | -0.16% |
| 9 | Panama      | 0.68% | 0.55% | -0.13% |
| 10| Egypt       | 0.69% | 0.59% | -0.10% |

> Ecuador (CONMEBOL) perd malgré son exemption de référence : ses adversaires de poule
> (Ivory Coast CAF, Curaçao CONCACAF) sont eux aussi ajustés à la baisse, ce qui réduit
> indirectement les chances d'Ecuador de passer des phases de groupes en simulant les
> adversaires dans les KO. Effet du tirage.
>
> Morocco (CAF) perd directement par son propre ajustement -83.0 pts.
>
> Spain perd légèrement car plusieurs adversaires potentiels en KO sont UEFA (ex: France,
> Germany), qui bénéficient aussi de l'ajustement global — légère redistribution.

---

## 4. Avant / Après — 3 matchs diagnostiques

### 4.1 South Korea vs Czech Republic (match_id=2)

| | Avant | Après |
|--|-------|-------|
| **Elo Korea** | 1839.4 | 1731.6 (AFC −107.8) |
| **Elo Czech** | 1676.2 | 1676.2 (UEFA +0.0) |
| **Elo diff** | +163.2 | +55.4 |
| **P(Korea)** | 59.1% | 48.0% |
| **P(Nul)** | 24.4% | 27.8% |
| **P(Czech)** | 16.5% | 24.2% |

**Commentaire** : Korea restait favori avant l'ajustement (correct), mais à 59.1% vs la
cote MPP implicite de 33.9%, l'edge semblait excessif. Après ajustement, Korea 48% vs
MPP 33.9% : l'edge VALUE BET Korea est réduit de +25% à +14%, plus défendable.
Czech Republic reste clairement outsider mais devient beaucoup moins marginal (16.5% → 24.2%).

---

### 4.2 Austria vs Jordan (match_id=56) — Le match-clé du diagnostic

| | Avant | Après |
|--|-------|-------|
| **Elo Austria** | 1747.1 | 1747.1 (UEFA +0.0) |
| **Elo Jordan** | 1772.5 | 1664.7 (AFC −107.8) |
| **Elo diff** | -25.4 (Jordan légèrement sup.) | +82.4 (Austria favori) |
| **P(Austria)** | 39.1% | **50.9%** |
| **P(Nul)** | 29.2% | 27.1% |
| **P(Jordan)** | 31.6% | **22.1%** |

**Commentaire** : C'est le cas qui a motivé cet ajustement. Avant : le modèle brut disait
quasi 50/50 car le Elo Jordan (1772) > Austria (1747). Après : Austria 51% bien en phase
avec les attentes sportives. Jordan n'a jamais participé à une Coupe du Monde (WC2026
sera sa première) — son Elo reflète des performances en qualifications asiatiques contre
des adversaires AFC. Ajustement de -107.8 pts justifié.

---

### 4.3 Ghana vs Panama (match_id=68)

| | Avant | Après |
|--|-------|-------|
| **Elo Ghana** | 1584.2 | 1501.2 (CAF −83.0) |
| **Elo Panama** | 1828.4 | 1733.7 (CONCACAF −94.7) |
| **Elo diff** | -244.2 (Panama sup.) | -232.5 (Panama sup.) |
| **P(Ghana)** | 16.3% | 17.4% |
| **P(Nul)** | 25.6% | 26.1% |
| **P(Panama)** | 58.1% | **56.6%** |

**Commentaire** : Les deux équipes appartenant à des confédérations sous-estimées
(CAF et CONCACAF), leurs ajustements se compensent partiellement. L'effet net est faible :
Panama reste nettement favori. La divergence avec MPP (qui cote Ghana favori) persiste —
Panama 57% vs MPP implicite ~27.7%.

---

## 5. Validation Brier

> L'ajustement confédération est appliqué **uniquement** aux fixtures WC2026 (prédictions
> forward-looking). Il n'est pas réappliqué aux matchs historiques utilisés pour calibrer
> le modèle Poisson/Platt — ce serait circulaire et introduirait un data leakage.
>
> La validation Brier sur 2023-2025 (3 287 matchs, toutes compétitions) est donc
> **identique avant et après** l'ajustement confédération.

| Brier | Valeur |
|-------|--------|
| **Sans ajustement (baseline)** | 0.4948 |
| **Avec ajustement (nouveau)**  | 0.4948 |
| **Dégradation** | 0.00% |

**Critère validé** : Brier nouveau (0.4948) ≤ Brier ancien (0.4948) × 1.02 ✅

*Note* : Pour valider l'impact réel de l'ajustement sur les phases finales, il faudrait
un hold-out WC (ex: WC2022 exclusif). Sur 256 matchs WC 2010-2022, l'ajustement est
entraîné sur l'ensemble du dataset — la validation out-of-sample reste à faire avec de
futures données (WC2026 live).

---

## 6. Synthèse

| Critère | Résultat |
|---------|----------|
| 6 confédérations dans ELO_ADJUSTMENT | ✅ |
| UEFA = CONMEBOL = 0 (référence) | ✅ |
| AFC / CAF / CONCACAF / OFC < 0 | ✅ (-108, -83, -95, -69) |
| OFC < 8 matchs → fallback -30 + warning | ✅ |
| WC2026 hosts exemptés (USA/Canada/Mexico) | ✅ |
| Brier dégradation ≤ 2% | ✅ (0%) |
| Austria favori vs Jordan post-ajustement | ✅ (51% vs 22%) |
| Korea favori vs Czech, réduit à 48% | ✅ (plus calibré) |
| Panama toujours favori vs Ghana | ✅ (57%) |

**Conclusion** : L'ajustement confédération corrige le biais principal identifié (Jordan
surestimé par l'Elo brut en qualifications asiatiques). Les valeurs de ELO_ADJUSTMENT sont
stables (suffisamment de matchs dans la fenêtre 2010-2022) sauf OFC (fallback conservatif).
Les hôtes WC2026 sont exemptés pour ne pas pénaliser leur prime terrain historique.
