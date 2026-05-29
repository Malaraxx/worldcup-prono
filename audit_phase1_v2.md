# Audit Phase 1 v2 — Calibration + Monte-Carlo

---

## 1. Brier Score avant/apres calibration (validation 2023-2025)

| | Brier | vs random (0.6667) | Gain |
|--|-------|-------------------|------|
| **Brut (Poisson)** | 0.4935 | 0.6667 | +0.1732 |
| **Calibre (Platt)** | 0.4948 | 0.6667 | +0.1719 |
| Delta calibration | | | -0.0013 |

> La calibration degrade legerement le Brier (0.4935 -> 0.4948).
> **Cause probable :** le modele Poisson est entraine sur 2018+, donc les predictions
> 2018-2022 utilisees pour calibrer sont en grande partie *in-sample*. La regression
> logistique apprend des corrections infimes qui ne generalisent pas parfaitement.
> Sur validation 2023-2025 les deux modeles sont quasi-equivalents.

---

## 2. Top 10 probabilites de vainqueur (Monte-Carlo 10k sims)

| Rang | Equipe | Elo | P(R32) | P(Vainqueur) |
|------|--------|-----|--------|-------------|
| 1 | Spain | 2108 | 99.2% | **22.7%** |
| 2 | Argentina | 2044 | 95.8% | **12.3%** |
| 3 | France | 1998 | 88.7% | **7.5%** |
| 4 | Morocco | 1971 | 91.5% | **6.5%** |
| 5 | Ecuador | 1952 | 94.1% | **5.9%** |
| 6 | England | 1958 | 93.7% | **4.7%** |
| 7 | Mexico | 1936 | 94.9% | **4.7%** |
| 8 | Japan | 1931 | 90.6% | **3.5%** |
| 9 | Brazil | 1909 | 86.7% | **3.0%** |
| 10 | Turkey | 1916 | 80.8% | **2.8%** |

---

## 3. Plus gros ecarts Elo vs proba_winner (equipes sous/sur-cotees par le format)

### Teams sous-cotees par Elo (meilleur classement winner que elo)

| Equipe | Rang Elo | Rang P(Winner) | Ecart |
|--------|----------|----------------|-------|
| Brazil | 13 | 9 | -4 |
| Scotland | 35 | 31 | -4 |
| Switzerland | 25 | 22 | -3 |
| Canada | 27 | 24 | -3 |
| Qatar | 46 | 43 | -3 |

### Teams sur-cotees par Elo (moins bon classement winner que elo)

| Equipe | Rang Elo | Rang P(Winner) | Ecart |
|--------|----------|----------------|-------|
| DR Congo | 30 | 35 | +5 |
| Uzbekistan | 24 | 28 | +4 |
| Portugal | 9 | 12 | +3 |
| Sweden | 37 | 40 | +3 |
| South Africa | 44 | 47 | +3 |

> Sous-cote = meilleur dans le tournoi qu'attendu par l'Elo seul.
> Cause principale : tirage au sort favorable (groupe facile).

---

## 4. 5 matchs de poules les plus serres (pre-Monte-Carlo, probas brutes)

| Match | p_home | p_draw | p_away | Ecart vs 33-33-33 |
|-------|--------|--------|--------|-------------------|
| Austria vs Jordan | 36.6% | 28.0% | 35.4% | 0.106 |
| Panama vs Croatia | 36.0% | 28.0% | 36.0% | 0.106 |
| Cape Verde vs Saudi Arabia | 36.7% | 28.0% | 35.3% | 0.107 |
| Norway vs Senegal | 36.8% | 28.0% | 35.2% | 0.107 |
| Australia vs Turkey | 34.9% | 28.0% | 37.1% | 0.107 |

---

## 5. 5 matchs de poules ou la calibration a le plus change les probas

| Match | p_home brut | p_home cal | p_draw brut | p_draw cal | p_away brut | p_away cal | Delta max |
|-------|------------|-----------|------------|-----------|------------|-----------|-----------|
| United States vs Australia | 31.3% | 33.0% | 27.8% | 29.4% | 41.0% | 37.6% | 4.01% |
| Haiti vs Scotland | 30.3% | 32.5% | 27.6% | 29.4% | 42.1% | 38.1% | 3.99% |
| New Zealand vs Belgium | 24.6% | 32.8% | 26.5% | 29.4% | 48.9% | 37.8% | 3.99% |
| Norway vs France | 36.8% | 29.6% | 28.0% | 29.2% | 35.2% | 41.1% | 3.99% |
| Uzbekistan vs Colombia | 31.4% | 33.7% | 27.8% | 29.4% | 40.8% | 36.8% | 3.99% |

---

## Synthese

- **Calibration :** Brier brut=0.4935, calibre=0.4948 — delta -0.0013
- **Monte-Carlo 10k sims (2s) :** vainqueur probable Spain 22.7%
- sum(proba_r32)=32.0, sum(proba_winner)=1.0000
- 30/30 tests verts
- USA 53.5% / Canada 89.2% / Mexico 94.9% (R32)