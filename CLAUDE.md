# worldcup-prono

## Objectif

Système **personnel** de prédiction des scores Mondial FIFA 2026 (48 équipes, 104 matchs, 11 juin–19 juillet 2026) pour aider Flo à pronostiquer sur **Mon Petit Prono**. App mono-utilisateur, pas de multi-user.

## Stack

Python 3.11+, pandas, scipy, BeautifulSoup4, Streamlit, Plotly. Dépendances figées dans `requirements.txt`.

## Architecture data (couches d'ingestion)

```
martj42 (historique 2010+)
  → Wikipedia (équipes, groupes, fixtures, squads)
    → Transfermarkt (valeurs marchandes)
      → FBref Big 5 (stats club 2025-26)
        → Understat (xG/xA précis, override FBref)
          → API-Football (live scores + xG officiel pendant le tournoi)
```

## Phases

| Phase | Contenu | Critère de passage |
|-------|---------|-------------------|
| **Phase 0** | Ingestion uniquement | `pytest tests/ -v` 100% vert |
| **Phase 1** | Baseline Elo + Poisson sur **résultats uniquement** (pas joueurs) | Calibration Brier score |
| **Phase 1.5a** | Ajustement Market Value (alpha) — **ROLLBACK** : alpha_optimal=0 | ✗ Dégradait Brier |
| **Phase 1.5b** | Dixon-Coles correction (rho) — **ROLLBACK** : rho=-0.019, Δbrier=+0.0002 | ✗ Insuffisant |
| **Phase 2** | Frontend Streamlit — distribution probabiliste des scores | App interactive |
| **Phase 3** | Deploy VPS OVH | Disponible H24 |
| **Phase 4** | Mise à jour bayésienne live pendant le tournoi (scores réels → recalibration) | WC live |

**Règle absolue** : ne pas anticiper les phases. Phase 0 = ingestion, rien d'autre.

## Noms d'équipes

Convention : **noms FIFA officiels 2026** partout. Le mapping martj42 → FIFA est dans `src/ingest/mappings.py::TEAM_NAME_MAP`. Valider ce mapping avec Flo avant tout commit.

## Scraping

- User-Agent : `Mozilla/5.0 (compatible; WCPronoBot/1.0)`
- Cache HTML 24h dans `data/cache/{domain}/{sha256(url)}.html`
- Rate limit : 1 req / 2 sec minimum (Transfermarkt : 2 sec strict)
- Retry : 3x avec backoff exponentiel
- En cas d'échec persistant : log warning + NaN + `continue` (jamais crash)

## API-Football

- Clé dans `.env` : `API_FOOTBALL_KEY=`
- **JAMAIS commit** la clé (`.env` dans `.gitignore`)
- league_id = 1 (FIFA World Cup), season = 2026
- Tester avec `scripts/test_api_football.py` avant usage production
- Valider la disponibilité des xG avec Flo avant d'intégrer

## Flashscore (module dormant)

`src/ingest/flashscore.py` — fallback **dormant**. À activer **manuellement** uniquement si API-Football tombe pendant le tournoi. Ne PAS intégrer dans la pipeline normale sans décision explicite.

## Refresh quotidien (jusqu'au 11 juin 2026)

Les listes définitives sortent progressivement — relancer chaque jour :

```bash
venv\Scripts\python scripts/refresh_squads.py
venv\Scripts\python scripts/refresh_market_value.py
```

## Commandes

```bash
# Tests
venv\Scripts\python -m pytest tests/ -v

# App Streamlit Phase 2 (http://localhost:8501)
venv\Scripts\python scripts/run_app.py
# ou directement :
venv\Scripts\streamlit run src/app/main.py --server.port 8501

# Pipeline prédictions (re-run si données changent)
venv\Scripts\python scripts/run_predictions.py
venv\Scripts\python src/strategy/optimal_pick.py
venv\Scripts\python scripts/gen_strategy_brief.py
```

## Architecture Streamlit (Phase 2)

```
src/app/
├── main.py                   # entry point : st.navigation 4 pages
├── utils.py                  # load_data(), get_match(), helpers
├── pages/
│   ├── 1_dashboard.py        # métriques, prochains matchs, top vainqueur
│   ├── 2_calendrier.py       # filtres + tableau dense toutes fixtures
│   ├── 3_detail_match.py     # heatmap 7x7, picks, modèle vs MPP
│   └── 4_bracket.py          # standings + bracket KO + pick vainqueur
└── components/
    ├── score_heatmap.py      # plotly heatmap P(i,j)
    ├── probas_bar.py         # barres comparatives modèle vs MPP
    └── pick_card.py          # HTML card safe/value/lottery
```

## État du modèle (02/06/2026)

**Pipeline actif** : Elo brut → conf_adj → Poisson → Platt scaling  
**Modèle** : Elo + Poisson indépendant + Platt scaling (calibration multinomiale)  
**Brier val 2023-2025** : 0.4926 (raw) / 0.4948 (calibré Platt)

| Module | Fichier | Statut |
|--------|---------|--------|
| Elo ratings | `src/model/elo.py` | ✅ actif |
| Régression Poisson | `src/model/poisson.py` | ✅ actif |
| Platt scaling | `src/model/calibration.py` | ✅ actif — entraîné 2018-2022 |
| Conf. adjustment | `src/model/confederation_adjustment.py` | ✅ actif — WC fixtures uniquement |
| Market Value (α) | `src/model/elo_mv_adjustment.py` | ✗ rollback — alpha_optimal=0 |
| Dixon-Coles (ρ) | `src/model/dixon_coles.py` | ✗ rollback — rho=-0.019, Δbrier=+0.0002 |

**Règle** : ne jamais modifier predict.py pour activer DC ou MV sans Δbrier confirmé ≥ 0.005.

## Ajustement confédération (Phase 1)

`src/model/confederation_adjustment.py` applique un décalage Elo UNIQUEMENT aux fixtures WC2026.
Calibré sur WC 2010-2022 : AFC=-108, CAF=-83, CONCACAF=-95, OFC=-69, UEFA=CONMEBOL=0.
Hôtes WC2026 (USA/Canada/Mexico) exemptés de l'ajustement CONCACAF.

## Workflow Claude Code

- **Stop à chaque étape** pour validation Flo, surtout sur :
  - Mapping noms équipes (TEAM_NAME_MAP)
  - Slugs + IDs Transfermarkt (TRANSFERMARKT_TEAMS)
  - Structure HTML à parser (Wikipedia, FBref, Understat)
- Modifications minimales et ciblées
- Commit Git après chaque étape validée : `feat(ingest): étape N — description`
- Afficher 5 lignes d'échantillon après chaque CSV produit
- En cas d'incertitude entre deux approches : expliquer les deux, laisser choisir

## Hors scope

- `venv/`, `__pycache__/`, `.pytest_cache/`
- `data/raw/`, `data/cache/`
- `.env`
- `*.pyc`, `.DS_Store`
