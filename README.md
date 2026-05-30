# worldcup-prono

Système personnel de prédiction des scores Mondial FIFA 2026 pour pronos Mon Petit Prono.

## Phases

- **Phase 0** : Ingestion data (historique, fixtures, squads, valeurs marchandes, stats club)
- **Phase 1** : Baseline Elo + Poisson sur résultats uniquement
- **Phase 2** : Frontend Streamlit
- **Phase 3** : Deploy VPS
- **Phase 4** : Dixon-Coles + mise à jour bayésienne pendant le tournoi

## Usage

```bash
# Installer les dépendances
python -m venv venv
venv\Scripts\pip install -r requirements.txt

# Tests
venv\Scripts\python -m pytest tests/ -v

# Rafraîchir les effectifs (quotidien jusqu'au 11 juin)
venv\Scripts\python scripts/refresh_squads.py
venv\Scripts\python scripts/refresh_market_value.py
```
