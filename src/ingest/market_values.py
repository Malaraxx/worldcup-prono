"""
Enrichit players.csv avec les valeurs marchandes depuis transfermarkt-api.fly.dev.
Stratégie : recherche par nom joueur → matching par nationalité → marketValue.
Cache 24h partagé avec http_utils (data/cache/).
"""

import json
import logging
import time
import unicodedata
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

from .http_utils import _load_cache, _save_cache

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"
TM_API_BASE   = "https://transfermarkt-api.fly.dev"
TM_API_DELAY  = 0.5  # secondes entre appels (API publique, pas Cloudflare direct)

# Noms d'équipes projet → nationalité Transfermarkt (exceptions seulement)
TEAM_TO_TM_NAT: dict[str, str] = {
    "United States":         "United States",
    "DR Congo":              "DR Congo",
    "Ivory Coast":           "Ivory Coast",
    "South Korea":           "South Korea",
    "Bosnia and Herzegovina":"Bosnia-Herzegovina",
    "Czech Republic":        "Czech Republic",
    "Curacao":               "Curaçao",
}

_http_session = requests.Session()
_http_session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; WCPronoBot/1.0)"})


def _strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


def _search_player(name: str) -> list[dict]:
    url = f"{TM_API_BASE}/players/search/{quote(name)}"
    cached = _load_cache(url)
    if cached is not None:
        try:
            return json.loads(cached).get("results", [])
        except Exception:
            pass

    time.sleep(TM_API_DELAY)
    try:
        resp = _http_session.get(url, timeout=15)
        if resp.status_code == 200:
            _save_cache(url, resp.text)
            return resp.json().get("results", [])
        logger.debug("TM API %s for player search: %s", resp.status_code, name)
    except Exception as exc:
        logger.warning("TM search error for %s: %s", name, exc)
    return []


def _match_value(results: list[dict], tm_nationality: str) -> int | None:
    """Retourne marketValue du résultat dont la nationalité correspond, ou None."""
    for r in results:
        if tm_nationality in r.get("nationalities", []):
            return r.get("marketValue")
    # Fallback : résultat unique, on le prend sans vérifier la nationalité
    if len(results) == 1:
        return results[0].get("marketValue")
    return None


def enrich_market_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute/met à jour la colonne market_value_eur dans df.
    Scrape uniquement les joueurs sans valeur déjà présente.
    """
    df = df.copy()
    if "market_value_eur" not in df.columns:
        df["market_value_eur"] = None

    to_enrich = df[df["market_value_eur"].isna()].index
    total = len(to_enrich)
    found = 0

    for i, idx in enumerate(to_enrich, 1):
        row  = df.loc[idx]
        name = row["name"]
        team = row["team"]
        tm_nat = TEAM_TO_TM_NAT.get(team, team)

        results = _search_player(name)
        if not results:
            stripped = _strip_accents(name)
            if stripped != name:
                results = _search_player(stripped)

        value = _match_value(results, tm_nat)
        if value is not None:
            df.at[idx, "market_value_eur"] = value
            found += 1

        if i % 50 == 0 or i == total:
            logger.info("Progression : %d/%d (%.0f%% couverts)", i, total,
                        found / i * 100)

    coverage = found / total if total > 0 else 0
    logger.info("Valeurs marchandes : %d/%d joueurs (%.1f%%)", found, total,
                coverage * 100)
    return df


def save_market_values(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Charge players.csv, enrichit avec valeurs TM, sauvegarde."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if df is None:
        path = PROCESSED_DIR / "players.csv"
        if not path.exists():
            raise FileNotFoundError(f"players.csv non trouvé : {path}")
        df = pd.read_csv(path)

    df = enrich_market_values(df)
    out = PROCESSED_DIR / "players.csv"
    df.to_csv(out, index=False)
    logger.info("Sauvegardé : %s (%d lignes)", out, len(df))
    return df
