"""
Wrapper API-Football v3 (api-sports.io).
Clé dans .env : API_FOOTBALL_KEY.
Usage : live scores + xG officiel pendant le tournoi (11 juin – 19 juillet 2026).
Avant le tournoi : retourne données vides sans crasher.
"""

import json
import logging
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

from .http_utils import _load_cache, _save_cache

logger = logging.getLogger(__name__)

load_dotenv()

BASE_URL = "https://v3.football.api-sports.io"
WC_LEAGUE_ID = 1
WC_SEASON    = 2026
RAW_DIR      = Path(__file__).parents[2] / "data" / "raw"


def _headers() -> dict:
    key = os.getenv("API_FOOTBALL_KEY", "")
    if not key:
        raise EnvironmentError("API_FOOTBALL_KEY manquante dans .env")
    return {
        "x-rapidapi-key":  key,
        "x-rapidapi-host": "v3.football.api-sports.io",
    }


def _get(endpoint: str, params: dict | None = None, use_cache: bool = True) -> dict:
    """GET API-Football, retourne le JSON ou {} en cas d'échec."""
    import urllib.parse
    url = f"{BASE_URL}/{endpoint}"
    if params:
        url += "?" + urllib.parse.urlencode(params)

    if use_cache:
        cached = _load_cache(url)
        if cached is not None:
            try:
                return json.loads(cached)
            except Exception:
                pass

    try:
        resp = requests.get(
            f"{BASE_URL}/{endpoint}",
            headers=_headers(),
            params=params,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        if use_cache:
            _save_cache(url, json.dumps(data))
        return data
    except EnvironmentError as exc:
        logger.warning("API-Football : %s", exc)
        return {}
    except Exception as exc:
        logger.warning("API-Football GET /%s : %s", endpoint, exc)
        return {}


def check_status() -> dict:
    """Vérifie que la clé API est valide. Retourne {} si pas de clé."""
    return _get("status", use_cache=False)


def get_fixtures(league: int = WC_LEAGUE_ID, season: int = WC_SEASON) -> list[dict]:
    """Retourne la liste des fixtures WC2026. Vide avant que l'API les publie."""
    data = _get("fixtures", {"league": league, "season": season})
    fixtures = data.get("response", [])
    logger.info("API-Football fixtures : %d matchs", len(fixtures))
    return fixtures


def get_fixture_xg(fixture_id: int) -> dict | None:
    """
    Retourne les stats xG pour un match donné.
    Format : {"home": xg_float, "away": xg_float} ou None si non dispo.
    """
    data = _get("fixtures/statistics", {"fixture": fixture_id, "type": "Expected Goals"})
    response = data.get("response", [])
    if len(response) < 2:
        return None
    try:
        home_xg = float(response[0]["statistics"][0]["value"] or 0)
        away_xg = float(response[1]["statistics"][0]["value"] or 0)
        return {"home": home_xg, "away": away_xg}
    except Exception:
        return None


def get_live_fixtures(league: int = WC_LEAGUE_ID) -> list[dict]:
    """Retourne les matchs en cours (live). À appeler pendant le tournoi."""
    data = _get("fixtures", {"live": "all", "league": league}, use_cache=False)
    return data.get("response", [])


def save_fixtures_raw() -> Path:
    """Sauvegarde les fixtures WC2026 brutes en JSON dans data/raw/."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    fixtures = get_fixtures()
    out = RAW_DIR / "api_football_fixtures.json"
    out.write_text(json.dumps(fixtures, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Sauvegardé : %s (%d fixtures)", out, len(fixtures))
    return out
