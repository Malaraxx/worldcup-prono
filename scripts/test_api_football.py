"""
Script de validation API-Football.
À lancer MANUELLEMENT une fois la clé API disponible dans .env.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from dotenv import load_dotenv
import os
import requests

load_dotenv()
API_KEY = os.getenv("API_FOOTBALL_KEY")

BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {
    "x-rapidapi-key": API_KEY or "",
    "x-rapidapi-host": "v3.football.api-sports.io",
}


def check_status():
    resp = requests.get(f"{BASE_URL}/status", headers=HEADERS, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    print("Status API:", data.get("response", {}).get("account", {}).get("firstname", "OK"))
    return data


def get_fixtures():
    resp = requests.get(
        f"{BASE_URL}/fixtures",
        headers=HEADERS,
        params={"league": 1, "season": 2026},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json().get("response", [])


def main():
    if not API_KEY:
        print("API_FOOTBALL_KEY manquante dans .env")
        sys.exit(1)

    print("1. Vérification clé API...")
    check_status()

    print("2. Récupération fixtures Mondial 2026...")
    fixtures = get_fixtures()
    print(f"   {len(fixtures)} fixtures trouvées")

    out = Path("data/raw/api_football_fixtures.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fixtures, indent=2, ensure_ascii=False))
    print(f"   Sauvegardé : {out}")

    if fixtures:
        sample = fixtures[:2]
        print("\nÉchantillon :")
        for f in sample:
            print(f"  {f.get('fixture', {}).get('date')} — "
                  f"{f.get('teams', {}).get('home', {}).get('name')} vs "
                  f"{f.get('teams', {}).get('away', {}).get('name')}")


if __name__ == "__main__":
    main()
