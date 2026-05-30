"""
Ingestion historique matchs internationaux depuis martj42/international_results.
Sources :
  - results.csv    : matchs depuis 1872
  - shootouts.csv  : tirs au but
  - goalscorers.csv: buteurs
"""

import logging
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"

MARTJ42_BASE = "https://raw.githubusercontent.com/martj42/international_results/master"
FILES = {
    "results": "results.csv",
    "shootouts": "shootouts.csv",
    "goalscorers": "goalscorers.csv",
}

TOURNAMENT_WEIGHTS: dict[str, float] = {
    "Friendly": 1.0,
    "FIFA World Cup": 4.0,
    "UEFA Euro": 3.5,
    "Copa América": 3.5,
    "African Cup of Nations": 3.5,
    "AFC Asian Cup": 3.5,
    "Gold Cup": 3.5,
    "UEFA Nations League": 3.0,
    "Confederations Cup": 3.0,
}
QUALIF_WEIGHT = 2.5
DEFAULT_WEIGHT = 1.5


def _download_raw(name: str) -> Path:
    filename = FILES[name]
    dest = RAW_DIR / filename
    if dest.exists():
        logger.info("Already downloaded: %s", filename)
        return dest
    url = f"{MARTJ42_BASE}/{filename}"
    logger.info("Downloading %s …", url)
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
        logger.info("Saved %s (%d bytes)", dest, len(resp.content))
    except Exception as exc:
        logger.warning("Download failed for %s: %s — utiliser le cache si dispo", filename, exc)
        # Ne pas crasher : si le fichier existe partiellement ou pas du tout,
        # les appelants recevront une FileNotFoundError à la lecture, qu'ils gèrent.
        raise FileNotFoundError(f"Impossible de télécharger {filename}: {exc}") from exc
    return dest


def load_matches(min_date: str = "2010-01-01") -> pd.DataFrame:
    """
    Charge results.csv depuis martj42, filtre à partir de min_date,
    normalise les types, assigne les poids.
    Retourne DataFrame avec colonnes :
      date, home_team, away_team, home_score, away_score,
      tournament, city, country, neutral, match_weight
    """
    path = _download_raw("results")
    df = pd.read_csv(path, parse_dates=["date"])
    total_before = len(df)
    logger.info("Matchs chargés (tout historique) : %d", total_before)

    df = df[df["date"] >= min_date].copy()
    logger.info("Matchs après filtre %s : %d", min_date, len(df))

    df["neutral"] = df["neutral"].astype(str).str.upper().map({"TRUE": True, "FALSE": False}).fillna(False)
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")

    df = assign_match_weight(df)

    cat_counts = df.groupby("tournament_category").size().sort_values(ascending=False)
    logger.info("Matchs par catégorie :\n%s", cat_counts.to_string())

    unique_teams = pd.concat([df["home_team"], df["away_team"]]).nunique()
    logger.info("Équipes uniques : %d", unique_teams)

    return df


def normalize_team_names(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Applique le mapping noms d'équipes (martj42 → convention projet)."""
    if not mapping:
        return df
    df = df.copy()
    df["home_team"] = df["home_team"].replace(mapping)
    df["away_team"] = df["away_team"].replace(mapping)
    return df


def assign_match_weight(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute match_weight et tournament_category selon le type de compétition.
    Hiérarchie : Friendly(1.0) < Default(1.5) < Qualif(2.5) < Confed/NL(3.0) < Continental(3.5) < WC(4.0)
    """
    df = df.copy()

    def _weight_and_cat(tournament: str) -> tuple[float, str]:
        if not isinstance(tournament, str):
            return DEFAULT_WEIGHT, "other"
        t = tournament.strip()
        if t == "Friendly":
            return TOURNAMENT_WEIGHTS["Friendly"], "friendly"
        if "qualification" in t.lower() or "qualifier" in t.lower():
            return QUALIF_WEIGHT, "qualification"
        if t == "FIFA World Cup":
            return TOURNAMENT_WEIGHTS["FIFA World Cup"], "world_cup"
        if t in ("UEFA Euro", "Copa América", "African Cup of Nations",
                 "AFC Asian Cup", "Gold Cup"):
            return TOURNAMENT_WEIGHTS[t], "continental"
        if t in ("UEFA Nations League", "Confederations Cup"):
            return TOURNAMENT_WEIGHTS[t], "nations_league"
        return DEFAULT_WEIGHT, "other"

    weights_cats = df["tournament"].apply(_weight_and_cat)
    df["match_weight"] = weights_cats.apply(lambda x: x[0])
    df["tournament_category"] = weights_cats.apply(lambda x: x[1])
    return df


def save_all() -> dict[str, pd.DataFrame]:
    """
    Télécharge les 3 CSVs martj42, filtre depuis 2010, sauvegarde dans data/processed/.
    Retourne dict {name: DataFrame}.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    # results.csv — filtré + enrichi
    matches = load_matches(min_date="2010-01-01")
    out = PROCESSED_DIR / "matches_historical.csv"
    matches.to_csv(out, index=False)
    logger.info("Sauvegardé : %s (%d lignes)", out, len(matches))
    results["matches"] = matches

    # shootouts.csv — brut
    path_s = _download_raw("shootouts")
    shootouts = pd.read_csv(path_s, parse_dates=["date"])
    out_s = PROCESSED_DIR / "shootouts.csv"
    shootouts.to_csv(out_s, index=False)
    logger.info("Sauvegardé : %s (%d lignes)", out_s, len(shootouts))
    results["shootouts"] = shootouts

    # goalscorers.csv — brut
    path_g = _download_raw("goalscorers")
    goalscorers = pd.read_csv(path_g, parse_dates=["date"])
    out_g = PROCESSED_DIR / "goalscorers.csv"
    goalscorers.to_csv(out_g, index=False)
    logger.info("Sauvegardé : %s (%d lignes)", out_g, len(goalscorers))
    results["goalscorers"] = goalscorers

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    dfs = save_all()
    print("\n=== Échantillon matches_historical.csv (5 lignes) ===")
    print(dfs["matches"].head(5).to_string(index=False))
    print("\n=== Échantillon shootouts.csv (5 lignes) ===")
    print(dfs["shootouts"].head(5).to_string(index=False))
    print("\n=== Échantillon goalscorers.csv (5 lignes) ===")
    print(dfs["goalscorers"].head(5).to_string(index=False))
