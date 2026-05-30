"""
Calcul des ratings Elo pour les équipes WC2026.
Source : matches_historical.csv (2010-2026, pondéré par match_weight).
"""

import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROCESSED_DIR  = Path(__file__).parents[2] / "data" / "processed"
DEFAULT_RATING = 1500.0
K_BASE         = 30.0   # K de base, multiplié par match_weight
HOME_ADV_ELO   = 60.0   # bonus Elo terrain (0 si match neutre)


def _expected(r_home: float, r_away: float, home_adv: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((r_away - r_home - home_adv) / 400.0))


def compute_elo(
    matches: pd.DataFrame,
    k_base: float = K_BASE,
    home_adv: float = HOME_ADV_ELO,
) -> dict[str, float]:
    """
    Parcourt les matchs chronologiquement et retourne {team: elo_final}.
    Utilise match_weight pour moduler K (WC = poids 4.0, amical = 1.0).
    Pas d'avantage terrain si neutral == True.
    """
    ratings: dict[str, float] = defaultdict(lambda: DEFAULT_RATING)
    matches_sorted = matches.sort_values("date")

    for _, row in matches_sorted.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        hs   = row["home_score"]
        as_  = row["away_score"]

        if pd.isna(hs) or pd.isna(as_):
            continue

        neutral = bool(row.get("neutral", False))
        adv     = 0.0 if neutral else home_adv
        k       = k_base * float(row.get("match_weight", 1.0))

        exp     = _expected(ratings[home], ratings[away], adv)
        result  = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
        delta   = k * (result - exp)

        ratings[home] += delta
        ratings[away] -= delta

    return dict(ratings)


def load_elo_ratings(recalculate: bool = False) -> pd.DataFrame:
    """
    Charge (ou recalcule) les ratings Elo depuis elo_ratings.csv.
    Retourne DataFrame : team, elo_rating.
    """
    out = PROCESSED_DIR / "elo_ratings.csv"

    if out.exists() and not recalculate:
        return pd.read_csv(out)

    matches = pd.read_csv(PROCESSED_DIR / "matches_historical.csv", parse_dates=["date"])
    ratings = compute_elo(matches)

    df = pd.DataFrame(
        sorted(ratings.items(), key=lambda x: -x[1]),
        columns=["team", "elo_rating"],
    )
    df["elo_rating"] = df["elo_rating"].round(1)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info("Elo calculé : %d équipes, sauvegardé %s", len(df), out)
    return df


def get_ratings_dict(recalculate: bool = False) -> dict[str, float]:
    df = load_elo_ratings(recalculate)
    return dict(zip(df["team"], df["elo_rating"]))
