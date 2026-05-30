"""
xG/xGA clubs Big 5 depuis Understat via soccerdata.
Agrège home_xg/away_xg par équipe sur la saison 2025-26.
Override des stats xG dans club_stats.csv (Understat est plus précis que FBref).
"""

import logging
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

PROCESSED_DIR       = Path(__file__).parents[2] / "data" / "processed"
UNDERSTAT_LEAGUES   = [
    "ENG-Premier League",
    "ESP-La Liga",
    "GER-Bundesliga",
    "ITA-Serie A",
    "FRA-Ligue 1",
]
SEASONS_TO_TRY = ("2526", "2425")

# Noms clubs Understat → noms clubs FBref (exceptions seulement)
CLUB_NAME_US_TO_FB: dict[str, str] = {
    # Premier League
    "Manchester United":       "Manchester Utd",
    "Tottenham":               "Tottenham Hotspur",
    "West Ham":                "West Ham United",
    "Wolverhampton Wanderers": "Wolves",
    "Leeds":                   "Leeds United",
    # La Liga
    "Alaves":                  "Alavés",
    "Atletico Madrid":         "Atlético Madrid",
    "Real Oviedo":             "Oviedo",
    # Bundesliga
    "Borussia Dortmund":       "Dortmund",
    "RasenBallsport Leipzig":  "RB Leipzig",
    "Bayer Leverkusen":        "Leverkusen",
    "VfB Stuttgart":           "Stuttgart",
    "FC Cologne":              "Köln",
    "FC Heidenheim":           "Heidenheim",
    "Borussia M.Gladbach":     "Gladbach",
    "St. Pauli":               "St Pauli",
    # Serie A
    "AC Milan":                "Milan",
    "Parma Calcio 1913":       "Parma",
    "Verona":                  "Hellas Verona",
    # Ligue 1
    "Paris Saint Germain":     "Paris Saint-Germain",
}


def _agg_xg_for_league(us, league_label: str) -> pd.DataFrame:
    """Agrège xG/xGA par équipe depuis le schedule Understat d'une ligue."""
    schedule = us.read_schedule()
    finished = schedule[schedule["is_result"] == True].copy()

    home = (
        finished.groupby("home_team")
        .agg(home_m=("game_id", "count"),
             home_xg=("home_xg", "sum"),
             home_xga=("away_xg", "sum"))
        .reset_index()
        .rename(columns={"home_team": "club"})
    )
    away = (
        finished.groupby("away_team")
        .agg(away_m=("game_id", "count"),
             away_xg=("away_xg", "sum"),
             away_xga=("home_xg", "sum"))
        .reset_index()
        .rename(columns={"away_team": "club"})
    )

    df = home.merge(away, on="club", how="outer").fillna(0)
    df["matches_us"] = df["home_m"]  + df["away_m"]
    df["xG"]         = df["home_xg"] + df["away_xg"]
    df["xGA"]        = df["home_xga"] + df["away_xga"]
    df["xG_p90"]     = (df["xG"]  / df["matches_us"]).round(3)
    df["xGA_p90"]    = (df["xGA"] / df["matches_us"]).round(3)
    df["league"]     = league_label
    # Normaliser les noms vers la convention FBref
    df["club"] = df["club"].replace(CLUB_NAME_US_TO_FB)

    return df[["club", "league", "matches_us", "xG", "xGA", "xG_p90", "xGA_p90"]]


def load_xg_stats(season: str | None = None) -> pd.DataFrame:
    """Retourne DataFrame xG/xGA pour les 96 clubs des Big 5."""
    import soccerdata as sd

    seasons = [season] if season else list(SEASONS_TO_TRY)
    all_frames = []

    for s in seasons:
        success = False
        for league in UNDERSTAT_LEAGUES:
            try:
                us = sd.Understat(leagues=league, seasons=s)
                league_label = league.split("-", 1)[1]
                df = _agg_xg_for_league(us, league_label)
                all_frames.append(df)
                logger.info("Understat %s %s : %d clubs", league_label, s, len(df))
                success = True
            except Exception as exc:
                logger.warning("Understat %s %s : %s", league, s, exc)
        if success:
            break

    if not all_frames:
        raise RuntimeError("Understat : toutes les ligues/saisons ont échoué")

    return pd.concat(all_frames, ignore_index=True)


def enrich_club_stats_xg(club_stats: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Charge club_stats.csv (si club_stats=None), y ajoute les colonnes xG/xGA,
    sauvegarde et retourne le DataFrame enrichi.
    """
    import soccerdata as sd

    if club_stats is None:
        path = PROCESSED_DIR / "club_stats.csv"
        if not path.exists():
            raise FileNotFoundError(f"club_stats.csv non trouvé : {path}")
        club_stats = pd.read_csv(path)

    xg = load_xg_stats()

    # Supprimer les anciennes colonnes xG pour éviter les conflits de merge
    for col in ["xG", "xGA", "xG_p90", "xGA_p90", "matches_us"]:
        if col in club_stats.columns:
            club_stats = club_stats.drop(columns=[col])

    merged = club_stats.merge(
        xg[["club", "league", "xG", "xGA", "xG_p90", "xGA_p90"]],
        on=["club", "league"],
        how="left",
    )

    matched = merged["xG_p90"].notna().sum()
    logger.info(
        "xG enrichi : %d/%d clubs (%d sans correspondance)",
        matched, len(merged), len(merged) - matched,
    )

    out = PROCESSED_DIR / "club_stats.csv"
    merged.to_csv(out, index=False)
    logger.info("Sauvegardé : %s (%d lignes)", out, len(merged))
    return merged


# Pour rétro-compat avec l'ancien stub
def enrich_understat(players_df: pd.DataFrame) -> pd.DataFrame:
    """Joint xG club sur players_df via la colonne 'club'."""
    xg = load_xg_stats()
    return players_df.merge(
        xg[["club", "xG_p90", "xGA_p90"]],
        on="club", how="left",
    )
