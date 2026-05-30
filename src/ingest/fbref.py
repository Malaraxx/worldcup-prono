"""
Stats clubs Big 5 depuis FBref via soccerdata (Cloudflare bypass headless Chrome).
Saison prioritaire : 2025-26. Fallback : 2024-25.
Colonnes retenues : poss, gls, gls_p90, ast_p90, sh_p90, sot_pct, mp.
"""

import logging
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

PROCESSED_DIR  = Path(__file__).parents[2] / "data" / "processed"
SEASONS_TO_TRY = ("2526", "2425")

# Fetch individuel par ligue pour garantir des noms corrects dans l'index
FBREF_LEAGUES = [
    ("ENG-Premier League", "Premier League"),
    ("ESP-La Liga",        "La Liga"),
    ("GER-Bundesliga",     "Bundesliga"),
    ("ITA-Serie A",        "Serie A"),
    ("FRA-Ligue 1",        "Ligue 1"),
]


def _fetch_standard(fbref) -> pd.DataFrame:
    raw  = fbref.read_team_season_stats(stat_type="standard")
    cols = raw.columns  # MultiIndex

    selected = {}
    for label, new_name in [
        (("Poss", ""),              "poss"),
        (("Playing Time", "MP"),    "mp"),
        (("Performance", "Gls"),    "gls"),
        (("Performance", "Ast"),    "ast"),
        (("Per 90 Minutes", "Gls"), "gls_p90"),
        (("Per 90 Minutes", "Ast"), "ast_p90"),
    ]:
        if label in cols:
            selected[new_name] = raw[label]

    return pd.DataFrame(selected, index=raw.index)


def _fetch_shooting(fbref) -> pd.DataFrame:
    raw  = fbref.read_team_season_stats(stat_type="shooting")
    cols = raw.columns

    selected = {}
    for label, new_name in [
        (("Standard", "Sh/90"),  "sh_p90"),
        (("Standard", "SoT%"),   "sot_pct"),
        (("Standard", "SoT/90"), "sot_p90"),
    ]:
        if label in cols:
            selected[new_name] = raw[label]

    return pd.DataFrame(selected, index=raw.index)


def load_club_stats(season: str | None = None) -> pd.DataFrame:
    """
    Retourne DataFrame avec colonnes :
      club, league, season, poss, mp, gls, ast, gls_p90, ast_p90, sh_p90, sot_pct, sot_p90
    """
    import soccerdata as sd

    seasons = [season] if season else list(SEASONS_TO_TRY)

    for s in seasons:
        frames = []
        for fbref_name, label in FBREF_LEAGUES:
            try:
                fbref = sd.FBref(leagues=fbref_name, seasons=s)
                std   = _fetch_standard(fbref)
                sh    = _fetch_shooting(fbref)
                df    = std.join(sh, how="left").reset_index()
                df    = df.rename(columns={"team": "club"})
                df["league"] = label  # nom explicite, pas de regex
                frames.append(df)
                logger.info("FBref %s %s : %d clubs", label, s, len(df))
            except Exception as exc:
                logger.warning("FBref %s %s : %s", label, s, exc)

        if frames:
            return pd.concat(frames, ignore_index=True)

    raise RuntimeError("FBref : toutes les saisons ont échoué")


def save_club_stats() -> pd.DataFrame:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df  = load_club_stats()
    out = PROCESSED_DIR / "club_stats.csv"
    df.to_csv(out, index=False)
    logger.info("Sauvegardé : %s (%d lignes)", out, len(df))
    return df


def enrich_club_stats(players_df: pd.DataFrame) -> pd.DataFrame:
    """Joint les stats clubs sur players_df via la colonne 'club'."""
    club_stats = load_club_stats()
    stat_cols  = ["club", "poss", "gls_p90", "ast_p90", "sh_p90", "sot_pct"]
    available  = [c for c in stat_cols if c in club_stats.columns]
    merged = players_df.merge(
        club_stats[available], on="club", how="left"
    )
    logger.info(
        "Club stats jointes : %d/%d joueurs enrichis",
        merged["gls_p90"].notna().sum() if "gls_p90" in merged else 0,
        len(merged),
    )
    return merged
