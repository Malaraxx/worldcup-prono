"""
Agrégation sources → team_stats.csv (une ligne par équipe WC2026).
Joint : effectifs + valeurs marchandes + stats clubs FBref/Understat.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"


def build_team_stats(players_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Produit team_stats.csv avec une ligne par équipe WC2026 (48 lignes).

    Colonnes :
      team, n_players, avg_age, avg_caps,
      squad_value_eur, avg_player_value_eur,
      poss, gls_p90, xG_p90, xGA_p90, sh_p90, sot_pct   ← moyenne pondérée par valeur
    """
    if players_df is None:
        path = PROCESSED_DIR / "players.csv"
        players_df = pd.read_csv(path)

    players_df = players_df.copy()
    players_df["market_value_eur"] = pd.to_numeric(
        players_df["market_value_eur"], errors="coerce"
    )

    # ── Stats de base par équipe ──────────────────────────────────────────────
    base = players_df.groupby("team").agg(
        n_players           = ("player_id", "count"),
        avg_age             = ("age",        "mean"),
        avg_caps            = ("caps",       "mean"),
        squad_value_eur     = ("market_value_eur", "sum"),
        avg_player_value_eur= ("market_value_eur", "mean"),
    ).reset_index()

    # ── Stats clubs FBref/Understat (pondérées par valeur marchande) ──────────
    club_stats_path = PROCESSED_DIR / "club_stats.csv"
    if club_stats_path.exists():
        club = pd.read_csv(club_stats_path)
        club_cols = ["club", "poss", "gls_p90", "xG_p90", "xGA_p90", "sh_p90", "sot_pct"]
        club = club[[c for c in club_cols if c in club.columns]]

        # Joindre les stats club sur chaque joueur
        enriched = players_df.merge(club, left_on="club", right_on="club", how="left")

        stat_cols = [c for c in ["poss", "gls_p90", "xG_p90", "xGA_p90", "sh_p90", "sot_pct"]
                     if c in enriched.columns]

        # Moyenne pondérée par valeur marchande (ou simple si valeur manquante)
        def weighted_mean(group: pd.DataFrame) -> pd.Series:
            weights = group["market_value_eur"].fillna(1.0)
            result = {}
            for col in stat_cols:
                valid = group[col].notna()
                if valid.any():
                    w = weights[valid]
                    result[col] = (group.loc[valid, col] * w).sum() / w.sum()
                else:
                    result[col] = float("nan")
            return pd.Series(result)

        club_agg = enriched.groupby("team").apply(weighted_mean).reset_index()
        base = base.merge(club_agg, on="team", how="left")
    else:
        logger.warning("club_stats.csv absent — colonnes xG/poss manquantes dans team_stats")

    # ── Équipes WC sans joueurs scrappés → garder dans teams.csv ─────────────
    teams_path = PROCESSED_DIR / "teams.csv"
    if teams_path.exists():
        all_teams = pd.read_csv(teams_path)[["team", "group", "confederation"]]
        base = all_teams.merge(base, on="team", how="left")

    # Arrondir les flottants
    float_cols = base.select_dtypes("float").columns
    base[float_cols] = base[float_cols].round(3)

    logger.info("team_stats : %d équipes, %d colonnes", len(base), len(base.columns))
    return base


def save_team_stats() -> pd.DataFrame:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df  = build_team_stats()
    out = PROCESSED_DIR / "team_stats.csv"
    df.to_csv(out, index=False)
    logger.info("Sauvegardé : %s (%d lignes)", out, len(df))
    return df
