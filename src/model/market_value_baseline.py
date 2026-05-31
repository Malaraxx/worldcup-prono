"""
Phase 1.5a — Baseline market value par équipe WC2026.

Logique :
  - mv_current = squad_value_eur dans team_stats.csv (Transfermarkt 2026)
  - mv_baseline = médiane des équipes de la même confédération
  - mv_ratio = mv_current / mv_baseline  (1.0 = dans la moyenne conf.)
  - OFC (1 seule équipe, NaN) → fallback médiane AFC (niveau économique comparable)
  - Équipes sans MV → fallback médiane confédération
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROCESSED = Path(__file__).parents[2] / "data" / "processed"


def compute_mv_baseline() -> pd.DataFrame:
    """
    Retourne DataFrame : team, confederation, mv_baseline_eur, mv_current_eur, mv_ratio.
    Sauvegarde data/processed/team_mv_baseline.csv.
    """
    ts = pd.read_csv(PROCESSED / "team_stats.csv")

    # Médiane par confédération (hors NaN et hors 0 — scraping failures)
    ts_valid = ts[ts["squad_value_eur"] > 0].copy()
    conf_median: dict[str, float] = (
        ts_valid.groupby("confederation")["squad_value_eur"]
        .median()
        .to_dict()
    )

    # OFC : une seule équipe (NZ) avec NaN → fallback médiane AFC
    if "OFC" not in conf_median or pd.isna(conf_median.get("OFC")):
        conf_median["OFC"] = conf_median.get("AFC", ts["squad_value_eur"].median())
        logger.warning("OFC : 1 équipe sans MV → fallback médiane AFC = %.0f €",
                       conf_median["OFC"])

    rows = []
    for _, r in ts.iterrows():
        conf    = r["confederation"]
        current = r["squad_value_eur"]
        base    = conf_median.get(conf, ts["squad_value_eur"].median())

        if pd.isna(current) or current <= 0:
            logger.warning("MV manquante/nulle pour %s (%s) → fallback médiane conf = %.0f €",
                           r["team"], conf, base)
            current = base

        mv_ratio = float(current) / float(base) if base > 0 else 1.0

        rows.append({
            "team":            r["team"],
            "confederation":   conf,
            "mv_baseline_eur": round(base),
            "mv_current_eur":  round(current),
            "mv_ratio":        round(mv_ratio, 4),
        })

    df = pd.DataFrame(rows).sort_values("mv_ratio", ascending=False)
    out = PROCESSED / "team_mv_baseline.csv"
    df.to_csv(out, index=False)
    logger.info("team_mv_baseline.csv : %d équipes, ratio min=%.2f max=%.2f",
                len(df), df["mv_ratio"].min(), df["mv_ratio"].max())
    return df


def load_mv_ratios() -> dict[str, float]:
    """Retourne {team: mv_ratio} depuis team_mv_baseline.csv (crée le fichier si absent)."""
    path = PROCESSED / "team_mv_baseline.csv"
    if not path.exists():
        compute_mv_baseline()
    df = pd.read_csv(path)
    return dict(zip(df["team"], df["mv_ratio"]))
