"""
Pipeline complet Phase 1 : Elo -> Poisson -> calibration -> prédictions WC2026.

Flux :
  1. Calcul Elo sur tout l'historique (2010-2026)
  2. Calcul elo_diff pour chaque match historique
  3. Fit Poisson sur matches 2018+ (période récente)
  4. Fit calibration Platt scaling sur 2018-2022
  5. Brier score brut et calibré sur validation (2023-2025)
  6. Prédictions pour les 104 fixtures WC2026 (raw + calibrées)
"""

import logging
from pathlib import Path

import pandas as pd

from .elo       import compute_elo, get_ratings_dict, DEFAULT_RATING, HOME_ADV_ELO
from .poisson   import fit, lambdas, score_matrix, outcome_probs, most_likely_score, brier_score
from .calibration import (
    fit_calibration, save_calibration, calibrate_dataframe,
    CAL_MIN_DATE, CAL_MAX_DATE,
)
from .confederation_adjustment import (
    compute_confederation_adjustments, get_adjusted_elo,
)

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"

FIT_MIN_DATE = "2018-01-01"
VAL_MIN_DATE = "2023-01-01"
VAL_MAX_DATE = "2025-12-31"


# ── Préparation des données ───────────────────────────────────────────────────

def _add_elo_diff(matches: pd.DataFrame, ratings: dict[str, float]) -> pd.DataFrame:
    df = matches.copy()
    df["elo_home"] = df["home_team"].map(ratings).fillna(DEFAULT_RATING)
    df["elo_away"] = df["away_team"].map(ratings).fillna(DEFAULT_RATING)
    adv = df["neutral"].map({True: 0.0, False: HOME_ADV_ELO}).fillna(0.0)
    df["elo_diff"] = df["elo_home"] + adv - df["elo_away"]
    return df


# ── Pipeline principal ────────────────────────────────────────────────────────

def run(recalculate_elo: bool = False) -> pd.DataFrame:
    """Exécute le pipeline complet et retourne predictions.csv (avec colonnes calibrées)."""
    # 1. Historique
    matches = pd.read_csv(
        PROCESSED_DIR / "matches_historical.csv",
        parse_dates=["date"],
    )
    matches["home_score"] = pd.to_numeric(matches["home_score"], errors="coerce")
    matches["away_score"] = pd.to_numeric(matches["away_score"], errors="coerce")
    matches = matches.dropna(subset=["home_score", "away_score"])

    # 2. Elo final
    ratings = get_ratings_dict(recalculate=recalculate_elo)
    logger.info("Elo ratings : %d équipes chargés", len(ratings))

    # 3. Enrichissement elo_diff
    matches = _add_elo_diff(matches, ratings)

    # 4. Fit Poisson sur 2018+
    fit_data = matches[matches["date"] >= FIT_MIN_DATE].copy()
    params   = fit(fit_data)

    # 4.5. Calibration Platt scaling sur 2018-2022
    cal_data = matches[
        (matches["date"] >= CAL_MIN_DATE) &
        (matches["date"] <= CAL_MAX_DATE)
    ].copy()
    cal_preds = _predict_dataframe(cal_data, ratings, params)
    cal_preds["actual_result"] = cal_preds.apply(
        lambda r: "H" if r["home_score"] > r["away_score"]
                  else ("D" if r["home_score"] == r["away_score"] else "A"),
        axis=1,
    )
    clf_cal = fit_calibration(cal_preds)
    save_calibration(clf_cal)
    logger.info("Calibration entraînée sur %d matchs 2018-2022", len(cal_preds))

    # 5. Brier brut et calibré sur validation 2023-2025
    val = matches[
        (matches["date"] >= VAL_MIN_DATE) &
        (matches["date"] <= VAL_MAX_DATE)
    ].copy()
    if not val.empty:
        val_preds = _predict_dataframe(val, ratings, params)
        val_preds["actual_result"] = val_preds.apply(
            lambda r: "H" if r["home_score"] > r["away_score"]
                      else ("D" if r["home_score"] == r["away_score"] else "A"),
            axis=1,
        )
        bs_raw = brier_score(val_preds)
        logger.info("Brier RAW 2023-2025 : %.4f", bs_raw)

        val_cal = calibrate_dataframe(val_preds, clf_cal)
        val_for_bs = val_cal.copy()
        val_for_bs["p_home_win"] = val_cal["p_home_win_cal"]
        val_for_bs["p_draw"]     = val_cal["p_draw_cal"]
        val_for_bs["p_away_win"] = val_cal["p_away_win_cal"]
        bs_cal = brier_score(val_for_bs)
        logger.info(
            "Brier CALIBRE 2023-2025 : %.4f  (gain +%.4f)",
            bs_cal, bs_raw - bs_cal,
        )

    # 6. Prédictions WC2026 + calibration (avec ajustement confédération)
    fixtures     = pd.read_csv(PROCESSED_DIR / "fixtures.csv")
    conf_adj     = compute_confederation_adjustments()
    teams_df     = pd.read_csv(PROCESSED_DIR / "teams.csv")
    conf_map     = dict(zip(teams_df["team"], teams_df["confederation"]))
    preds        = _predict_fixtures(fixtures, ratings, params, conf_map, conf_adj)
    preds        = calibrate_dataframe(preds, clf_cal)

    out = PROCESSED_DIR / "predictions.csv"
    preds.to_csv(out, index=False)
    logger.info("Prédictions sauvegardées : %s (%d matchs)", out, len(preds))
    return preds


def _predict_dataframe(df: pd.DataFrame,
                       ratings: dict[str, float],
                       params: dict) -> pd.DataFrame:
    """Ajoute colonnes de prédiction (raw) sur un DataFrame de matchs historiques."""
    df = _add_elo_diff(df, ratings)
    rows = []
    for _, r in df.iterrows():
        lh, la        = lambdas(r["elo_diff"], params)
        mat           = score_matrix(lh, la)
        phw, pd_, paw = outcome_probs(mat)
        sh, sa, pp    = most_likely_score(mat)
        rows.append({
            **r.to_dict(),
            "lambda_home":     round(lh, 3),
            "lambda_away":     round(la, 3),
            "p_home_win":      round(phw, 4),
            "p_draw":          round(pd_, 4),
            "p_away_win":      round(paw, 4),
            "pred_score_home": sh,
            "pred_score_away": sa,
            "p_pred_score":    round(pp, 4),
        })
    return pd.DataFrame(rows)


def _predict_fixtures(
    fixtures: pd.DataFrame,
    ratings: dict[str, float],
    params: dict,
    conf_map: dict[str, str] | None = None,
    adjustments: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Prédit les 104 matchs WC2026 (tous terrains neutres).

    conf_map et adjustments : si fournis, applique l'ajustement confédération
    aux ratings Elo avant calcul (Elo ajusté = base + delta confédération).
    """
    rows = []
    for _, f in fixtures.iterrows():
        home_slot = f.get("home_slot", f.get("home_team", ""))
        away_slot = f.get("away_slot", f.get("away_team", ""))

        base_h = ratings.get(home_slot, DEFAULT_RATING)
        base_a = ratings.get(away_slot, DEFAULT_RATING)

        if conf_map is not None and adjustments is not None:
            conf_h = conf_map.get(home_slot, "")
            conf_a = conf_map.get(away_slot, "")
            elo_h  = get_adjusted_elo(home_slot, base_h, conf_h, adjustments)
            elo_a  = get_adjusted_elo(away_slot, base_a, conf_a, adjustments)
        else:
            elo_h, elo_a = base_h, base_a

        elo_diff = elo_h - elo_a

        lh, la        = lambdas(elo_diff, params)
        mat           = score_matrix(lh, la)
        phw, pd_, paw = outcome_probs(mat)
        sh, sa, pp    = most_likely_score(mat)

        rows.append({
            "match_id":        f["match_id"],
            "date":            f.get("kickoff_utc", f.get("date_local", "")),
            "stage":           f["stage"],
            "group":           f.get("group", ""),
            "home_slot":       home_slot,
            "away_slot":       away_slot,
            "venue":           f.get("venue", ""),
            "elo_home":        round(base_h, 1),
            "elo_away":        round(base_a, 1),
            "elo_home_adj":    round(elo_h, 1),
            "elo_away_adj":    round(elo_a, 1),
            "elo_diff":        round(elo_diff, 1),
            "lambda_home":     round(lh, 3),
            "lambda_away":     round(la, 3),
            "p_home_win":      round(phw, 4),
            "p_draw":          round(pd_, 4),
            "p_away_win":      round(paw, 4),
            "pred_score_home": sh,
            "pred_score_away": sa,
            "p_pred_score":    round(pp, 4),
        })

    return pd.DataFrame(rows)
