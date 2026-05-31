"""
Phase 1.5a — Calibration empirique de alpha (ajustement MV).

Teste une grille d'alpha sur les matchs 2024-2025.
Pour chaque alpha :
  - Elo de chaque équipe = Elo_final + conf_adj + alpha * log(mv_ratio)
  - Poisson → probas → calibration Platt → Brier
Retourne alpha_optimal et le delta Brier vs baseline.

Garde-fou : si aucun alpha n'améliore le Brier de > THRESHOLD_PCT, alpha_optimal = 0.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.model.elo import get_ratings_dict, DEFAULT_RATING, HOME_ADV_ELO
from src.model.poisson import fit, lambdas, outcome_probs, score_matrix, brier_score
from src.model.calibration import (
    CAL_MIN_DATE, CAL_MAX_DATE, fit_calibration, load_calibration, calibrate_dataframe,
)
from src.model.confederation_adjustment import (
    compute_confederation_adjustments, get_adjusted_elo,
)
from src.model.market_value_baseline import load_mv_ratios
from src.model.elo_mv_adjustment import adjust_elo_by_mv

logger = logging.getLogger(__name__)

PROCESSED    = Path(__file__).parents[2] / "data" / "processed"
FIT_MIN_DATE = "2018-01-01"
VAL_MIN_DATE = "2024-01-01"   # MV 2026 les plus proches de cette période
VAL_MAX_DATE = "2025-12-31"
ALPHA_GRID   = [0, 10, 25, 50, 75, 100, 150]
THRESHOLD_PCT = 0.5           # amélioration minimale exigée (%)


def _add_elo_with_adjustments(
    df: pd.DataFrame,
    ratings: dict[str, float],
    conf_adj: dict[str, float],
    conf_map: dict[str, str],
    mv_ratios: dict[str, float],
    alpha: float,
) -> pd.DataFrame:
    """Ajoute elo_diff en appliquant conf + MV adjustments."""
    out = df.copy()
    elo_home_vals, elo_away_vals = [], []

    for _, r in df.iterrows():
        home = r["home_team"]
        away = r["away_team"]

        base_h = ratings.get(home, DEFAULT_RATING)
        base_a = ratings.get(away, DEFAULT_RATING)

        conf_h = conf_map.get(home, "")
        conf_a = conf_map.get(away, "")

        # 1. Ajustement confédération
        elo_h = get_adjusted_elo(home, base_h, conf_h, conf_adj)
        elo_a = get_adjusted_elo(away, base_a, conf_a, conf_adj)

        # 2. Ajustement market value (mv_ratio=None → no-op)
        ratio_h = mv_ratios.get(home)
        ratio_a = mv_ratios.get(away)
        elo_h = adjust_elo_by_mv(elo_h, ratio_h, alpha)
        elo_a = adjust_elo_by_mv(elo_a, ratio_a, alpha)

        elo_home_vals.append(elo_h)
        elo_away_vals.append(elo_a)

    out["elo_home"] = elo_home_vals
    out["elo_away"] = elo_away_vals
    neutral = out["neutral"].map({True: 0.0, False: HOME_ADV_ELO}).fillna(0.0)
    out["elo_diff"] = out["elo_home"] + neutral - out["elo_away"]
    return out


def _predict_with_alpha(
    val: pd.DataFrame,
    ratings: dict[str, float],
    params: dict,
    conf_adj: dict[str, float],
    conf_map: dict[str, str],
    mv_ratios: dict[str, float],
    alpha: float,
    clf_cal,
) -> float:
    """Calcule le Brier sur val pour un alpha donné. Retourne Brier score."""
    df = _add_elo_with_adjustments(val, ratings, conf_adj, conf_map, mv_ratios, alpha)

    rows = []
    for _, r in df.iterrows():
        lh, la = lambdas(r["elo_diff"], params)
        mat = score_matrix(lh, la)
        phw, pd_, paw = outcome_probs(mat)
        rows.append({**r.to_dict(), "p_home_win": phw, "p_draw": pd_, "p_away_win": paw})
    preds = pd.DataFrame(rows)

    preds_cal = calibrate_dataframe(preds, clf_cal)
    preds_cal["p_home_win"] = preds_cal["p_home_win_cal"]
    preds_cal["p_draw"]     = preds_cal["p_draw_cal"]
    preds_cal["p_away_win"] = preds_cal["p_away_win_cal"]
    return brier_score(preds_cal)


def run_calibration() -> dict:
    """
    Exécute la grille d'alpha et retourne :
    {
        "results": [{alpha, brier, delta_pct}, ...],
        "alpha_optimal": int,
        "brier_baseline": float,
        "brier_best": float,
        "delta_pct": float,
        "applied": bool,
    }
    """
    # Données
    matches = pd.read_csv(PROCESSED / "matches_historical.csv", parse_dates=["date"])
    matches["home_score"] = pd.to_numeric(matches["home_score"], errors="coerce")
    matches["away_score"] = pd.to_numeric(matches["away_score"], errors="coerce")
    matches = matches.dropna(subset=["home_score", "away_score"])
    matches["actual_result"] = matches.apply(
        lambda r: "H" if r["home_score"] > r["away_score"]
                  else ("D" if r["home_score"] == r["away_score"] else "A"),
        axis=1,
    )

    # Elo final
    ratings = get_ratings_dict()

    # Poisson params (fit sur 2018+)
    fit_data = matches[matches["date"] >= FIT_MIN_DATE].copy()
    fit_data["elo_home"] = fit_data["home_team"].map(ratings).fillna(DEFAULT_RATING)
    fit_data["elo_away"] = fit_data["away_team"].map(ratings).fillna(DEFAULT_RATING)
    neutral = fit_data["neutral"].map({True: 0.0, False: HOME_ADV_ELO}).fillna(0.0)
    fit_data["elo_diff"] = fit_data["elo_home"] + neutral - fit_data["elo_away"]
    params = fit(fit_data)

    # Calibration Platt (2018-2022)
    try:
        clf_cal = load_calibration()
    except FileNotFoundError:
        cal_data = matches[
            (matches["date"] >= CAL_MIN_DATE) & (matches["date"] <= CAL_MAX_DATE)
        ].copy()
        cal_data["elo_home"] = cal_data["home_team"].map(ratings).fillna(DEFAULT_RATING)
        cal_data["elo_away"] = cal_data["away_team"].map(ratings).fillna(DEFAULT_RATING)
        neutral_cal = cal_data["neutral"].map({True: 0.0, False: HOME_ADV_ELO}).fillna(0.0)
        cal_data["elo_diff"] = cal_data["elo_home"] + neutral_cal - cal_data["elo_away"]
        cal_rows = []
        for _, r in cal_data.iterrows():
            lh, la = lambdas(r["elo_diff"], params)
            mat = score_matrix(lh, la)
            phw, pd_, paw = outcome_probs(mat)
            cal_rows.append({**r.to_dict(), "p_home_win": phw, "p_draw": pd_, "p_away_win": paw})
        clf_cal = fit_calibration(pd.DataFrame(cal_rows))

    # Données confédération et MV
    conf_adj  = compute_confederation_adjustments()
    teams_df  = pd.read_csv(PROCESSED / "teams.csv")
    conf_map  = dict(zip(teams_df["team"], teams_df["confederation"]))
    mv_ratios = load_mv_ratios()

    # Validation 2024-2025
    val = matches[
        (matches["date"] >= VAL_MIN_DATE) & (matches["date"] <= VAL_MAX_DATE)
    ].copy()
    logger.info("Validation : %d matchs 2024-2025", len(val))

    # Grille alpha
    results = []
    brier_base = None
    for alpha in ALPHA_GRID:
        b = _predict_with_alpha(val, ratings, params, conf_adj, conf_map, mv_ratios, alpha, clf_cal)
        if brier_base is None:
            brier_base = b  # alpha=0 est la baseline
        delta_pct = (brier_base - b) / brier_base * 100  # positif = amélioration
        results.append({"alpha": alpha, "brier": round(b, 6), "delta_pct": round(delta_pct, 3)})
        logger.info("alpha=%3d → Brier=%.6f  Δ=%+.3f%%", alpha, b, delta_pct)

    # Meilleur alpha (excluant 0)
    best = min(results, key=lambda x: x["brier"])
    alpha_optimal = best["alpha"]
    brier_best    = best["brier"]
    delta_pct     = best["delta_pct"]

    applied = alpha_optimal > 0 and delta_pct >= THRESHOLD_PCT
    if not applied:
        logger.warning(
            "Rollback : meilleur alpha=%d améliore de %.3f%% < %.1f%% seuil — alpha=0 retenu",
            alpha_optimal, delta_pct, THRESHOLD_PCT,
        )
        alpha_optimal = 0

    return {
        "results":        results,
        "alpha_optimal":  alpha_optimal,
        "brier_baseline": round(brier_base, 6),
        "brier_best":     round(brier_best, 6),
        "delta_pct":      round(delta_pct, 3),
        "applied":        applied,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = run_calibration()
    print("\n=== RÉSULTAT CALIBRATION ALPHA ===")
    for r in result["results"]:
        print(f"  alpha={r['alpha']:3d}  Brier={r['brier']:.6f}  Δ={r['delta_pct']:+.3f}%")
    print(f"\nAlpha optimal : {result['alpha_optimal']}")
    print(f"Brier baseline (alpha=0) : {result['brier_baseline']:.6f}")
    print(f"Brier best     : {result['brier_best']:.6f}")
    print(f"Amélioration   : {result['delta_pct']:+.3f}%")
    print(f"Appliqué       : {'OUI' if result['applied'] else 'NON (rollback)'}")
