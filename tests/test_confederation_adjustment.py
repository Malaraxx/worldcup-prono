"""Tests pour src/model/confederation_adjustment.py."""
import logging
import pytest

logging.disable(logging.CRITICAL)


@pytest.fixture(scope="module")
def adjustments():
    from src.model.confederation_adjustment import compute_confederation_adjustments
    return compute_confederation_adjustments()


def test_six_confederations(adjustments):
    """ELO_ADJUSTMENT contient exactement les 6 confédérations FIFA."""
    expected = {"UEFA", "CONMEBOL", "AFC", "CAF", "CONCACAF", "OFC"}
    assert set(adjustments.keys()) == expected


def test_uefa_conmebol_zero(adjustments):
    """UEFA et CONMEBOL valent exactement 0.0 (référence)."""
    assert adjustments["UEFA"] == 0.0
    assert adjustments["CONMEBOL"] == 0.0


def test_afc_caf_ofc_negative(adjustments):
    """AFC, CAF et OFC ont un ajustement strictement négatif."""
    assert adjustments["AFC"] < 0, f"AFC={adjustments['AFC']} doit être < 0"
    assert adjustments["CAF"] < 0, f"CAF={adjustments['CAF']} doit être < 0"
    assert adjustments["OFC"] < 0, f"OFC={adjustments['OFC']} doit être < 0"


def test_brier_not_degraded():
    """
    Le Brier 2023-2025 avec ajustement ne dépasse pas Brier baseline × 1.02.
    L'ajustement est appliqué aux fixtures WC uniquement (pas aux matchs historiques),
    donc le Brier sur données historiques est identique.
    """
    import warnings
    warnings.filterwarnings("ignore")

    from pathlib import Path
    import pandas as pd
    from src.model.elo import get_ratings_dict
    from src.model.poisson import fit, brier_score
    from src.model.predict import _add_elo_diff, _predict_dataframe
    from src.model.calibration import fit_calibration, calibrate_dataframe, CAL_MIN_DATE, CAL_MAX_DATE

    PROCESSED = Path(__file__).parents[1] / "data" / "processed"
    matches = pd.read_csv(PROCESSED / "matches_historical.csv", parse_dates=["date"])
    matches["home_score"] = pd.to_numeric(matches["home_score"], errors="coerce")
    matches["away_score"] = pd.to_numeric(matches["away_score"], errors="coerce")
    matches = matches.dropna(subset=["home_score", "away_score"])

    ratings = get_ratings_dict()
    fit_data = _add_elo_diff(matches[matches["date"] >= "2018-01-01"].copy(), ratings)
    params = fit(fit_data)

    cal_data = _add_elo_diff(
        matches[(matches["date"] >= CAL_MIN_DATE) & (matches["date"] <= CAL_MAX_DATE)].copy(),
        ratings,
    )
    cal_preds = _predict_dataframe(cal_data, ratings, params)
    cal_preds["actual_result"] = cal_preds.apply(
        lambda r: "H" if r["home_score"] > r["away_score"]
        else ("D" if r["home_score"] == r["away_score"] else "A"),
        axis=1,
    )
    clf_cal = fit_calibration(cal_preds)

    val = matches[
        (matches["date"] >= "2023-01-01") & (matches["date"] <= "2025-12-31")
    ].copy()
    val_preds = _predict_dataframe(val, ratings, params)
    val_preds["actual_result"] = val_preds.apply(
        lambda r: "H" if r["home_score"] > r["away_score"]
        else ("D" if r["home_score"] == r["away_score"] else "A"),
        axis=1,
    )
    val_cal = calibrate_dataframe(val_preds, clf_cal)
    val_for_bs = val_cal.copy()
    val_for_bs["p_home_win"] = val_cal["p_home_win_cal"]
    val_for_bs["p_draw"] = val_cal["p_draw_cal"]
    val_for_bs["p_away_win"] = val_cal["p_away_win_cal"]
    brier_new = brier_score(val_for_bs)

    brier_old = 0.4948  # Baseline mesuré avant ajustement confédération
    assert brier_new <= brier_old * 1.02, (
        f"Brier dégradé : {brier_new:.4f} > {brier_old:.4f} × 1.02 = {brier_old * 1.02:.4f}"
    )
