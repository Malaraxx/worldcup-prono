"""
Platt scaling multinomial pour calibrer les probas brutes du modèle Poisson.
Entraîné sur 2018-2022, appliqué à WC2026.
"""

import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"
MODEL_PATH    = PROCESSED_DIR / "calibration_model.pkl"
CAL_MIN_DATE  = "2018-01-01"
CAL_MAX_DATE  = "2022-12-31"


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def fit_calibration(cal_df: pd.DataFrame) -> LogisticRegression:
    """
    Entraîne une régression logistique multinomiale sur logit(p_home, p_draw, p_away).
    cal_df requiert : p_home_win, p_draw, p_away_win, actual_result (H/D/A).
    """
    df = cal_df.dropna(subset=["actual_result"]).copy()
    X = np.column_stack([
        _logit(df["p_home_win"].values),
        _logit(df["p_draw"].values),
        _logit(df["p_away_win"].values),
    ])
    label_map = {"H": 0, "D": 1, "A": 2}
    y = df["actual_result"].map(label_map).values

    clf = LogisticRegression(
        solver="lbfgs", max_iter=2000, C=10.0, random_state=42,
    )
    clf.fit(X, y)
    logger.info(
        "Calibration fit : %d matchs (H=%d D=%d A=%d)",
        len(df), (y == 0).sum(), (y == 1).sum(), (y == 2).sum(),
    )
    return clf


def save_calibration(clf: LogisticRegression) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    logger.info("Calibration model sauvegardé : %s", MODEL_PATH)


def load_calibration() -> LogisticRegression:
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def calibrate(
    p_home: float,
    p_draw: float,
    p_away: float,
    clf: LogisticRegression | None = None,
) -> tuple[float, float, float]:
    """Calibre un triplet de probas. Retourne (p_home_cal, p_draw_cal, p_away_cal)."""
    if clf is None:
        clf = load_calibration()
    X = np.array([[
        _logit(np.array([p_home]))[0],
        _logit(np.array([p_draw]))[0],
        _logit(np.array([p_away]))[0],
    ]])
    proba = clf.predict_proba(X)[0]
    classes = list(clf.classes_)
    return (
        float(proba[classes.index(0)]),
        float(proba[classes.index(1)]),
        float(proba[classes.index(2)]),
    )


def calibrate_dataframe(
    df: pd.DataFrame,
    clf: LogisticRegression | None = None,
) -> pd.DataFrame:
    """
    Calibration batch sur DataFrame.
    Requiert colonnes : p_home_win, p_draw, p_away_win.
    Ajoute colonnes : p_home_win_cal, p_draw_cal, p_away_win_cal.
    """
    if clf is None:
        clf = load_calibration()
    X = np.column_stack([
        _logit(df["p_home_win"].values),
        _logit(df["p_draw"].values),
        _logit(df["p_away_win"].values),
    ])
    proba = clf.predict_proba(X)
    classes = list(clf.classes_)
    idx_h = classes.index(0)
    idx_d = classes.index(1)
    idx_a = classes.index(2)

    result = df.copy()
    result["p_home_win_cal"] = proba[:, idx_h].round(4)
    result["p_draw_cal"]     = proba[:, idx_d].round(4)
    result["p_away_win_cal"] = proba[:, idx_a].round(4)
    return result
