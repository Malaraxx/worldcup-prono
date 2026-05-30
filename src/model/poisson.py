"""
Régression Poisson pour calibrer λ_home / λ_away depuis le différentiel Elo.
Modèle :
  log(λ_home) = α_h + β_h * elo_diff
  log(λ_away) = α_a - β_a * elo_diff   (symétrique)
où elo_diff = elo_home - elo_away (terrain inclus le cas échéant).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson as poisson_dist

logger = logging.getLogger(__name__)

MAX_GOALS = 8   # troncature pour les distributions de scores

# ── MLE Poisson ──────────────────────────────────────────────────────────────

def _neg_ll(params: np.ndarray, elo_diff: np.ndarray,
            goals: np.ndarray, weights: np.ndarray,
            sign: float = 1.0) -> float:
    """log-vraisemblance négative pondérée pour une équipe (home ou away)."""
    alpha, beta = params
    lam = np.exp(alpha + sign * beta * elo_diff)
    ll  = weights * (goals * np.log(lam + 1e-12) - lam)
    return -ll.sum()


def fit(matches: pd.DataFrame) -> dict:
    """
    Ajuste le modèle Poisson sur les matchs fournis.
    matches doit avoir : elo_diff, home_score, away_score, match_weight.
    Retourne dict {alpha_h, beta_h, alpha_a, beta_a}.
    """
    elo_diff = matches["elo_diff"].values.astype(float)
    home_g   = matches["home_score"].values.astype(float)
    away_g   = matches["away_score"].values.astype(float)
    weights  = matches["match_weight"].values.astype(float)

    x0 = np.array([0.2, 0.001])

    res_h = minimize(_neg_ll, x0, args=(elo_diff, home_g, weights,  1.0),
                     method="Nelder-Mead", options={"xatol": 1e-7, "fatol": 1e-7})
    res_a = minimize(_neg_ll, x0, args=(elo_diff, away_g, weights, -1.0),
                     method="Nelder-Mead", options={"xatol": 1e-7, "fatol": 1e-7})

    params = {
        "alpha_h": res_h.x[0], "beta_h": res_h.x[1],
        "alpha_a": res_a.x[0], "beta_a": res_a.x[1],
    }
    lam_h_mean = np.exp(params["alpha_h"])
    lam_a_mean = np.exp(params["alpha_a"])
    logger.info(
        "Poisson fit : α_h=%.3f β_h=%.4f α_a=%.3f β_a=%.4f  "
        "(λ moyen : home=%.2f away=%.2f)",
        params["alpha_h"], params["beta_h"],
        params["alpha_a"], params["beta_a"],
        lam_h_mean, lam_a_mean,
    )
    return params


# ── Prédiction ────────────────────────────────────────────────────────────────

def lambdas(elo_diff: float, params: dict) -> tuple[float, float]:
    """Retourne (λ_home, λ_away) pour un différentiel Elo donné."""
    lam_h = np.exp(params["alpha_h"] + params["beta_h"] * elo_diff)
    lam_a = np.exp(params["alpha_a"] - params["beta_a"] * elo_diff)
    return float(lam_h), float(lam_a)


def score_matrix(lam_h: float, lam_a: float, max_g: int = MAX_GOALS) -> np.ndarray:
    """Matrice (max_g+1 × max_g+1) de probabilités P(H goals = i, A goals = j)."""
    ph = np.array([poisson_dist.pmf(k, lam_h) for k in range(max_g + 1)])
    pa = np.array([poisson_dist.pmf(k, lam_a) for k in range(max_g + 1)])
    return np.outer(ph, pa)


def outcome_probs(mat: np.ndarray) -> tuple[float, float, float]:
    """Retourne (p_home_win, p_draw, p_away_win) depuis la matrice de scores."""
    n = mat.shape[0]
    p_hw  = sum(mat[h, a] for h in range(n) for a in range(n) if h > a)
    p_d   = sum(mat[i, i] for i in range(n))
    p_aw  = 1.0 - p_hw - p_d
    return float(p_hw), float(p_d), float(p_aw)


def most_likely_score(mat: np.ndarray) -> tuple[int, int, float]:
    """Retourne (h_goals, a_goals, probability) du score le plus probable."""
    idx = np.unravel_index(mat.argmax(), mat.shape)
    return int(idx[0]), int(idx[1]), float(mat[idx])


def brier_score(predictions: pd.DataFrame) -> float:
    """
    Brier score sur les résultats réels du DataFrame predictions.
    Colonnes requises : p_home_win, p_draw, p_away_win, actual_result (H/D/A).
    """
    def _outcome_vec(r: str) -> np.ndarray:
        return {"H": [1, 0, 0], "D": [0, 1, 0], "A": [0, 0, 1]}.get(r, [0, 0, 0])

    scores = []
    for _, row in predictions.iterrows():
        if pd.isna(row.get("actual_result")):
            continue
        p = np.array([row["p_home_win"], row["p_draw"], row["p_away_win"]])
        y = np.array(_outcome_vec(row["actual_result"]))
        scores.append(np.sum((p - y) ** 2))

    return float(np.mean(scores)) if scores else float("nan")
