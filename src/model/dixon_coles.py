"""
Correction Dixon-Coles (1997) pour les scores faibles en football.

Le modèle Poisson suppose l'indépendance entre les buts des deux équipes,
ce qui sous-estime les scores 1-1, 2-1, 1-2 et surestime 0-0, 1-0, 0-1
sur les matchs équilibrés. Dixon-Coles ajustent les 4 scores faibles via
un paramètre rho (ρ ≤ 0 pour le football international).

Référence :
  Dixon, M. & Coles, S. (1997). Modelling association football scores and
  inefficiencies in the football betting market. Applied Statistics 46(2), 265-280.
"""

import logging
from pathlib import Path

import numpy as np
from scipy.stats import poisson as poisson_dist

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"
RHO_FILE = PROCESSED_DIR / "dixon_coles_rho.txt"


# ── Correction tau ────────────────────────────────────────────────────────────

def tau(i: int, j: int, lambda_h: float, lambda_a: float, rho: float) -> float:
    """
    Facteur de correction Dixon-Coles pour les 4 scores faibles.
    Retourne 1.0 pour tous les autres scores (pas de correction).

    Contrainte : rho doit être tel que tau reste > 0 pour tous les scores.
    En pratique, rho ∈ [-0.2, 0] est sûr pour λ_h, λ_a ∈ [0.5, 3.0].
    """
    if i == 0 and j == 0:
        return 1 - lambda_h * lambda_a * rho
    if i == 0 and j == 1:
        return 1 + lambda_h * rho
    if i == 1 and j == 0:
        return 1 + lambda_a * rho
    if i == 1 and j == 1:
        return 1 - rho
    return 1.0


# ── Probabilité d'un score ────────────────────────────────────────────────────

def dixon_coles_probability(i: int, j: int,
                            lambda_h: float, lambda_a: float,
                            rho: float) -> float:
    """
    Probabilité du score (i, j) sous Dixon-Coles :
      P_DC(i,j) = P_Poisson(i, λ_h) × P_Poisson(j, λ_a) × τ(i,j,λ_h,λ_a,ρ)
    """
    p_indep = float(poisson_dist.pmf(i, lambda_h) * poisson_dist.pmf(j, lambda_a))
    return p_indep * tau(i, j, lambda_h, lambda_a, rho)


# ── Matrice de scores ─────────────────────────────────────────────────────────

def score_matrix_dixon_coles(lambda_h: float, lambda_a: float,
                              rho: float, max_goals: int = 8) -> np.ndarray:
    """
    Matrice (max_goals+1) × (max_goals+1) de probabilités score-par-score DC.
    Normalisée pour que la somme vaille 1.
    """
    mat = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            mat[i, j] = dixon_coles_probability(i, j, lambda_h, lambda_a, rho)
    total = mat.sum()
    return mat / total if total > 0 else mat


# ── Chargement du rho calibré ─────────────────────────────────────────────────

def load_rho() -> float:
    """
    Charge rho depuis data/processed/dixon_coles_rho.txt.
    Retourne 0.0 si le fichier est absent (fallback Poisson pur — tau = 1 partout).
    """
    if RHO_FILE.exists():
        try:
            rho = float(RHO_FILE.read_text().strip())
            logger.debug("Dixon-Coles rho chargé : %.4f", rho)
            return rho
        except (ValueError, OSError):
            logger.warning("Impossible de lire %s — fallback rho=0.0", RHO_FILE)
    return 0.0
