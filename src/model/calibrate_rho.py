"""
Calibration du paramètre rho Dixon-Coles par maximum de vraisemblance.

Méthode :
  1. Charger matches historiques, calculer Elo ratings et Poisson params
  2. Optimiser rho sur log-likelihood DC — fit set 2018-2022 (L-BFGS-B, bornes [-0.2, 0])
  3. Valider Brier sur 2023-2025
  4. Sauvegarder rho si amélioration >= BRIER_THRESHOLD (0.005 absolu)
     Sinon : rollback — predict.py reste inchangé

Ordre du pipeline : Elo brut → conf_adj (fixtures WC uniquement) → Poisson → DC
Pour l'historique, pas de conf_adj (cohérent avec predict._predict_dataframe).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson as poisson_dist

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"
RHO_FILE      = PROCESSED_DIR / "dixon_coles_rho.txt"

FIT_MIN_DATE = "2018-01-01"
FIT_MAX_DATE = "2022-12-31"
VAL_MIN_DATE = "2023-01-01"
VAL_MAX_DATE = "2025-12-31"

BRIER_THRESHOLD = 0.005   # amélioration absolue minimale requise


# ── Préparation des données ───────────────────────────────────────────────────

def _prepare_matches() -> pd.DataFrame:
    """
    Charge les matches historiques, ajoute elo_diff et lambdas Poisson.
    Utilise les mêmes paramètres que predict.py (pas de conf_adj sur l'historique).
    """
    from .elo    import get_ratings_dict, DEFAULT_RATING, HOME_ADV_ELO
    from .poisson import fit as fit_poisson

    matches = pd.read_csv(PROCESSED_DIR / "matches_historical.csv", parse_dates=["date"])
    matches["home_score"] = pd.to_numeric(matches["home_score"], errors="coerce")
    matches["away_score"] = pd.to_numeric(matches["away_score"], errors="coerce")
    matches = matches.dropna(subset=["home_score", "away_score"]).copy()

    ratings = get_ratings_dict()

    matches["elo_home"] = matches["home_team"].map(ratings).fillna(DEFAULT_RATING)
    matches["elo_away"] = matches["away_team"].map(ratings).fillna(DEFAULT_RATING)
    adv = matches["neutral"].map({True: 0.0, False: HOME_ADV_ELO}).fillna(0.0)
    matches["elo_diff"] = matches["elo_home"] + adv - matches["elo_away"]

    # Fit Poisson sur 2018+ (même période que predict.py)
    fit_data = matches[matches["date"] >= FIT_MIN_DATE].copy()
    params   = fit_poisson(fit_data)

    # Lambdas vectorisées (évite une boucle Python sur 15k lignes)
    matches["lambda_home"] = np.exp(params["alpha_h"] + params["beta_h"] * matches["elo_diff"])
    matches["lambda_away"] = np.exp(params["alpha_a"] - params["beta_a"] * matches["elo_diff"])

    return matches


# ── Log-vraisemblance Dixon-Coles ─────────────────────────────────────────────

def _neg_log_likelihood(rho: float, df: pd.DataFrame) -> float:
    """
    Log-likelihood négative DC vectorisée sur un DataFrame de matches.
    Seuls les 4 scores faibles reçoivent la correction tau ; les autres ont tau=1.
    """
    lh = df["lambda_home"].values
    la = df["lambda_away"].values
    i  = df["home_score"].values.astype(int)
    j  = df["away_score"].values.astype(int)

    p_indep  = poisson_dist.pmf(i, lh) * poisson_dist.pmf(j, la)

    tau_vals = np.ones(len(df))
    tau_vals[(i == 0) & (j == 0)] = 1 - lh[(i == 0) & (j == 0)] * la[(i == 0) & (j == 0)] * rho
    tau_vals[(i == 0) & (j == 1)] = 1 + lh[(i == 0) & (j == 1)] * rho
    tau_vals[(i == 1) & (j == 0)] = 1 + la[(i == 1) & (j == 0)] * rho
    tau_vals[(i == 1) & (j == 1)] = 1 - rho

    p_dc = p_indep * tau_vals
    return -float(np.sum(np.log(np.maximum(p_dc, 1e-10))))


# ── Brier score ───────────────────────────────────────────────────────────────

def _brier(rho: float, df: pd.DataFrame) -> float:
    """
    Brier score raw (pre-Platt) avec correction DC pour un rho donné.
    rho=0.0 → Poisson pur (tau=1 partout).
    """
    from .dixon_coles import score_matrix_dixon_coles
    from .poisson     import outcome_probs

    results = []
    for _, row in df.iterrows():
        mat           = score_matrix_dixon_coles(row["lambda_home"], row["lambda_away"], rho)
        phw, pd_, paw = outcome_probs(mat)
        actual = "H" if row["home_score"] > row["away_score"] else (
                 "D" if row["home_score"] == row["away_score"] else "A")
        y = {"H": [1, 0, 0], "D": [0, 1, 0], "A": [0, 0, 1]}[actual]
        results.append(np.sum((np.array([phw, pd_, paw]) - np.array(y)) ** 2))
    return float(np.mean(results))


# ── Calibration principale ────────────────────────────────────────────────────

def calibrate(save: bool = True) -> dict:
    """
    Calibre rho par MLE sur 2018-2022, valide Brier sur 2023-2025.

    Returns:
        dict : rho_optimal, brier_before, brier_after, improvement, improved, n_fit, n_val
    """
    logger.info("=== Calibration rho Dixon-Coles ===")
    matches = _prepare_matches()

    fit_df = matches[
        (matches["date"] >= FIT_MIN_DATE) & (matches["date"] <= FIT_MAX_DATE)
    ].reset_index(drop=True)
    val_df = matches[
        (matches["date"] >= VAL_MIN_DATE) & (matches["date"] <= VAL_MAX_DATE)
    ].reset_index(drop=True)

    logger.info("Fit set : %d matches (%s – %s)", len(fit_df), FIT_MIN_DATE, FIT_MAX_DATE)
    logger.info("Val set : %d matches (%s – %s)", len(val_df), VAL_MIN_DATE, VAL_MAX_DATE)

    # Optimisation rho par L-BFGS-B sur [-0.2, 0]
    result = minimize(
        fun=lambda x: _neg_log_likelihood(float(x[0]), fit_df),
        x0=[-0.1],
        method="L-BFGS-B",
        bounds=[(-0.2, 0.0)],
        options={"ftol": 1e-9, "gtol": 1e-7},
    )
    rho_optimal = float(np.clip(result.x[0], -0.2, 0.0))
    logger.info("rho optimal : %.4f  (converged=%s, nit=%d)",
                rho_optimal, result.success, result.nit)

    # Brier avant (Poisson pur) et après (DC) sur validation
    logger.info("Calcul Brier val set...")
    brier_before = _brier(0.0, val_df)
    brier_after  = _brier(rho_optimal, val_df)
    improvement  = brier_before - brier_after
    improved     = improvement >= BRIER_THRESHOLD

    logger.info(
        "Brier 2023-2025 : avant=%.4f  après=%.4f  delta=%.4f  threshold=%.4f",
        brier_before, brier_after, improvement, BRIER_THRESHOLD,
    )

    if improved:
        logger.info("✓ Amélioration suffisante — Dixon-Coles validé")
        if save:
            RHO_FILE.write_text(f"{rho_optimal:.6f}")
            logger.info("rho sauvegardé : %s", RHO_FILE)
    else:
        logger.warning(
            "✗ Amélioration insuffisante (%.4f < %.4f) — rollback, predict.py inchangé",
            improvement, BRIER_THRESHOLD,
        )

    return {
        "rho_optimal":  rho_optimal,
        "brier_before": brier_before,
        "brier_after":  brier_after,
        "improvement":  improvement,
        "improved":     improved,
        "n_fit":        len(fit_df),
        "n_val":        len(val_df),
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parents[2]))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    r = calibrate()
    print(f"\n{'='*52}")
    print(f"  rho optimal  : {r['rho_optimal']:.4f}")
    print(f"  Brier avant  : {r['brier_before']:.4f}")
    print(f"  Brier après  : {r['brier_after']:.4f}")
    print(f"  Amélioration : {r['improvement']:+.4f}")
    print(f"  Résultat     : {'✓ DC validé' if r['improved'] else '✗ Rollback'}")
    print(f"{'='*52}")
