"""Tests Phase 1.5b — Dixon-Coles correction."""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import poisson as poisson_dist

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.model.dixon_coles import (
    tau, dixon_coles_probability, score_matrix_dixon_coles, load_rho,
)

PROCESSED = Path(__file__).parents[1] / "data" / "processed"


# ── tau ───────────────────────────────────────────────────────────────────────

def test_tau_no_correction_for_other_scores():
    """Tous les scores hors (0,0),(0,1),(1,0),(1,1) ont tau = 1.0."""
    for i, j in [(2, 0), (0, 2), (2, 1), (1, 2), (3, 3), (5, 0)]:
        assert tau(i, j, 1.5, 1.0, -0.1) == 1.0, f"tau({i},{j}) != 1.0"


def test_tau_correction_zero_zero():
    """tau(0,0) != 1.0 quand rho != 0."""
    t = tau(0, 0, 1.5, 1.0, -0.1)
    # rho=-0.1 → tau = 1 - 1.5*1.0*(-0.1) = 1.15
    assert t != 1.0
    assert abs(t - 1.15) < 1e-9


def test_tau_boosts_1_1_when_rho_negative():
    """tau(1,1) = 1 - rho > 1 quand rho < 0."""
    t = tau(1, 1, 1.5, 1.0, -0.1)
    assert t == pytest.approx(1.1, abs=1e-9)
    assert t > 1.0


def test_tau_reduces_01_when_rho_negative():
    """tau(0,1) = 1 + lambda_h * rho < 1 quand rho < 0 et lambda_h > 0."""
    t = tau(0, 1, 1.5, 1.0, -0.1)
    # 1 + 1.5*(-0.1) = 0.85
    assert t == pytest.approx(0.85, abs=1e-9)
    assert t < 1.0


def test_tau_four_special_scores_covered():
    """Les 4 scores spéciaux ont des corrections non triviales quand rho != 0."""
    rho, lh, la = -0.05, 1.2, 1.1
    assert tau(0, 0, lh, la, rho) != 1.0
    assert tau(0, 1, lh, la, rho) != 1.0
    assert tau(1, 0, lh, la, rho) != 1.0
    assert tau(1, 1, lh, la, rho) != 1.0


# ── dixon_coles_probability ───────────────────────────────────────────────────

def test_dc_boosts_1_1():
    """P_DC(1,1) > P_Poisson(1,1) quand rho < 0 (tau(1,1) = 1 - rho > 1)."""
    lh, la, rho = 1.0, 1.0, -0.1
    p_dc  = dixon_coles_probability(1, 1, lh, la, rho)
    p_poi = float(poisson_dist.pmf(1, lh) * poisson_dist.pmf(1, la))
    assert p_dc > p_poi


def test_dc_boosts_0_0():
    """P_DC(0,0) > P_Poisson(0,0) quand rho < 0 (tau(0,0) = 1 - lh*la*rho > 1)."""
    lh, la, rho = 1.0, 1.0, -0.1
    p_dc  = dixon_coles_probability(0, 0, lh, la, rho)
    p_poi = float(poisson_dist.pmf(0, lh) * poisson_dist.pmf(0, la))
    assert p_dc > p_poi   # rho<0 → tau(0,0) = 1+|rho|*lh*la > 1


def test_dc_reduces_0_1():
    """P_DC(0,1) < P_Poisson(0,1) quand rho < 0 (tau(0,1) = 1 + lh*rho < 1)."""
    lh, la, rho = 1.0, 1.0, -0.1
    p_dc  = dixon_coles_probability(0, 1, lh, la, rho)
    p_poi = float(poisson_dist.pmf(0, lh) * poisson_dist.pmf(1, la))
    assert p_dc < p_poi


def test_dc_equals_poisson_when_rho_zero():
    """DC avec rho=0 est identique au Poisson indépendant."""
    for i, j in [(0, 0), (1, 1), (2, 3), (0, 1), (1, 0)]:
        p_dc  = dixon_coles_probability(i, j, 1.3, 1.1, 0.0)
        p_poi = float(poisson_dist.pmf(i, 1.3) * poisson_dist.pmf(j, 1.1))
        assert p_dc == pytest.approx(p_poi, rel=1e-8)


# ── score_matrix_dixon_coles ──────────────────────────────────────────────────

def test_matrix_dimensions():
    """score_matrix_dixon_coles retourne une matrice (max_goals+1) × (max_goals+1)."""
    mat = score_matrix_dixon_coles(1.3, 1.1, -0.05, max_goals=6)
    assert mat.shape == (7, 7)

    mat8 = score_matrix_dixon_coles(1.3, 1.1, -0.05, max_goals=8)
    assert mat8.shape == (9, 9)


def test_matrix_sums_to_one():
    """La matrice DC normalisée somme à 1.0 (tolérance 0.001)."""
    for rho in [-0.2, -0.1, -0.05, 0.0]:
        mat = score_matrix_dixon_coles(1.3, 1.1, rho, max_goals=6)
        assert abs(mat.sum() - 1.0) < 0.001, f"rho={rho}: sum={mat.sum()}"


def test_matrix_all_non_negative():
    """Toutes les probabilités sont >= 0."""
    mat = score_matrix_dixon_coles(1.5, 1.0, -0.2, max_goals=8)
    assert (mat >= 0).all()


def test_matrix_equals_poisson_when_rho_zero():
    """Avec rho=0, la distribution DC est proportionnelle à Poisson.
    DC normalise (sum=1), Poisson tronqué ne l'est pas exactement — on compare
    après normalisation des deux."""
    from src.model.poisson import score_matrix
    lh, la = 1.3, 1.1
    mat_dc  = score_matrix_dixon_coles(lh, la, 0.0, max_goals=8)
    mat_poi = score_matrix(lh, la, max_g=8)
    mat_poi_norm = mat_poi / mat_poi.sum()
    np.testing.assert_allclose(mat_dc, mat_poi_norm, rtol=1e-6)


# ── load_rho ──────────────────────────────────────────────────────────────────

def test_load_rho_returns_float():
    """load_rho() retourne toujours un float."""
    rho = load_rho()
    assert isinstance(rho, float)


def test_load_rho_fallback():
    """Sans fichier rho ou avec rho=0, load_rho() retourne 0.0 ou la valeur du fichier."""
    rho = load_rho()
    # Doit être dans [-0.2, 0] si fichier existe, ou 0.0 si absent
    assert -0.2 <= rho <= 0.0


# ── Tests conditionnels (si fichier rho calibré existe) ───────────────────────

def _rho_file():
    return PROCESSED / "dixon_coles_rho.txt"


@pytest.mark.skipif(not _rho_file().exists(), reason="rho non calibré (rollback)")
def test_rho_bounded():
    """rho fitté doit être dans [-0.2, 0]."""
    rho = float(_rho_file().read_text().strip())
    assert -0.2 <= rho <= 0.0, f"rho={rho} hors bornes [-0.2, 0]"


@pytest.mark.skipif(not _rho_file().exists(), reason="rho non calibré (rollback)")
def test_brier_not_degraded():
    """Brier DC <= Brier Poisson + 0.005 sur 2023-2025."""
    import pandas as pd
    from src.model.calibrate_rho import _prepare_matches, _brier, VAL_MIN_DATE, VAL_MAX_DATE

    rho = float(_rho_file().read_text().strip())
    matches = _prepare_matches()
    val_df  = matches[
        (matches["date"] >= VAL_MIN_DATE) & (matches["date"] <= VAL_MAX_DATE)
    ].reset_index(drop=True)

    brier_poi = _brier(0.0,  val_df)
    brier_dc  = _brier(rho,  val_df)
    assert brier_dc <= brier_poi + 0.005, (
        f"Brier DC ({brier_dc:.4f}) dégrade trop vs Poisson ({brier_poi:.4f})"
    )
