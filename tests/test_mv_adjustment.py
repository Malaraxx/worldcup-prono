"""Tests Phase 1.5a — Ajustement Elo par valeur marchande."""
import math
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.model.elo_mv_adjustment import adjust_elo_by_mv


def test_adjust_elo_none_safe():
    assert adjust_elo_by_mv(1500, None, 50) == 1500.0


def test_adjust_elo_nan_safe():
    assert adjust_elo_by_mv(1500, float("nan"), 50) == 1500.0


def test_adjust_elo_zero_ratio_safe():
    assert adjust_elo_by_mv(1500, 0, 50) == 1500.0


def test_adjust_elo_neutral():
    assert adjust_elo_by_mv(1500, 1.0, 50) == pytest.approx(1500.0, abs=1e-9)


def test_adjust_elo_alpha_zero():
    assert adjust_elo_by_mv(1500, 2.0, 0) == 1500.0


def test_adjust_elo_positive():
    result = adjust_elo_by_mv(1500, 1.5, 50)
    assert result > 1500
    assert result == pytest.approx(1500 + 50 * math.log(1.5), rel=1e-9)


def test_adjust_elo_negative():
    result = adjust_elo_by_mv(1500, 0.7, 50)
    assert result < 1500
    assert result == pytest.approx(1500 + 50 * math.log(0.7), rel=1e-9)


def test_no_overflow():
    """mv_ratio=100 (extrême) : l'ajustement doit rester fini et raisonnable."""
    result = adjust_elo_by_mv(1500, 100, 50)
    assert math.isfinite(result)
    assert result < 1500 + 300   # 50 * log(100) ≈ 230 pts


def test_brier_not_degraded():
    """Brier 2024-2025 après ajustement avec alpha optimal ne doit pas dépasser
    Brier baseline + 0.5% (régression interdite)."""
    from src.model.calibrate_alpha import run_calibration
    result = run_calibration()
    baseline = result["brier_baseline"]
    best     = result["brier_best"]
    # La règle : soit on améliore, soit on n'applique pas → pas de régression possible
    assert best <= baseline * 1.005, (
        f"Brier dégradé : baseline={baseline:.6f} best={best:.6f}"
    )


def test_mv_baseline_48_teams():
    """team_mv_baseline.csv doit couvrir les 48 équipes WC2026."""
    import pandas as pd
    from src.model.market_value_baseline import compute_mv_baseline
    df = compute_mv_baseline()
    assert len(df) == 48
    assert df["mv_ratio"].notna().all()
    assert (df["mv_ratio"] > 0).all()


def test_mv_baseline_ratio_neutral_for_median():
    """Au moins une équipe par conf doit avoir mv_ratio proche de 1.0."""
    from src.model.market_value_baseline import compute_mv_baseline
    df = compute_mv_baseline()
    for conf, grp in df.groupby("confederation"):
        close_to_one = ((grp["mv_ratio"] - 1.0).abs() < 0.2).any()
        assert close_to_one, f"Conf {conf} : aucune équipe avec ratio ≈ 1.0"
