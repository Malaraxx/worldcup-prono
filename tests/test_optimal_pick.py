"""Tests pour le module src/strategy/optimal_pick."""
import pytest
import pandas as pd
from pathlib import Path

PROCESSED  = Path(__file__).parents[1] / "data" / "processed"
PICKS_PATH = PROCESSED / "optimal_picks.csv"
VALID_MODES = {"safe", "value", "lottery"}


@pytest.fixture(scope="module")
def picks_df():
    if not PICKS_PATH.exists():
        pytest.skip("optimal_picks.csv non généré — lancez src/strategy/optimal_pick.py d'abord")
    return pd.read_csv(PICKS_PATH)


def test_24_lignes(picks_df):
    assert len(picks_df) == 24


def test_aucun_nan(picks_df):
    assert picks_df.isnull().sum().sum() == 0


def test_safe_wr_moyen_ge_50pct(picks_df):
    assert picks_df["safe_wr"].mean() >= 0.50


def test_value_ev_ge_safe_ev_en_moyenne(picks_df):
    assert picks_df["value_ev"].mean() >= picks_df["safe_ev"].mean()


def test_modes_valides(picks_df):
    assert set(picks_df["mode_recommended"].unique()).issubset(VALID_MODES)
