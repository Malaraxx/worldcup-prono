"""Tests minimaux pour l'app — vérifie que load_data() ne crash pas."""
import pytest
import pandas as pd
from pathlib import Path


@pytest.fixture(scope="module")
def data():
    import sys
    sys.path.insert(0, str(Path(__file__).parents[1]))
    from src.app.utils import load_data
    return load_data()


def test_load_data_all_keys(data):
    expected = {"fixtures", "predictions", "picks", "distributions", "tp", "teams", "group_sims", "ko"}
    assert set(data.keys()) == expected


def test_fixtures_shape(data):
    df = data["fixtures"]
    assert len(df) == 104
    assert "match_id" in df.columns
    assert "kickoff_dt" in df.columns


def test_predictions_shape(data):
    df = data["predictions"]
    assert len(df) == 104
    assert "p_home_win_cal" in df.columns
    assert "elo_home_adj" in df.columns


def test_picks_shape(data):
    df = data["picks"]
    assert len(df) == 24
    assert "mode_recommended" in df.columns


def test_distributions_shape(data):
    df = data["distributions"]
    assert len(df) == 24 * 49


def test_tp_shape(data):
    df = data["tp"]
    assert len(df) == 48
    assert "proba_winner" in df.columns


def test_get_match_returns_dict(data):
    from src.app.utils import get_match
    m = get_match(2)
    assert m is not None
    assert m["home"] == "South Korea"
    assert m["away"] == "Czech Republic"
    assert "pred" in m
    assert "picks" in m


def test_get_match_invalid_id():
    from src.app.utils import get_match
    assert get_match(9999) is None


def test_flag_helper():
    from src.app.utils import flag
    assert "🇫🇷" in flag("France")
    assert flag("Unknown Team") == "🏳️"


def test_mpp_implied_sums_to_one():
    from src.app.utils import mpp_implied
    ih, id_, ia = mpp_implied(96, 107, 91)
    assert abs(ih + id_ + ia - 1.0) < 1e-9
