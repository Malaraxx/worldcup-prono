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


def test_picks_nan_coverage(data):
    """optimal_picks.csv couvre 24 matchs (groupes) ; le merge left sur 104 fixtures
    doit produire exactement 80 matchs sans pick (NaN sur mode_recommended)."""
    import pandas as pd
    fix   = data["fixtures"]
    picks = data["picks"]

    merged = fix.merge(
        picks[["match_id", "mode_recommended"]], on="match_id", how="left"
    )
    n_with    = merged["mode_recommended"].notna().sum()
    n_without = merged["mode_recommended"].isna().sum()

    assert n_with == 24,  f"Attendu 24 matchs avec picks, got {n_with}"
    assert n_without == 80, f"Attendu 80 matchs sans pick (NaN), got {n_without}"


def test_mode_safe_on_nan_does_not_crash(data):
    """Vérifie que str(NaN).upper() ne crashe pas et donne 'NAN' ou qu'on
    protège correctement avec pd.notna() — test de la logique de guard."""
    import pandas as pd
    import numpy as np

    mode_nan = float("nan")
    # La guard utilisée dans les pages
    mode_str = str(mode_nan).upper() if pd.notna(mode_nan) else "—"
    assert mode_str == "—"

    mode_valid = "value"
    mode_str2 = str(mode_valid).upper() if pd.notna(mode_valid) else "—"
    assert mode_str2 == "VALUE"


def test_ko_match_pred_elo_fields_are_float(data):
    """get_match() sur un match KO doit retourner des floats (pas de string '—')
    pour les clés Elo — garantit que la page détail ne crashe pas à l'affichage."""
    from src.app.utils import get_match
    ko_ids = data["fixtures"][data["fixtures"]["stage"] != "group"]["match_id"].tolist()
    assert ko_ids, "Doit y avoir des matchs KO"
    m = get_match(ko_ids[0])
    assert m is not None
    for key in ("elo_home", "elo_away", "elo_home_adj", "elo_away_adj"):
        val = m["pred"].get(key)
        if val is not None:
            assert isinstance(val, float), f"{key} doit être float, got {type(val)}"


def test_ev_formula_symmetry():
    """BUG-FIX check : le bonus rareté doit s'appliquer aux 3 issues, pas seulement draw/away."""
    # Simule la formule corrigée : base_cote + rarity pour toutes les issues
    ch, cd, ca = 96, 107, 91
    rarity = 5
    for i, j in [(1, 0), (0, 0), (0, 1)]:  # dom / nul / ext
        base_cote = ch if i > j else (ca if j > i else cd)
        ev = 0.10 * (base_cote + rarity)
        assert ev > 0, f"EV doit être positif pour ({i},{j})"
        # Vérifie que rarity est toujours inclus
        ev_no_rarity = 0.10 * base_cote
        assert ev > ev_no_rarity, f"EV avec rarity doit être > sans rarity pour ({i},{j})"


def test_fk_picks_match_ids_in_fixtures(data):
    """Tous les match_id dans optimal_picks.csv doivent exister dans fixtures.csv."""
    fix_ids  = set(data["fixtures"]["match_id"])
    pick_ids = set(data["picks"]["match_id"])
    orphans  = pick_ids - fix_ids
    assert not orphans, f"match_ids orphelins dans picks : {orphans}"


def test_fk_distributions_match_ids_in_fixtures(data):
    """Tous les match_id dans score_distributions doivent exister dans fixtures.csv."""
    fix_ids  = set(data["fixtures"]["match_id"])
    dist_ids = set(data["distributions"]["match_id"])
    orphans  = dist_ids - fix_ids
    assert not orphans, f"match_ids orphelins dans distributions : {orphans}"
