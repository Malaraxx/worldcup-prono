"""
Tests pour scripts/update_tournament.py
Couvrent : validate_results, inject_results, backup, sens Elo.
"""

import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

# Chemins
ROOT      = Path(__file__).parents[1]
PROCESSED = ROOT / "data" / "processed"
RAW       = ROOT / "data" / "raw"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from update_tournament import (  # noqa: E402
    backup_matches_historical,
    inject_results,
    validate_results,
)


# ── Fixtures pytest ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def fixtures_df():
    return pd.read_csv(PROCESSED / "fixtures.csv")


@pytest.fixture(scope="module")
def group_fixtures(fixtures_df):
    return fixtures_df[fixtures_df["stage"] == "group"].copy()


@pytest.fixture
def restore_hist():
    """Backup matches_historical.csv avant le test, restaure après."""
    src    = PROCESSED / "matches_historical.csv"
    backup = src.with_name("matches_historical._pytest_backup")
    shutil.copy2(src, backup)
    yield
    shutil.copy2(backup, src)
    backup.unlink(missing_ok=True)


# ── validate_results ──────────────────────────────────────────────────────────

def test_validate_file_not_found(tmp_path, fixtures_df):
    with pytest.raises(SystemExit):
        validate_results(tmp_path / "nonexistent.csv", fixtures_df)


def test_validate_empty_returns_empty(tmp_path, fixtures_df):
    p = tmp_path / "results.csv"
    p.write_text("match_id,home_score,away_score\n")
    df = validate_results(p, fixtures_df)
    assert df.empty


def test_validate_valid_rows(tmp_path, fixtures_df, group_fixtures):
    mids = group_fixtures["match_id"].head(3).tolist()
    p = tmp_path / "results.csv"
    rows = "\n".join(f"{mid},1,0" for mid in mids)
    p.write_text(f"match_id,home_score,away_score\n{rows}\n")
    df = validate_results(p, fixtures_df)
    assert len(df) == 3
    assert list(df.dtypes[["home_score", "away_score"]]) == [
        pd.api.types.pandas_dtype("int64"),
        pd.api.types.pandas_dtype("int64"),
    ]


def test_validate_unknown_match_id(tmp_path, fixtures_df):
    p = tmp_path / "results.csv"
    p.write_text("match_id,home_score,away_score\n9999,1,0\n")
    with pytest.raises(SystemExit):
        validate_results(p, fixtures_df)


def test_validate_negative_score(tmp_path, fixtures_df, group_fixtures):
    mid = int(group_fixtures.iloc[0]["match_id"])
    p = tmp_path / "results.csv"
    p.write_text(f"match_id,home_score,away_score\n{mid},-1,0\n")
    with pytest.raises(SystemExit):
        validate_results(p, fixtures_df)


def test_validate_duplicate_match_id(tmp_path, fixtures_df, group_fixtures):
    mid = int(group_fixtures.iloc[0]["match_id"])
    p = tmp_path / "results.csv"
    p.write_text(f"match_id,home_score,away_score\n{mid},1,0\n{mid},2,1\n")
    with pytest.raises(SystemExit):
        validate_results(p, fixtures_df)


def test_validate_non_integer_score(tmp_path, fixtures_df, group_fixtures):
    mid = int(group_fixtures.iloc[0]["match_id"])
    p = tmp_path / "results.csv"
    p.write_text(f"match_id,home_score,away_score\n{mid},abc,0\n")
    with pytest.raises(SystemExit):
        validate_results(p, fixtures_df)


def test_validate_missing_column(tmp_path, fixtures_df, group_fixtures):
    mid = int(group_fixtures.iloc[0]["match_id"])
    p = tmp_path / "results.csv"
    p.write_text(f"match_id,home_score\n{mid},1\n")
    with pytest.raises(SystemExit):
        validate_results(p, fixtures_df)


# ── backup_matches_historical ─────────────────────────────────────────────────

def test_backup_creates_file():
    bak = backup_matches_historical()
    try:
        assert bak.exists(), f"Fichier backup non créé : {bak}"
        assert bak.suffix == ".bak"
        # Contenu identique à l'original
        src_content = (PROCESSED / "matches_historical.csv").read_text(encoding="utf-8")
        bak_content = bak.read_text(encoding="utf-8")
        assert src_content == bak_content
    finally:
        bak.unlink(missing_ok=True)


# ── inject_results ────────────────────────────────────────────────────────────

def test_inject_two_group_results(restore_hist, fixtures_df, group_fixtures):
    rows = group_fixtures.head(2)
    results_df = pd.DataFrame({
        "match_id":   rows["match_id"].tolist(),
        "home_score": [2, 1],
        "away_score": [1, 0],
    })

    report, n_changed = inject_results(results_df, fixtures_df)

    assert n_changed == 2
    assert all(r["status"] == "injected" for r in report)

    # Vérifier que les scores sont bien dans matches_historical.csv
    hist = pd.read_csv(PROCESSED / "matches_historical.csv")
    for i, frow in rows.reset_index(drop=True).iterrows():
        mask = (
            (hist["home_team"] == frow["home_slot"]) &
            (hist["away_team"] == frow["away_slot"]) &
            (hist["tournament"] == "FIFA World Cup") &
            (hist["date"] >= "2026-06-01")
        )
        matched = hist[mask]
        assert len(matched) == 1
        expected_hs = results_df.iloc[i]["home_score"]
        expected_as = results_df.iloc[i]["away_score"]
        assert int(float(matched.iloc[0]["home_score"])) == expected_hs
        assert int(float(matched.iloc[0]["away_score"])) == expected_as


def test_inject_idempotent(restore_hist, fixtures_df, group_fixtures):
    row = group_fixtures.head(1)
    results_df = pd.DataFrame({
        "match_id":   row["match_id"].tolist(),
        "home_score": [3],
        "away_score": [2],
    })

    _, n1 = inject_results(results_df, fixtures_df)
    assert n1 == 1, "Premier appel : 1 ligne doit être modifiée"

    _, n2 = inject_results(results_df, fixtures_df)
    assert n2 == 0, "Deuxième appel avec mêmes données : aucune modification"

    # Le fichier est identique entre les deux lectures
    hist_after = pd.read_csv(PROCESSED / "matches_historical.csv")
    frow = row.iloc[0]
    mask = (
        (hist_after["home_team"] == frow["home_slot"]) &
        (hist_after["away_team"] == frow["away_slot"]) &
        (hist_after["tournament"] == "FIFA World Cup") &
        (hist_after["date"] >= "2026-06-01")
    )
    assert int(float(hist_after[mask].iloc[0]["home_score"])) == 3
    assert int(float(hist_after[mask].iloc[0]["away_score"])) == 2


def test_inject_empty_df_returns_zero(fixtures_df):
    empty = pd.DataFrame(columns=["match_id", "home_score", "away_score"])
    report, n = inject_results(empty, fixtures_df)
    assert n == 0
    assert report == []


def test_inject_overwrite_updates(restore_hist, fixtures_df, group_fixtures):
    row = group_fixtures.head(1)
    mid = int(row.iloc[0]["match_id"])

    df1 = pd.DataFrame({"match_id": [mid], "home_score": [1], "away_score": [0]})
    df2 = pd.DataFrame({"match_id": [mid], "home_score": [2], "away_score": [1]})

    _, n1 = inject_results(df1, fixtures_df)
    assert n1 == 1

    _, n2 = inject_results(df2, fixtures_df)
    assert n2 == 1  # overwrite avec nouveau score

    hist = pd.read_csv(PROCESSED / "matches_historical.csv")
    frow = row.iloc[0]
    mask = (
        (hist["home_team"] == frow["home_slot"]) &
        (hist["away_team"] == frow["away_slot"]) &
        (hist["tournament"] == "FIFA World Cup") &
        (hist["date"] >= "2026-06-01")
    )
    assert int(float(hist[mask].iloc[0]["home_score"])) == 2
    assert int(float(hist[mask].iloc[0]["away_score"])) == 1


# ── Sens Elo ──────────────────────────────────────────────────────────────────

def test_elo_moves_correct_direction(fixtures_df):
    """
    Injecter un résultat surprise (outsider gagne) doit :
    - augmenter l'Elo de l'outsider (away team, South Africa)
    - diminuer l'Elo du favori (home team, Mexico)
    Utilise compute_elo() directement, sans I/O sur elo_ratings.csv.
    """
    from src.model.elo import compute_elo

    hist = pd.read_csv(PROCESSED / "matches_historical.csv", parse_dates=["date"])
    hist["home_score"] = pd.to_numeric(hist["home_score"], errors="coerce")
    hist["away_score"] = pd.to_numeric(hist["away_score"], errors="coerce")

    # Recherche Mexico vs South Africa (match 1 du tournoi)
    mask_fix = (
        (fixtures_df["home_slot"] == "Mexico") &
        (fixtures_df["away_slot"] == "South Africa")
    )
    if not mask_fix.any():
        pytest.skip("Mexico vs South Africa introuvable dans fixtures.csv")

    baseline = compute_elo(hist)

    # Inject South Africa 1-0 Mexico (surprise — Mexico est favori)
    hist_mod = hist.copy()
    mask_hist = (
        (hist_mod["home_team"] == "Mexico") &
        (hist_mod["away_team"] == "South Africa") &
        (hist_mod["tournament"] == "FIFA World Cup") &
        (hist_mod["date"] >= "2026-06-01")
    )
    if not mask_hist.any():
        pytest.skip("Ligne Mexico vs South Africa introuvable dans matches_historical.csv")

    hist_mod.loc[mask_hist, "home_score"] = 0.0
    hist_mod.loc[mask_hist, "away_score"] = 1.0

    new_ratings = compute_elo(hist_mod)

    sa_base = baseline.get("South Africa", 1500.0)
    sa_new  = new_ratings.get("South Africa", 1500.0)
    mx_base = baseline.get("Mexico", 1500.0)
    mx_new  = new_ratings.get("Mexico", 1500.0)

    assert sa_new > sa_base, (
        f"South Africa devrait gagner de l'Elo (victoire surprise) : "
        f"{sa_base:.1f} → {sa_new:.1f}"
    )
    assert mx_new < mx_base, (
        f"Mexico devrait perdre de l'Elo (défaite surprise) : "
        f"{mx_base:.1f} → {mx_new:.1f}"
    )
