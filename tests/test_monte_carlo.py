"""Tests Phase 2 : Monte-Carlo WC2026."""

import pytest
from pathlib import Path

PROCESSED = Path(__file__).parents[1] / "data" / "processed"
TP_PATH   = PROCESSED / "tournament_probabilities.csv"
GS_PATH   = PROCESSED / "group_stage_simulations.csv"
KO_PATH   = PROCESSED / "ko_predictions.csv"

requires_mc = pytest.mark.skipif(
    not TP_PATH.exists(),
    reason="tournament_probabilities.csv non généré — lancer src/simulation/monte_carlo.py"
)


@requires_mc
def test_proba_r32_sums_to_32():
    """La somme des proba_r32 sur les 48 équipes doit être 32 (±0.5)."""
    import pandas as pd
    df = pd.read_csv(TP_PATH)
    total = df["proba_r32"].sum()
    assert abs(total - 32) < 0.5, f"Somme proba_r32 = {total:.3f} (attendu 32)"


@requires_mc
def test_proba_winner_sums_to_one():
    """La somme des proba_winner sur les 48 équipes doit être 1 (±0.01)."""
    import pandas as pd
    df = pd.read_csv(TP_PATH)
    total = df["proba_winner"].sum()
    assert abs(total - 1.0) < 0.01, f"Somme proba_winner = {total:.4f} (attendu 1.0)"


@requires_mc
def test_host_nations_r32_proba():
    """Les 3 pays hôtes (USA, Canada, Mexique) ont chacun proba_r32 > 50%."""
    import pandas as pd
    df = pd.read_csv(TP_PATH)
    hosts = {"United States": "USA", "Canada": "Canada", "Mexico": "Mexique"}
    for team, label in hosts.items():
        row = df[df["team"] == team]
        assert not row.empty, f"{team} introuvable dans tournament_probabilities.csv"
        p = row.iloc[0]["proba_r32"]
        assert p > 0.50, f"{label} proba_r32 = {p:.1%} (attendu > 50%)"


@requires_mc
def test_elite_teams_in_top_winner():
    """Le top 3 proba_winner contient au moins 2 équipes parmi Spain/Argentina/France/Brazil/England."""
    import pandas as pd
    df = pd.read_csv(TP_PATH)
    elite = {"Spain", "Argentina", "France", "Brazil", "England"}
    top3  = set(df.nlargest(3, "proba_winner")["team"])
    found = elite & top3
    assert len(found) >= 2, (
        f"Seulement {len(found)} équipe(s) élite dans le top 3 : {top3}"
    )
