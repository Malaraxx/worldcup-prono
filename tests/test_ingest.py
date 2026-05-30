"""
Tests d'intégration ingestion — Phase 0.
Les tests sur données réelles seront activés au fur et à mesure des étapes.
"""

import pytest
from pathlib import Path

PROCESSED = Path(__file__).parents[1] / "data" / "processed"


# ── Étape 2 : historique ──────────────────────────────────────────────────────

@pytest.mark.skipif(
    not (PROCESSED / "matches_historical.csv").exists(),
    reason="matches_historical.csv non généré (Étape 2)"
)
def test_matches_since_2010():
    import pandas as pd
    df = pd.read_csv(PROCESSED / "matches_historical.csv", parse_dates=["date"])
    assert (df["date"] >= "2010-01-01").all(), "Matchs antérieurs à 2010 présents"


@pytest.mark.skipif(
    not (PROCESSED / "matches_historical.csv").exists(),
    reason="matches_historical.csv non généré (Étape 2)"
)
def test_matches_min_count():
    import pandas as pd
    df = pd.read_csv(PROCESSED / "matches_historical.csv")
    assert len(df) >= 5000, f"Seulement {len(df)} matchs (minimum 5000)"


@pytest.mark.skipif(
    not (PROCESSED / "matches_historical.csv").exists(),
    reason="matches_historical.csv non généré (Étape 2)"
)
def test_match_weight_no_nan():
    import pandas as pd
    df = pd.read_csv(PROCESSED / "matches_historical.csv")
    assert df["match_weight"].notna().all(), "NaN dans match_weight"


@pytest.mark.skipif(
    not (PROCESSED / "matches_historical.csv").exists(),
    reason="matches_historical.csv non généré (Étape 2)"
)
def test_match_weight_distribution():
    import pandas as pd
    df = pd.read_csv(PROCESSED / "matches_historical.csv")
    friendly_avg = df[df["tournament"] == "Friendly"]["match_weight"].mean()
    qualif_avg = df[df["tournament"].str.contains("qualification", case=False, na=False)]["match_weight"].mean()
    wc_avg = df[df["tournament"] == "FIFA World Cup"]["match_weight"].mean()
    assert friendly_avg < qualif_avg < wc_avg, (
        f"Hiérarchie poids incorrecte : friendly={friendly_avg:.2f}, "
        f"qualif={qualif_avg:.2f}, worldcup={wc_avg:.2f}"
    )


# ── Étape 3 : équipes et fixtures ────────────────────────────────────────────

@pytest.mark.skipif(
    not (PROCESSED / "teams.csv").exists(),
    reason="teams.csv non généré (Étape 3)"
)
def test_48_teams_in_csv():
    import pandas as pd
    df = pd.read_csv(PROCESSED / "teams.csv")
    assert len(df) == 48, f"{len(df)} équipes (attendu 48)"


@pytest.mark.skipif(
    not (PROCESSED / "teams.csv").exists(),
    reason="teams.csv non généré (Étape 3)"
)
def test_12_groups_4_teams_each():
    import pandas as pd
    df = pd.read_csv(PROCESSED / "teams.csv")
    groups = df.groupby("group").size()
    assert len(groups) == 12, f"{len(groups)} groupes (attendu 12)"
    assert (groups == 4).all(), f"Groupes non équilibrés : {groups.to_dict()}"


@pytest.mark.skipif(
    not (PROCESSED / "fixtures.csv").exists(),
    reason="fixtures.csv non généré (Étape 3)"
)
def test_fixtures_104_total():
    import pandas as pd
    df = pd.read_csv(PROCESSED / "fixtures.csv")
    assert len(df) == 104, f"{len(df)} matchs (attendu 104)"


@pytest.mark.skipif(
    not (PROCESSED / "fixtures.csv").exists(),
    reason="fixtures.csv non généré (Étape 3)"
)
def test_fixtures_72_group_stage():
    import pandas as pd
    df = pd.read_csv(PROCESSED / "fixtures.csv")
    group_matches = df[df["stage"] == "group"]
    assert len(group_matches) == 72, f"{len(group_matches)} matchs de poule (attendu 72)"


@pytest.mark.skipif(
    not (PROCESSED / "fixtures.csv").exists(),
    reason="fixtures.csv non généré (Étape 3)"
)
def test_fixtures_32_knockout():
    import pandas as pd
    df = pd.read_csv(PROCESSED / "fixtures.csv")
    knockout = df[df["stage"] != "group"]
    assert len(knockout) == 32, f"{len(knockout)} matchs à élimination directe (attendu 32)"


# ── Étape 4 : effectifs ──────────────────────────────────────────────────────

@pytest.mark.skipif(
    not (PROCESSED / "players.csv").exists(),
    reason="players.csv non généré (Étape 4)"
)
def test_players_per_team_in_range():
    import pandas as pd
    df = pd.read_csv(PROCESSED / "players.csv")
    counts = df.groupby("team").size()
    assert (counts >= 23).all(), f"Équipes avec < 23 joueurs : {counts[counts < 23].to_dict()}"
    assert (counts <= 60).all(), f"Équipes avec > 60 joueurs : {counts[counts > 60].to_dict()}"


# ── Étape 5 : valeurs marchandes ─────────────────────────────────────────────

@pytest.mark.skipif(
    not (PROCESSED / "players.csv").exists(),
    reason="players.csv non généré (Étape 5)"
)
def test_market_value_coverage():
    import pandas as pd
    df = pd.read_csv(PROCESSED / "players.csv")
    if "market_value_eur" not in df.columns:
        pytest.skip("Colonne market_value_eur absente (Étape 5 non faite)")
    coverage = df["market_value_eur"].notna().mean()
    assert coverage >= 0.80, f"Couverture market_value_eur : {coverage:.1%} (minimum 80%)"


# ── Étape 6 : club_stats FBref ───────────────────────────────────────────────

@pytest.mark.skipif(
    not (PROCESSED / "club_stats.csv").exists(),
    reason="club_stats.csv non généré (Étape 6)"
)
def test_club_stats_96_rows():
    import pandas as pd
    df = pd.read_csv(PROCESSED / "club_stats.csv")
    assert len(df) == 96, f"{len(df)} lignes dans club_stats.csv (attendu 96)"


@pytest.mark.skipif(
    not (PROCESSED / "club_stats.csv").exists(),
    reason="club_stats.csv non généré (Étape 6)"
)
def test_club_stats_no_nan_league():
    import pandas as pd
    df = pd.read_csv(PROCESSED / "club_stats.csv")
    assert df["league"].notna().all(), "NaN dans la colonne league de club_stats"


@pytest.mark.skipif(
    not (PROCESSED / "club_stats.csv").exists(),
    reason="club_stats.csv non généré (Étapes 6-7)"
)
def test_club_stats_xg_coverage():
    import pandas as pd
    df = pd.read_csv(PROCESSED / "club_stats.csv")
    if "xG_p90" not in df.columns:
        pytest.skip("Colonne xG_p90 absente (Étape 7 non faite)")
    coverage = df["xG_p90"].notna().mean()
    assert coverage >= 0.95, f"Couverture xG_p90 : {coverage:.1%} (minimum 95%)"


# ── Étape 10 : team_stats ────────────────────────────────────────────────────

@pytest.mark.skipif(
    not (PROCESSED / "team_stats.csv").exists(),
    reason="team_stats.csv non généré (Étape 10)"
)
def test_team_stats_48_rows():
    import pandas as pd
    df = pd.read_csv(PROCESSED / "team_stats.csv")
    assert len(df) == 48, f"{len(df)} lignes dans team_stats.csv (attendu 48)"


@pytest.mark.skipif(
    not (PROCESSED / "team_stats.csv").exists(),
    reason="team_stats.csv non généré (Étape 10)"
)
def test_team_stats_required_cols():
    import pandas as pd
    df = pd.read_csv(PROCESSED / "team_stats.csv")
    required = ["team", "group", "n_players", "squad_value_eur"]
    missing = [c for c in required if c not in df.columns]
    assert not missing, f"Colonnes manquantes dans team_stats : {missing}"
