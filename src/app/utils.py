"""Chargement centralisé et helpers pour l'app Mon Petit Prono."""
from pathlib import Path

import pandas as pd
import streamlit as st

PROCESSED = Path(__file__).parents[2] / "data" / "processed"

STAGE_LABELS = {
    "group": "Groupes",
    "r32": "R32",
    "r16": "R16",
    "qf": "Quarts",
    "sf": "Demies",
    "3rd": "3e place",
    "final": "Finale",
}

STAGE_ORDER = ["group", "r32", "r16", "qf", "sf", "3rd", "final"]

FLAG: dict[str, str] = {
    "Mexico": "🇲🇽", "South Korea": "🇰🇷", "South Africa": "🇿🇦",
    "Czech Republic": "🇨🇿", "Canada": "🇨🇦", "Switzerland": "🇨🇭",
    "Qatar": "🇶🇦", "Bosnia and Herzegovina": "🇧🇦", "Brazil": "🇧🇷",
    "Morocco": "🇲🇦", "Haiti": "🇭🇹", "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿",
    "United States": "🇺🇸", "Paraguay": "🇵🇾", "Australia": "🇦🇺",
    "Turkey": "🇹🇷", "Germany": "🇩🇪", "Ecuador": "🇪🇨",
    "Ivory Coast": "🇨🇮", "Curaçao": "🇨🇼", "Netherlands": "🇳🇱",
    "Japan": "🇯🇵", "Sweden": "🇸🇪", "Tunisia": "🇹🇳",
    "Belgium": "🇧🇪", "Egypt": "🇪🇬", "Iran": "🇮🇷",
    "New Zealand": "🇳🇿", "Spain": "🇪🇸", "Uruguay": "🇺🇾",
    "Saudi Arabia": "🇸🇦", "Cape Verde": "🇨🇻", "France": "🇫🇷",
    "Senegal": "🇸🇳", "Iraq": "🇮🇶", "Norway": "🇳🇴",
    "Argentina": "🇦🇷", "Austria": "🇦🇹", "Algeria": "🇩🇿",
    "Jordan": "🇯🇴", "Portugal": "🇵🇹", "Colombia": "🇨🇴",
    "Uzbekistan": "🇺🇿", "DR Congo": "🇨🇩", "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    "Croatia": "🇭🇷", "Panama": "🇵🇦", "Ghana": "🇬🇭",
}


def flag(team: str) -> str:
    return FLAG.get(team, "🏳️")


def format_pct(x: float) -> str:
    return f"{x:.1%}"


def format_ev(x: float) -> str:
    return f"{x:.1f}"


def mpp_implied(cote_h: int, cote_d: int, cote_a: int) -> tuple[float, float, float]:
    s = 1 / cote_h + 1 / cote_d + 1 / cote_a
    return 1 / cote_h / s, 1 / cote_d / s, 1 / cote_a / s


# ── Loaders ──────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_fixtures() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED / "fixtures.csv")
    df["kickoff_dt"] = pd.to_datetime(df["kickoff_utc"], utc=True)
    df["date_local"] = df["kickoff_dt"].dt.tz_convert("Europe/Paris")
    return df


@st.cache_data(ttl=3600)
def load_predictions() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "predictions.csv")


@st.cache_data(ttl=3600)
def load_picks() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "optimal_picks.csv")


@st.cache_data(ttl=3600)
def load_score_distributions() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "score_distributions.csv")


@st.cache_data(ttl=3600)
def load_tournament_probabilities() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "tournament_probabilities.csv")


@st.cache_data(ttl=3600)
def load_teams() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "teams.csv")


@st.cache_data(ttl=3600)
def load_group_simulations() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "group_stage_simulations.csv")


@st.cache_data(ttl=3600)
def load_ko_predictions() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "ko_predictions.csv")


def load_data() -> dict:
    """Retourne tous les datasets en un dict (usage externe / tests)."""
    return {
        "fixtures":      load_fixtures(),
        "predictions":   load_predictions(),
        "picks":         load_picks(),
        "distributions": load_score_distributions(),
        "tp":            load_tournament_probabilities(),
        "teams":         load_teams(),
        "group_sims":    load_group_simulations(),
        "ko":            load_ko_predictions(),
    }


def get_match(match_id: int) -> dict | None:
    """Retourne un dict complet (fixture + prediction + picks + distributions) pour un match."""
    fix  = load_fixtures()
    pred = load_predictions()
    picks = load_picks()
    dist = load_score_distributions()
    teams = load_teams()

    fix_row = fix[fix["match_id"] == match_id]
    if fix_row.empty:
        return None
    fix_row = fix_row.iloc[0].to_dict()

    pred_row  = pred[pred["match_id"] == match_id]
    pred_dict = pred_row.iloc[0].to_dict() if not pred_row.empty else {}

    picks_row  = picks[picks["match_id"] == match_id]
    picks_dict = picks_row.iloc[0].to_dict() if not picks_row.empty else {}

    dist_mat = dist[dist["match_id"] == match_id].copy()

    home = fix_row.get("home_slot", "")
    away = fix_row.get("away_slot", "")

    conf_home = teams.loc[teams["team"] == home, "confederation"].values
    conf_away = teams.loc[teams["team"] == away, "confederation"].values

    return {
        "match_id":   match_id,
        "home":       home,
        "away":       away,
        "home_flag":  flag(home),
        "away_flag":  flag(away),
        "home_conf":  conf_home[0] if len(conf_home) else "",
        "away_conf":  conf_away[0] if len(conf_away) else "",
        "kickoff_dt": fix_row.get("kickoff_dt"),
        "date_local": fix_row.get("date_local"),
        "venue":      fix_row.get("venue", ""),
        "city":       fix_row.get("city", ""),
        "country":    fix_row.get("country", ""),
        "stage":      fix_row.get("stage", ""),
        "group":      fix_row.get("group", ""),
        "pred":       pred_dict,
        "picks":      picks_dict,
        "dist":       dist_mat,
    }


def model_update_time() -> str:
    path = PROCESSED / "predictions.csv"
    if path.exists():
        import datetime
        ts = datetime.datetime.fromtimestamp(path.stat().st_mtime)
        return ts.strftime("%d/%m/%Y %H:%M")
    return "—"


def load_my_score() -> int | None:
    path = PROCESSED / "my_score.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        return int(df["score"].sum())
    except Exception:
        return None


def load_my_winner_pick() -> str | None:
    path = PROCESSED / "my_winner_pick.txt"
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip() or None


def save_my_winner_pick(team: str) -> None:
    (PROCESSED / "my_winner_pick.txt").write_text(team, encoding="utf-8")
