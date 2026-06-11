"""Chargement centralisé et helpers pour l'app Mon Petit Prono."""
from pathlib import Path

import pandas as pd
import streamlit as st

PROCESSED = Path(__file__).parents[2] / "data" / "processed"
RAW       = Path(__file__).parents[2] / "data" / "raw"

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


@st.cache_data(ttl=3600)
def load_team_stats() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "team_stats.csv")


def get_match(match_id: int) -> dict | None:
    """Retourne un dict complet (fixture + prediction + picks + distributions) pour un match."""
    fix        = load_fixtures()
    pred       = load_predictions()
    picks      = load_picks()
    dist       = load_score_distributions()
    teams      = load_teams()
    team_stats = load_team_stats()

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

    def _team_info(team):
        row_t = teams.loc[teams["team"] == team]
        row_s = team_stats.loc[team_stats["team"] == team]
        conf = row_t["confederation"].values[0] if not row_t.empty else ""
        pot  = int(row_t["pot"].values[0]) if not row_t.empty and "pot" in row_t.columns else None
        xg   = float(row_s["xG_p90"].values[0]) if not row_s.empty and pd.notna(row_s["xG_p90"].values[0] if not row_s.empty else None) else None
        xga  = float(row_s["xGA_p90"].values[0]) if not row_s.empty and pd.notna(row_s["xGA_p90"].values[0] if not row_s.empty else None) else None
        mv   = float(row_s["squad_value_eur"].values[0]) if not row_s.empty and pd.notna(row_s["squad_value_eur"].values[0] if not row_s.empty else None) else None
        return conf, pot, xg, xga, mv

    home_conf, home_pot, home_xg, home_xga, home_mv = _team_info(home)
    away_conf, away_pot, away_xg, away_xga, away_mv = _team_info(away)

    return {
        "match_id":   match_id,
        "home":       home,
        "away":       away,
        "home_flag":  flag(home),
        "away_flag":  flag(away),
        "home_conf":  home_conf,
        "away_conf":  away_conf,
        "home_pot":   home_pot,
        "away_pot":   away_pot,
        "home_xg":    home_xg,
        "home_xga":   home_xga,
        "home_mv":    home_mv,
        "away_xg":    away_xg,
        "away_xga":   away_xga,
        "away_mv":    away_mv,
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


@st.cache_data(ttl=3600)
def load_mv_baseline() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "team_mv_baseline.csv")


@st.cache_data(ttl=300)
def load_results() -> pd.DataFrame:
    """
    Charge data/raw/wc2026_results.csv (scores réels saisis manuellement).
    Retourne un DataFrame vide si le fichier n'existe pas ou est vide.
    """
    path = RAW / "wc2026_results.csv"
    if not path.exists():
        return pd.DataFrame(columns=["match_id", "home_score", "away_score"])
    try:
        df = pd.read_csv(path)
        if df.empty or "match_id" not in df.columns:
            return pd.DataFrame(columns=["match_id", "home_score", "away_score"])
        df["match_id"]   = df["match_id"].astype(int)
        df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
        df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
        return df[df["home_score"].notna() & df["away_score"].notna()].copy()
    except Exception:
        return pd.DataFrame(columns=["match_id", "home_score", "away_score"])
