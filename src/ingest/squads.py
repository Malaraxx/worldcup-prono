"""
Scraping effectifs des 48 équipes depuis Wikipedia.

Primaire  : https://en.wikipedia.org/wiki/{Team}_at_the_2026_FIFA_World_Cup
Fallback  : https://en.wikipedia.org/wiki/{Team}_national_football_team  (provisional=True)

Parsing via l'API Wikipedia (wikitext brut) → templates {{nat fs g player|...}}.
"""

import json
import logging
import re
import uuid
from datetime import date
from pathlib import Path
from urllib.parse import quote

import pandas as pd

from .http_utils import CachedSession
from .mappings import CLUBNAT_TO_LEAGUE, WIKI_NATIONAL_TEAM_PAGE

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"
WIKI_API      = "https://en.wikipedia.org/w/api.php"
WC_START      = date(2026, 6, 11)

PLAYER_COLS = [
    "player_id", "name", "team", "position",
    "shirt_number", "dob", "age",
    "caps", "intl_goals",
    "club", "league", "provisional",
]


# ── Helpers Wikipedia API ────────────────────────────────────────────────────

def _wiki_page_title_wc(team: str) -> str:
    return quote(team.replace(" ", "_") + "_at_the_2026_FIFA_World_Cup", safe="")


def _wiki_page_title_national(team: str) -> str:
    override = WIKI_NATIONAL_TEAM_PAGE.get(team)
    if override:
        return override
    return quote(team.replace(" ", "_") + "_national_football_team", safe="")


def _api_get(session: CachedSession, params: dict) -> dict | None:
    parts = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{WIKI_API}?{parts}"
    raw = session.get(url)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except Exception as exc:
        logger.warning("JSON parse error for %s: %s", url, exc)
        return None


def _find_squad_section(session: CachedSession, page_title: str) -> str | None:
    """Retourne l'index de section 'Current squad' ou None."""
    data = _api_get(session, {
        "action": "parse",
        "page": page_title,
        "prop": "sections",
        "format": "json",
    })
    if data is None or "error" in data:
        return None
    sections = data.get("parse", {}).get("sections", [])
    for sec in sections:
        line = sec.get("line", "").lower()
        if "current squad" in line or line == "squad":
            return sec["index"]
    return None


def _fetch_wikitext(session: CachedSession, page_title: str, section_idx: str) -> str | None:
    data = _api_get(session, {
        "action": "parse",
        "page": page_title,
        "section": section_idx,
        "prop": "wikitext",
        "format": "json",
    })
    if data is None or "error" in data:
        return None
    return data.get("parse", {}).get("wikitext", {}).get("*")


# ── Parsing wikitext ─────────────────────────────────────────────────────────

def _extract_player_templates(wikitext: str) -> list[str]:
    """Extrait tous les blocs {{nat fs g player|...}} en gérant les templates imbriqués."""
    results = []
    pattern = re.compile(r"\{\{nat\s+fs\s+g\s+player\b", re.IGNORECASE)
    for m in pattern.finditer(wikitext):
        start = m.start()
        depth = 0
        i = start
        while i < len(wikitext) - 1:
            chunk = wikitext[i: i + 2]
            if chunk == "{{":
                depth += 1
                i += 2
            elif chunk == "}}":
                depth -= 1
                i += 2
                if depth == 0:
                    results.append(wikitext[start:i])
                    break
            else:
                i += 1
    return results


def _get_field(template: str, field: str) -> str:
    """Extrait la valeur d'un champ, en gérant les templates imbriqués."""
    pat = re.compile(rf"\|{re.escape(field)}=", re.IGNORECASE)
    m = pat.search(template)
    if not m:
        return ""
    start = m.end()
    depth = 0
    i = start
    while i < len(template) - 1:
        chunk = template[i: i + 2]
        if chunk in ("{{", "[["):
            depth += 1
            i += 2
        elif chunk in ("}}", "]]"):
            depth -= 1
            if depth < 0:
                break
            i += 2
        elif template[i] == "|" and depth == 0:
            break
        else:
            i += 1
    return template[start:i].strip()


def _clean_wikilink(raw: str) -> str:
    """[[Page|Display]] ou [[Page]] → Display ou Page."""
    m = re.search(r"\[\[(?:[^\]|]+\|)?([^\]|]+)\]\]", raw)
    if m:
        return m.group(1).strip()
    # Supprimer les éventuels templates restants
    cleaned = re.sub(r"\{\{[^}]+\}\}", "", raw).strip()
    return cleaned or raw.strip()


def _extract_dob(template: str) -> str | None:
    """Extrait la date de naissance depuis {{bda|Y|M|D}} ou {{birth date and age|df=y|Y|M|D}}."""
    m = re.search(
        r"\{\{(?:bda|birth[\s_]date[\s_]and[\s_]age)\b[^}]*?"
        r"\|(?:df=[a-z]+\|)?(\d{4})\|(\d{1,2})\|(\d{1,2})",
        template,
        re.IGNORECASE,
    )
    if m:
        y, mo, d = m.groups()
        return f"{y}-{int(mo):02d}-{int(d):02d}"
    return None


def _compute_age(dob_str: str | None) -> int | None:
    if not dob_str:
        return None
    try:
        dob = date.fromisoformat(dob_str)
        delta = WC_START - dob
        return int(delta.days / 365.25)
    except Exception:
        return None


def _parse_player(template: str, team: str, provisional: bool) -> dict | None:
    name_raw = _get_field(template, "name")
    name = _clean_wikilink(name_raw) if name_raw else None
    if not name:
        return None

    pos_raw = _get_field(template, "pos").upper()
    pos_map = {"GK": "GK", "DF": "DF", "MF": "MF", "FW": "FW",
               "G": "GK", "D": "DF", "M": "MF", "F": "FW"}
    position = pos_map.get(pos_raw, pos_raw or None)

    dob = _extract_dob(template)

    club_raw  = _get_field(template, "club")
    club_name = _clean_wikilink(club_raw) if club_raw else None
    clubnat   = _get_field(template, "clubnat").strip().upper()
    league    = CLUBNAT_TO_LEAGUE.get(clubnat)

    caps_raw  = _get_field(template, "caps")
    goals_raw = _get_field(template, "goals")

    def _to_int(s: str) -> int | None:
        try:
            return int(s.strip())
        except (ValueError, AttributeError):
            return None

    return {
        "player_id":   str(uuid.uuid4()),
        "name":        name,
        "team":        team,
        "position":    position,
        "shirt_number": _to_int(_get_field(template, "no")),
        "dob":         dob,
        "age":         _compute_age(dob),
        "caps":        _to_int(caps_raw),
        "intl_goals":  _to_int(goals_raw),
        "club":        club_name,
        "league":      league,
        "provisional": provisional,
    }


# ── Scraping par équipe ───────────────────────────────────────────────────────

def _squad_for_team(team: str, session: CachedSession) -> list[dict]:
    """
    Tente primaire (WC2026) puis fallback (national).
    Retourne liste de dicts joueurs, [] si rien trouvé.
    """
    # 1. Tentative page WC2026
    for title, provisional in [
        (_wiki_page_title_wc(team), False),
        (_wiki_page_title_national(team), True),
    ]:
        section = _find_squad_section(session, title)
        if section is None:
            logger.debug("Pas de section squad : %s", title)
            continue
        wikitext = _fetch_wikitext(session, title, section)
        if not wikitext:
            logger.debug("Wikitext vide : %s (section %s)", title, section)
            continue

        templates = _extract_player_templates(wikitext)
        if not templates:
            logger.debug("Aucun template joueur : %s", title)
            continue

        players = []
        for tmpl in templates:
            p = _parse_player(tmpl, team, provisional)
            if p:
                players.append(p)

        if players:
            src = "WC2026" if not provisional else "national team (provisional)"
            logger.info("%-30s → %2d joueurs (%s)", team, len(players), src)
            return players

    logger.warning("%-30s → aucun effectif trouvé", team)
    return []


# ── Entrées publiques ─────────────────────────────────────────────────────────

def load_squads(teams: list[str]) -> pd.DataFrame:
    """
    Scrape les effectifs des 48 équipes depuis Wikipedia.
    Retourne DataFrame avec colonnes PLAYER_COLS.
    """
    session = CachedSession()
    all_players: list[dict] = []
    for team in teams:
        players = _squad_for_team(team, session)
        all_players.extend(players)

    if not all_players:
        logger.warning("Aucun effectif récupéré — retour DataFrame vide")
        return pd.DataFrame(columns=PLAYER_COLS)

    df = pd.DataFrame(all_players, columns=PLAYER_COLS)
    logger.info("Total : %d joueurs pour %d équipes", len(df), df["team"].nunique())
    return df


def save_all(teams: list[str]) -> pd.DataFrame:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = load_squads(teams)
    out = PROCESSED_DIR / "players.csv"
    df.to_csv(out, index=False)
    logger.info("Sauvegardé : %s (%d lignes)", out, len(df))
    return df


if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
        stream=sys.stdout,
    )
    from .wc2026 import build_teams_df
    teams = build_teams_df()["team"].tolist()
    df = save_all(teams)
    print(f"\n=== players.csv ({len(df)} joueurs) — 5 premières lignes ===")
    print(df.head(5).to_string(index=False))
