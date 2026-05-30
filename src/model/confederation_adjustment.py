"""
Ajustement Elo par confédération calibré sur les WC 2010-2022.

Les confédérations UEFA et CONMEBOL servent de référence (ajustement = 0.0).
Les autres confédérations reçoivent un décalage négatif si elles sous-performent
leur prédiction Elo en phase finale mondiale (biais connu : Elo gonflé par des
qualifications contre adversaires plus faibles de la même confédération).

L'ajustement est appliqué UNIQUEMENT aux fixtures WC2026, pas au calcul Elo historique.
"""
import logging
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.model.elo import DEFAULT_RATING, K_BASE, HOME_ADV_ELO, _expected

logger = logging.getLogger(__name__)

PROCESSED = Path(__file__).parents[2] / "data" / "processed"

# WC2026 hosts : exemptés de l'ajustement CONCACAF
WC2026_HOSTS: set[str] = {"United States", "Canada", "Mexico"}

# Confederations des équipes présentes en WC 2010-2022 mais absentes du WC2026
_HISTORICAL_CONF: dict[str, str] = {
    "Cameroon":    "CAF",
    "Chile":       "CONMEBOL",
    "Costa Rica":  "CONCACAF",
    "Denmark":     "UEFA",
    "Greece":      "UEFA",
    "Honduras":    "CONCACAF",
    "Iceland":     "UEFA",
    "Italy":       "UEFA",
    "Nigeria":     "CAF",
    "North Korea": "AFC",
    "Peru":        "CONMEBOL",
    "Poland":      "UEFA",
    "Russia":      "UEFA",
    "Serbia":      "UEFA",
    "Slovakia":    "UEFA",
    "Slovenia":    "UEFA",
    "Wales":       "UEFA",
}

_CONFEDERATIONS = ["UEFA", "CONMEBOL", "AFC", "CAF", "CONCACAF", "OFC"]
_MIN_MATCHES    = 8
# OFC : seulement 3 matchs cross-conf. en WC 2010-2022 (NZ 2010).
# -30 pts bruts → après normalisation UEFA/CONMEBOL ≈ -69 pts finaux.
# Valeur heuristique conservatrice : mi-chemin entre AFC (-108) et UEFA (0).
_FALLBACK_ADJ   = -30.0


def _build_confederation_map() -> dict[str, str]:
    """Retourne {team: confederation} pour toutes les équipes WC 2010-2026."""
    teams_df = pd.read_csv(PROCESSED / "teams.csv")
    conf_map = dict(zip(teams_df["team"], teams_df["confederation"]))
    conf_map.update(_HISTORICAL_CONF)
    return conf_map


def _replay_wc_snapshots(matches: pd.DataFrame) -> list[dict]:
    """
    Rejoue le calcul Elo sur tout l'historique (même algorithme que compute_elo).
    Capture les ratings pré-match pour chaque WC 2010-2022.
    """
    ratings: dict[str, float] = defaultdict(lambda: DEFAULT_RATING)
    snapshots: list[dict] = []

    for _, row in matches.sort_values("date").iterrows():
        home = row["home_team"]
        away = row["away_team"]
        hs   = row["home_score"]
        as_  = row["away_score"]

        if pd.isna(hs) or pd.isna(as_):
            continue

        neutral = bool(row.get("neutral", False))
        adv     = 0.0 if neutral else HOME_ADV_ELO
        k       = K_BASE * float(row.get("match_weight", 1.0))
        year    = pd.Timestamp(row["date"]).year

        # Snapshot pré-match (avant update Elo) pour les WC 2010-2022
        if row.get("tournament") == "FIFA World Cup" and 2010 <= year <= 2022:
            snapshots.append({
                "home":       home,
                "away":       away,
                "elo_home":   float(ratings[home]),
                "elo_away":   float(ratings[away]),
                "home_score": float(hs),
                "away_score": float(as_),
            })

        exp    = _expected(ratings[home], ratings[away], adv)
        result = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
        delta  = k * (result - exp)
        ratings[home] += delta
        ratings[away] -= delta

    return snapshots


def compute_confederation_adjustments() -> dict[str, float]:
    """
    Calcule ELO_ADJUSTMENT par confédération (en points Elo).

    Méthode :
      1. Rejouer l'Elo sur tout l'historique, capturer snapshots WC 2010-2022.
      2. Pour chaque match cross-confédération : delta = actual - expected.
      3. Biais par conf = moyenne des deltas.
      4. Conversion en Elo : 400 * log10((0.5+b) / (0.5-b)).
      5. Normalisation : UEFA=0, CONMEBOL=0 (reference).

    < _MIN_MATCHES matchs cross-conf → _FALLBACK_ADJ + warning.
    """
    matches = pd.read_csv(PROCESSED / "matches_historical.csv", parse_dates=["date"])
    matches["home_score"] = pd.to_numeric(matches["home_score"], errors="coerce")
    matches["away_score"] = pd.to_numeric(matches["away_score"], errors="coerce")
    matches = matches.dropna(subset=["home_score", "away_score"])

    conf_map  = _build_confederation_map()
    snapshots = _replay_wc_snapshots(matches)

    conf_deltas: dict[str, list[float]] = defaultdict(list)

    for snap in snapshots:
        conf_h = conf_map.get(snap["home"])
        conf_a = conf_map.get(snap["away"])

        if conf_h is None:
            logger.warning("Confédération inconnue pour '%s' — ignoré", snap["home"])
            continue
        if conf_a is None:
            logger.warning("Confédération inconnue pour '%s' — ignoré", snap["away"])
            continue
        if conf_h == conf_a:
            continue

        # WC = terrain neutre → home_adv = 0
        exp_h    = _expected(snap["elo_home"], snap["elo_away"], 0.0)
        actual_h = (1.0 if snap["home_score"] > snap["away_score"]
                    else 0.5 if snap["home_score"] == snap["away_score"] else 0.0)
        delta_h  = actual_h - exp_h

        conf_deltas[conf_h].append(delta_h)
        conf_deltas[conf_a].append(-delta_h)

    raw_adj: dict[str, float] = {}
    for conf in _CONFEDERATIONS:
        deltas = conf_deltas.get(conf, [])
        n = len(deltas)
        if n < _MIN_MATCHES:
            logger.warning(
                "Confédération %s : %d matchs cross-conf < %d — ajustement conservatif %.0f pts",
                conf, n, _MIN_MATCHES, _FALLBACK_ADJ,
            )
            raw_adj[conf] = _FALLBACK_ADJ
        else:
            b = sum(deltas) / n
            b = max(-0.48, min(0.48, b))
            raw_adj[conf] = 400.0 * math.log10((0.5 + b) / (0.5 - b))
            logger.info(
                "Confédération %s : n=%d, biais=%.4f → raw_adj=%.1f pts",
                conf, n, b, raw_adj[conf],
            )

    # Normaliser : référence = moyenne UEFA + CONMEBOL
    reference = (raw_adj["UEFA"] + raw_adj["CONMEBOL"]) / 2.0
    adj = {conf: round(v - reference, 1) for conf, v in raw_adj.items()}

    # Forcer références à exactement 0.0
    adj["UEFA"]     = 0.0
    adj["CONMEBOL"] = 0.0

    logger.info("ELO_ADJUSTMENT final : %s", adj)
    return adj


def get_adjusted_elo(
    team: str,
    base_elo: float,
    confederation: str,
    adjustments: dict[str, float],
) -> float:
    """
    Retourne l'Elo ajusté pour la phase finale mondiale.
    Les hôtes WC2026 (USA, Canada, Mexico) sont exemptés de l'ajustement CONCACAF.
    """
    if team in WC2026_HOSTS:
        return base_elo
    return base_elo + adjustments.get(confederation, 0.0)
