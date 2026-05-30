"""Mapping noms d'équipes MPP → noms FIFA officiels WC2026."""
import logging

logger = logging.getLogger(__name__)

MPP_TO_FIFA: dict[str, str] = {
    "Côte d'Ivoire": "Ivory Coast",
    "RD Congo":      "DR Congo",
    "USA":           "United States",
    "Czechia":       "Czech Republic",
    "Bosnia":        "Bosnia and Herzegovina",
}


def normalize(name: str) -> str:
    """Retourne le nom FIFA correspondant au nom MPP (identité si absent du dict)."""
    return MPP_TO_FIFA.get(name, name)


def check_against_teams(mpp_names: list[str], teams_set: set[str]) -> None:
    """Log un warning pour chaque équipe MPP introuvable dans teams_set après mapping."""
    for name in mpp_names:
        mapped = normalize(name)
        if mapped not in teams_set:
            logger.warning("MPP team '%s' → '%s' : non trouvée dans teams", name, mapped)
