"""
Fallback Flashscore — MODULE DORMANT.

À activer MANUELLEMENT uniquement si API-Football tombe pendant le tournoi.
Voir CLAUDE.md section "Flashscore".

Ne PAS appeler ces fonctions en production sans activation explicite.
"""


def get_match_score(team_a: str, team_b: str, date: str) -> dict:
    """Retourne le score final d'un match depuis Flashscore."""
    raise NotImplementedError(
        "Module Flashscore dormant. Activer manuellement si API-Football indisponible. Voir CLAUDE.md."
    )
