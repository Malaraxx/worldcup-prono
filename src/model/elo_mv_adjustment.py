"""
Phase 1.5a — Ajustement Elo par valeur marchande du squad.

Formule : elo_ajusté = elo_brut + alpha * log(mv_ratio)
  - alpha = 0   → pas d'ajustement
  - alpha = 50  → +20 pts si squad 50% plus cher (mv_ratio=1.5)
  - alpha = 50  → -18 pts si squad 30% moins cher (mv_ratio=0.7)

Calibré empiriquement via calibrate_alpha.py sur les matchs 2024-2025.
NaN-safe : si mv_ratio est None/0/NaN, retourne elo_brut inchangé.
"""

import math


def adjust_elo_by_mv(elo_brut: float, mv_ratio: float | None, alpha: float) -> float:
    """
    Ajuste l'Elo brut en fonction du ratio valeur marchande.

    elo_brut  : rating Elo pré-ajustement (après conf. adjustment si applicable)
    mv_ratio  : mv_current / mv_baseline_confederation (1.0 = dans la moyenne)
    alpha     : coefficient d'intensité (0 = désactivé, 50 = modéré, 100 = fort)
    """
    if mv_ratio is None or mv_ratio <= 0 or not math.isfinite(mv_ratio):
        return float(elo_brut)
    if alpha == 0:
        return float(elo_brut)
    return float(elo_brut) + alpha * math.log(mv_ratio)
