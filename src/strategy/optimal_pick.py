"""Calcul de pronos optimaux pour Mon Petit Prono (WC2026)."""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson as poisson_dist

from src.strategy.mpp_mapping import normalize, check_against_teams

logger = logging.getLogger(__name__)

PROCESSED = Path(__file__).parents[2] / "data" / "processed"
RAW       = Path(__file__).parents[2] / "data" / "raw"
MAX_G     = 6   # scores i,j ∈ [0,6]


# ── Matrice de scores ─────────────────────────────────────────────────────────

def _score_mat(lam_h: float, lam_a: float) -> np.ndarray:
    """Matrice 7×7 P(home=i, away=j) pour i,j ∈ [0,6]."""
    ph = np.array([poisson_dist.pmf(k, lam_h) for k in range(MAX_G + 1)])
    pa = np.array([poisson_dist.pmf(k, lam_a) for k in range(MAX_G + 1)])
    return np.outer(ph, pa)


# ── Rareté MPP ────────────────────────────────────────────────────────────────

def estimate_rarity(home_score: int, away_score: int,
                    cote_h: int, cote_d: int, cote_a: int) -> int:
    """Bonus de rareté du score selon les cotes MPP."""
    favori     = "H" if cote_h < cote_a else ("A" if cote_a < cote_h else "D")
    score_diff = home_score - away_score
    total      = home_score + away_score

    # Score consensus sur favori clair
    if min(cote_h, cote_a) <= 50:
        if favori == "H" and (home_score, away_score) in [(1, 0), (2, 0), (2, 1)]:
            return 5
        if favori == "A" and (home_score, away_score) in [(0, 1), (0, 2), (1, 2)]:
            return 5

    # Nul standard sur match équilibré
    if (home_score, away_score) in [(1, 1), (0, 0)] and abs(cote_h - cote_a) < 30:
        return 10

    # Score modéré
    if (home_score, away_score) in [(2, 1), (1, 2), (3, 1), (1, 3), (2, 2)]:
        return 30

    # Gros score décisif
    if total >= 4 or abs(score_diff) >= 3:
        return 60

    # Inversion totale : outsider gagne par 2+ buts
    if (favori == "H" and away_score > home_score and (away_score - home_score) >= 2) \
       or (favori == "A" and home_score > away_score and (home_score - away_score) >= 2):
        return 100

    return 30


# ── Table EV ──────────────────────────────────────────────────────────────────

def _build_ev_df(mat: np.ndarray,
                 cote_h: int, cote_d: int, cote_a: int) -> tuple[pd.DataFrame, float, float, float]:
    """
    Construit la table EV 7×7.
    Retourne (ev_df, p_home, p_draw, p_away).
    ev_df colonnes : i, j, result, proba, cote_result, rarity, ev, proba_result.
    """
    rows = []
    p_home = p_draw = p_away = 0.0

    for i in range(MAX_G + 1):
        for j in range(MAX_G + 1):
            p = float(mat[i, j])
            if i > j:
                result, cote_r = "H", cote_h
                p_home += p
            elif i < j:
                result, cote_r = "A", cote_a
                p_away += p
            else:
                result, cote_r = "D", cote_d
                p_draw += p
            rarity = estimate_rarity(i, j, cote_h, cote_d, cote_a)
            rows.append({
                "i": i, "j": j,
                "result": result,
                "proba": p,
                "cote_result": cote_r,
                "rarity": rarity,
                "ev": p * (cote_r + rarity),
            })

    df = pd.DataFrame(rows)
    pr_map = {"H": p_home, "D": p_draw, "A": p_away}
    df["proba_result"] = df["result"].map(pr_map)
    return df, p_home, p_draw, p_away


# ── Modes de sélection ────────────────────────────────────────────────────────

def _safe_pick(ev_df: pd.DataFrame,
               p_home: float, p_draw: float, p_away: float) -> dict:
    best_r, best_pr = max(
        [("H", p_home), ("D", p_draw), ("A", p_away)], key=lambda x: x[1]
    )
    sub = ev_df[ev_df["result"] == best_r]
    row = sub.loc[sub["proba"].idxmax()]
    return {
        "score": f"{int(row['i'])}-{int(row['j'])}",
        "ev":    float(row["ev"]),
        "wr":    float(best_pr),
    }


def _value_pick(ev_df: pd.DataFrame) -> dict | None:
    """Argmax ev parmi (i,j) avec proba_result ≥ 35%. None si aucun candidat."""
    cands = ev_df[ev_df["proba_result"] >= 0.35]
    if cands.empty:
        return None
    row = cands.loc[cands["ev"].idxmax()]
    return {
        "score": f"{int(row['i'])}-{int(row['j'])}",
        "ev":    float(row["ev"]),
        "wr":    float(row["proba_result"]),
    }


def _lottery_pick(ev_df: pd.DataFrame) -> dict:
    """Argmax ev sans contrainte."""
    row = ev_df.loc[ev_df["ev"].idxmax()]
    return {
        "score": f"{int(row['i'])}-{int(row['j'])}",
        "ev":    float(row["ev"]),
        "wr":    float(row["proba_result"]),
    }


def _recommend_mode(safe: dict, value: dict, lottery: dict) -> tuple[str, float]:
    """Retourne (mode, edge_value_vs_safe_pct)."""
    edge = (value["ev"] - safe["ev"]) / safe["ev"] * 100 if safe["ev"] > 0 else 0.0
    if lottery["ev"] > 1.3 * value["ev"]:
        return "lottery", edge
    if edge < 10:
        return "safe", edge
    return "value", edge


# ── Chargement données ────────────────────────────────────────────────────────

def load_merged() -> pd.DataFrame:
    """Charge et fusionne mpp_cotes.csv + predictions.csv (poules uniquement)."""
    mpp  = pd.read_csv(RAW / "mpp_cotes.csv")
    pred = pd.read_csv(PROCESSED / "predictions.csv")
    pred_g = pred[pred["stage"] == "group"].copy()

    mpp["home_fifa"] = mpp["home"].apply(normalize)
    mpp["away_fifa"] = mpp["away"].apply(normalize)

    all_teams = set(pred_g["home_slot"]) | set(pred_g["away_slot"])
    check_against_teams(
        mpp["home"].tolist() + mpp["away"].tolist(), all_teams
    )

    merged = mpp.merge(
        pred_g[[
            "match_id", "date", "home_slot", "away_slot",
            "lambda_home", "lambda_away",
            "p_home_win_cal", "p_draw_cal", "p_away_win_cal",
        ]].rename(columns={"home_slot": "home_fifa", "away_slot": "away_fifa"}),
        on=["home_fifa", "away_fifa"],
        how="left",
    )

    unmatched = merged[merged["match_id"].isna()]
    if not unmatched.empty:
        logger.warning("Matchs MPP sans correspondance : %s",
                       unmatched[["home", "away"]].to_string(index=False))

    return merged


# ── Run principal ─────────────────────────────────────────────────────────────

def run() -> pd.DataFrame:
    """Génère score_distributions.csv et optimal_picks.csv. Retourne picks_df."""
    merged = load_merged()

    dist_rows  = []
    picks_rows = []

    for _, mrow in merged.iterrows():
        match_id = int(mrow["match_id"])
        cote_h = int(mrow["cote_home"])
        cote_d = int(mrow["cote_draw"])
        cote_a = int(mrow["cote_away"])
        lam_h  = float(mrow["lambda_home"])
        lam_a  = float(mrow["lambda_away"])

        mat = _score_mat(lam_h, lam_a)

        # Score distributions
        for i in range(MAX_G + 1):
            for j in range(MAX_G + 1):
                dist_rows.append({
                    "match_id": match_id,
                    "home": mrow["home_fifa"],
                    "away": mrow["away_fifa"],
                    "i": i, "j": j,
                    "proba": float(mat[i, j]),
                })

        # Picks
        ev_df, p_home, p_draw, p_away = _build_ev_df(mat, cote_h, cote_d, cote_a)
        safe    = _safe_pick(ev_df, p_home, p_draw, p_away)
        value   = _value_pick(ev_df) or safe.copy()
        lottery = _lottery_pick(ev_df)
        mode, edge = _recommend_mode(safe, value, lottery)

        picks_rows.append({
            "match_id":   match_id,
            "home":       mrow["home"],
            "away":       mrow["away"],
            "kickoff":    mrow["date"],
            "cote_home":  cote_h,
            "cote_draw":  cote_d,
            "cote_away":  cote_a,
            "p_home":     round(float(mrow["p_home_win_cal"]), 4),
            "p_draw":     round(float(mrow["p_draw_cal"]),     4),
            "p_away":     round(float(mrow["p_away_win_cal"]), 4),
            "safe_score":  safe["score"],
            "safe_ev":     round(safe["ev"],  4),
            "safe_wr":     round(safe["wr"],  4),
            "value_score": value["score"],
            "value_ev":    round(value["ev"], 4),
            "value_wr":    round(value["wr"], 4),
            "lottery_score": lottery["score"],
            "lottery_ev":    round(lottery["ev"], 4),
            "lottery_wr":    round(lottery["wr"], 4),
            "mode_recommended":       mode,
            "edge_value_vs_safe_pct": round(edge, 2),
        })

    dist_df  = pd.DataFrame(dist_rows)
    picks_df = pd.DataFrame(picks_rows)

    dist_df.to_csv(PROCESSED / "score_distributions.csv",  index=False)
    picks_df.to_csv(PROCESSED / "optimal_picks.csv", index=False)

    logger.info("score_distributions.csv : %d lignes", len(dist_rows))
    logger.info("optimal_picks.csv       : %d lignes", len(picks_df))
    return picks_df


# ── Fonctions utilitaires ─────────────────────────────────────────────────────

def recommend_bonus_x2(picks_df: pd.DataFrame) -> pd.DataFrame:
    """Top 5 matchs pour le bonus x2 (value_wr ≥ 40%, trié par value_ev desc)."""
    return (
        picks_df[picks_df["value_wr"] >= 0.40]
        .sort_values("value_ev", ascending=False)
        .head(5)
    )


def recommend_winner_pick(tp_df: pd.DataFrame) -> pd.DataFrame:
    """Top candidats vainqueur par proba_winner (sans cotes MPP vainqueur)."""
    logger.warning(
        "Cotes MPP vainqueur non disponibles — à recalculer quand cotes MPP dispo"
    )
    return (
        tp_df[tp_df["proba_winner"] >= 0.06]
        .sort_values("proba_winner", ascending=False)
        .head(10)
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    picks = run()
    print(f"\nOK optimal_picks.csv : {len(picks)} lignes\n")
    print(picks[["home", "away", "mode_recommended",
                 "value_score", "value_ev", "value_wr"]].to_string(index=False))
