"""
Simulation Monte-Carlo du Mondial FIFA 2026.
10 000 simulations en < 2 min. Seed fixe = 42.

Sorties :
  data/processed/tournament_probabilities.csv   -- 48 équipes, probas par round
  data/processed/group_stage_simulations.csv    -- standings de poules
  data/processed/ko_predictions.csv            -- bracket KO annoté
"""

import logging
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from src.model.elo     import get_ratings_dict, DEFAULT_RATING
from src.model.poisson import fit, lambdas as compute_lambdas, score_matrix, outcome_probs, most_likely_score
from src.model.predict import _add_elo_diff

logger = logging.getLogger(__name__)
PROCESSED = Path(__file__).parents[2] / "data" / "processed"

GROUPS   = list("ABCDEFGHIJKL")
N_GROUPS = 12

# ── 3rd-place slots R32 ───────────────────────────────────────────────────────
# (match_id, frozenset of eligible group indices 0-11)
# A=0 B=1 C=2 D=3 E=4 F=5 G=6 H=7 I=8 J=9 K=10 L=11
THIRD_SLOTS = [
    (74, frozenset([0, 1, 2, 3, 5])),    # 3rd Group A/B/C/D/F
    (77, frozenset([2, 3, 5, 6, 7])),    # 3rd Group C/D/F/G/H
    (79, frozenset([2, 4, 5, 7, 8])),    # 3rd Group C/E/F/H/I
    (80, frozenset([4, 7, 8, 9, 10])),   # 3rd Group E/H/I/J/K
    (81, frozenset([1, 4, 5, 8, 9])),    # 3rd Group B/E/F/I/J
    (82, frozenset([0, 4, 7, 8, 9])),    # 3rd Group A/E/H/I/J
    (85, frozenset([4, 5, 6, 8, 9])),    # 3rd Group E/F/G/I/J
    (87, frozenset([3, 4, 8, 9, 11])),   # 3rd Group D/E/I/J/L
]
THIRD_SLOT_MATCH_IDS = [mid for mid, _ in THIRD_SLOTS]
THIRD_SLOT_ELIGIBLE  = [es  for _, es in THIRD_SLOTS]


def _backtrack_assign(
    avail: set,
    slot_remaining: list,
    eligible_sets: list,
    result: dict,
) -> bool:
    """Backtracking pour assigner 8 groupes qualifiés aux 8 slots 3e. O(8!) worst case."""
    if not slot_remaining:
        return True
    # Slot le plus contraint en premier (heuristique)
    best = min(slot_remaining, key=lambda s: len(eligible_sets[s] & avail))
    candidates = eligible_sets[best] & avail
    if not candidates:
        return False
    remaining = [s for s in slot_remaining if s != best]
    for chosen in sorted(candidates):
        result[best] = chosen
        if _backtrack_assign(avail - {chosen}, remaining, eligible_sets, result):
            return True
    result[best] = -1
    return False


def _precompute_lookup() -> dict:
    """Pré-calcule les 495 assignations possibles (C(12,8) = 495)."""
    lookup = {}
    errors = 0
    for combo in combinations(range(N_GROUPS), 8):
        key    = frozenset(combo)
        result = {}
        ok = _backtrack_assign(
            avail=set(combo),
            slot_remaining=list(range(len(THIRD_SLOT_ELIGIBLE))),
            eligible_sets=THIRD_SLOT_ELIGIBLE,
            result=result,
        )
        if not ok or -1 in result.values():
            errors += 1
            # Fallback : assignation aléatoire (ne devrait pas arriver sur un bracket FIFA correct)
            avail = list(combo)
            for si in range(len(THIRD_SLOT_ELIGIBLE)):
                if si not in result or result[si] == -1:
                    result[si] = avail.pop(0) if avail else -1
        lookup[key] = result
    if errors:
        logger.warning("THIRD_LOOKUP : %d combinaisons sans assignation parfaite (fallback)", errors)
    return lookup


THIRD_LOOKUP = _precompute_lookup()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_group_info(group_fixtures: pd.DataFrame, group_preds: pd.DataFrame,
                      team_idx: dict) -> dict:
    """Pour chaque groupe : teams, indices locaux/globaux, matchs."""
    info = {}
    for group in GROUPS:
        gf = group_fixtures[group_fixtures["group"] == group].sort_values("match_id")
        gp = group_preds[group_preds["group"] == group].sort_values("match_id")

        teams  = sorted(set(gf["home_slot"].tolist() + gf["away_slot"].tolist()))
        local  = {t: i for i, t in enumerate(teams)}
        l2g    = np.array([team_idx.get(t, -1) for t in teams], dtype=np.int16)

        matches = []
        for (_, fr), (_, pr) in zip(gf.iterrows(), gp.iterrows()):
            matches.append((
                local[fr["home_slot"]],
                local[fr["away_slot"]],
                int(pr.name),           # position 0-71 dans le tableau des 72 matchs
            ))

        info[group] = {"teams": teams, "l2g": l2g, "matches": matches}
    return info


def _simulate_ko_match(
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    elo_array: np.ndarray,
    params: dict,
    n_sims: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Simule n_sims matchs KO : Poisson 90min → ET 30min → tirs au but 50/50."""
    elo_diff = elo_array[home_idx] - elo_array[away_idx]
    lam_h    = np.exp(params["alpha_h"] + params["beta_h"] * elo_diff)
    lam_a    = np.exp(params["alpha_a"] - params["beta_a"] * elo_diff)

    g_h = np.random.poisson(lam_h)
    g_a = np.random.poisson(lam_a)

    draw90 = g_h == g_a
    if draw90.any():
        g_h[draw90] += np.random.poisson(lam_h[draw90] / 3)
        g_a[draw90] += np.random.poisson(lam_a[draw90] / 3)

    draw_et = g_h == g_a
    if draw_et.any():
        pen = np.random.random(draw_et.sum()) < 0.5
        g_h[draw_et] += pen.astype(np.int32)

    home_adv = g_h > g_a
    winners  = np.where(home_adv, home_idx, away_idx).astype(np.int16)
    losers   = np.where(home_adv, away_idx, home_idx).astype(np.int16)
    return winners, losers


# ── Pipeline principal ────────────────────────────────────────────────────────

def run_tournament(n_simulations: int = 10000, seed: int = 42) -> dict:
    """
    Simule n_simulations fois le Mondial WC2026.
    Sauvegarde 3 CSV dans data/processed/.
    """
    np.random.seed(seed)
    logger.info("Monte-Carlo WC2026 : %d simulations (seed=%d)", n_simulations, seed)

    # ── Chargement ────────────────────────────────────────────────────────────
    teams_df  = pd.read_csv(PROCESSED / "teams.csv")
    team_list = teams_df["team"].tolist()
    n_teams   = len(team_list)
    team_idx  = {t: i for i, t in enumerate(team_list)}

    ratings   = get_ratings_dict()
    elo_array = np.array([ratings.get(t, DEFAULT_RATING) for t in team_list])

    fixtures    = pd.read_csv(PROCESSED / "fixtures.csv")
    predictions = pd.read_csv(PROCESSED / "predictions.csv")

    group_fix  = fixtures[fixtures["stage"] == "group"].sort_values("match_id").reset_index(drop=True)
    group_pred = predictions[predictions["stage"] == "group"].sort_values("match_id").reset_index(drop=True)

    # Re-fit Poisson (rapide, pour calcul lambdas KO)
    hist = pd.read_csv(PROCESSED / "matches_historical.csv", parse_dates=["date"])
    hist["home_score"] = pd.to_numeric(hist["home_score"], errors="coerce")
    hist["away_score"] = pd.to_numeric(hist["away_score"], errors="coerce")
    hist = hist.dropna(subset=["home_score", "away_score"])
    hist = _add_elo_diff(hist, ratings)
    params = fit(hist[hist["date"] >= "2018-01-01"].copy())

    # ── PHASE DE POULES ───────────────────────────────────────────────────────
    lam_h_arr = group_pred["lambda_home"].values   # (72,)
    lam_a_arr = group_pred["lambda_away"].values

    home_g = np.random.poisson(lam=lam_h_arr[np.newaxis, :], size=(n_simulations, 72))
    away_g = np.random.poisson(lam=lam_a_arr[np.newaxis, :], size=(n_simulations, 72))

    group_info    = _build_group_info(group_fix, group_pred, team_idx)
    group_results = {}
    points_hist   = {t: np.zeros(10, dtype=np.int32) for t in team_list}

    for group in GROUPS:
        info   = group_info[group]
        n_loc  = len(info["teams"])

        pts = np.zeros((n_simulations, n_loc), dtype=np.int16)
        gd  = np.zeros((n_simulations, n_loc), dtype=np.int16)
        gf  = np.zeros((n_simulations, n_loc), dtype=np.int16)

        for hi, ai, gi in info["matches"]:
            hg = home_g[:, gi];  ag = away_g[:, gi]
            hw = hg > ag;  dr = hg == ag;  aw = hg < ag
            pts[:, hi] += (3 * hw + dr).astype(np.int16)
            pts[:, ai] += (3 * aw + dr).astype(np.int16)
            gd[:, hi]  += (hg - ag).astype(np.int16)
            gd[:, ai]  += (ag - hg).astype(np.int16)
            gf[:, hi]  += hg.astype(np.int16)
            gf[:, ai]  += ag.astype(np.int16)

        noise = np.random.random((n_simulations, n_loc)) * 0.9
        key   = pts.astype(float) * 1e6 + gd.astype(float) * 1e3 + gf.astype(float) + noise
        # ranks[sim, rank] = LOCAL team index → ordre décroissant de performance
        ranks        = np.argsort(-key, axis=1).astype(np.int8)
        # ranked[sim, rank] = GLOBAL team index
        ranked_global = info["l2g"][ranks]

        group_results[group] = {
            "ranked": ranked_global,   # shape (n_sims, 4) — GLOBAL idx
            "ranks":  ranks,           # shape (n_sims, 4) — LOCAL idx
            "pts": pts, "gd": gd, "gf": gf,
            "l2g": info["l2g"],
            "teams": info["teams"],
        }

        # Points distribution
        for li, tname in enumerate(info["teams"]):
            tp = pts[:, li]
            for p in range(10):
                points_hist[tname][p] += int((tp == p).sum())

    # ── 8 MEILLEURS 3es ───────────────────────────────────────────────────────
    thirds_global = np.zeros((n_simulations, N_GROUPS), dtype=np.int16)
    thirds_pts    = np.zeros((n_simulations, N_GROUPS), dtype=np.int16)
    thirds_gd     = np.zeros((n_simulations, N_GROUPS), dtype=np.int16)
    thirds_gf     = np.zeros((n_simulations, N_GROUPS), dtype=np.int16)

    for g_idx, group in enumerate(GROUPS):
        gr          = group_results[group]
        third_local = gr["ranks"][:, 2].astype(int)          # LOCAL idx du 3e
        thirds_global[:, g_idx] = gr["l2g"][third_local]     # → GLOBAL
        thirds_pts[:, g_idx]    = gr["pts"][np.arange(n_simulations), third_local]
        thirds_gd[:, g_idx]     = gr["gd"][np.arange(n_simulations),  third_local]
        thirds_gf[:, g_idx]     = gr["gf"][np.arange(n_simulations),  third_local]

    noise3       = np.random.random((n_simulations, N_GROUPS)) * 0.9
    key3         = thirds_pts.astype(float) * 1e6 + thirds_gd.astype(float) * 1e3 + thirds_gf.astype(float) + noise3
    thirds_order = np.argsort(-key3, axis=1)        # (n_sims, 12), groupes triés
    qual_g_idx   = thirds_order[:, :8]               # top 8 group indices (0-11)

    # ── ASSIGNATION SLOTS 3e ──────────────────────────────────────────────────
    qual_keys   = [frozenset(qual_g_idx[s].tolist()) for s in range(n_simulations)]
    assignments = [THIRD_LOOKUP[k] for k in qual_keys]

    all_assigns = np.array(
        [[a.get(si, -1) for si in range(8)] for a in assignments],
        dtype=np.int8,
    )

    third_slot_teams = np.full((n_simulations, 8), -1, dtype=np.int16)
    sims_all = np.arange(n_simulations)
    for sp in range(8):
        gi   = all_assigns[:, sp].astype(int)
        valid = gi >= 0
        sv    = sims_all[valid]
        third_slot_teams[sv, sp] = thirds_global[sv, gi[sv]]

    # ── MATCHS KO ─────────────────────────────────────────────────────────────
    ko_fix = fixtures[fixtures["stage"] != "group"].sort_values("match_id")

    match_winners: dict[int, np.ndarray] = {}
    match_losers:  dict[int, np.ndarray] = {}

    def resolve(slot_str: str) -> np.ndarray:
        if slot_str.startswith("Winner Group "):
            return group_results[slot_str[-1]]["ranked"][:, 0].copy()
        if slot_str.startswith("Runner-up Group "):
            return group_results[slot_str[-1]]["ranked"][:, 1].copy()
        if slot_str.startswith("3rd Group "):
            eligible_str = set(slot_str.replace("3rd Group ", "").split("/"))
            for si, (_, es) in enumerate(THIRD_SLOTS):
                if {GROUPS[i] for i in es} == eligible_str:
                    return third_slot_teams[:, si].copy()
            raise ValueError(f"Slot 3rd inconnu : {slot_str}")
        if slot_str.startswith("Winner Match "):
            return match_winners[int(slot_str.split()[-1])].copy()
        if slot_str.startswith("Loser Match "):
            return match_losers[int(slot_str.split()[-1])].copy()
        raise ValueError(f"Slot inconnu : {slot_str}")

    for _, row in ko_fix.iterrows():
        mid   = int(row["match_id"])
        stage = row["stage"]
        if stage == "3rd":
            ht = match_losers[int(row["home_slot"].split()[-1])]
            at = match_losers[int(row["away_slot"].split()[-1])]
        else:
            ht = resolve(row["home_slot"])
            at = resolve(row["away_slot"])

        ht = np.where(ht < 0, 0, ht).astype(np.int16)
        at = np.where(at < 0, 0, at).astype(np.int16)
        w, lo = _simulate_ko_match(ht, at, elo_array, params, n_simulations)
        match_winners[mid] = w
        match_losers[mid]  = lo

    # ── AGRÉGATION ────────────────────────────────────────────────────────────
    proba_r32    = np.zeros(n_teams, dtype=float)
    proba_r16    = np.zeros(n_teams, dtype=float)
    proba_qf     = np.zeros(n_teams, dtype=float)
    proba_sf     = np.zeros(n_teams, dtype=float)
    proba_final  = np.zeros(n_teams, dtype=float)
    proba_winner = np.zeros(n_teams, dtype=float)

    # R32 qualifiés : top 2 de chaque groupe
    for group in GROUPS:
        for rank in range(2):
            for tid in group_results[group]["ranked"][:, rank]:
                proba_r32[tid] += 1
    # R32 qualifiés : 8 meilleurs 3es
    for sp in range(8):
        for tid in third_slot_teams[:, sp]:
            if tid >= 0:
                proba_r32[tid] += 1

    # Rounds KO
    stage_mids = {
        "r32":   sorted(ko_fix[ko_fix["stage"] == "r32"]["match_id"]),
        "r16":   sorted(ko_fix[ko_fix["stage"] == "r16"]["match_id"]),
        "qf":    sorted(ko_fix[ko_fix["stage"] == "qf"]["match_id"]),
        "sf":    sorted(ko_fix[ko_fix["stage"] == "sf"]["match_id"]),
        "final": sorted(ko_fix[ko_fix["stage"] == "final"]["match_id"]),
    }
    for mid in stage_mids["r32"]:
        for tid in match_winners[mid]: proba_r16[tid] += 1
    for mid in stage_mids["r16"]:
        for tid in match_winners[mid]: proba_qf[tid] += 1
    for mid in stage_mids["qf"]:
        for tid in match_winners[mid]: proba_sf[tid] += 1
    for mid in stage_mids["sf"]:
        for tid in match_winners[mid]: proba_final[tid] += 1
    for mid in stage_mids["final"]:
        for tid in match_winners[mid]: proba_winner[tid] += 1

    for arr in [proba_r32, proba_r16, proba_qf, proba_sf, proba_final, proba_winner]:
        arr /= n_simulations

    # ── tournament_probabilities.csv ──────────────────────────────────────────
    tp_df = pd.DataFrame({
        "team":         team_list,
        "elo_rating":   elo_array.round(1),
        "proba_r32":    proba_r32.round(4),
        "proba_r16":    proba_r16.round(4),
        "proba_qf":     proba_qf.round(4),
        "proba_sf":     proba_sf.round(4),
        "proba_final":  proba_final.round(4),
        "proba_winner": proba_winner.round(4),
    }).sort_values("proba_winner", ascending=False).reset_index(drop=True)
    tp_df.to_csv(PROCESSED / "tournament_probabilities.csv", index=False)

    # ── group_stage_simulations.csv ───────────────────────────────────────────
    gs_rows = []
    for group in GROUPS:
        gr = group_results[group]
        for li, tname in enumerate(gr["teams"]):
            global_tid = int(gr["l2g"][li])
            # Rang de ce team dans chaque sim (colonne dans ranked_global)
            rank_per_sim = np.where(gr["ranked"] == global_tid)[1]
            row = {
                "team":       tname,
                "group":      group,
                "elo_rating": round(float(elo_array[global_tid]), 1),
                "proba_1st":  round(float((rank_per_sim == 0).mean()), 4),
                "proba_2nd":  round(float((rank_per_sim == 1).mean()), 4),
                "proba_3rd":  round(float((rank_per_sim == 2).mean()), 4),
                "proba_elim": round(float((rank_per_sim == 3).mean()), 4),
            }
            for p in range(10):
                row[f"pts_{p}"] = round(float(points_hist[tname][p] / n_simulations), 4)
            gs_rows.append(row)

    gs_df = pd.DataFrame(gs_rows)
    gs_df.to_csv(PROCESSED / "group_stage_simulations.csv", index=False)

    # ── ko_predictions.csv ────────────────────────────────────────────────────
    ko_rows = []
    for _, row in ko_fix.iterrows():
        mid   = int(row["match_id"])
        stage = row["stage"]
        hslot = row["home_slot"]
        aslot = row["away_slot"]

        if stage == "3rd":
            ht_sims = match_losers[int(hslot.split()[-1])]
            at_sims = match_losers[int(aslot.split()[-1])]
        else:
            ht_sims = resolve(hslot)
            at_sims = resolve(aslot)

        def top3(arr):
            arr = np.where(arr < 0, n_teams - 1, arr).astype(int)
            counts = np.bincount(arr, minlength=n_teams)
            top_idx = np.argsort(-counts)[:3]
            return [(team_list[i], round(counts[i] / n_simulations, 4)) for i in top_idx if counts[i] > 0]

        h3 = top3(ht_sims)
        a3 = top3(at_sims)

        best_home = h3[0][0] if h3 else ""
        best_away = a3[0][0] if a3 else ""
        pred_h = pred_a = p_hw = p_d = p_aw = float("nan")
        if best_home and best_away:
            elo_h = ratings.get(best_home, DEFAULT_RATING)
            elo_a = ratings.get(best_away, DEFAULT_RATING)
            lh, la = compute_lambdas(elo_h - elo_a, params)
            mat    = score_matrix(lh, la)
            p_hw, p_d, p_aw = outcome_probs(mat)
            pred_h, pred_a, _ = most_likely_score(mat)

        kr = {
            "match_id":   mid, "stage": stage,
            "home_slot":  hslot, "away_slot": aslot,
        }
        for i, (tn, prob) in enumerate(h3[:3]):
            kr[f"home_team_{i+1}"] = tn; kr[f"home_team_{i+1}_prob"] = prob
        for i, (tn, prob) in enumerate(a3[:3]):
            kr[f"away_team_{i+1}"] = tn; kr[f"away_team_{i+1}_prob"] = prob
        kr.update({
            "pred_score_home": pred_h if pred_h == pred_h else None,
            "pred_score_away": pred_a if pred_a == pred_a else None,
            "p_home_win": round(p_hw, 4) if p_hw == p_hw else None,
            "p_draw":     round(p_d,  4) if p_d  == p_d  else None,
            "p_away_win": round(p_aw, 4) if p_aw == p_aw else None,
        })
        ko_rows.append(kr)

    ko_df = pd.DataFrame(ko_rows)
    ko_df.to_csv(PROCESSED / "ko_predictions.csv", index=False)

    logger.info("Monte-Carlo termine : 3 CSV sauvegardes.")
    return {
        "tournament_probabilities": tp_df,
        "group_stage_simulations":  gs_df,
        "ko_predictions":           ko_df,
    }
