"""Audit santé du projet worldcup-prono — données, picks, WR, cohérence."""
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parents[1]
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"

errors = 0
warns = 0


def err(msg):
    global errors
    errors += 1
    print(f"  [ERREUR] {msg}")


def warn(msg):
    global warns
    warns += 1
    print(f"  [WARN]   {msg}")


def ok(msg):
    print(f"  [OK]     {msg}")


# 1. CSV processed présents et non vides
print("=== CSV PROCESSED ===")
files = [
    "fixtures.csv", "predictions.csv", "optimal_picks.csv", "score_distributions.csv",
    "tournament_probabilities.csv", "group_stage_simulations.csv", "ko_predictions.csv",
    "elo_ratings.csv", "matches_historical.csv", "team_stats.csv", "teams.csv",
]
for f in files:
    p = PROCESSED / f
    if not p.exists():
        err(f"{f} MANQUANT")
    else:
        df = pd.read_csv(p)
        if df.empty:
            err(f"{f} VIDE")
        else:
            ok(f"{f} ({len(df)} lignes)")

# 2. Résultats — doublons, scores négatifs, match_id valides
print("\n=== RESULTS ===")
results = pd.read_csv(RAW / "wc2026_results.csv")
fixtures = pd.read_csv(PROCESSED / "fixtures.csv")
valid_ids = set(fixtures["match_id"])
ok(f"{len(results)} résultats saisis")

dups = results[results.duplicated(subset=["match_id"])]
if not dups.empty:
    err(f"doublons match_id: {dups['match_id'].tolist()}")
else:
    ok("pas de doublons")

bad_ids = set(results["match_id"]) - valid_ids
if bad_ids:
    err(f"match_id inconnus: {sorted(bad_ids)}")
else:
    ok("tous les match_id existent dans fixtures")

neg = results[(results["home_score"] < 0) | (results["away_score"] < 0)]
if not neg.empty:
    err(f"scores négatifs: {neg['match_id'].tolist()}")
else:
    ok("aucun score négatif")

# 3. Picks — NaN, modes valides, FK
print("\n=== PICKS ===")
picks = pd.read_csv(PROCESSED / "optimal_picks.csv")
nulls = picks.isnull().sum()
nulls = nulls[nulls > 0]
if not nulls.empty:
    warn(f"NaN: {nulls.to_dict()}")
else:
    ok("aucun NaN")

modes_ok = set(picks["mode_recommended"]) <= {"safe", "value", "lottery"}
if modes_ok:
    ok("modes_recommended valides")
else:
    err(f"modes invalides: {set(picks['mode_recommended'])}")

bad_pick_ids = set(picks["match_id"]) - valid_ids
if bad_pick_ids:
    err(f"picks avec match_id inconnu: {sorted(bad_pick_ids)}")
else:
    ok(f"{len(picks)} picks, tous les match_id valides")

# 4. WR live
print("\n=== WR LIVE ===")


def score_to_dir(s):
    try:
        h, a = map(int, str(s).split("-"))
        return "H" if h > a else ("D" if h == a else "A")
    except Exception:
        return None


merged = picks.merge(results, on="match_id")
correct = 0
for _, r in merged.iterrows():
    actual = "H" if r["home_score"] > r["away_score"] else ("D" if r["home_score"] == r["away_score"] else "A")
    pick_dir = score_to_dir(r.get(r["mode_recommended"] + "_score"))
    if pick_dir == actual:
        correct += 1
if len(merged) > 0:
    ok(f"{correct}/{len(merged)} = {correct / len(merged) * 100:.1f}% WR ({len(merged)}/24 picks joués)")
else:
    warn("aucun pick joué")

# 5. Phase de groupes — progression
print("\n=== PROGRESSION ===")
group_fix = fixtures[fixtures["stage"] == "group"]
played_group = len(set(results["match_id"]) & set(group_fix["match_id"]))
ok(f"phase de groupes: {played_group}/{len(group_fix)} matchs joués")

print("\n" + "=" * 50)
print(f"RÉSULTAT: {errors} erreur(s), {warns} warning(s)")
sys.exit(1 if errors else 0)
