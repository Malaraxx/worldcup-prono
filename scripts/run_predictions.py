"""
Pipeline Phase 1 : Elo + Poisson → predictions.csv
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.model.predict import run
from src.model.elo import load_elo_ratings


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--recalculate-elo", action="store_true",
                        help="Recalcule les ratings Elo depuis zéro")
    args = parser.parse_args()

    # Elo ratings
    elo_df = load_elo_ratings(recalculate=args.recalculate_elo)
    print(f"\nTop 20 ratings Elo :")
    print(elo_df.head(20).to_string(index=False))

    # Pipeline complet
    preds = run(recalculate_elo=args.recalculate_elo)

    print(f"\nPrédictions WC2026 ({len(preds)} matchs) :")
    group = preds[preds["stage"] == "group"].copy()
    group["match"] = group["home_slot"] + " vs " + group["away_slot"]
    cols = ["match_id", "match", "elo_diff", "lambda_home", "lambda_away",
            "p_home_win", "p_draw", "p_away_win", "pred_score_home", "pred_score_away"]
    print(group[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
