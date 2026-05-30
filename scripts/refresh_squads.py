"""
Rafraîchit les effectifs des 48 équipes depuis Wikipedia.
À lancer quotidiennement jusqu'au 11 juin 2026.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.ingest.squads import save_all
from src.ingest.wc2026 import build_teams_df


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    teams = build_teams_df()["team"].tolist()
    print(f"Rafraîchissement effectifs pour {len(teams)} équipes...")
    df = save_all(teams)
    print(f"\nTerminé : {len(df)} joueurs, {df['team'].nunique()} équipes")
    print(f"Provisoires : {df['provisional'].sum()}/{len(df)}")
    print("\nJoueurs par équipe :")
    print(df.groupby("team").size().sort_values().to_string())


if __name__ == "__main__":
    main()
