"""
Enrichit players.csv avec les valeurs marchandes Transfermarkt.
Source : API publique transfermarkt-api.fly.dev
Cache 24h → relancer quotidiennement pour mises à jour.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.ingest.market_values import save_market_values


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = save_market_values()
    df["market_value_eur"] = pd.to_numeric(df["market_value_eur"], errors="coerce")

    covered = df["market_value_eur"].notna().sum()
    total   = len(df)
    print(f"\nTerminé : {covered}/{total} joueurs avec valeur marchande "
          f"({covered / total:.1%})")

    print("\nTop 10 par valeur marchande :")
    top = (
        df.dropna(subset=["market_value_eur"])
          .nlargest(10, "market_value_eur")
          [["name", "team", "club", "market_value_eur"]]
    )
    top["market_value_eur"] = top["market_value_eur"].apply(
        lambda v: f"€{v/1_000_000:.1f}M"
    )
    print(top.to_string(index=False))

    print("\nValeur totale par équipe (top 10) :")
    by_team = (
        df.groupby("team")["market_value_eur"]
          .sum()
          .sort_values(ascending=False)
          .head(10)
          .apply(lambda v: f"€{v/1_000_000:.0f}M")
    )
    print(by_team.to_string())


if __name__ == "__main__":
    main()
