"""Liste les matchs dont le coup d'envoi est passé mais sans score saisi."""
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).parents[1]

fix = pd.read_csv(ROOT / "data" / "processed" / "fixtures.csv")
res = pd.read_csv(ROOT / "data" / "raw" / "wc2026_results.csv")
fix["ko"] = pd.to_datetime(fix["kickoff_utc"], utc=True)
now = datetime.now(timezone.utc)
played = set(res["match_id"])

grp = fix[fix["stage"] == "group"].copy()
manquants = grp[(grp["ko"] < now) & (~grp["match_id"].isin(played))].sort_values("ko")

print("MATCHS JOUÉS SANS SCORE SAISI :")
for _, r in manquants.iterrows():
    print(f"  #{int(r['match_id']):>2}  {r['ko'].strftime('%d/%m %H:%M')}  {r['home_slot']} vs {r['away_slot']}")
print(f"\nTotal manquants : {len(manquants)}")
