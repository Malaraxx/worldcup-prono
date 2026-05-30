import pandas as pd
ko = pd.read_csv("data/processed/ko_predictions.csv")
tp = pd.read_csv("data/processed/tournament_probabilities.csv")
teams = pd.read_csv("data/processed/teams.csv")
gs = pd.read_csv("data/processed/group_stage_simulations.csv")
elo = pd.read_csv("data/processed/elo_ratings.csv")

# Spain KO path: R32=84, R16=93, QF=98, SF=101, F=104
print("=== SPAIN PATH ===")
for mid in [84, 93, 98, 101, 104]:
    r = ko[ko["match_id"] == mid].iloc[0]
    ht1 = r.get("home_team_1", "?")
    hp1 = r.get("home_team_1_prob", 0)
    ht2 = r.get("home_team_2", "?")
    hp2 = r.get("home_team_2_prob", 0)
    at1 = r.get("away_team_1", "?")
    ap1 = r.get("away_team_1_prob", 0)
    at2 = r.get("away_team_2", "?")
    ap2 = r.get("away_team_2_prob", 0)
    phw = r.get("p_home_win", float("nan"))
    pd_ = r.get("p_draw", float("nan"))
    paw = r.get("p_away_win", float("nan"))
    print(f"Match {mid} ({r['stage']}): {r['home_slot']} vs {r['away_slot']}")
    print(f"  Home: {ht1}({hp1:.0%}) {ht2}({hp2:.0%})")
    print(f"  Away: {at1}({ap1:.0%}) {at2}({ap2:.0%})")
    print(f"  Odds for most likely matchup: {phw:.1%}/{pd_:.1%}/{paw:.1%}")
    print()

# Elo ranks
elo["elo_rank"] = elo["elo_rating"].rank(ascending=False).astype(int)
print("=== ELO RANKS FOR KEY TEAMS ===")
for t in ["Spain","Brazil","England","Portugal","Germany","Netherlands","Belgium"]:
    row = elo[elo["team"] == t]
    if not row.empty:
        print(t, "elo_rank=", row.iloc[0]["elo_rank"], "elo=", row.iloc[0]["elo_rating"])

# Merge all for full table
merged = (tp
    .merge(teams[["team","group","fifa_ranking"]], on="team", how="left")
    .merge(elo[["team","elo_rank"]], on="team", how="left")
)
merged["winner_rank"] = merged["proba_winner"].rank(ascending=False).astype(int)
merged["rank_delta"] = merged["winner_rank"] - merged["elo_rank"]
merged = merged.sort_values("proba_winner", ascending=False).reset_index(drop=True)

print()
print("=== TOP 15 ===")
cols = ["team","group","fifa_ranking","elo_rating","elo_rank","proba_r32","proba_r16","proba_qf","proba_sf","proba_final","proba_winner","winner_rank"]
print(merged[cols].head(15).to_string(index=False))

print()
print("=== OVERCOTED (winner_rank > elo_rank) ===")
print(merged.nlargest(5,"rank_delta")[["team","elo_rank","proba_winner","winner_rank","rank_delta"]].to_string(index=False))
print()
print("=== UNDERCOTED (winner_rank < elo_rank) ===")
print(merged.nsmallest(5,"rank_delta")[["team","elo_rank","proba_winner","winner_rank","rank_delta"]].to_string(index=False))
