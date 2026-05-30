import pandas as pd
tp    = pd.read_csv("data/processed/tournament_probabilities.csv")
teams = pd.read_csv("data/processed/teams.csv")
elo   = pd.read_csv("data/processed/elo_ratings.csv")
gs    = pd.read_csv("data/processed/group_stage_simulations.csv")

# Check if fifa_ranking has data
print("FIFA ranking sample:", teams["fifa_ranking"].head(10).tolist())
print("NaN count:", teams["fifa_ranking"].isna().sum())
print()

# Elo ranks within the 312-team universe
elo["elo_rank_global"] = elo["elo_rating"].rank(ascending=False).astype(int)

# Build elo rank within WC 48 teams only
wc_teams = teams["team"].tolist()
elo_wc = elo[elo["team"].isin(wc_teams)].copy()
elo_wc["elo_rank_wc"] = elo_wc["elo_rating"].rank(ascending=False).astype(int)

merged = (tp
    .merge(teams[["team","group","fifa_ranking"]], on="team", how="left")
    .merge(elo_wc[["team","elo_rating","elo_rank_global","elo_rank_wc"]], on="team", how="left")
)
merged["winner_rank_wc"] = merged["proba_winner"].rank(ascending=False).astype(int)
merged["rank_delta_wc"]  = merged["winner_rank_wc"] - merged["elo_rank_wc"]
merged = merged.sort_values("proba_winner", ascending=False).reset_index(drop=True)

print("=== Position de Germany, Netherlands, Belgium ===")
for t in ["Germany", "Netherlands", "Belgium", "Brazil", "England", "Portugal"]:
    r = merged[merged["team"]==t].iloc[0]
    print(f"{t}: elo_rank_wc={int(r['elo_rank_wc'])} winner_rank={int(r['winner_rank_wc'])} "
          f"r32={r['proba_r32']:.1%} r16={r['proba_r16']:.1%} winner={r['proba_winner']:.1%} "
          f"group={r['group']}")

print()
print("=== Sur-cotes (winner_rank_wc > elo_rank_wc) ===")
print(merged[merged["proba_winner"]>0.001].nlargest(5,"rank_delta_wc")
      [["team","elo_rank_wc","winner_rank_wc","rank_delta_wc","proba_winner","group"]].to_string(index=False))

print()
print("=== Sous-cotes (winner_rank_wc < elo_rank_wc) ===")
print(merged[merged["proba_winner"]>0.001].nsmallest(5,"rank_delta_wc")
      [["team","elo_rank_wc","winner_rank_wc","rank_delta_wc","proba_winner","group"]].to_string(index=False))

print()
print("=== Conditional P(winner | R32) top 10 ===")
merged["p_win_given_r32"] = merged["proba_winner"] / merged["proba_r32"]
print(merged[["team","proba_r32","proba_winner","p_win_given_r32"]].head(10).to_string(index=False))

print()
print("=== Spain group stage ===")
print(gs[gs["team"]=="Spain"][["team","group","proba_1st","proba_2nd","proba_3rd","proba_elim"]].to_string(index=False))
print("=== Group H full ===")
print(gs[gs["group"]=="H"][["team","proba_1st","proba_2nd","proba_3rd","proba_elim","elo_rating"]].to_string(index=False))
