import pandas as pd
tp = pd.read_csv("data/processed/tournament_probabilities.csv")
for t in ["United States", "Canada", "Mexico"]:
    row = tp[tp["team"] == t]
    if not row.empty:
        r = row.iloc[0]
        print(t, "r32={:.1%}".format(r["proba_r32"]),
              "r16={:.1%}".format(r["proba_r16"]),
              "winner={:.1%}".format(r["proba_winner"]))

print()
print("sum proba_r32:", round(tp["proba_r32"].sum(), 3))
print("sum proba_winner:", round(tp["proba_winner"].sum(), 4))
print()
print("Top 10 winner:")
print(tp[["team","proba_r32","proba_winner"]].head(10).to_string(index=False))
