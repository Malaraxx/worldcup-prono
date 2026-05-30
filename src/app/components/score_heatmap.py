"""Heatmap Plotly 7×7 de distribution de scores."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def score_heatmap(dist: pd.DataFrame, home_name: str, away_name: str) -> go.Figure:
    """Heatmap P(home=i, away=j), i lignes (home), j colonnes (away)."""
    mat = np.zeros((7, 7))
    for _, row in dist.iterrows():
        i, j = int(row["i"]), int(row["j"])
        if 0 <= i <= 6 and 0 <= j <= 6:
            mat[i, j] = float(row["proba"])

    text = [[f"{mat[i, j]:.1%}" for j in range(7)] for i in range(7)]

    fig = go.Figure(data=go.Heatmap(
        z=mat,
        x=[str(j) for j in range(7)],
        y=[str(i) for i in range(7)],
        colorscale="Blues",
        showscale=True,
        colorbar=dict(title="Proba", tickformat=".0%"),
        text=text,
        texttemplate="%{text}",
        textfont={"size": 9},
        hovertemplate=(
            f"{home_name} %{{y}} — {away_name} %{{x}}<br>"
            "Proba : %{text}<extra></extra>"
        ),
    ))

    fig.update_layout(
        xaxis=dict(title=f"Buts {away_name}", side="bottom"),
        yaxis=dict(title=f"Buts {home_name}", autorange="reversed"),
        margin=dict(l=50, r=30, t=10, b=50),
        height=380,
        plot_bgcolor="white",
    )
    return fig
