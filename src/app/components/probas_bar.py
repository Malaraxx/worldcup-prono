"""Graphique comparatif modèle Elo vs probabilités implicites MPP."""
import plotly.graph_objects as go


def probas_bar(
    ph: float, pd_: float, pa: float,
    home_name: str = "Home",
    away_name: str = "Away",
    ip_h: float | None = None,
    ip_d: float | None = None,
    ip_a: float | None = None,
) -> go.Figure:
    """
    Barres groupées modèle / MPP implicite pour les 3 issues.
    ip_* = probabilités implicites MPP (optionnel, absent si match KO sans cotes).
    """
    cats = [home_name, "Nul", away_name]
    model_vals = [ph * 100, pd_ * 100, pa * 100]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Modèle Elo+Poisson",
        x=cats,
        y=model_vals,
        marker_color=["#1565C0", "#757575", "#BF360C"],
        text=[f"{v:.1f}%" for v in model_vals],
        textposition="outside",
        cliponaxis=False,
    ))

    if ip_h is not None and ip_d is not None and ip_a is not None:
        mpp_vals = [ip_h * 100, ip_d * 100, ip_a * 100]
        fig.add_trace(go.Bar(
            name="MPP implicite",
            x=cats,
            y=mpp_vals,
            marker_color=["#90CAF9", "#BDBDBD", "#FFAB91"],
            text=[f"{v:.1f}%" for v in mpp_vals],
            textposition="outside",
            cliponaxis=False,
        ))

    fig.update_layout(
        barmode="group",
        yaxis=dict(title="Probabilité (%)", range=[0, 105]),
        margin=dict(l=20, r=20, t=10, b=20),
        height=260,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig
