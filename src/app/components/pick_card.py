"""Carte Safe / Value / Lottery pour un match."""
import streamlit as st


def pick_card(
    mode: str,
    score: str,
    ev: float,
    wr: float,
    recommended: bool = False,
    edge_pct: float | None = None,
) -> None:
    icons  = {"safe": "🛡️", "value": "💎", "lottery": "🎰"}
    colors = {"safe": "#1565C0", "value": "#2E7D32", "lottery": "#6A1B9A"}

    icon  = icons.get(mode, "")
    color = colors.get(mode, "#424242")
    border = f"3px solid {color}" if recommended else "1px solid #ddd"
    bg     = "#fffde7" if recommended else "#fafafa"
    badge  = " &nbsp;⭐ <b>RECOMMANDÉ</b>" if recommended else ""
    edge_html = (
        f'<span style="color:#2E7D32;font-weight:600">Edge +{edge_pct:.0f}%</span>'
        if edge_pct and mode == "value" else ""
    )

    st.markdown(f"""
<div style="border:{border};border-radius:8px;padding:14px 16px;background:{bg};margin-bottom:8px">
  <div style="font-size:0.72rem;color:{color};font-weight:700;letter-spacing:1.2px;margin-bottom:6px">
    {icon} {mode.upper()}{badge}
  </div>
  <div style="font-size:2.2rem;font-weight:800;letter-spacing:2px;line-height:1.1">{score}</div>
  <div style="display:flex;gap:20px;margin-top:8px;font-size:0.85rem;color:#555">
    <span>EV <b style="color:#111">{ev:.1f}</b></span>
    <span>WR <b style="color:#111">{wr:.0%}</b></span>
    {edge_html}
  </div>
</div>
""", unsafe_allow_html=True)
