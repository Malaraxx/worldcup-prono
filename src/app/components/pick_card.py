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
    icons      = {"safe": "🛡️", "value": "💎", "lottery": "🎰"}
    colors     = {"safe": "#1565C0", "value": "#2E7D32", "lottery": "#6A1B9A"}
    bg_light   = {"safe": "#E3F2FD", "value": "#E8F5E9", "lottery": "#EDE7F6"}

    icon   = icons.get(mode, "")
    color  = colors.get(mode, "#424242")
    bg     = bg_light.get(mode, "#F5F5F5") if recommended else "#FAFAFA"
    border = f"2px solid {color}" if recommended else "1px solid #E0E0E0"
    shadow = "box-shadow:0 4px 16px rgba(0,0,0,0.10);" if recommended else ""

    badge = (
        f'<span style="background:{color};color:#fff;border-radius:4px;'
        f'padding:2px 7px;font-size:0.62rem;font-weight:700;letter-spacing:0.8px">RECOMMANDÉ ⭐</span>'
        if recommended else ""
    )
    edge_html = (
        f'<span style="background:#E8F5E9;color:#2E7D32;border-radius:4px;'
        f'padding:2px 8px;font-size:0.72rem;font-weight:700">+{edge_pct:.0f}% edge</span>'
        if edge_pct and mode == "value" else ""
    )
    ev_color = "#1B5E20" if ev >= 15 else ("#E65100" if ev >= 8 else "#757575")

    st.markdown(f"""
<div style="border:{border};border-radius:10px;padding:16px 18px;background:{bg};
            margin-bottom:10px;{shadow}">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px">
    <span style="font-size:0.72rem;color:{color};font-weight:700;letter-spacing:1.2px">
      {icon} {mode.upper()}
    </span>
    {badge}
  </div>
  <div style="font-size:2.4rem;font-weight:800;letter-spacing:3px;line-height:1;color:#111">
    {score}
  </div>
  <div style="display:flex;gap:20px;margin-top:12px;align-items:flex-end">
    <div>
      <div style="font-size:0.6rem;color:#999;letter-spacing:1px;margin-bottom:2px">EV</div>
      <div style="font-size:1.15rem;font-weight:800;color:{ev_color}">{ev:.1f}</div>
    </div>
    <div>
      <div style="font-size:0.6rem;color:#999;letter-spacing:1px;margin-bottom:2px">WIN RATE</div>
      <div style="font-size:1.15rem;font-weight:800;color:#424242">{wr:.0%}</div>
    </div>
    {f'<div style="margin-left:4px">{edge_html}</div>' if edge_html else ''}
  </div>
</div>
""", unsafe_allow_html=True)
