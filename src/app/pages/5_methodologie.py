"""Page 5 — Méthodologie : comment fonctionne le modèle."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[3]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import poisson as poisson_dist

from src.app.utils import load_tournament_probabilities, flag


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#0A2342 0%,#1565C0 55%,#1976D2 100%);
            border-radius:14px;padding:22px 28px;margin-bottom:20px;
            box-shadow:0 4px 20px rgba(21,101,192,0.25)">
  <div style="font-size:1.9rem;font-weight:800;color:#fff;letter-spacing:0.5px">
    📐 Méthodologie
  </div>
  <div style="font-size:0.9rem;color:rgba(255,255,255,0.65);margin-top:4px">
    Comment le modèle prédit les scores · Elo · Poisson · Monte-Carlo · Stratégie MPP
  </div>
</div>
""", unsafe_allow_html=True)


# ── Pipeline ───────────────────────────────────────────────────────────────────
st.markdown("### 🔄 Pipeline complet")

st.markdown("""
<div style="display:flex;align-items:center;gap:0;flex-wrap:wrap;margin:16px 0 24px">
  <div style="background:#E3F2FD;border:2px solid #1565C0;border-radius:10px;padding:12px 16px;text-align:center;min-width:110px">
    <div style="font-size:1.4rem">📊</div>
    <div style="font-weight:700;font-size:0.8rem;color:#1565C0;margin-top:4px">Historique</div>
    <div style="font-size:0.68rem;color:#555;margin-top:2px">15 742 matchs<br>2010 → 2026</div>
  </div>
  <div style="font-size:1.5rem;color:#90CAF9;padding:0 8px">→</div>
  <div style="background:#E3F2FD;border:2px solid #1565C0;border-radius:10px;padding:12px 16px;text-align:center;min-width:110px">
    <div style="font-size:1.4rem">🏅</div>
    <div style="font-weight:700;font-size:0.8rem;color:#1565C0;margin-top:4px">Elo Rating</div>
    <div style="font-size:0.68rem;color:#555;margin-top:2px">Force relative<br>de chaque équipe</div>
  </div>
  <div style="font-size:1.5rem;color:#90CAF9;padding:0 8px">→</div>
  <div style="background:#E3F2FD;border:2px solid #1565C0;border-radius:10px;padding:12px 16px;text-align:center;min-width:110px">
    <div style="font-size:1.4rem">📐</div>
    <div style="font-weight:700;font-size:0.8rem;color:#1565C0;margin-top:4px">Poisson</div>
    <div style="font-size:0.68rem;color:#555;margin-top:2px">Distribution<br>des buts</div>
  </div>
  <div style="font-size:1.5rem;color:#90CAF9;padding:0 8px">→</div>
  <div style="background:#E3F2FD;border:2px solid #1565C0;border-radius:10px;padding:12px 16px;text-align:center;min-width:110px">
    <div style="font-size:1.4rem">🎯</div>
    <div style="font-weight:700;font-size:0.8rem;color:#1565C0;margin-top:4px">Calibration</div>
    <div style="font-size:0.68rem;color:#555;margin-top:2px">Platt scaling<br>sur 2018–2022</div>
  </div>
  <div style="font-size:1.5rem;color:#90CAF9;padding:0 8px">→</div>
  <div style="background:#E8F5E9;border:2px solid #2E7D32;border-radius:10px;padding:12px 16px;text-align:center;min-width:110px">
    <div style="font-size:1.4rem">🎲</div>
    <div style="font-weight:700;font-size:0.8rem;color:#2E7D32;margin-top:4px">Monte-Carlo</div>
    <div style="font-size:0.68rem;color:#555;margin-top:2px">10 000 sims<br>bracket KO</div>
  </div>
  <div style="font-size:1.5rem;color:#90CAF9;padding:0 8px">→</div>
  <div style="background:#EDE7F6;border:2px solid #6A1B9A;border-radius:10px;padding:12px 16px;text-align:center;min-width:110px">
    <div style="font-size:1.4rem">💡</div>
    <div style="font-weight:700;font-size:0.8rem;color:#6A1B9A;margin-top:4px">Picks MPP</div>
    <div style="font-size:0.68rem;color:#555;margin-top:2px">SAFE / VALUE<br>LOTTERY</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏅 Elo Rating",
    "📐 Modèle Poisson",
    "🌍 Conf. Adjustment",
    "🎲 Monte-Carlo",
    "💡 Stratégie MPP",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ELO
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_text, col_chart = st.columns([1, 1])

    with col_text:
        st.markdown("#### Qu'est-ce que le rating Elo ?")
        st.markdown("""
Le **système Elo** mesure la force relative de chaque équipe sur une échelle commune.
Inventé pour les échecs par Arpad Elo, il est ici adapté au football international.

**Principe :** chaque match met à jour les ratings des deux équipes selon le résultat.
Une victoire contre un adversaire fort rapporte beaucoup de points ; contre un faible, peu.
""")

        st.markdown("""
<div style="background:#F8FAFF;border-left:4px solid #1565C0;padding:14px 18px;
            border-radius:0 8px 8px 0;margin:12px 0;font-family:monospace">
  <div style="font-size:0.75rem;color:#888;margin-bottom:6px">FORMULE</div>
  <div style="font-size:0.95rem">
    R<sub>new</sub> = R<sub>old</sub> + K × (S − E)
  </div>
  <div style="margin-top:10px;font-size:0.78rem;color:#555;line-height:1.8">
    <b>K</b> = 30 (amplitude d'apprentissage)<br>
    <b>S</b> = résultat réel (1 victoire, 0.5 nul, 0 défaite)<br>
    <b>E</b> = résultat attendu selon l'écart de rating<br>
    <b>Home advantage</b> = +60 pts Elo si terrain non neutre
  </div>
</div>
""", unsafe_allow_html=True)

        st.markdown("""
**Paramètres du modèle :**
- Entraîné sur **15 742 matchs** depuis 2010
- Match weight : tournois majeurs pondérés plus fort
- Ratings initiaux : 1500 pour les équipes sans historique
""")

    with col_chart:
        st.markdown("#### Top 20 — Ratings Elo WC2026")
        try:
            elo_df = pd.read_csv(
                Path(__file__).parents[3] / "data" / "processed" / "elo_ratings.csv"
            ).head(20)
            fig = go.Figure(go.Bar(
                x=elo_df["elo_rating"],
                y=elo_df["team"],
                orientation="h",
                marker_color="#1565C0",
                text=elo_df["elo_rating"].round(0).astype(int),
                textposition="outside",
            ))
            fig.update_layout(
                height=500,
                margin=dict(l=10, r=60, t=10, b=10),
                yaxis=dict(autorange="reversed"),
                xaxis=dict(range=[1700, 2200]),
                plot_bgcolor="white",
                paper_bgcolor="white",
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.info("Chargement ratings Elo impossible.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — POISSON
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    col_text, col_viz = st.columns([1, 1.2])

    with col_text:
        st.markdown("#### Régression de Poisson")
        st.markdown("""
Le modèle convertit l'écart de rating Elo en **taux de buts attendus** (λ).
La distribution de Poisson modélise ensuite la probabilité de chaque score.
""")

        st.markdown("""
<div style="background:#F8FAFF;border-left:4px solid #1565C0;padding:14px 18px;
            border-radius:0 8px 8px 0;margin:12px 0;font-family:monospace">
  <div style="font-size:0.75rem;color:#888;margin-bottom:8px">MODÈLE</div>
  <div style="font-size:0.9rem;line-height:2">
    log(λ<sub>home</sub>) = α<sub>h</sub> + β<sub>h</sub> × ΔElo<br>
    log(λ<sub>away</sub>) = α<sub>a</sub> − β<sub>a</sub> × ΔElo
  </div>
  <div style="margin-top:10px;font-size:0.78rem;color:#555;line-height:1.8">
    <b>λ_home</b> ≈ 1.24 buts/match (moyenne)<br>
    <b>λ_away</b> ≈ 1.11 buts/match (moyenne)<br>
    <b>ΔElo</b> = Elo_home_adj − Elo_away_adj<br><br>
    Paramètres fit par MLE sur matchs 2018+
  </div>
</div>
""", unsafe_allow_html=True)

        st.markdown("""
<div style="background:#F8FAFF;border-left:4px solid #1565C0;padding:14px 18px;
            border-radius:0 8px 8px 0;margin:12px 0;font-family:monospace">
  <div style="font-size:0.75rem;color:#888;margin-bottom:8px">SCORE (i, j)</div>
  <div style="font-size:0.9rem;line-height:2">
    P(i buts home, j buts away)<br>
    = Poisson(i, λ_h) × Poisson(j, λ_a)
  </div>
  <div style="margin-top:8px;font-size:0.78rem;color:#555">
    Hypothèse : buts des deux équipes indépendants
  </div>
</div>
""", unsafe_allow_html=True)

    with col_viz:
        st.markdown("#### Exemple interactif")
        c1, c2 = st.columns(2)
        elo_diff_ex = c1.slider("ΔElo (favori)", -400, 400, 200, 50,
                                help="Positif = équipe home favorite")
        with c2:
            st.markdown("")
            st.markdown("")

        # Paramètres fitées
        alpha_h, beta_h = 0.219, 0.0019
        alpha_a, beta_a = 0.102, 0.0019
        lh = np.exp(alpha_h + beta_h * elo_diff_ex)
        la = np.exp(alpha_a - beta_a * elo_diff_ex)

        c1.metric("λ home (buts attendus)", f"{lh:.2f}")
        c2.metric("λ away (buts attendus)", f"{la:.2f}")

        # Heatmap Poisson
        MAX_G = 6
        mat = np.zeros((MAX_G + 1, MAX_G + 1))
        for i in range(MAX_G + 1):
            for j in range(MAX_G + 1):
                mat[i, j] = poisson_dist.pmf(i, lh) * poisson_dist.pmf(j, la)

        p_home = sum(mat[i, j] for i in range(MAX_G + 1) for j in range(MAX_G + 1) if i > j)
        p_draw = sum(mat[i, i] for i in range(MAX_G + 1))
        p_away = 1 - p_home - p_draw

        text_mat = [[f"{mat[i,j]:.1%}" for j in range(MAX_G + 1)] for i in range(MAX_G + 1)]
        fig = go.Figure(go.Heatmap(
            z=mat, x=[str(j) for j in range(MAX_G + 1)],
            y=[str(i) for i in range(MAX_G + 1)],
            colorscale="Blues", showscale=False,
            text=text_mat, texttemplate="%{text}", textfont={"size": 9},
            hovertemplate="Score %{y}-%{x} : %{text}<extra></extra>",
        ))
        fig.update_layout(
            xaxis=dict(title="Buts away", side="bottom"),
            yaxis=dict(title="Buts home", autorange="reversed"),
            margin=dict(l=40, r=10, t=10, b=40),
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)

        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("P(Home win)", f"{p_home:.1%}")
        col_r2.metric("P(Nul)", f"{p_draw:.1%}")
        col_r3.metric("P(Away win)", f"{p_away:.1%}")

        st.caption("""
**Brier score** (précision modèle) : **0.493** sur val. 2023-2025
(0 = parfait, 0.667 = hasard pur — gain +0.174 vs aléatoire)
""")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CONF ADJUSTMENT
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    col_text, col_chart = st.columns([1, 1])

    with col_text:
        st.markdown("#### Pourquoi ajuster par confédération ?")
        st.markdown("""
Le rating Elo est calculé sur **tous types de matchs** (qualifications, amicaux,
tournois continentaux). Mais en phase finale de Coupe du Monde, certaines
confédérations surperforment ou sous-performent leur niveau Elo apparent.

**Exemple** : une équipe africaine (CAF) avec Elo 1800 perd historiquement
plus souvent en phase finale de Mondial qu'une équipe européenne (UEFA) avec le
même Elo 1800 — probablement à cause de la différence de niveau des compétitions
qualificatives.

L'ajustement **décale le rating Elo** uniquement pour les 104 fixtures WC2026,
calibré sur les WC 2010-2022.
""")

        st.markdown("""
<div style="background:#FFF8E1;border-left:4px solid #F57F17;padding:14px 18px;
            border-radius:0 8px 8px 0;margin:12px 0">
  <div style="font-size:0.75rem;color:#888;margin-bottom:6px">RÈGLE SPÉCIALE</div>
  <div style="font-size:0.85rem;color:#555;line-height:1.7">
    <b>USA, Canada, Mexico</b> sont exemptés de l'ajustement CONCACAF.<br>
    En tant qu'hôtes WC2026, ils bénéficient d'un contexte différent et
    leurs Elo sont conservés tels quels.
  </div>
</div>
""", unsafe_allow_html=True)

    with col_chart:
        st.markdown("#### Ajustements par confédération")
        confs = ["UEFA", "CONMEBOL", "AFC", "CAF", "CONCACAF", "OFC"]
        deltas = [0, 0, -108, -83, -95, -69]
        colors = ["#2E7D32" if d == 0 else "#1565C0" if d > -50 else
                  "#E65100" if d > -100 else "#B71C1C" for d in deltas]

        fig = go.Figure(go.Bar(
            x=confs, y=deltas,
            marker_color=["#4CAF50", "#4CAF50", "#F44336", "#FF9800", "#F44336", "#FF5722"],
            text=[f"{d:+d}" for d in deltas],
            textposition="outside",
        ))
        fig.add_hline(y=0, line_color="#333", line_width=1)
        fig.update_layout(
            yaxis=dict(title="Δ Elo WC (points)", range=[-140, 30]),
            margin=dict(l=10, r=10, t=10, b=10),
            height=300,
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
| Conf. | Ajustement | Interprétation |
|-------|-----------|----------------|
| UEFA | 0 | Référence |
| CONMEBOL | 0 | Référence |
| AFC | **-108** | Forte sur-cotation Elo |
| CONCACAF | **-95** | Forte sur-cotation Elo |
| CAF | **-83** | Sur-cotation modérée |
| OFC | **-69** | Sur-cotation modérée |
""")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MONTE-CARLO
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    col_text, col_chart = st.columns([1, 1])

    with col_text:
        st.markdown("#### Simulation Monte-Carlo du bracket")
        st.markdown("""
La phase de groupes produit 48 qualifiés pour le tableau final.
Mais **on ne sait pas encore** quelles équipes se qualifieront ni dans quel ordre.

La simulation **Monte-Carlo** résout ce problème en simulant **10 000 fois**
l'intégralité du tournoi :

1. **Phase de groupes** : chaque match est simulé en tirant un score aléatoirement
   selon la distribution Poisson de ce match
2. **Qualification** : les 2 premiers de chaque groupe avancent en R32, plus les
   4 meilleurs 3e (format WC2026)
3. **Phase KO** : les matchs R32 → R16 → QF → SF → Finale sont simulés de même,
   avec chaque équipe qualifiée depuis l'étape précédente
4. **Agrégation** : sur 10 000 simulations, on calcule la fréquence à laquelle
   chaque équipe atteint chaque stade

Le résultat donne des **probabilités conditionnelles** réalistes qui tiennent
compte du chemin potentiel de chaque équipe dans le tableau.
""")

        st.markdown("""
<div style="background:#E8F5E9;border-left:4px solid #2E7D32;padding:14px 18px;
            border-radius:0 8px 8px 0;margin:12px 0">
  <div style="font-size:0.75rem;color:#888;margin-bottom:6px">POURQUOI 10 000 ?</div>
  <div style="font-size:0.85rem;color:#555;line-height:1.7">
    10 000 itérations donnent une erreur standard d'environ <b>±0.5%</b>
    sur les probabilités. C'est suffisamment précis pour classer les favoris
    tout en restant rapide (~2s de calcul).
  </div>
</div>
""", unsafe_allow_html=True)

    with col_chart:
        st.markdown("#### Top 10 — P(Vainqueur) Monte-Carlo")
        try:
            tp = load_tournament_probabilities().sort_values(
                "proba_winner", ascending=False
            ).head(10)

            fig = go.Figure()
            stages = [
                ("proba_winner", "Vainqueur", "#1565C0"),
                ("proba_final",  "Finale",    "#1976D2"),
                ("proba_sf",     "Demies",    "#42A5F5"),
                ("proba_r32",    "R32",       "#BBDEFB"),
            ]
            for col, label, color in stages:
                if col in tp.columns:
                    fig.add_trace(go.Bar(
                        name=label,
                        y=[f"{flag(t)} {t}" for t in tp["team"]],
                        x=tp[col] * 100,
                        orientation="h",
                        marker_color=color,
                    ))

            fig.update_layout(
                barmode="overlay",
                xaxis=dict(title="Probabilité (%)"),
                yaxis=dict(autorange="reversed"),
                legend=dict(orientation="h", y=-0.15, font_size=10),
                margin=dict(l=10, r=10, t=10, b=60),
                height=420,
                plot_bgcolor="white",
                paper_bgcolor="white",
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.info("Données Monte-Carlo non disponibles.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — STRATÉGIE MPP
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("#### Comment exploiter les cotes MPP ?")

    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown("""
Sur **Mon Petit Prono**, chaque match a des cotes (style pari). Le score exact donne
des points, avec un multiplicateur selon la rareté du score choisi.

Le modèle calcule la **probabilité implicite MPP** de chaque issue et la compare
à la probabilité du modèle pour détecter des **écarts d'estimation** (value bets).
""")

        st.markdown("""
<div style="background:#F8FAFF;border-left:4px solid #1565C0;padding:14px 18px;
            border-radius:0 8px 8px 0;margin:12px 0;font-family:monospace">
  <div style="font-size:0.75rem;color:#888;margin-bottom:8px">PROBABILITÉ IMPLICITE</div>
  <div style="font-size:0.9rem;line-height:2">
    P<sub>impl</sub>(Home) = (1/cote_H) / Σ(1/cotes)
  </div>
  <div style="margin-top:8px;font-size:0.78rem;color:#555">
    Normalisation pour retirer la marge MPP (sum > 1 brut)
  </div>
</div>
""", unsafe_allow_html=True)

        st.markdown("""
<div style="background:#F8FAFF;border-left:4px solid #1565C0;padding:14px 18px;
            border-radius:0 8px 8px 0;margin:12px 0;font-family:monospace">
  <div style="font-size:0.75rem;color:#888;margin-bottom:8px">ESPÉRANCE DE VALEUR (EV)</div>
  <div style="font-size:0.9rem;line-height:2">
    EV(i, j) = P_modèle(i,j) × (cote_result + bonus_rareté)
  </div>
  <div style="margin-top:8px;font-size:0.78rem;color:#555;line-height:1.7">
    <b>cote_result</b> = cote de l'issue (H/N/A) du score (i,j)<br>
    <b>bonus_rareté</b> = 5 à 100 pts selon la rareté du score exact
  </div>
</div>
""", unsafe_allow_html=True)

    with col_b:
        st.markdown("#### Les 3 modes de sélection")

        st.markdown("""
<div style="border:2px solid #1565C0;border-radius:10px;padding:14px 18px;
            background:#E3F2FD;margin-bottom:10px">
  <div style="font-size:0.8rem;font-weight:700;color:#1565C0;letter-spacing:1px">
    🛡️ SAFE
  </div>
  <div style="font-size:0.85rem;color:#333;margin-top:6px;line-height:1.6">
    Score le plus probable dans l'issue la plus probable.<br>
    <b>Priorité</b> : maximiser la proba de donner le bon résultat (H/N/A).<br>
    <b>Quand</b> : favori clair, match prévisible.
  </div>
</div>

<div style="border:2px solid #2E7D32;border-radius:10px;padding:14px 18px;
            background:#E8F5E9;margin-bottom:10px">
  <div style="font-size:0.8rem;font-weight:700;color:#2E7D32;letter-spacing:1px">
    💎 VALUE
  </div>
  <div style="font-size:0.85rem;color:#333;margin-top:6px;line-height:1.6">
    Meilleur EV parmi les scores avec proba résultat ≥ 35%.<br>
    <b>Priorité</b> : maximiser l'espérance tout en gardant un WR décent.<br>
    <b>Quand</b> : écart modèle/MPP > 10% (value bet détecté).
  </div>
</div>

<div style="border:2px solid #6A1B9A;border-radius:10px;padding:14px 18px;
            background:#EDE7F6;margin-bottom:10px">
  <div style="font-size:0.8rem;font-weight:700;color:#6A1B9A;letter-spacing:1px">
    🎰 LOTTERY
  </div>
  <div style="font-size:0.85rem;color:#333;margin-top:6px;line-height:1.6">
    Meilleur EV absolu, sans contrainte sur le WR.<br>
    <b>Priorité</b> : maximiser les gains si le score rare sort.<br>
    <b>Quand</b> : EV lottery > 1.3× EV value — prime de rareté MPP très forte.
  </div>
</div>
""", unsafe_allow_html=True)

    st.divider()
    st.markdown("#### Arbre de décision — Quel mode est recommandé ?")

    st.markdown("""
<div style="background:#F8FAFF;border-radius:10px;padding:20px 24px;
            font-family:monospace;font-size:0.82rem;line-height:2.2;
            border:1px solid #DDE3F0">
  EV_lottery > 1.3 × EV_value ?
  <br>&nbsp;&nbsp;&nbsp;&nbsp;✅ OUI → <b style="color:#6A1B9A">LOTTERY</b>
  <br>&nbsp;&nbsp;&nbsp;&nbsp;❌ NON → Edge value vs safe > 10% ?
  <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;✅ OUI → <b style="color:#2E7D32">VALUE</b>
  <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;❌ NON → <b style="color:#1565C0">SAFE</b>
</div>
""", unsafe_allow_html=True)

    st.divider()
    st.markdown("#### Indicateurs clés")

    c1, c2, c3 = st.columns(3)
    c1.markdown("""
<div style="background:#fff;border:1px solid #DDE3F0;border-radius:10px;
            padding:16px;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.06)">
  <div style="font-size:2rem;margin-bottom:6px">📊</div>
  <div style="font-weight:700;font-size:0.9rem;color:#1565C0">EV (Espérance)</div>
  <div style="font-size:0.78rem;color:#666;margin-top:6px;line-height:1.5">
    Points espérés si le prono est exact.<br>
    <b style="color:#1B5E20">≥ 15 pts</b> = excellent<br>
    <b style="color:#E65100">8–15 pts</b> = correct<br>
    <b style="color:#9E9E9E">< 8 pts</b> = faible
  </div>
</div>
""", unsafe_allow_html=True)

    c2.markdown("""
<div style="background:#fff;border:1px solid #DDE3F0;border-radius:10px;
            padding:16px;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.06)">
  <div style="font-size:2rem;margin-bottom:6px">🎯</div>
  <div style="font-weight:700;font-size:0.9rem;color:#1565C0">WR (Win Rate)</div>
  <div style="font-size:0.78rem;color:#666;margin-top:6px;line-height:1.5">
    Probabilité que le résultat H/N/A<br>
    soit correct (pas le score exact).<br>
    Mesure la robustesse du prono.
  </div>
</div>
""", unsafe_allow_html=True)

    c3.markdown("""
<div style="background:#fff;border:1px solid #DDE3F0;border-radius:10px;
            padding:16px;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.06)">
  <div style="font-size:2rem;margin-bottom:6px">⚠️</div>
  <div style="font-weight:700;font-size:0.9rem;color:#1565C0">Value Bet</div>
  <div style="font-size:0.78rem;color:#666;margin-top:6px;line-height:1.5">
    Écart modèle/MPP > 15%.<br>
    MPP sous- ou sur-estime une issue<br>
    par rapport au modèle → opportunité.
  </div>
</div>
""", unsafe_allow_html=True)

st.divider()
st.caption(
    "Modèle : Elo + Poisson indépendant + Platt scaling · "
    "Brier 2023-2025 : 0.4926 raw / 0.4948 calibré · "
    "Monte-Carlo 10 000 itérations · "
    "Dixon-Coles Phase 1.5b : rho=-0.019 (rollback — amélioration insuffisante)"
)
