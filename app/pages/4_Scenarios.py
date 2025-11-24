"""
Page Simulation de Scenarios - Modelisation de l'impact des actions marketing.

Cette page permet de simuler differents scenarios d'amelioration des KPIs
et de projeter leur impact sur les revenus et la valeur client.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
from pathlib import Path
import io

# Imports locaux
sys.path.append(str(Path(__file__).parent.parent.parent))
import config
import utils

# CONFIGURATION DE LA PAGE
st.set_page_config(
    page_title="Simulation de Scenarios - Marketing Analytics",
    page_icon=":material/science:",
    layout="wide"
)


# FONCTIONS UTILITAIRES
def format_currency(value):
    """Formate une valeur monetaire en livres sterling."""
    return f"£{value:,.0f}"


def format_number(value):
    """Formate un nombre avec separateurs de milliers."""
    return f"{value:,.0f}"


def format_percentage(value, decimals=1):
    """Formate un nombre en pourcentage."""
    return f"{value:.{decimals}f}%"


def calculate_roi(revenue_gain, cost):
    """Calcule le retour sur investissement."""
    if cost == 0:
        return 0
    return ((revenue_gain - cost) / cost) * 100


def generate_monthly_projection(current_revenue, growth_rate, months):
    """Genere une projection mensuelle de revenu."""
    monthly_revenue = current_revenue / 12
    projections = []

    for month in range(months + 1):
        # Croissance progressive
        growth_factor = 1 + (growth_rate * (month / months))
        projected = monthly_revenue * growth_factor * (month + 1)
        projections.append(projected)

    return projections


def create_comparison_bar_chart(current_kpis, projected_kpis):
    """Cree un graphique comparatif des KPIs."""

    metrics = ['Revenu Annuel', 'Nombre de Clients', 'Panier Moyen', 'Frequence d\'Achat']
    current_values = [
        current_kpis['revenue'],
        current_kpis['customers'],
        current_kpis['aov'],
        current_kpis['frequency']
    ]
    projected_values = [
        projected_kpis['revenue'],
        projected_kpis['customers'],
        projected_kpis['aov'],
        projected_kpis['frequency']
    ]

    # Normaliser les valeurs pour comparaison visuelle
    current_normalized = []
    projected_normalized = []

    for curr, proj in zip(current_values, projected_values):
        current_normalized.append(100)
        projected_normalized.append((proj / curr) * 100 if curr > 0 else 100)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Actuel',
        x=metrics,
        y=current_normalized,
        marker_color='#7f7f7f',
        text=[format_currency(v) if i == 0 else format_number(v) if i in [1] else f"{v:.2f}" for i, v in enumerate(current_values)],
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>Actuel: %{text}<extra></extra>'
    ))

    fig.add_trace(go.Bar(
        name='Projete',
        x=metrics,
        y=projected_normalized,
        marker_color='#2ca02c',
        text=[format_currency(v) if i == 0 else format_number(v) if i in [1] else f"{v:.2f}" for i, v in enumerate(projected_values)],
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>Projete: %{text}<extra></extra>'
    ))

    fig.update_layout(
        title="Comparaison des KPIs : Actuel vs Projete",
        xaxis_title="Indicateurs",
        yaxis_title="Indice (Base 100 = Actuel)",
        barmode='group',
        height=400,
        showlegend=True,
        hovermode='x unified'
    )

    return fig


def create_evolution_chart(current_revenue, projected_revenue, months):
    """Cree un graphique d'evolution dans le temps."""

    # Calculer les taux de croissance
    current_growth = 0.0  # Croissance baseline (stagnation)
    projected_growth = (projected_revenue - current_revenue) / current_revenue

    # Generer les projections mensuelles
    months_list = list(range(months + 1))
    baseline_projections = generate_monthly_projection(current_revenue, current_growth, months)
    scenario_projections = generate_monthly_projection(current_revenue, projected_growth, months)

    # Calculer l'intervalle de confiance (±10%)
    upper_bound = [p * 1.1 for p in scenario_projections]
    lower_bound = [p * 0.9 for p in scenario_projections]

    fig = go.Figure()

    # Ligne baseline
    fig.add_trace(go.Scatter(
        x=months_list,
        y=baseline_projections,
        mode='lines',
        name='Tendance Actuelle',
        line=dict(color='#7f7f7f', width=2, dash='dash'),
        hovertemplate='Mois %{x}<br>Revenu: £%{y:,.0f}<extra></extra>'
    ))

    # Intervalle de confiance
    fig.add_trace(go.Scatter(
        x=months_list + months_list[::-1],
        y=upper_bound + lower_bound[::-1],
        fill='toself',
        fillcolor='rgba(44, 160, 44, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=True,
        name='Intervalle de Confiance (±10%)',
        hoverinfo='skip'
    ))

    # Ligne scenario
    fig.add_trace(go.Scatter(
        x=months_list,
        y=scenario_projections,
        mode='lines+markers',
        name='Projection Scenario',
        line=dict(color='#2ca02c', width=3),
        marker=dict(size=6),
        hovertemplate='Mois %{x}<br>Revenu projete: £%{y:,.0f}<extra></extra>'
    ))

    fig.update_layout(
        title=f"Evolution du Revenu Cumule - Projection sur {months} mois",
        xaxis_title="Mois",
        yaxis_title="Revenu Cumule (£)",
        height=450,
        hovermode='x unified',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


def create_sensitivity_chart(params, impacts):
    """Cree un graphique de sensibilite (tornado chart)."""

    # Trier les parametres par impact
    sorted_items = sorted(zip(params, impacts), key=lambda x: abs(x[1]), reverse=True)
    params_sorted, impacts_sorted = zip(*sorted_items)

    # Determiner les couleurs (positif = vert, negatif = rouge)
    colors = ['#2ca02c' if impact > 0 else '#d62728' for impact in impacts_sorted]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=impacts_sorted,
        y=params_sorted,
        orientation='h',
        marker_color=colors,
        text=[f"{impact:+.1f}%" for impact in impacts_sorted],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Impact: %{x:.2f}%<extra></extra>'
    ))

    fig.update_layout(
        title="Analyse de Sensibilite - Impact de Chaque Parametre",
        xaxis_title="Impact sur le Revenu (%)",
        yaxis_title="Parametres",
        height=400,
        showlegend=False
    )

    # Ajouter une ligne verticale à 0
    fig.add_vline(x=0, line_width=2, line_color="black", line_dash="solid")

    return fig


def generate_insights(results, params):
    """Genere des insights automatiques bases sur les resultats."""

    insights = []

    # Analyse du gain de revenu
    revenue_gain_pct = results['delta']['revenue_pct']
    if revenue_gain_pct > 20:
        insights.append(f"**Impact Majeur** : Le scenario projette une augmentation de revenu de {revenue_gain_pct:.1f}%, ce qui represente un gain significatif.")
    elif revenue_gain_pct > 10:
        insights.append(f"**Impact Important** : Le scenario projette une augmentation de revenu de {revenue_gain_pct:.1f}%, un resultat tres positif.")
    elif revenue_gain_pct > 5:
        insights.append(f"**Impact Modere** : Le scenario projette une augmentation de revenu de {revenue_gain_pct:.1f}%, un progres notable.")
    else:
        insights.append(f"**Impact Limite** : Le scenario projette une augmentation de revenu de {revenue_gain_pct:.1f}%, un impact modeste.")

    # Identifier le levier le plus important
    param_impacts = {
        'Retention': params.get('retention_delta', 0) * results['current']['revenue'] * 1.5,
        'AOV': params.get('aov_increase', 0) * results['current']['revenue'],
        'Frequence': params.get('frequency_increase', 0) * results['current']['revenue'] * 1.2,
        'Croissance Client': params.get('customer_growth', 0) * results['current']['revenue']
    }

    max_lever = max(param_impacts.items(), key=lambda x: abs(x[1]))
    insights.append(f"**Levier Principal** : L'amelioration de '{max_lever[0]}' est le facteur le plus impactant de ce scenario.")

    # Analyse de la retention
    retention_delta = params.get('retention_delta', 0)
    if retention_delta > 0.05:
        insights.append(f"**Retention Cle** : Une amelioration de +{retention_delta*100:.0f}% de la retention aura un effet multiplicateur sur la valeur client long-terme.")

    # Analyse de la faisabilite
    if revenue_gain_pct > 30:
        insights.append("**Faisabilite** : Ce scenario est tres ambitieux. Assurez-vous d'avoir les ressources et la strategie necessaires pour atteindre ces objectifs.")
    elif revenue_gain_pct > 15:
        insights.append("**Faisabilite** : Ce scenario est realiste avec une execution rigoureuse et des investissements cibles.")
    else:
        insights.append("**Faisabilite** : Ce scenario est conservateur et devrait etre realisable avec des optimisations incrementales.")

    # Recommandations d'actions
    insights.append("\n**Actions Prioritaires Recommandees** :")
    if params.get('retention_delta', 0) > 0:
        insights.append("- Lancer un programme de fidelisation pour ameliorer la retention")
    if params.get('aov_increase', 0) > 0:
        insights.append("- Implementer des techniques d'upselling et cross-selling")
    if params.get('frequency_increase', 0) > 0:
        insights.append("- Augmenter la frequence des communications marketing (emails, notifications)")
    if params.get('customer_growth', 0) > 0:
        insights.append("- Investir dans l'acquisition de nouveaux clients (SEA, SEO, partenariats)")

    return insights


# EN-TETE DE LA PAGE
st.title("Simulation de Scenarios Marketing")
st.markdown("""
Simulez l'impact de differentes strategies marketing sur vos KPIs
et projetez les revenus futurs bases sur des hypotheses d'amelioration.
""")

st.divider()


# VERIFICATION DES DONNEES
if not st.session_state.get('data_loaded', False):
    st.warning("Veuillez d'abord charger les donnees depuis la page d'accueil.")
    st.stop()


# KPIS ACTUELS (BASELINE)
st.header("Situation Actuelle (Baseline)")

df = st.session_state.get('df_clean', None)

if df is not None:
    # Appliquer les filtres globaux
    active_filters = st.session_state.get('active_filters', {})
    df = utils.apply_global_filters(df, active_filters)

    # Calculer les KPIs actuels
    with st.spinner("Calcul des KPIs actuels..."):
        baseline_kpis = utils.calculate_kpis(df)

    # Stocker en session state pour reutilisation
    st.session_state.baseline_kpis = baseline_kpis

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Revenu annuel",
            value=format_currency(baseline_kpis['total_revenue']),
            help="Revenu total sur la periode analysee"
        )

    with col2:
        st.metric(
            label="Nombre de clients",
            value=format_number(baseline_kpis['total_customers']),
            help="Nombre de clients uniques"
        )

    with col3:
        st.metric(
            label="Taux de retention",
            value=format_percentage(baseline_kpis['retention_rate'] * 100),
            help="Pourcentage de clients qui effectuent plus d'un achat"
        )

    with col4:
        st.metric(
            label="Panier moyen",
            value=format_currency(baseline_kpis['avg_order_value']),
            help="Valeur moyenne d'une commande"
        )

    # KPIs additionnels en expander
    with st.expander("Voir plus de metriques"):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="Nombre de transactions",
                value=format_number(baseline_kpis['total_transactions']),
                help="Nombre total de commandes"
            )

        with col2:
            st.metric(
                label="Frequence d'achat",
                value=f"{baseline_kpis['purchase_frequency']:.2f}",
                help="Nombre moyen de commandes par client"
            )

        with col3:
            st.metric(
                label="CLV moyenne",
                value=format_currency(baseline_kpis['avg_clv']),
                help="Valeur vie client moyenne estimee"
            )

        with col4:
            st.metric(
                label="Clients actifs (90j)",
                value=format_number(baseline_kpis['active_customers']),
                help="Clients ayant achete dans les 90 derniers jours"
            )

else:
    st.error("Erreur lors du chargement des donnees")
    st.stop()

st.divider()


# SELECTION DU SCENARIO
st.header("Configuration du Scenario")

# Choix entre scenarios predefinis ou personnalise
scenario_type = st.radio(
    "Type de scenario",
    ["Scenarios predefinis", "Scenario personnalise"],
    horizontal=True,
    help="Choisir un scenario predefini ou creer votre propre scenario"
)

st.divider()

# Initialiser les parametres de simulation
simulation_params = None

if scenario_type == "Scenarios predefinis":
    # SCENARIOS PREDEFINIS
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Scenario Optimiste")
        st.markdown(f"""
        **Hypotheses :**
        - Retention : +{config.SIMULATION_SCENARIOS['optimistic']['retention_increase']*100:.0f}%
        - Panier moyen : +{config.SIMULATION_SCENARIOS['optimistic']['aov_increase']*100:.0f}%
        - Frequence : +{config.SIMULATION_SCENARIOS['optimistic']['frequency_increase']*100:.0f}%

        **Contexte :**
        Lancement reussi de programme fidelite
        + campagne marketing majeure
        """)

        if st.button("Simuler Optimiste", use_container_width=True, type="primary"):
            simulation_params = {
                'retention_delta': config.SIMULATION_SCENARIOS['optimistic']['retention_increase'],
                'aov_increase': config.SIMULATION_SCENARIOS['optimistic']['aov_increase'],
                'frequency_increase': config.SIMULATION_SCENARIOS['optimistic']['frequency_increase'],
                'customer_growth': 0.10,
                'margin_pct': 0.4,
                'discount_pct': 0.0
            }
            st.session_state.current_scenario = 'optimistic'
            st.session_state.simulation_params = simulation_params

    with col2:
        st.subheader("Scenario Realiste")
        st.markdown(f"""
        **Hypotheses :**
        - Retention : +{config.SIMULATION_SCENARIOS['realistic']['retention_increase']*100:.0f}%
        - Panier moyen : +{config.SIMULATION_SCENARIOS['realistic']['aov_increase']*100:.0f}%
        - Frequence : +{config.SIMULATION_SCENARIOS['realistic']['frequency_increase']*100:.0f}%

        **Contexte :**
        Amelioration progressive
        + optimisations incrementales
        """)

        if st.button("Simuler Realiste", use_container_width=True):
            simulation_params = {
                'retention_delta': config.SIMULATION_SCENARIOS['realistic']['retention_increase'],
                'aov_increase': config.SIMULATION_SCENARIOS['realistic']['aov_increase'],
                'frequency_increase': config.SIMULATION_SCENARIOS['realistic']['frequency_increase'],
                'customer_growth': 0.05,
                'margin_pct': 0.4,
                'discount_pct': 0.0
            }
            st.session_state.current_scenario = 'realistic'
            st.session_state.simulation_params = simulation_params

    with col3:
        st.subheader("Scenario Conservateur")
        st.markdown(f"""
        **Hypotheses :**
        - Retention : +{config.SIMULATION_SCENARIOS['conservative']['retention_increase']*100:.0f}%
        - Panier moyen : +{config.SIMULATION_SCENARIOS['conservative']['aov_increase']*100:.0f}%
        - Frequence : +{config.SIMULATION_SCENARIOS['conservative']['frequency_increase']*100:.0f}%

        **Contexte :**
        Petites ameliorations
        + approche prudente
        """)

        if st.button("Simuler Conservateur", use_container_width=True):
            simulation_params = {
                'retention_delta': config.SIMULATION_SCENARIOS['conservative']['retention_increase'],
                'aov_increase': config.SIMULATION_SCENARIOS['conservative']['aov_increase'],
                'frequency_increase': config.SIMULATION_SCENARIOS['conservative']['frequency_increase'],
                'customer_growth': 0.02,
                'margin_pct': 0.4,
                'discount_pct': 0.0
            }
            st.session_state.current_scenario = 'conservative'
            st.session_state.simulation_params = simulation_params

else:
    # SCENARIO PERSONNALISE
    st.subheader("Parametres personnalises")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Parametres d'Amelioration")

        st.markdown("**Taux de retention**")
        st.caption("Impact sur la fidelite client et les achats repetes")
        retention_increase = st.slider(
            "Variation du taux de retention",
            min_value=-20,
            max_value=50,
            value=5,
            step=1,
            format="%d%%",
            help="Augmentation ou diminution du taux de retention (en points de pourcentage)",
            label_visibility="collapsed"
        )

        st.markdown("**Panier moyen (AOV)**")
        st.caption("Impact sur le montant depense par transaction")
        aov_increase = st.slider(
            "Variation du panier moyen",
            min_value=-10,
            max_value=30,
            value=10,
            step=1,
            format="%d%%",
            help="Augmentation ou diminution du montant moyen par commande",
            label_visibility="collapsed"
        )

        st.markdown("**Frequence d'achat**")
        st.caption("Impact sur le nombre de commandes par client")
        frequency_increase = st.slider(
            "Variation de la frequence d'achat",
            min_value=-10,
            max_value=40,
            value=15,
            step=1,
            format="%d%%",
            help="Augmentation ou diminution du nombre de commandes par client",
            label_visibility="collapsed"
        )

    with col2:
        st.markdown("#### Parametres de Marche")

        st.markdown("**Croissance de la base client**")
        st.caption("Impact sur le nombre total de clients")
        customer_growth = st.slider(
            "Variation du nombre de clients",
            min_value=-10,
            max_value=50,
            value=10,
            step=1,
            format="%d%%",
            help="Augmentation ou diminution du nombre total de clients",
            label_visibility="collapsed"
        )

        st.markdown("**Marge brute**")
        st.caption("Pourcentage de marge sur les ventes")
        margin_pct = st.slider(
            "Marge brute",
            min_value=10,
            max_value=80,
            value=40,
            step=5,
            format="%d%%",
            help="Marge brute moyenne sur les produits",
            label_visibility="collapsed"
        )

        st.markdown("**Remise moyenne**")
        st.caption("Impact des promotions et reductions")
        discount_pct = st.slider(
            "Remise moyenne appliquee",
            min_value=0,
            max_value=30,
            value=0,
            step=1,
            format="%d%%",
            help="Pourcentage moyen de remise applique",
            label_visibility="collapsed"
        )

    st.divider()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("#### Parametres Financiers")

        col_a, col_b = st.columns(2)

        with col_a:
            forecast_months = st.number_input(
                "Periode de projection (mois)",
                min_value=3,
                max_value=36,
                value=12,
                step=3,
                help="Duree de la projection en mois"
            )

        with col_b:
            strategy_cost = st.number_input(
                "Cout estime de la strategie (£)",
                min_value=0,
                max_value=1000000,
                value=50000,
                step=10000,
                help="Investissement necessaire pour atteindre ces objectifs"
            )

    with col2:
        st.markdown("#### Lancer la Simulation")
        st.markdown("")  # Spacing
        if st.button("Simuler le Scenario", use_container_width=True, type="primary"):
            simulation_params = {
                'retention_delta': retention_increase / 100,
                'aov_increase': aov_increase / 100,
                'frequency_increase': frequency_increase / 100,
                'customer_growth': customer_growth / 100,
                'margin_pct': margin_pct / 100,
                'discount_pct': discount_pct / 100,
                'forecast_months': forecast_months,
                'strategy_cost': strategy_cost
            }
            st.session_state.current_scenario = 'custom'
            st.session_state.simulation_params = simulation_params

st.divider()


# RESULTATS DE LA SIMULATION
st.header("Resultats de la Simulation")

# Verifier si une simulation a ete lancee
if 'simulation_params' in st.session_state and st.session_state.simulation_params is not None:

    params = st.session_state.simulation_params

    # Executer la simulation
    with st.spinner("Simulation en cours..."):
        results = utils.simulate_scenario(df, params)

    # Stocker les resultats
    st.session_state.simulation_results = results

    # Calculer ROI si cout fourni
    strategy_cost = params.get('strategy_cost', 0)
    revenue_gain = results['delta']['revenue']
    margin_pct = params.get('margin_pct', 0.4)
    profit_gain = revenue_gain * margin_pct

    if strategy_cost > 0:
        roi = calculate_roi(profit_gain, strategy_cost)
    else:
        roi = 0

    # Affichage des resultats comparatifs
    st.subheader("Impact Projete")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Revenu projete",
            value=format_currency(results['projected']['revenue']),
            delta=format_currency(results['delta']['revenue']),
            delta_color="normal",
            help="Revenu total apres application du scenario"
        )

    with col2:
        st.metric(
            label="Clients projetes",
            value=format_number(results['projected']['customers']),
            delta=f"+{results['delta']['customers']:,.0f}",
            help="Nombre de clients apres croissance"
        )

    with col3:
        st.metric(
            label="CLV projetee",
            value=format_currency(results['projected']['clv']),
            delta=format_currency(results['delta']['clv']),
            help="Customer Lifetime Value moyenne projetee"
        )

    with col4:
        if strategy_cost > 0:
            st.metric(
                label="ROI",
                value=f"{roi:,.0f}%",
                delta=None,
                help=f"Retour sur investissement (Gain: {format_currency(profit_gain)}, Cout: {format_currency(strategy_cost)})"
            )
        else:
            st.metric(
                label="Gain de revenu",
                value=format_percentage(results['delta']['revenue_pct']),
                delta=None,
                help="Augmentation relative du revenu"
            )

    st.divider()

    # Graphiques comparatifs
    st.subheader("Visualisations Comparatives")

    tab1, tab2, tab3 = st.tabs(["Comparaison KPIs", "Evolution projetee", "Sensibilite"])

    with tab1:
        # Bar chart comparant baseline vs projete
        fig_comparison = create_comparison_bar_chart(results['current'], results['projected'])
        st.plotly_chart(fig_comparison, use_container_width=True)

        # Tableau de comparaison detaille
        st.markdown("#### Comparaison Detaillee")

        comparison_data = {
            'KPI': [
                'Revenu Annuel (£)',
                'Nombre de Clients',
                'Nombre de Transactions',
                'Panier Moyen (£)',
                'Frequence d\'Achat',
                'Taux de Retention',
                'CLV Moyenne (£)'
            ],
            'Actuel': [
                format_number(results['current']['revenue']),
                format_number(results['current']['customers']),
                format_number(results['current']['transactions']),
                f"{results['current']['aov']:.2f}",
                f"{results['current']['frequency']:.2f}",
                format_percentage(results['current']['retention_rate'] * 100),
                format_number(results['current']['clv'])
            ],
            'Projete': [
                format_number(results['projected']['revenue']),
                format_number(results['projected']['customers']),
                format_number(results['projected']['transactions']),
                f"{results['projected']['aov']:.2f}",
                f"{results['projected']['frequency']:.2f}",
                format_percentage(results['projected']['retention_rate'] * 100),
                format_number(results['projected']['clv'])
            ],
            'Delta Absolu': [
                format_currency(results['projected']['revenue'] - results['current']['revenue']),
                format_number(results['projected']['customers'] - results['current']['customers']),
                format_number(results['projected']['transactions'] - results['current']['transactions']),
                f"{results['projected']['aov'] - results['current']['aov']:.2f}",
                f"{results['projected']['frequency'] - results['current']['frequency']:.2f}",
                f"{(results['projected']['retention_rate'] - results['current']['retention_rate'])*100:+.1f}%",
                format_currency(results['projected']['clv'] - results['current']['clv'])
            ],
            'Delta %': [
                format_percentage((results['projected']['revenue'] / results['current']['revenue'] - 1) * 100 if results['current']['revenue'] > 0 else 0),
                format_percentage((results['projected']['customers'] / results['current']['customers'] - 1) * 100 if results['current']['customers'] > 0 else 0),
                format_percentage((results['projected']['transactions'] / results['current']['transactions'] - 1) * 100 if results['current']['transactions'] > 0 else 0),
                format_percentage((results['projected']['aov'] / results['current']['aov'] - 1) * 100 if results['current']['aov'] > 0 else 0),
                format_percentage((results['projected']['frequency'] / results['current']['frequency'] - 1) * 100 if results['current']['frequency'] > 0 else 0),
                format_percentage((results['projected']['retention_rate'] / results['current']['retention_rate'] - 1) * 100 if results['current']['retention_rate'] > 0 else 0),
                format_percentage((results['projected']['clv'] / results['current']['clv'] - 1) * 100 if results['current']['clv'] > 0 else 0)
            ]
        }

        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    with tab2:
        # Line chart montrant l'evolution projetee
        forecast_months = params.get('forecast_months', 12)
        fig_evolution = create_evolution_chart(
            results['current']['revenue'],
            results['projected']['revenue'],
            forecast_months
        )
        st.plotly_chart(fig_evolution, use_container_width=True)

        st.info(f"""
        **Interpretation** : Ce graphique montre l'evolution cumulative du revenu sur {forecast_months} mois.

        - **Ligne grise pointillee** : Tendance actuelle (stagnation)
        - **Ligne verte** : Projection avec le scenario simule
        - **Zone verte claire** : Intervalle de confiance (±10%)

        L'ecart entre les deux lignes represente le gain potentiel de la strategie.
        """)

    with tab3:
        # Analyse de sensibilite
        st.markdown("#### Impact de Chaque Parametre")

        # Calculer l'impact de chaque parametre individuellement
        param_names = []
        param_impacts = []

        # Impact retention
        if params.get('retention_delta', 0) != 0:
            retention_only = {'retention_delta': params['retention_delta'], 'aov_increase': 0, 'frequency_increase': 0, 'customer_growth': 0, 'margin_pct': params.get('margin_pct', 0.4), 'discount_pct': 0}
            result_retention = utils.simulate_scenario(df, retention_only)
            param_names.append('Amelioration Retention')
            param_impacts.append(result_retention['delta']['revenue_pct'])

        # Impact AOV
        if params.get('aov_increase', 0) != 0:
            aov_only = {'retention_delta': 0, 'aov_increase': params['aov_increase'], 'frequency_increase': 0, 'customer_growth': 0, 'margin_pct': params.get('margin_pct', 0.4), 'discount_pct': params.get('discount_pct', 0)}
            result_aov = utils.simulate_scenario(df, aov_only)
            param_names.append('Augmentation Panier Moyen')
            param_impacts.append(result_aov['delta']['revenue_pct'])

        # Impact Frequency
        if params.get('frequency_increase', 0) != 0:
            freq_only = {'retention_delta': 0, 'aov_increase': 0, 'frequency_increase': params['frequency_increase'], 'customer_growth': 0, 'margin_pct': params.get('margin_pct', 0.4), 'discount_pct': 0}
            result_freq = utils.simulate_scenario(df, freq_only)
            param_names.append('Augmentation Frequence')
            param_impacts.append(result_freq['delta']['revenue_pct'])

        # Impact Customer Growth
        if params.get('customer_growth', 0) != 0:
            growth_only = {'retention_delta': 0, 'aov_increase': 0, 'frequency_increase': 0, 'customer_growth': params['customer_growth'], 'margin_pct': params.get('margin_pct', 0.4), 'discount_pct': 0}
            result_growth = utils.simulate_scenario(df, growth_only)
            param_names.append('Croissance Base Client')
            param_impacts.append(result_growth['delta']['revenue_pct'])

        if len(param_names) > 0:
            fig_sensitivity = create_sensitivity_chart(param_names, param_impacts)
            st.plotly_chart(fig_sensitivity, use_container_width=True)

            # Identifier le levier principal
            max_impact_idx = np.argmax([abs(x) for x in param_impacts])
            st.success(f"""
            **Levier Principal** : '{param_names[max_impact_idx]}' a l'impact le plus important
            sur le revenu avec une contribution de {param_impacts[max_impact_idx]:+.1f}%.
            """)
        else:
            st.warning("Aucun parametre modifie. Ajustez les parametres pour voir l'analyse de sensibilite.")

    st.divider()

    # Insights et recommandations
    st.subheader("Insights et Recommandations")

    with st.expander("Analyse Automatique de la Simulation", expanded=True):
        insights = generate_insights(results, params)
        for insight in insights:
            st.markdown(insight)

else:
    st.info("Selectionnez et lancez un scenario ci-dessus pour voir les resultats de la simulation.")

st.divider()


# COMPARAISON DE SCENARIOS
st.header("Comparaison de Scenarios")

st.markdown("""
Comparez plusieurs scenarios cote a cote pour identifier la meilleure strategie.
""")

# Initialiser le stockage des scenarios
if 'saved_scenarios' not in st.session_state:
    st.session_state.saved_scenarios = []

col1, col2 = st.columns([3, 1])

with col1:
    if 'simulation_results' in st.session_state:
        scenario_name = st.text_input(
            "Nom du scenario",
            value=st.session_state.get('current_scenario', 'Mon scenario').replace('_', ' ').title(),
            help="Donnez un nom a ce scenario pour le sauvegarder"
        )

with col2:
    st.markdown("")  # Spacing
    st.markdown("")  # Spacing
    if st.button("Sauvegarder ce scenario", use_container_width=True):
        if 'simulation_results' in st.session_state:
            scenario_data = {
                'name': scenario_name,
                'results': st.session_state.simulation_results,
                'params': st.session_state.simulation_params,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M")
            }
            st.session_state.saved_scenarios.append(scenario_data)
            st.success(f"Scenario '{scenario_name}' sauvegarde !")
        else:
            st.warning("Aucune simulation a sauvegarder. Lancez d'abord un scenario.")

# Afficher les scenarios sauvegardes
if len(st.session_state.saved_scenarios) > 0:
    st.markdown(f"**{len(st.session_state.saved_scenarios)} scenario(s) sauvegarde(s)**")

    # Creer un tableau comparatif
    comparison_list = []
    for scenario in st.session_state.saved_scenarios:
        comparison_list.append({
            'Scenario': scenario['name'],
            'Date': scenario['timestamp'],
            'Revenu Projete': format_currency(scenario['results']['projected']['revenue']),
            'Gain Revenu': format_currency(scenario['results']['delta']['revenue']),
            'Gain %': format_percentage(scenario['results']['delta']['revenue_pct']),
            'Clients Projetes': format_number(scenario['results']['projected']['customers']),
            'CLV Projetee': format_currency(scenario['results']['projected']['clv'])
        })

    comparison_df = pd.DataFrame(comparison_list)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    # Option pour effacer les scenarios
    if st.button("Effacer tous les scenarios sauvegardes"):
        st.session_state.saved_scenarios = []
        st.rerun()

    # Radar chart si au moins 2 scenarios
    if len(st.session_state.saved_scenarios) >= 2:
        st.markdown("#### Comparaison Radar")

        fig_radar = go.Figure()

        categories = ['Revenu', 'Clients', 'Frequence', 'Panier Moyen', 'Retention']

        for scenario in st.session_state.saved_scenarios:
            # Normaliser les valeurs pour le radar
            baseline = st.session_state.baseline_kpis
            r = scenario['results']

            values = [
                (r['projected']['revenue'] / baseline['total_revenue']) * 100,
                (r['projected']['customers'] / baseline['total_customers']) * 100,
                (r['projected']['frequency'] / baseline['purchase_frequency']) * 100,
                (r['projected']['aov'] / baseline['avg_order_value']) * 100,
                (r['projected']['retention_rate'] / baseline['retention_rate']) * 100
            ]

            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=scenario['name']
            ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[80, 130]
                )
            ),
            showlegend=True,
            title="Comparaison Multi-Scenarios (Base 100 = Actuel)"
        )

        st.plotly_chart(fig_radar, use_container_width=True)

else:
    st.info("Aucun scenario sauvegarde. Lancez et sauvegardez des scenarios pour les comparer.")

st.divider()


# EXPORT
st.header("Export de la Simulation")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Exporter resultats (CSV)", use_container_width=True):
        if 'simulation_results' in st.session_state:
            results = st.session_state.simulation_results

            # Creer un DataFrame avec les resultats
            export_data = {
                'Metrique': ['Revenu', 'Clients', 'Transactions', 'AOV', 'Frequence', 'Retention', 'CLV'],
                'Valeur_Actuelle': [
                    results['current']['revenue'],
                    results['current']['customers'],
                    results['current']['transactions'],
                    results['current']['aov'],
                    results['current']['frequency'],
                    results['current']['retention_rate'],
                    results['current']['clv']
                ],
                'Valeur_Projetee': [
                    results['projected']['revenue'],
                    results['projected']['customers'],
                    results['projected']['transactions'],
                    results['projected']['aov'],
                    results['projected']['frequency'],
                    results['projected']['retention_rate'],
                    results['projected']['clv']
                ],
                'Delta_Absolu': [
                    results['projected']['revenue'] - results['current']['revenue'],
                    results['projected']['customers'] - results['current']['customers'],
                    results['projected']['transactions'] - results['current']['transactions'],
                    results['projected']['aov'] - results['current']['aov'],
                    results['projected']['frequency'] - results['current']['frequency'],
                    results['projected']['retention_rate'] - results['current']['retention_rate'],
                    results['projected']['clv'] - results['current']['clv']
                ]
            }

            export_df = pd.DataFrame(export_data)

            # Convertir en CSV
            csv = export_df.to_csv(index=False)

            st.download_button(
                label="Telecharger CSV",
                data=csv,
                file_name=f"simulation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.warning("Aucune simulation a exporter")

with col2:
    if st.button("Exporter comparaison (CSV)", use_container_width=True):
        if len(st.session_state.saved_scenarios) > 0:
            # Exporter le tableau de comparaison
            csv = comparison_df.to_csv(index=False)

            st.download_button(
                label="Telecharger Comparaison CSV",
                data=csv,
                file_name=f"scenarios_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.warning("Aucun scenario sauvegarde a exporter")

with col3:
    if st.button("Guide d'interpretation", use_container_width=True):
        st.info("""
        **Guide d'interpretation des resultats** :

        1. **Revenu projete** : Revenu total estime apres application du scenario
        2. **Delta** : Difference entre la situation actuelle et projetee
        3. **ROI** : Retour sur investissement = (Gain - Cout) / Cout
        4. **Sensibilite** : Montre quel levier a le plus d'impact
        5. **Evolution** : Projection mois par mois avec intervalle de confiance

        Les scenarios sont bases sur des hypotheses. Validez-les avec votre equipe avant implementation.
        """)


# FOOTER
st.divider()
st.caption("Page Simulation de Scenarios - Derniere mise a jour : " + datetime.now().strftime("%Y-%m-%d %H:%M"))
