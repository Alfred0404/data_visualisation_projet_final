"""
Page Overview - Vue d'ensemble des KPIs marketing.

Cette page affiche les indicateurs clés de performance (KPIs) globaux
et des visualisations synthétiques pour avoir une vue d'ensemble
de l'activité commerciale.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Imports locaux
sys.path.append(str(Path(__file__).parent.parent.parent))
import config
import utils


# ==============================================================================
# CONFIGURATION DE LA PAGE
# ==============================================================================

st.set_page_config(
    page_title="Overview - Marketing Analytics",
    page_icon="🏠",
    layout="wide"
)


# ==============================================================================
# EN-TETE DE LA PAGE
# ==============================================================================

st.title("🏠 Vue d'ensemble - KPIs Marketing")
st.markdown("""
Cette page présente une vue synthétique de vos principaux indicateurs de performance
et l'évolution de votre activité commerciale.
""")

st.divider()


# ==============================================================================
# FILTRES SPECIFIQUES A LA PAGE
# ==============================================================================

with st.sidebar:
    st.subheader("🎯 Filtres - Overview")

    # TODO: Ajouter des filtres spécifiques
    # - Période de comparaison (MoM, QoQ, YoY)
    # - Segments à inclure
    # - etc.

    st.divider()


# ==============================================================================
# VERIFICATION DES DONNEES
# ==============================================================================

if not st.session_state.get('data_loaded', False):
    st.warning("⚠️ Veuillez d'abord charger les données depuis la page d'accueil.")
    st.stop()


# ==============================================================================
# KPIS PRINCIPAUX
# ==============================================================================

st.header("📊 KPIs Principaux")

df = st.session_state.get('df_clean', None)

if df is not None:
    kpis = st.session_state.get('kpis', {})
    if not kpis:
        kpis = utils.calculate_kpis(df)
        st.session_state.kpis = kpis

    # Ligne 1 de KPIs
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="👥 Clients Totaux",
            value=f"{kpis.get('total_customers', 0):,}",
            help="Nombre total de clients uniques"
        )

    with col2:
        st.metric(
            label="💰 Revenu Total",
            value=utils.format_currency(kpis.get('total_revenue', 0)),
            help="Chiffre d'affaires total sur la période"
        )

    with col3:
        st.metric(
            label="🛒 Panier Moyen",
            value=utils.format_currency(kpis.get('avg_order_value', 0)),
            help="Valeur moyenne d'une transaction (AOV)"
        )

    with col4:
        st.metric(
            label="🔁 Fréquence d'Achat",
            value=f"{kpis.get('purchase_frequency', 0):.2f}",
            help="Nombre moyen de transactions par client"
        )

    st.divider()

    # Ligne 2 de KPIs
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="📈 Taux de Rétention",
            value=utils.format_percentage(kpis.get('retention_rate', 0)),
            help="Pourcentage de clients qui reviennent"
        )

    with col2:
        churn_rate = utils.calculate_churn_rate(df)
        st.metric(
            label="📉 Taux de Churn",
            value=utils.format_percentage(churn_rate),
            delta_color="inverse",
            help="Pourcentage de clients perdus"
        )

    with col3:
        st.metric(
            label="💎 CLV Moyenne",
            value=utils.format_currency(kpis.get('avg_clv', 0)),
            help="Customer Lifetime Value moyenne"
        )

    with col4:
        st.metric(
            label="📦 Transactions",
            value=f"{kpis.get('total_transactions', 0):,}",
            help="Nombre total de transactions"
        )

else:
    st.error("Erreur lors du chargement des données")

st.divider()


# ==============================================================================
# VISUALISATIONS PRINCIPALES
# ==============================================================================

st.header("📈 Évolution de l'Activité")

# Layout en 2 colonnes
col1, col2 = st.columns(2)

with col1:
    st.subheader("Revenu mensuel")
    df_monthly_revenue = df.set_index('InvoiceDate').resample('M')['TotalAmount'].sum().reset_index()
    fig = px.line(df_monthly_revenue, x='InvoiceDate', y='TotalAmount', title="Revenu mensuel")
    fig.update_layout(xaxis_title="Mois", yaxis_title="Revenu")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Nombre de clients actifs")
    df_monthly_customers = df.set_index('InvoiceDate').resample('M')['Customer ID'].nunique().reset_index()
    fig = px.bar(df_monthly_customers, x='InvoiceDate', y='Customer ID', title="Clients actifs par mois")
    fig.update_layout(xaxis_title="Mois", yaxis_title="Nombre de clients")
    st.plotly_chart(fig, use_container_width=True)

st.divider()


# ==============================================================================
# ANALYSE PAR PAYS
# ==============================================================================

st.header("🌍 Répartition Géographique")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Revenu par pays (Top 10)")
    df_country_revenue = df.groupby('Country')['TotalAmount'].sum().nlargest(10).reset_index()
    fig = px.bar(df_country_revenue, x='TotalAmount', y='Country', orientation='h', title="Top 10 des pays par revenu")
    fig.update_layout(xaxis_title="Revenu", yaxis_title="Pays")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Distribution des clients")
    df_country_customers = df.groupby('Country')['Customer ID'].nunique().nlargest(10).reset_index()
    fig = px.pie(df_country_customers, values='Customer ID', names='Country', title="Distribution des clients par pays (Top 10)")
    st.plotly_chart(fig, use_container_width=True)

st.divider()


# ==============================================================================
# ANALYSE TEMPORELLE
# ==============================================================================

st.header("⏰ Analyse Temporelle")

# Tabs pour différentes analyses temporelles
tab1, tab2, tab3 = st.tabs(["📅 Évolution mensuelle", "📊 Saisonnalité", "📈 Tendances"])

with tab1:
    st.subheader("Évolution mensuelle des principaux KPIs")

    # TODO: Créer un graphique multi-lignes avec :
    # - Revenu mensuel (axe gauche)
    # - Nombre de clients (axe droit)
    # - Panier moyen (axe gauche)

    st.info("TODO: Graphique évolution mensuelle multi-KPIs")

with tab2:
    st.subheader("Analyse de la saisonnalité")

    # TODO: Créer un heatmap ou un graphique montrant :
    # - Revenu par mois de l'année (tous les mois de janvier, février, etc.)
    # - Identifier les pics saisonniers

    st.info("TODO: Analyse de saisonnalité")

with tab3:
    st.subheader("Tendances et prévisions")

    # TODO: Créer un graphique avec :
    # - Données historiques
    # - Ligne de tendance (régression)
    # - Optionnel : prévision simple

    st.info("TODO: Graphique de tendances")

st.divider()


# ==============================================================================
# TOP PERFORMERS
# ==============================================================================

st.header("🏆 Top Performers")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Top 10 Produits (par revenu)")

    # TODO: Afficher un tableau des 10 produits les plus vendus
    # - Grouper par StockCode/Description
    # - Calculer le revenu total
    # - Afficher avec st.dataframe() avec formatting

    st.info("TODO: Table top produits")

with col2:
    st.subheader("Top 10 Clients (par revenu)")

    # TODO: Afficher un tableau des 10 meilleurs clients
    # - Grouper par Customer ID
    # - Calculer le revenu total
    # - Afficher avec st.dataframe()

    st.info("TODO: Table top clients")

st.divider()


# ==============================================================================
# ALERTES ET RECOMMANDATIONS
# ==============================================================================

st.header("⚠️ Alertes et Recommandations")

# TODO: Implémenter un système d'alertes basé sur :
# - Taux de churn élevé (> seuil dans config)
# - Baisse de revenu significative
# - Segments à risque
# - Opportunités d'amélioration

# Exemple de structure :
with st.expander("📊 État de santé des KPIs", expanded=True):
    st.info("""
    **TODO: Alertes automatiques**
    - ✅ Taux de rétention : Normal
    - ⚠️ Taux de churn : Au-dessus du seuil
    - ✅ Revenu : Croissance stable
    - 💡 Recommandation : Activer campagne de réactivation
    """)

st.divider()


# ==============================================================================
# EXPORT
# ==============================================================================

st.header("📥 Export")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 Exporter les KPIs (CSV)", use_container_width=True):
        # TODO: Exporter les KPIs en CSV avec utils.export_to_csv()
        st.success("TODO: Implémenter l'export CSV")

with col2:
    if st.button("📈 Exporter les graphiques (PNG)", use_container_width=True):
        # TODO: Exporter les graphiques avec utils.export_chart_to_png()
        st.success("TODO: Implémenter l'export PNG")

with col3:
    if st.button("📄 Générer rapport PDF", use_container_width=True):
        # TODO: Générer un rapport PDF complet
        st.info("TODO: Implémenter la génération de rapport PDF")


# ==============================================================================
# FOOTER
# ==============================================================================

st.divider()
st.caption("Page Overview - Dernière mise à jour : " + datetime.now().strftime("%Y-%m-%d %H:%M"))
