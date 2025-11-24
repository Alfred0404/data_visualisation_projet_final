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


# CONFIGURATION DE LA PAGE
st.set_page_config(
    page_title="Overview - Marketing Analytics",
    page_icon=":material/home:",
    layout="wide"
)


# EN-TETE DE LA PAGE
st.title("Vue d'ensemble - KPIs Marketing")
st.markdown("""
Cette page présente une vue synthétique de vos principaux indicateurs de performance
et l'évolution de votre activité commerciale.
""")

st.divider()


# VERIFICATION DES DONNEES
if not st.session_state.get('data_loaded', False):
    st.warning("Veuillez d'abord charger les données depuis la page d'accueil.")
    st.stop()


# KPIS PRINCIPAUX
st.header("KPIs Principaux")

df = st.session_state.get('df_clean', None)

if df is not None:
    # Appliquer les filtres globaux
    active_filters = st.session_state.get('active_filters', {})
    df = utils.apply_global_filters(df, active_filters)
    kpis = st.session_state.get('kpis', {})
    if not kpis:
        kpis = utils.calculate_kpis(df)
        st.session_state.kpis = kpis

    # Ligne 1 de KPIs
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Clients Totaux",
            value=f"{kpis.get('total_customers', 0):,}",
            help="Nombre total de clients uniques"
        )

    with col2:
        st.metric(
            label="Revenu Total",
            value=utils.format_currency(kpis.get('total_revenue', 0)),
            help="Chiffre d'affaires total sur la période"
        )

    with col3:
        st.metric(
            label="Panier Moyen",
            value=utils.format_currency(kpis.get('avg_order_value', 0)),
            help="Valeur moyenne d'une transaction (AOV)"
        )

    with col4:
        st.metric(
            label="Fréquence d'Achat",
            value=f"{kpis.get('purchase_frequency', 0):.2f}",
            help="Nombre moyen de transactions par client"
        )

    st.divider()

    # Ligne 2 de KPIs
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Taux de Rétention",
            value=utils.format_percentage(kpis.get('retention_rate', 0)),
            help="Pourcentage de clients qui reviennent"
        )

    with col2:
        churn_rate = utils.calculate_churn_rate(df)
        st.metric(
            label="Taux de Churn",
            value=utils.format_percentage(churn_rate),
            delta_color="inverse",
            help="Pourcentage de clients perdus"
        )

    with col3:
        st.metric(
            label="CLV Moyenne",
            value=utils.format_currency(kpis.get('avg_clv', 0)),
            help="Customer Lifetime Value moyenne"
        )

    with col4:
        st.metric(
            label="Transactions",
            value=f"{kpis.get('total_transactions', 0):,}",
            help="Nombre total de transactions"
        )

else:
    st.error("Erreur lors du chargement des données")

st.divider()


# VISUALISATIONS PRINCIPALES
st.header("Évolution de l'Activité")

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


# ANALYSE PAR PAYS
st.header("Répartition Géographique")

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


# ANALYSE TEMPORELLE
st.header("Analyse Temporelle")

# Tabs pour différentes analyses temporelles
tab1, tab2, tab3 = st.tabs(["Évolution mensuelle", "Saisonnalité", "Tendances"])

with tab1:
    st.subheader("Évolution mensuelle des principaux KPIs")

    try:
        # Préparer les données mensuelles
        df_monthly = df[~df['IsReturn']].copy()
        df_monthly_agg = df_monthly.set_index('InvoiceDate').resample('M').agg({
            'TotalAmount': 'sum',
            'Customer ID': 'nunique',
            'Invoice': 'nunique'
        }).reset_index()

        # Calculer le panier moyen
        df_monthly_agg['AOV'] = df_monthly_agg['TotalAmount'] / df_monthly_agg['Invoice']

        # Créer le graphique avec double axe Y
        fig = go.Figure()

        # Revenu (axe gauche)
        fig.add_trace(go.Scatter(
            x=df_monthly_agg['InvoiceDate'],
            y=df_monthly_agg['TotalAmount'],
            name='Revenu',
            mode='lines+markers',
            line=dict(color='#1f77b4', width=2),
            yaxis='y'
        ))

        # Panier moyen (axe gauche)
        fig.add_trace(go.Scatter(
            x=df_monthly_agg['InvoiceDate'],
            y=df_monthly_agg['AOV'],
            name='Panier Moyen',
            mode='lines+markers',
            line=dict(color='#2ca02c', width=2),
            yaxis='y'
        ))

        # Nombre de clients (axe droit)
        fig.add_trace(go.Scatter(
            x=df_monthly_agg['InvoiceDate'],
            y=df_monthly_agg['Customer ID'],
            name='Clients Actifs',
            mode='lines+markers',
            line=dict(color='#ff7f0e', width=2),
            yaxis='y2'
        ))

        # Configuration des axes
        fig.update_layout(
            title="Évolution des KPIs Mensuels",
            xaxis=dict(title="Mois"),
            yaxis=dict(
                title="Revenu (GBP) / Panier Moyen (GBP)",
                side='left'
            ),
            yaxis2=dict(
                title="Nombre de Clients",
                overlaying='y',
                side='right'
            ),
            hovermode='x unified',
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Erreur lors de la création du graphique d'évolution mensuelle: {str(e)}")

with tab2:
    st.subheader("Analyse de la saisonnalité")

    try:
        # Préparer les données pour la heatmap de saisonnalité
        df_season = df[~df['IsReturn']].copy()
        df_season['YearMonth'] = df_season['InvoiceDate'].dt.to_period('M')

        # Agréger par année et mois
        season_data = df_season.groupby(['Year', 'Month'])['TotalAmount'].sum().reset_index()

        # Créer une matrice pivot pour la heatmap
        season_pivot = season_data.pivot(index='Year', columns='Month', values='TotalAmount')

        # Créer la heatmap
        fig = go.Figure(data=go.Heatmap(
            z=season_pivot.values,
            x=['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin',
               'Juil', 'Août', 'Sep', 'Oct', 'Nov', 'Déc'],
            y=season_pivot.index,
            colorscale='RdYlGn',
            text=season_pivot.values,
            texttemplate='%{text:,.0f}',
            textfont={"size": 10},
            colorbar=dict(title="Revenu (GBP)")
        ))

        fig.update_layout(
            title="Heatmap de Saisonnalité - Revenu par Mois et Année",
            xaxis_title="Mois",
            yaxis_title="Année",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        # Analyse complémentaire : revenu moyen par mois (tous les ans confondus)
        st.subheader("Revenu moyen par mois (toutes années confondues)")

        monthly_avg = season_data.groupby('Month')['TotalAmount'].mean().reset_index()

        fig2 = px.bar(
            monthly_avg,
            x='Month',
            y='TotalAmount',
            title="Revenu moyen par mois calendaire",
            labels={'TotalAmount': 'Revenu Moyen (GBP)', 'Month': 'Mois'}
        )

        fig2.update_xaxes(
            tickmode='array',
            tickvals=list(range(1, 13)),
            ticktext=['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin',
                     'Juil', 'Août', 'Sep', 'Oct', 'Nov', 'Déc']
        )

        st.plotly_chart(fig2, use_container_width=True)

        # Identifier le meilleur et le pire mois
        best_month = monthly_avg.loc[monthly_avg['TotalAmount'].idxmax()]
        worst_month = monthly_avg.loc[monthly_avg['TotalAmount'].idxmin()]

        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="Meilleur Mois",
                value=f"Mois {int(best_month['Month'])}",
                help=f"Revenu moyen: {utils.format_currency(best_month['TotalAmount'])}"
            )
        with col2:
            st.metric(
                label="Mois le Plus Faible",
                value=f"Mois {int(worst_month['Month'])}",
                help=f"Revenu moyen: {utils.format_currency(worst_month['TotalAmount'])}"
            )

    except Exception as e:
        st.error(f"Erreur lors de l'analyse de saisonnalité: {str(e)}")

with tab3:
    st.subheader("Tendances et prévisions")

    try:
        # Préparer les données mensuelles
        df_trend = df[~df['IsReturn']].copy()
        df_trend_monthly = df_trend.set_index('InvoiceDate').resample('M')['TotalAmount'].sum().reset_index()

        # Calculer la ligne de tendance (régression linéaire)
        df_trend_monthly['Month_Num'] = range(len(df_trend_monthly))

        # Régression linéaire simple
        x = df_trend_monthly['Month_Num'].values
        y = df_trend_monthly['TotalAmount'].values

        # Calcul des coefficients de régression
        n = len(x)
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)

        slope = numerator / denominator if denominator != 0 else 0
        intercept = y_mean - slope * x_mean

        # Calculer la ligne de tendance
        df_trend_monthly['Trend'] = slope * df_trend_monthly['Month_Num'] + intercept

        # Créer des prévisions pour les 3 prochains mois
        last_month_num = df_trend_monthly['Month_Num'].max()
        future_months = pd.DataFrame({
            'Month_Num': [last_month_num + 1, last_month_num + 2, last_month_num + 3]
        })

        # Calculer les dates futures
        last_date = df_trend_monthly['InvoiceDate'].max()
        future_months['InvoiceDate'] = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=3,
            freq='MS'
        )

        future_months['Forecast'] = slope * future_months['Month_Num'] + intercept

        # Créer le graphique
        fig = go.Figure()

        # Données historiques
        fig.add_trace(go.Scatter(
            x=df_trend_monthly['InvoiceDate'],
            y=df_trend_monthly['TotalAmount'],
            name='Revenu Réel',
            mode='lines+markers',
            line=dict(color='#1f77b4', width=2)
        ))

        # Ligne de tendance
        fig.add_trace(go.Scatter(
            x=df_trend_monthly['InvoiceDate'],
            y=df_trend_monthly['Trend'],
            name='Tendance',
            mode='lines',
            line=dict(color='#ff7f0e', width=2, dash='dash')
        ))

        # Prévisions
        fig.add_trace(go.Scatter(
            x=future_months['InvoiceDate'],
            y=future_months['Forecast'],
            name='Prévision (3 mois)',
            mode='lines+markers',
            line=dict(color='#2ca02c', width=2, dash='dot'),
            marker=dict(size=8)
        ))

        fig.update_layout(
            title="Tendance du Revenu et Prévisions",
            xaxis_title="Mois",
            yaxis_title="Revenu (GBP)",
            hovermode='x unified',
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        # Afficher les statistiques de tendance
        col1, col2, col3 = st.columns(3)

        with col1:
            trend_direction = "Croissance" if slope > 0 else "Décroissance"
            st.metric(
                label="Tendance",
                value=trend_direction,
                delta=f"{abs(slope):,.2f} GBP/mois"
            )

        with col2:
            growth_rate = (slope / y_mean) * 100 if y_mean != 0 else 0
            st.metric(
                label="Taux de Croissance",
                value=f"{growth_rate:.2f}%",
                help="Croissance mensuelle moyenne en pourcentage"
            )

        with col3:
            forecast_next_month = future_months.iloc[0]['Forecast']
            st.metric(
                label="Prévision M+1",
                value=utils.format_currency(forecast_next_month),
                help="Prévision pour le mois prochain"
            )

    except Exception as e:
        st.error(f"Erreur lors de la création du graphique de tendances: {str(e)}")

st.divider()


# TOP PERFORMERS
st.header("Top Performers")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Top 10 Produits (par revenu)")

    try:
        # Filtrer les données valides (sans retours)
        df_products = df[~df['IsReturn']].copy()

        # Grouper par produit
        top_products = df_products.groupby(['StockCode', 'Description']).agg({
            'TotalAmount': 'sum',
            'Quantity': 'sum',
            'Invoice': 'nunique'
        }).reset_index()

        # Renommer les colonnes
        top_products.columns = ['Code', 'Produit', 'Revenu', 'Quantité', 'Nb Commandes']

        # Trier par revenu
        top_products = top_products.nlargest(10, 'Revenu')

        # Formater le revenu
        top_products['Revenu'] = top_products['Revenu'].apply(lambda x: f"£{x:,.2f}")
        top_products['Quantité'] = top_products['Quantité'].apply(lambda x: f"{int(x):,}")

        # Afficher le tableau
        st.dataframe(
            top_products,
            hide_index=True,
            use_container_width=True,
            height=400
        )

    except Exception as e:
        st.error(f"Erreur lors du chargement des produits: {str(e)}")

with col2:
    st.subheader("Top 10 Clients (par revenu)")

    try:
        # Filtrer les données avec Customer ID
        df_customers = df[df['HasCustomerID'] & ~df['IsReturn']].copy()

        # Grouper par client
        top_customers = df_customers.groupby('Customer ID').agg({
            'TotalAmount': 'sum',
            'Invoice': 'nunique',
            'Country': 'first'
        }).reset_index()

        # Renommer les colonnes
        top_customers.columns = ['Client ID', 'Revenu', 'Nb Commandes', 'Pays']

        # Trier par revenu
        top_customers = top_customers.nlargest(10, 'Revenu')

        # Formater le revenu
        top_customers['Revenu'] = top_customers['Revenu'].apply(lambda x: f"£{x:,.2f}")

        # Afficher le tableau
        st.dataframe(
            top_customers,
            hide_index=True,
            use_container_width=True,
            height=400
        )

    except Exception as e:
        st.error(f"Erreur lors du chargement des clients: {str(e)}")

st.divider()


# ALERTES ET RECOMMANDATIONS
st.header("Alertes et Recommandations")

try:
    alerts = []
    recommendations = []

    # Analyser le taux de churn
    churn_rate = utils.calculate_churn_rate(df)
    if churn_rate > 0.30:  # 30% de churn
        alerts.append(("warning", f"Taux de churn élevé: {utils.format_percentage(churn_rate)}"))
        recommendations.append("Lancer une campagne de réactivation pour les clients inactifs")
    else:
        alerts.append(("success", f"Taux de churn acceptable: {utils.format_percentage(churn_rate)}"))

    # Analyser la rétention
    retention_rate = kpis.get('retention_rate', 0)
    if retention_rate < 0.20:  # Moins de 20% de rétention
        alerts.append(("warning", f"Taux de rétention faible: {utils.format_percentage(retention_rate)}"))
        recommendations.append("Améliorer le programme de fidélité et l'expérience client")
    else:
        alerts.append(("success", f"Taux de rétention satisfaisant: {utils.format_percentage(retention_rate)}"))

    # Analyser la tendance du revenu
    df_last_3m = df[df['InvoiceDate'] >= (df['InvoiceDate'].max() - pd.Timedelta(days=90))]
    df_prev_3m = df[(df['InvoiceDate'] >= (df['InvoiceDate'].max() - pd.Timedelta(days=180))) &
                    (df['InvoiceDate'] < (df['InvoiceDate'].max() - pd.Timedelta(days=90)))]

    revenue_last_3m = df_last_3m['TotalAmount'].sum()
    revenue_prev_3m = df_prev_3m['TotalAmount'].sum()

    if revenue_prev_3m > 0:
        revenue_change = ((revenue_last_3m - revenue_prev_3m) / revenue_prev_3m) * 100
        if revenue_change < -10:  # Baisse de plus de 10%
            alerts.append(("error", f"Baisse significative du revenu: {revenue_change:.1f}%"))
            recommendations.append("Analyser les causes de la baisse et ajuster la stratégie commerciale")
        elif revenue_change > 10:  # Croissance de plus de 10%
            alerts.append(("success", f"Forte croissance du revenu: +{revenue_change:.1f}%"))
            recommendations.append("Capitaliser sur cette dynamique positive avec des promotions ciblées")
        else:
            alerts.append(("info", f"Revenu stable: {revenue_change:+.1f}%"))

    # Afficher les alertes
    with st.expander("État de santé des KPIs", expanded=True):
        st.subheader("Alertes")

        for alert_type, message in alerts:
            if alert_type == "success":
                st.success(message)
            elif alert_type == "warning":
                st.warning(message)
            elif alert_type == "error":
                st.error(message)
            else:
                st.info(message)

        

except Exception as e:
    st.error(f"Erreur lors de la génération des alertes: {str(e)}")

st.divider()


# EXPORT
st.header("Export")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Exporter les données")

    try:
        # Préparer les KPIs pour l'export
        kpis_df = pd.DataFrame([kpis])

        # Convertir en CSV
        csv_kpis = kpis_df.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="Télécharger KPIs (CSV)",
            data=csv_kpis,
            file_name=f"kpis_overview_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

        # Export des top produits
        if 'top_products' in locals():
            csv_products = top_products.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Télécharger Top Produits (CSV)",
                data=csv_products,
                file_name=f"top_products_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

    except Exception as e:
        st.error(f"Erreur lors de l'export: {str(e)}")

with col2:
    st.subheader("Informations")
    st.info("""
    **Exports disponibles:**
    - KPIs au format CSV
    - Top 10 Produits
    - Top 10 Clients

    Pour exporter les visualisations, utilisez le menu de Plotly (icône appareil photo) en haut à droite de chaque graphique.
    """)


# FOOTER
st.divider()
st.caption("Page Overview - Dernière mise à jour : " + datetime.now().strftime("%Y-%m-%d %H:%M"))
