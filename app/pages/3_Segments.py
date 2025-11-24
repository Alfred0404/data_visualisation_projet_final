"""
Page Segmentation RFM - Segmentation client basée sur Récence, Fréquence et Montant.

Cette page permet de segmenter les clients selon la méthodologie RFM
et d'identifier les segments à forte valeur pour optimiser les actions marketing.
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


# FONCTIONS AUXILIAIRES
@st.cache_data
def calculate_rfm_cached(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule RFM avec cache pour optimiser les performances."""
    df_customers = df[df['HasCustomerID']].copy()
    rfm = utils.calculate_rfm(df_customers)
    return rfm


def get_segment_recommendations(segment: str) -> dict:
    """Retourne les recommandations marketing pour un segment donné."""
    recommendations = {
        "Champions": {
            "actions": [
                "Programme VIP exclusif avec avantages premium",
                "Early access aux nouveaux produits et collections",
                "Personnalisation maximale de l'expérience",
                "Incentives pour recommandations et parrainage"
            ],
            "channels": ["Email personnalisé", "SMS VIP", "Téléphone"],
            "offers": "Offres exclusives, cadeaux premium, services sur-mesure",
            "objective": "Fidélisation maximale et augmentation du LTV",
            "priority": "Très Haute",
            "budget": "25-30% du budget marketing"
        },
        "Loyal Customers": {
            "actions": [
                "Programme de fidélité avec points et récompenses",
                "Cross-sell et up-sell personnalisés",
                "Communication régulière et ciblée",
                "Enquêtes de satisfaction et feedback"
            ],
            "channels": ["Email", "Application mobile", "Notifications"],
            "offers": "Réductions fidélité, offres groupées, avant-premières",
            "objective": "Maintenir l'engagement et augmenter la fréquence",
            "priority": "Haute",
            "budget": "20-25% du budget marketing"
        },
        "Potential Loyalists": {
            "actions": [
                "Nurturing avec contenu éducatif de valeur",
                "Onboarding personnalisé et accompagnement",
                "Incitations à la deuxième et troisième commande",
                "Programme de découverte produits"
            ],
            "channels": ["Email séquencé", "Retargeting", "Contenu blog"],
            "offers": "Réductions progressives, livraison offerte, bundles découverte",
            "objective": "Conversion en Loyal Customers sous 3-6 mois",
            "priority": "Haute",
            "budget": "15-20% du budget marketing"
        },
        "New Customers": {
            "actions": [
                "Onboarding structuré avec séquence de bienvenue",
                "Éducation sur les produits et la marque",
                "Incitation à la deuxième commande rapide",
                "Collecte de préférences et feedback"
            ],
            "channels": ["Email de bienvenue", "Retargeting", "SMS"],
            "offers": "Code de bienvenue, livraison gratuite, guide débutant",
            "objective": "Activation rapide et conversion en clients réguliers",
            "priority": "Moyenne-Haute",
            "budget": "10-15% du budget marketing"
        },
        "At Risk": {
            "actions": [
                "Campagne de réactivation urgente et ciblée",
                "Enquête de satisfaction pour comprendre l'inactivité",
                "Offre win-back agressive et limitée dans le temps",
                "Communication multi-canal coordonnée"
            ],
            "channels": ["Email urgent", "SMS", "Retargeting display", "Direct mail"],
            "offers": "Réduction importante (20-30%), frais de port offerts, bonus",
            "objective": "Réactivation immédiate et prévention du churn",
            "priority": "Très Haute",
            "budget": "15-20% du budget marketing"
        },
        "Cannot Lose Them": {
            "actions": [
                "Contact direct et personnalisé (téléphone)",
                "Analyse approfondie des raisons d'inactivité",
                "Offre de reconquête exceptionnelle et sur-mesure",
                "Garantie satisfaction et engagement renouvelé"
            ],
            "channels": ["Téléphone", "Email PDG", "Courrier personnalisé"],
            "offers": "Offre VIP de reconquête, cadeaux, consultation gratuite",
            "objective": "Reconquête à tout prix - ROI à long terme",
            "priority": "Critique",
            "budget": "10-15% du budget marketing (ciblé)"
        },
        "Hibernating": {
            "actions": [
                "Campagne de réactivation à bas coût",
                "Contenu de rappel de marque",
                "Segmentation pour identifier les plus prometteurs",
                "Test de différentes approches"
            ],
            "channels": ["Email automatisé", "Display retargeting"],
            "offers": "Réduction modérée, nouveautés, rappel de compte",
            "objective": "Réactivation opportuniste avec ROI positif",
            "priority": "Basse",
            "budget": "5-8% du budget marketing"
        },
        "Lost": {
            "actions": [
                "Campagne de reconquête à coût minimal",
                "Enquête de sortie pour apprentissage",
                "Offre de dernière chance automatisée",
                "Exclusion des listes actives pour optimiser coûts"
            ],
            "channels": ["Email générique", "Display"],
            "offers": "Grande réduction si automatisée, sinon supprimer",
            "objective": "Reconquête opportuniste ou nettoyage de base",
            "priority": "Très Basse",
            "budget": "2-5% du budget marketing"
        },
        "Others": {
            "actions": [
                "Analyse approfondie pour recatégoriser",
                "Monitoring du comportement",
                "Test de différentes approches",
                "Segmentation plus fine si volume important"
            ],
            "channels": ["Mix de canaux selon profil"],
            "offers": "Tests A/B de différentes offres",
            "objective": "Identification du potentiel et segmentation appropriée",
            "priority": "Moyenne",
            "budget": "5-10% du budget marketing"
        }
    }

    return recommendations.get(segment, recommendations["Others"])


def create_segment_color_map():
    """Crée une palette de couleurs cohérente pour les segments RFM."""
    return {
        "Champions": "#2ca02c",              # Vert foncé
        "Loyal Customers": "#7fbc41",        # Vert clair
        "Potential Loyalists": "#4daf4a",    # Vert moyen
        "New Customers": "#1f77b4",          # Bleu
        "At Risk": "#ff7f0e",                # Orange
        "Cannot Lose Them": "#d62728",       # Rouge
        "Hibernating": "#bcbd22",            # Jaune-vert
        "Lost": "#8c564b",                   # Marron
        "Others": "#7f7f7f"                  # Gris
    }


# CONFIGURATION DE LA PAGE
st.set_page_config(
    page_title="Segmentation RFM - Marketing Analytics",
    page_icon=":material/target:",
    layout="wide"
)


# EN-TETE DE LA PAGE
st.title("Segmentation RFM")
st.markdown("""
La segmentation RFM (Recency, Frequency, Monetary) permet d'identifier
les clients les plus précieux et de personnaliser les stratégies marketing.
""")

st.divider()


# EXPLICATION RFM
with st.expander("Comprendre la méthodologie RFM", expanded=False):
    st.markdown("""
    ### Qu'est-ce que le RFM ?

    Le **RFM** est une méthode de segmentation client basée sur trois dimensions :

    - **R - Recency (Récence)** : Quand le client a-t-il acheté pour la dernière fois ?
      - Plus récent = meilleur score (4)
      - Moins récent = score faible (1)

    - **F - Frequency (Fréquence)** : Combien de fois le client a-t-il acheté ?
      - Plus de transactions = meilleur score (4)
      - Peu de transactions = score faible (1)

    - **M - Monetary (Montant)** : Combien le client a-t-il dépensé au total ?
      - Montant élevé = meilleur score (4)
      - Montant faible = score faible (1)

    ### Comment ça fonctionne ?

    1. Chaque client reçoit un score de 1 à 4 pour chaque dimension
    2. Les scores sont combinés (ex: "444" = meilleur client)
    3. Les clients sont regroupés en segments marketing
    4. Chaque segment nécessite une stratégie adaptée

    ### Les segments principaux

    - **Champions (444)** : Meilleurs clients - fidélisation premium
    - **Loyal Customers** : Clients fidèles - programmes de fidélité
    - **Potential Loyalists** : Clients prometteurs - nurturing
    - **At Risk** : Clients à risque - campagnes de réactivation
    - **Lost** : Clients perdus - campagnes de reconquête
    """)

st.divider()


# FILTRES SPECIFIQUES
with st.sidebar:
    st.subheader("Filtres - RFM")

    reference_date = st.date_input(
        "Date de référence pour le calcul RFM",
        value=datetime.now().date(),
        help="Date à partir de laquelle calculer la récence"
    )

    st.divider()


# VERIFICATION DES DONNEES
if not st.session_state.get('data_loaded', False):
    st.warning("Veuillez d'abord charger les données depuis la page d'accueil.")
    st.stop()


# CALCUL RFM
st.header("Calcul des Scores RFM")

df = st.session_state.get('df_clean', None)

if df is not None:
    # Appliquer les filtres globaux
    active_filters = st.session_state.get('active_filters', {})
    df = utils.apply_global_filters(df, active_filters)

    try:
        # Calculer RFM avec cache
        with st.spinner("Calcul des scores RFM en cours..."):
            df_rfm = calculate_rfm_cached(df)
            st.session_state.df_rfm = df_rfm

        # Jointure avec les données originales pour avoir le revenu
        df_customers = df[df['HasCustomerID']].copy()
        customer_revenue = df_customers.groupby('Customer ID')['TotalAmount'].sum().reset_index()
        customer_revenue.columns = ['Customer ID', 'Total_Revenue']

        df_rfm_full = df_rfm.merge(customer_revenue, on='Customer ID', how='left')

        # Métriques globales RFM
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            n_segments = df_rfm['Segment'].nunique()
            st.metric(
                label="Nombre de segments",
                value=n_segments,
                help="Nombre de segments RFM identifiés"
            )

        with col2:
            n_champions = len(df_rfm[df_rfm['Segment'] == 'Champions'])
            st.metric(
                label="Champions",
                value=f"{n_champions:,}",
                help="Clients avec score RFM le plus élevé"
            )

        with col3:
            n_at_risk = len(df_rfm[df_rfm['Segment'] == 'At Risk'])
            st.metric(
                label="At Risk",
                value=f"{n_at_risk:,}",
                help="Clients à risque de churn"
            )

        with col4:
            n_lost = len(df_rfm[df_rfm['Segment'] == 'Lost'])
            st.metric(
                label="Lost",
                value=f"{n_lost:,}",
                help="Clients perdus (score faible)"
            )

    except Exception as e:
        st.error(f"Erreur lors du calcul RFM: {str(e)}")
        st.stop()

else:
    st.error("Erreur lors du chargement des données")
    st.stop()

st.divider()


# DISTRIBUTION DES SEGMENTS
st.header("Distribution des Segments")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Répartition des clients par segment")

    try:
        # Compter les clients par segment
        segment_counts = df_rfm['Segment'].value_counts().reset_index()
        segment_counts.columns = ['Segment', 'Count']

        # Créer un treemap
        color_map = create_segment_color_map()
        fig = px.treemap(
            segment_counts,
            path=['Segment'],
            values='Count',
            title="Distribution des clients par segment RFM",
            color='Segment',
            color_discrete_map=color_map
        )

        fig.update_traces(textinfo="label+value+percent parent")
        fig.update_layout(height=500)

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Erreur lors de la création du treemap: {str(e)}")

with col2:
    st.subheader("Contribution au revenu")

    try:
        # Calculer la contribution au revenu par segment
        segment_revenue = df_rfm_full.groupby('Segment')['Total_Revenue'].sum().reset_index()
        segment_revenue = segment_revenue.sort_values('Total_Revenue', ascending=False)

        # Créer un pie chart
        fig = px.pie(
            segment_revenue,
            values='Total_Revenue',
            names='Segment',
            title="Contribution au revenu par segment",
            color='Segment',
            color_discrete_map=color_map
        )

        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=500)

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Erreur lors de la création du pie chart: {str(e)}")

st.divider()


# MATRICE RFM
st.header("Matrice RFM")

st.markdown("""
Visualisation en 3D ou matricielle des scores RFM pour identifier les patterns.
""")

# Choix du type de visualisation
viz_type = st.radio(
    "Type de visualisation",
    ["Scatter 3D", "Heatmap R-F", "Heatmap R-M", "Heatmap F-M"],
    horizontal=True,
    help="Choisir le type de visualisation de la matrice RFM"
)

if viz_type == "Scatter 3D":
    try:
        # Créer un scatter plot 3D
        fig = px.scatter_3d(
            df_rfm_full,
            x='R_Score',
            y='F_Score',
            z='M_Score',
            color='Segment',
            color_discrete_map=color_map,
            hover_data=['Customer ID', 'Recency', 'Frequency', 'Monetary'],
            title="Matrice 3D des scores RFM",
            labels={'R_Score': 'Score Récence', 'F_Score': 'Score Fréquence', 'M_Score': 'Score Montant'}
        )

        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Erreur lors de la création du scatter 3D: {str(e)}")

else:
    try:
        # Déterminer les dimensions selon le choix
        if viz_type == "Heatmap R-F":
            x_col, y_col = 'F_Score', 'R_Score'
            x_label, y_label = 'Score Fréquence', 'Score Récence'
        elif viz_type == "Heatmap R-M":
            x_col, y_col = 'M_Score', 'R_Score'
            x_label, y_label = 'Score Montant', 'Score Récence'
        else:  # Heatmap F-M
            x_col, y_col = 'M_Score', 'F_Score'
            x_label, y_label = 'Score Montant', 'Score Fréquence'

        # Créer la matrice de comptage
        heatmap_data = df_rfm.groupby([y_col, x_col]).size().reset_index(name='Count')
        heatmap_pivot = heatmap_data.pivot(index=y_col, columns=x_col, values='Count').fillna(0)

        # Créer la heatmap
        fig = px.imshow(
            heatmap_pivot,
            labels=dict(x=x_label, y=y_label, color="Nombre de clients"),
            title=f"Heatmap {viz_type.replace('Heatmap ', '')}",
            color_continuous_scale='Blues',
            text_auto=True
        )

        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Erreur lors de la création de la heatmap: {str(e)}")

st.divider()


# PROFILS DETAILLES DES SEGMENTS
st.header("Profils Détaillés des Segments")

# Liste des segments disponibles
segments_available = sorted(df_rfm['Segment'].unique().tolist())

# Sélection du segment à analyser
selected_segment = st.selectbox(
    "Choisir un segment à analyser",
    segments_available,
    help="Sélectionner un segment pour voir son profil détaillé"
)

if selected_segment:
    try:
        # Filtrer les données pour le segment sélectionné
        segment_data = df_rfm_full[df_rfm_full['Segment'] == selected_segment]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Caractéristiques")

            n_customers = len(segment_data)
            pct_customers = (n_customers / len(df_rfm)) * 100

            st.metric("Nombre de clients", f"{n_customers:,}")
            st.metric("% du total", f"{pct_customers:.1f}%")
            st.metric("Score R moyen", f"{segment_data['R_Score'].mean():.2f}")
            st.metric("Score F moyen", f"{segment_data['F_Score'].mean():.2f}")
            st.metric("Score M moyen", f"{segment_data['M_Score'].mean():.2f}")

        with col2:
            st.subheader("Performance")

            total_revenue = segment_data['Total_Revenue'].sum()
            avg_revenue = segment_data['Total_Revenue'].mean()
            pct_revenue = (total_revenue / df_rfm_full['Total_Revenue'].sum()) * 100

            st.metric("Revenu total", f"£{total_revenue:,.2f}")
            st.metric("Revenu moyen/client", f"£{avg_revenue:,.2f}")
            st.metric("% du revenu total", f"{pct_revenue:.1f}%")

        with col3:
            st.subheader("Comportement")

            avg_recency = segment_data['Recency'].mean()
            avg_frequency = segment_data['Frequency'].mean()
            avg_monetary = segment_data['Monetary'].mean()

            st.metric("Récence moyenne", f"{avg_recency:.0f} jours")
            st.metric("Fréquence moyenne", f"{avg_frequency:.1f} achats")
            st.metric("Montant moyen", f"£{avg_monetary:,.2f}")

        # Graphiques supplémentaires pour le segment
        st.subheader(f"Analyses détaillées - {selected_segment}")

        tab1, tab2, tab3 = st.tabs(["Distribution", "Comparaison", "Recommandations"])

        with tab1:
            # Histogrammes de distribution
            fig = go.Figure()

            fig.add_trace(go.Histogram(
                x=segment_data['Recency'],
                name='Récence (jours)',
                marker_color='#1f77b4',
                opacity=0.7
            ))

            fig.update_layout(
                title=f"Distribution de la Récence - {selected_segment}",
                xaxis_title="Jours depuis dernier achat",
                yaxis_title="Nombre de clients",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

            # Second graphique pour Frequency et Monetary
            col_a, col_b = st.columns(2)

            with col_a:
                fig2 = px.histogram(
                    segment_data,
                    x='Frequency',
                    title="Distribution de la Fréquence",
                    labels={'Frequency': 'Nombre d\'achats'}
                )
                fig2.update_layout(height=300)
                st.plotly_chart(fig2, use_container_width=True)

            with col_b:
                fig3 = px.histogram(
                    segment_data,
                    x='Monetary',
                    title="Distribution du Montant",
                    labels={'Monetary': 'Montant total (£)'}
                )
                fig3.update_layout(height=300)
                st.plotly_chart(fig3, use_container_width=True)

        with tab2:
            # Comparaison avec les autres segments
            # Créer un radar chart comparatif
            all_segments_stats = df_rfm.groupby('Segment').agg({
                'R_Score': 'mean',
                'F_Score': 'mean',
                'M_Score': 'mean'
            }).reset_index()

            # Ajouter la moyenne du segment sélectionné
            selected_stats = all_segments_stats[all_segments_stats['Segment'] == selected_segment].iloc[0]

            # Créer un bar chart comparatif
            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=['Récence', 'Fréquence', 'Montant'],
                y=[selected_stats['R_Score'], selected_stats['F_Score'], selected_stats['M_Score']],
                name=selected_segment,
                marker_color=create_segment_color_map()[selected_segment]
            ))

            # Ajouter la moyenne globale
            fig.add_trace(go.Bar(
                x=['Récence', 'Fréquence', 'Montant'],
                y=[df_rfm['R_Score'].mean(), df_rfm['F_Score'].mean(), df_rfm['M_Score'].mean()],
                name='Moyenne globale',
                marker_color='gray',
                opacity=0.5
            ))

            fig.update_layout(
                title=f"Comparaison {selected_segment} vs Moyenne Globale",
                yaxis_title="Score moyen",
                height=400,
                barmode='group'
            )

            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            # Afficher les recommandations pour ce segment
            recs = get_segment_recommendations(selected_segment)

            st.markdown(f"### Stratégie pour {selected_segment}")

            col_r1, col_r2 = st.columns(2)

            with col_r1:
                st.markdown("**Priorité:**")
                st.info(recs['priority'])

                st.markdown("**Budget suggéré:**")
                st.info(recs['budget'])

                st.markdown("**Objectif:**")
                st.success(recs['objective'])

            with col_r2:
                st.markdown("**Canaux de communication:**")
                for channel in recs['channels']:
                    st.markdown(f"- {channel}")

                st.markdown("**Offres suggérées:**")
                st.markdown(recs['offers'])

            st.markdown("**Actions recommandées:**")
            for i, action in enumerate(recs['actions'], 1):
                st.markdown(f"{i}. {action}")

    except Exception as e:
        st.error(f"Erreur lors de l'analyse du segment: {str(e)}")

st.divider()


# TABLE COMPLETE RFM
st.header("Tableau Complet RFM")

st.markdown("""
Vue détaillée de tous les segments avec leurs métriques agrégées.
""")

try:
    # Créer le tableau agrégé
    segment_summary = df_rfm_full.groupby('Segment').agg({
        'Customer ID': 'count',
        'R_Score': 'mean',
        'F_Score': 'mean',
        'M_Score': 'mean',
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean',
        'Total_Revenue': ['sum', 'mean']
    }).reset_index()

    # Aplatir les colonnes multi-index
    segment_summary.columns = ['Segment', 'Nb_Clients', 'R_Moyen', 'F_Moyen', 'M_Moyen',
                                'Recency_Moy', 'Frequency_Moy', 'Monetary_Moy',
                                'Revenue_Total', 'Revenue_Moyen']

    # Calculer le pourcentage
    segment_summary['Pct_Clients'] = (segment_summary['Nb_Clients'] / segment_summary['Nb_Clients'].sum()) * 100
    segment_summary['Pct_Revenue'] = (segment_summary['Revenue_Total'] / segment_summary['Revenue_Total'].sum()) * 100

    # Formater les colonnes
    segment_summary_display = segment_summary.copy()
    segment_summary_display['R_Moyen'] = segment_summary_display['R_Moyen'].round(2)
    segment_summary_display['F_Moyen'] = segment_summary_display['F_Moyen'].round(2)
    segment_summary_display['M_Moyen'] = segment_summary_display['M_Moyen'].round(2)
    segment_summary_display['Recency_Moy'] = segment_summary_display['Recency_Moy'].round(0).astype(int)
    segment_summary_display['Frequency_Moy'] = segment_summary_display['Frequency_Moy'].round(1)
    segment_summary_display['Monetary_Moy'] = segment_summary_display['Monetary_Moy'].round(2)
    segment_summary_display['Revenue_Total'] = segment_summary_display['Revenue_Total'].apply(lambda x: f"£{x:,.2f}")
    segment_summary_display['Revenue_Moyen'] = segment_summary_display['Revenue_Moyen'].apply(lambda x: f"£{x:,.2f}")
    segment_summary_display['Pct_Clients'] = segment_summary_display['Pct_Clients'].round(1).astype(str) + '%'
    segment_summary_display['Pct_Revenue'] = segment_summary_display['Pct_Revenue'].round(1).astype(str) + '%'

    # Options de personnalisation
    col1, col2, col3 = st.columns(3)

    with col1:
        sort_by = st.selectbox(
            "Trier par",
            ["Segment", "Nb_Clients", "Revenue_Total", "Revenue_Moyen"],
            help="Critère de tri"
        )

    with col2:
        filter_segments = st.multiselect(
            "Segments à afficher",
            segments_available,
            default=segments_available,
            help="Filtrer les segments à afficher"
        )

    with col3:
        show_scores = st.checkbox(
            "Afficher le détail des scores",
            value=True,
            help="Afficher les scores R, F, M individuels"
        )

    # Appliquer le filtre
    if filter_segments:
        segment_summary_display = segment_summary_display[segment_summary_display['Segment'].isin(filter_segments)]

    # Colonnes à afficher
    if show_scores:
        columns_to_show = ['Segment', 'Nb_Clients', 'Pct_Clients', 'R_Moyen', 'F_Moyen', 'M_Moyen',
                          'Revenue_Total', 'Pct_Revenue', 'Revenue_Moyen']
    else:
        columns_to_show = ['Segment', 'Nb_Clients', 'Pct_Clients',
                          'Revenue_Total', 'Pct_Revenue', 'Revenue_Moyen']

    # Afficher le tableau
    st.dataframe(
        segment_summary_display[columns_to_show],
        hide_index=True,
        use_container_width=True,
        height=400
    )

except Exception as e:
    st.error(f"Erreur lors de la création du tableau: {str(e)}")

st.divider()


# RECOMMANDATIONS PAR SEGMENT
st.header("Recommandations Marketing par Segment")

with st.expander("Stratégies recommandées", expanded=True):
    st.markdown("""
    ### Guide des stratégies marketing par segment

    Les recommandations ci-dessous sont basées sur les meilleures pratiques du marketing RFM.
    """)

    for segment in segments_available:
        recs = get_segment_recommendations(segment)

        st.markdown(f"#### {segment}")
        st.markdown(f"**Priorité:** {recs['priority']} | **Budget:** {recs['budget']}")
        st.markdown(f"**Objectif:** {recs['objective']}")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Actions clés:**")
            for action in recs['actions'][:2]:  # Limiter à 2 actions pour la vue générale
                st.markdown(f"- {action}")

        with col_b:
            st.markdown("**Canaux:** " + ", ".join(recs['channels']))
            st.markdown(f"**Offres:** {recs['offers']}")

        st.divider()

st.divider()


# EXPORT
st.header("Export des Analyses RFM")

col1, col2 = st.columns(2)

with col1:
    try:
        # Export RFM complet
        csv_rfm = df_rfm_full.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="Télécharger scores RFM (CSV)",
            data=csv_rfm,
            file_name=f"rfm_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

        # Export du résumé par segment
        csv_summary = segment_summary.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="Télécharger résumé segments (CSV)",
            data=csv_summary,
            file_name=f"rfm_segments_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    except Exception as e:
        st.error(f"Erreur lors de l'export: {str(e)}")

with col2:
    st.subheader("Informations")
    st.info("""
    **Exports disponibles:**
    - Scores RFM complets par client
    - Résumé agrégé par segment

    Pour exporter les visualisations, utilisez le menu de Plotly (icône appareil photo) en haut à droite de chaque graphique.
    """)


# FOOTER
st.divider()
st.caption("Page Segmentation RFM - Dernière mise à jour : " + datetime.now().strftime("%Y-%m-%d %H:%M"))
