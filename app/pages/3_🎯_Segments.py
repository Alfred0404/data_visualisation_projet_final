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

# Imports locaux
sys.path.append(str(Path(__file__).parent.parent.parent))
import config
import utils


# ==============================================================================
# CONFIGURATION DE LA PAGE
# ==============================================================================

st.set_page_config(
    page_title="Segmentation RFM - Marketing Analytics",
    page_icon="🎯",
    layout="wide"
)


# ==============================================================================
# EN-TETE DE LA PAGE
# ==============================================================================

st.title("🎯 Segmentation RFM")
st.markdown("""
La segmentation RFM (Recency, Frequency, Monetary) permet d'identifier
les clients les plus précieux et de personnaliser les stratégies marketing.
""")

st.divider()


# ==============================================================================
# EXPLICATION RFM
# ==============================================================================

with st.expander("ℹ️ Comprendre la méthodologie RFM", expanded=False):
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


# ==============================================================================
# FILTRES SPECIFIQUES
# ==============================================================================

with st.sidebar:
    st.subheader("🎯 Filtres - RFM")

    # TODO: Ajouter des filtres spécifiques
    # - Date de référence pour le calcul RFM
    # - Segments à afficher
    # - Seuils personnalisés

    st.divider()


# ==============================================================================
# VERIFICATION DES DONNEES
# ==============================================================================

if not st.session_state.get('data_loaded', False):
    st.warning("⚠️ Veuillez d'abord charger les données depuis la page d'accueil.")
    st.stop()


# ==============================================================================
# CALCUL RFM
# ==============================================================================

st.header("🔍 Calcul des Scores RFM")

df = st.session_state.get('df_clean', None)

if df is not None:
    # TODO: Calculer RFM avec utils.calculate_rfm()
    # df_rfm = utils.calculate_rfm(df)
    # st.session_state.df_rfm = df_rfm

    # Métriques globales RFM
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="📊 Nombre de segments",
            value="TBD",  # TODO: Nombre de segments distincts
            help="Nombre de segments RFM identifiés"
        )

    with col2:
        st.metric(
            label="👑 Champions",
            value="TBD",  # TODO: Nombre de champions
            help="Clients avec score RFM le plus élevé"
        )

    with col3:
        st.metric(
            label="⚠️ At Risk",
            value="TBD",  # TODO: Nombre de clients at risk
            help="Clients à risque de churn"
        )

    with col4:
        st.metric(
            label="❌ Lost",
            value="TBD",  # TODO: Nombre de clients perdus
            help="Clients perdus (score faible)"
        )

else:
    st.error("Erreur lors du chargement des données")

st.divider()


# ==============================================================================
# DISTRIBUTION DES SEGMENTS
# ==============================================================================

st.header("📊 Distribution des Segments")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Répartition des clients par segment")

    # TODO: Créer un treemap ou sunburst chart
    # - Afficher la répartition des clients par segment RFM
    # - Taille proportionnelle au nombre de clients
    # - Couleurs selon la valeur du segment

    st.info("TODO: Treemap de distribution des segments")

with col2:
    st.subheader("Contribution au revenu")

    # TODO: Créer un pie chart
    # - Montrer la contribution de chaque segment au revenu total
    # - Mettre en évidence les segments les plus profitables

    st.info("TODO: Pie chart contribution revenue")

st.divider()


# ==============================================================================
# MATRICE RFM
# ==============================================================================

st.header("🔲 Matrice RFM")

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
    # TODO: Créer un scatter plot 3D
    # - X: Recency
    # - Y: Frequency
    # - Z: Monetary
    # - Couleur: Segment RFM
    # - Taille: Valeur client

    st.info("TODO: Scatter 3D des scores RFM")

else:
    # TODO: Créer une heatmap 2D selon le choix
    # - Agréger les données selon 2 dimensions
    # - Afficher le nombre de clients ou le revenu moyen

    st.info(f"TODO: Heatmap {viz_type}")

st.divider()


# ==============================================================================
# PROFILS DETAILLES DES SEGMENTS
# ==============================================================================

st.header("👥 Profils Détaillés des Segments")

# Sélection du segment à analyser
selected_segment = st.selectbox(
    "Choisir un segment à analyser",
    [],  # TODO: Liste des segments disponibles
    help="Sélectionner un segment pour voir son profil détaillé"
)

if selected_segment:
    # TODO: Afficher les détails du segment sélectionné
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📊 Caractéristiques")

        # TODO: Afficher les métriques du segment
        # - Nombre de clients
        # - % du total
        # - Scores RFM moyens (R, F, M)

        st.info("TODO: Métriques du segment")

    with col2:
        st.subheader("💰 Performance")

        # TODO: Afficher la performance financière
        # - Revenu total
        # - Revenu moyen par client
        # - % du revenu total
        # - CLV moyenne

        st.info("TODO: Performance du segment")

    with col3:
        st.subheader("📈 Comportement")

        # TODO: Afficher le comportement d'achat
        # - Fréquence d'achat moyenne
        # - Panier moyen
        # - Dernière transaction moyenne
        # - Taux de rétention

        st.info("TODO: Comportement du segment")

    # Graphiques supplémentaires pour le segment
    st.subheader(f"📊 Analyses détaillées - {selected_segment}")

    tab1, tab2, tab3 = st.tabs(["Distribution", "Évolution", "Comparaison"])

    with tab1:
        # TODO: Créer des histogrammes de distribution
        # - Distribution de Recency
        # - Distribution de Frequency
        # - Distribution de Monetary

        st.info("TODO: Histogrammes de distribution")

    with tab2:
        # TODO: Créer un graphique d'évolution temporelle
        # - Évolution du nombre de clients dans ce segment
        # - Évolution du revenu généré

        st.info("TODO: Évolution temporelle du segment")

    with tab3:
        # TODO: Créer une comparaison avec les autres segments
        # - Radar chart ou bar chart comparatif
        # - Benchmarking des métriques clés

        st.info("TODO: Comparaison inter-segments")

st.divider()


# ==============================================================================
# TABLE COMPLETE RFM
# ==============================================================================

st.header("📋 Tableau Complet RFM")

st.markdown("""
Vue détaillée de tous les segments avec leurs métriques agrégées.
""")

# TODO: Créer un DataFrame agrégé par segment avec :
# - Segment
# - Nombre de clients
# - % du total
# - Scores R, F, M moyens
# - Revenu total
# - Revenu moyen par client
# - CLV moyenne
# - Taux de rétention

# Afficher avec st.dataframe() avec formatting

st.info("TODO: Table agrégée par segment")

# Options de personnalisation
col1, col2, col3 = st.columns(3)

with col1:
    st.selectbox(
        "Trier par",
        ["Segment", "Nb clients", "Revenu total", "CLV moyenne"],
        help="Critère de tri"
    )

with col2:
    st.multiselect(
        "Segments à afficher",
        [],  # TODO: Liste des segments
        help="Filtrer les segments à afficher"
    )

with col3:
    st.checkbox(
        "Afficher le détail des scores",
        value=True,
        help="Afficher les scores R, F, M individuels"
    )

st.divider()


# ==============================================================================
# RECOMMANDATIONS PAR SEGMENT
# ==============================================================================

st.header("💡 Recommandations Marketing par Segment")

# TODO: Créer un tableau ou des cards avec recommandations
# - Pour chaque segment
# - Actions marketing recommandées
# - Canaux de communication suggérés
# - Offres adaptées
# - Objectifs KPIs

with st.expander("📊 Stratégies recommandées", expanded=True):
    st.markdown("""
    **TODO: Générer des recommandations automatiques**

    Exemples de recommandations :

    **Champions (444)**
    - ✅ Programme VIP exclusif
    - ✅ Early access aux nouveaux produits
    - ✅ Personnalisation premium
    - 🎯 Objectif : Fidélisation maximale

    **At Risk**
    - ⚠️ Campagne de réactivation urgente
    - ⚠️ Offre spéciale win-back
    - ⚠️ Email personnalisé
    - 🎯 Objectif : Réduire le churn

    **Potential Loyalists**
    - 💎 Programme de fidélité
    - 💎 Cross-sell ciblé
    - 💎 Contenu éducatif
    - 🎯 Objectif : Conversion en Loyal Customers

    Ces recommandations seront générées automatiquement en fonction
    des caractéristiques de chaque segment.
    """)

st.divider()


# ==============================================================================
# ANALYSE DES TRANSITIONS
# ==============================================================================

st.header("🔄 Analyse des Transitions de Segments")

st.markdown("""
Suivre comment les clients évoluent d'un segment à l'autre au fil du temps.
""")

# TODO: Créer un Sankey diagram montrant :
# - Les mouvements de clients entre segments
# - Entre deux périodes (ex: T-1 vs T)
# - Identifier les flux principaux

st.info("TODO: Sankey diagram des transitions")

st.divider()


# ==============================================================================
# EXPORT
# ==============================================================================

st.header("📥 Export des Analyses RFM")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 Exporter scores RFM (CSV)", use_container_width=True):
        # TODO: Exporter le DataFrame RFM complet
        st.success("TODO: Export RFM CSV")

with col2:
    if st.button("📈 Exporter visualisations (PNG)", use_container_width=True):
        # TODO: Exporter les graphiques
        st.success("TODO: Export graphiques PNG")

with col3:
    if st.button("📄 Rapport segmentation (PDF)", use_container_width=True):
        # TODO: Générer rapport PDF complet
        st.info("TODO: Génération rapport PDF")


# ==============================================================================
# FOOTER
# ==============================================================================

st.divider()
st.caption("Page Segmentation RFM - Dernière mise à jour : " + datetime.now().strftime("%Y-%m-%d %H:%M"))
