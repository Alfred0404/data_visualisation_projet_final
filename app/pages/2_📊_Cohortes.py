"""
Page Analyse de Cohortes - Étude de la rétention client par cohorte d'acquisition.

Cette page permet d'analyser l'évolution des cohortes de clients dans le temps
et de calculer les taux de rétention pour optimiser les stratégies d'acquisition
et de fidélisation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
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
    page_title="Analyse de Cohortes - Marketing Analytics",
    page_icon="📊",
    layout="wide"
)


# ==============================================================================
# EN-TETE DE LA PAGE
# ==============================================================================

st.title("📊 Analyse de Cohortes")
st.markdown("""
L'analyse de cohortes permet de suivre l'évolution de groupes de clients acquis
durant la même période et d'évaluer leur comportement au fil du temps.
""")

st.divider()


# ==============================================================================
# FILTRES SPECIFIQUES
# ==============================================================================

with st.sidebar:
    st.subheader("🎯 Filtres - Cohortes")

    # TODO: Ajouter des filtres spécifiques
    # - Période de cohortes à analyser
    # - Taille minimale de cohorte
    # - Type de visualisation (%, nombres absolus)

    st.divider()


# ==============================================================================
# VERIFICATION DES DONNEES
# ==============================================================================

if not st.session_state.get('data_loaded', False):
    st.warning("⚠️ Veuillez d'abord charger les données depuis la page d'accueil.")
    st.stop()


# ==============================================================================
# CREATION DES COHORTES
# ==============================================================================

st.header("🔍 Création des Cohortes")

df = st.session_state.get('df_clean', None)

if df is not None:
    # TODO: Créer les cohortes avec utils.create_cohorts()
    # df_cohorts = utils.create_cohorts(df)
    # st.session_state.df_cohorts = df_cohorts

    # Informations sur les cohortes
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="📅 Nombre de cohortes",
            value="TBD",  # TODO: Nombre de cohortes uniques
            help="Nombre de mois d'acquisition différents"
        )

    with col2:
        st.metric(
            label="👥 Taille moyenne",
            value="TBD",  # TODO: Taille moyenne des cohortes
            help="Nombre moyen de clients par cohorte"
        )

    with col3:
        st.metric(
            label="📊 Plus grande cohorte",
            value="TBD",  # TODO: Taille de la plus grande cohorte
            help="Nombre de clients dans la cohorte la plus importante"
        )

    with col4:
        st.metric(
            label="📆 Période d'analyse",
            value="TBD",  # TODO: Nombre de mois analysés
            help="Durée de la période d'analyse en mois"
        )

else:
    st.error("Erreur lors du chargement des données")

st.divider()


# ==============================================================================
# HEATMAP DE RETENTION
# ==============================================================================

st.header("🔥 Heatmap de Rétention")

st.markdown("""
Cette heatmap montre le pourcentage de clients de chaque cohorte qui sont revenus
effectuer un achat lors des mois suivants leur acquisition.
""")

# TODO: Calculer la matrice de rétention avec utils.calculate_retention()
# retention_matrix = utils.calculate_retention(df_cohorts)

# Options de visualisation
col1, col2 = st.columns([3, 1])

with col2:
    st.subheader("Options")

    # Type de valeurs
    value_type = st.radio(
        "Type de valeurs",
        ["Pourcentage", "Nombre absolu"],
        help="Afficher les taux en % ou le nombre de clients"
    )

    # Format de la heatmap
    colormap = st.selectbox(
        "Palette de couleurs",
        ["RdYlGn", "YlOrRd", "Blues", "Greens"],
        help="Choisir la palette de couleurs pour la heatmap"
    )

with col1:
    # TODO: Créer la heatmap avec Plotly ou Seaborn
    # - Utiliser la matrice de rétention
    # - Appliquer la palette de couleurs sélectionnée
    # - Ajouter les annotations (valeurs dans les cellules)

    st.info("TODO: Créer la heatmap de rétention")

    # Exemple de structure :
    # fig = px.imshow(
    #     retention_matrix,
    #     labels=dict(x="Mois depuis acquisition", y="Cohorte", color="Rétention (%)"),
    #     color_continuous_scale=colormap,
    #     aspect="auto"
    # )
    # st.plotly_chart(fig, use_container_width=True)

st.divider()


# ==============================================================================
# COURBES DE RETENTION
# ==============================================================================

st.header("📈 Courbes de Rétention par Cohorte")

st.markdown("""
Ces courbes montrent l'évolution de la rétention pour chaque cohorte au fil des mois.
Elles permettent d'identifier les cohortes les plus performantes.
""")

# TODO: Créer un line chart avec :
# - Une ligne par cohorte
# - X : Mois depuis acquisition (CohortIndex)
# - Y : Taux de rétention (%)
# - Légende : Mois de la cohorte

st.info("TODO: Créer les courbes de rétention")

# Options d'affichage
with st.expander("⚙️ Options d'affichage", expanded=False):
    col1, col2 = st.columns(2)

    with col1:
        # Sélection des cohortes à afficher
        st.multiselect(
            "Cohortes à afficher",
            [],  # TODO: Liste des cohortes disponibles
            help="Sélectionner les cohortes à inclure dans le graphique"
        )

    with col2:
        # Nombre de mois à afficher
        st.slider(
            "Mois à afficher",
            min_value=3,
            max_value=12,
            value=6,
            help="Nombre de mois depuis acquisition à afficher"
        )

st.divider()


# ==============================================================================
# ANALYSE COMPARATIVE
# ==============================================================================

st.header("⚖️ Analyse Comparative des Cohortes")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Rétention M1 par cohorte")

    # TODO: Créer un bar chart de la rétention au mois 1
    # - X : Cohorte
    # - Y : Taux de rétention M1
    # - Couleur : selon performance (vert si > moyenne, rouge sinon)

    st.info("TODO: Graphique rétention M1")

with col2:
    st.subheader("Rétention M3 par cohorte")

    # TODO: Créer un bar chart de la rétention au mois 3
    # - Structure similaire à M1

    st.info("TODO: Graphique rétention M3")

st.divider()


# ==============================================================================
# METRIQUES DE RETENTION
# ==============================================================================

st.header("📊 Métriques de Rétention Globales")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Rétention moyenne")

    # TODO: Calculer et afficher la rétention moyenne
    # - Par période (M1, M3, M6, M12)
    # - Avec évolution vs période précédente

    st.info("TODO: Métriques de rétention moyenne")

with col2:
    st.subheader("Meilleure/Pire cohorte")

    # TODO: Identifier et afficher :
    # - La cohorte avec la meilleure rétention
    # - La cohorte avec la pire rétention
    # - L'écart entre les deux

    st.info("TODO: Identification des cohortes extrêmes")

with col3:
    st.subheader("Tendance de rétention")

    # TODO: Créer un petit graphique sparkline montrant :
    # - L'évolution de la rétention M1 dans le temps
    # - Tendance : amélioration ou dégradation

    st.info("TODO: Graphique de tendance")

st.divider()


# ==============================================================================
# TABLE DETAILLEE DES COHORTES
# ==============================================================================

st.header("📋 Tableau Détaillé des Cohortes")

st.markdown("""
Vue tabulaire détaillée de toutes les cohortes avec leurs métriques clés.
""")

# TODO: Créer un DataFrame avec pour chaque cohorte :
# - Mois de la cohorte
# - Nombre de clients
# - Rétention M1, M3, M6, M12
# - Revenu total de la cohorte
# - CLV moyenne de la cohorte

# Afficher avec st.dataframe() avec formatting et possibilité de tri

st.info("TODO: Table détaillée des cohortes")

# Options de tri et filtrage
col1, col2, col3 = st.columns(3)

with col1:
    st.selectbox(
        "Trier par",
        ["Mois", "Taille", "Rétention M1", "Rétention M3", "Revenu"],
        help="Critère de tri du tableau"
    )

with col2:
    st.selectbox(
        "Ordre",
        ["Décroissant", "Croissant"],
        help="Ordre de tri"
    )

with col3:
    st.number_input(
        "Taille minimale",
        min_value=0,
        value=config.MIN_COHORT_SIZE,
        help="Filtrer les cohortes avec un minimum de clients"
    )

st.divider()


# ==============================================================================
# INSIGHTS ET RECOMMANDATIONS
# ==============================================================================

st.header("💡 Insights et Recommandations")

with st.expander("📊 Analyse des cohortes", expanded=True):
    st.markdown("""
    **TODO: Générer des insights automatiques**

    Exemples d'insights à générer :
    - Cohortes avec la meilleure rétention et leurs caractéristiques
    - Périodes d'acquisition optimales
    - Tendances de rétention (amélioration/dégradation)
    - Comparaison avec les benchmarks du secteur
    - Recommandations d'actions marketing ciblées

    Ces insights seront générés automatiquement en analysant :
    - Les patterns dans les données de cohortes
    - Les variations saisonnières
    - Les corrélations avec les campagnes marketing
    """)

st.divider()


# ==============================================================================
# EXPORT
# ==============================================================================

st.header("📥 Export des Analyses")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 Exporter matrice de rétention (CSV)", use_container_width=True):
        # TODO: Exporter la matrice de rétention
        st.success("TODO: Export matrice CSV")

with col2:
    if st.button("📈 Exporter heatmap (PNG)", use_container_width=True):
        # TODO: Exporter la heatmap
        st.success("TODO: Export heatmap PNG")

with col3:
    if st.button("📄 Rapport cohortes (PDF)", use_container_width=True):
        # TODO: Générer rapport PDF
        st.info("TODO: Génération rapport PDF")


# ==============================================================================
# FOOTER
# ==============================================================================

st.divider()
st.caption("Page Analyse de Cohortes - Dernière mise à jour : " + datetime.now().strftime("%Y-%m-%d %H:%M"))
