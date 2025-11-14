"""
Application Streamlit d'aide à la décision marketing.

Cette application permet d'analyser les données de ventes pour optimiser
les stratégies marketing à travers l'analyse de cohortes, la segmentation RFM,
le calcul de CLV et la simulation de scénarios.

Auteur: Projet Marketing Analytics
Date: 2024
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
from pathlib import Path

# Ajouter le répertoire parent au path pour importer config et utils
sys.path.append(str(Path(__file__).parent.parent))
import config

import utils


# ==============================================================================
# CONFIGURATION DE LA PAGE STREAMLIT
# ==============================================================================

st.set_page_config(
    page_title=config.PAGE_CONFIG["page_title"],
    page_icon=config.PAGE_CONFIG["page_icon"],
    layout=config.PAGE_CONFIG["layout"],
    initial_sidebar_state=config.PAGE_CONFIG["initial_sidebar_state"],
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': f"# {config.APP_TITLE}\n{config.APP_DESCRIPTION}"
    }
)


# ==============================================================================
# GESTION DE L'ETAT DE SESSION
# ==============================================================================

def init_session_state():
    """
    Initialise les variables de session Streamlit.

    Cette fonction crée les variables de session nécessaires pour maintenir
    l'état de l'application entre les interactions utilisateur.
    """
    # Données chargées
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False

    if 'df_raw' not in st.session_state:
        st.session_state.df_raw = None

    if 'df_clean' not in st.session_state:
        st.session_state.df_clean = None

    # Données calculées
    if 'df_cohorts' not in st.session_state:
        st.session_state.df_cohorts = None

    if 'df_rfm' not in st.session_state:
        st.session_state.df_rfm = None

    # Filtres actifs
    if 'active_filters' not in st.session_state:
        st.session_state.active_filters = {}

    # KPIs
    if 'kpis' not in st.session_state:
        st.session_state.kpis = {}


# ==============================================================================
# FONCTIONS DE CHARGEMENT DES DONNEES
# ==============================================================================

@st.cache_data(show_spinner="Chargement des données en cours...")
def load_and_cache_data(file_path):
    """
    Charge et met en cache les données.

    Parameters
    ----------
    file_path : str or Path
        Chemin vers le fichier de données

    Returns
    -------
    pd.DataFrame
        DataFrame chargé

    Notes
    -----
    Le décorateur @st.cache_data permet de mettre en cache les données
    pour éviter de les recharger à chaque interaction.
    """
    return utils.load_data(file_path)


@st.cache_data(show_spinner="Nettoyage des données en cours...")
def clean_and_cache_data(_df):
    """
    Nettoie et met en cache les données.

    Parameters
    ----------
    _df : pd.DataFrame
        DataFrame brut (préfixe _ pour éviter le hashing par Streamlit)

    Returns
    -------
    pd.DataFrame
        DataFrame nettoyé
    """
    return utils.clean_data(_df)


# ==============================================================================
# SIDEBAR - NAVIGATION ET FILTRES GLOBAUX
# ==============================================================================

def render_sidebar():
    """
    Affiche la barre latérale avec navigation et filtres globaux.

    Cette fonction crée :
    - Le menu de navigation entre les pages
    - Les filtres globaux applicables à toutes les analyses
    - Les informations sur les données chargées
    """
    with st.sidebar:
        # En-tête
        st.title("📊 Navigation")

        # Informations sur les données
        if st.session_state.data_loaded:
            st.success("✅ Données chargées")

            if st.session_state.df_clean is not None:
                st.info(f"""
                **Données disponibles:**
                - {len(st.session_state.df_clean):,} transactions
                - {st.session_state.df_clean['Customer ID'].nunique():,} clients
                - Période: {st.session_state.df_clean['InvoiceDate'].min().strftime('%Y-%m-%d')}
                  à {st.session_state.df_clean['InvoiceDate'].max().strftime('%Y-%m-%d')}
                """)
        else:
            st.warning("⚠️ Données non chargées")

        st.divider()

        # Filtres globaux
        st.subheader("🎯 Filtres globaux")

        if st.session_state.data_loaded and st.session_state.df_clean is not None:
            df_clean = st.session_state.df_clean

            # Date range selector
            min_date = df_clean['InvoiceDate'].min().date()
            max_date = df_clean['InvoiceDate'].max().date()
            date_range = st.date_input(
                "Sélection de période",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
            )

            # Country multiselect
            all_countries = df_clean['Country'].unique()
            selected_countries = st.multiselect(
                "Sélection de pays",
                options=all_countries,
                default=st.session_state.active_filters.get('countries', list(all_countries))
            )

            # Minimum transaction amount slider
            min_amount = st.slider(
                "Montant minimum de transaction",
                min_value=0,
                max_value=int(df_clean['TotalAmount'].max()),
                value=st.session_state.active_filters.get('min_amount', 0)
            )

            st.session_state.active_filters = {
                'date_range': date_range,
                'countries': selected_countries,
                'min_amount': min_amount
            }
        else:
            st.info("Chargez les données pour voir les filtres.")

        st.divider()

        # Actions
        st.subheader("⚙️ Actions")

        if st.button("🔄 Recharger les données", use_container_width=True):
            st.cache_data.clear()
            st.session_state.data_loaded = False
            st.rerun()

        # Informations
        st.divider()
        st.caption(f"Version 1.0.0 | {datetime.now().year}")


# ==============================================================================
# PAGE D'ACCUEIL
# ==============================================================================

def render_home_page():
    """
    Affiche la page d'accueil de l'application.

    Cette page présente :
    - Le titre et la description de l'application
    - Les fonctionnalités disponibles
    - Les instructions de navigation
    - Un aperçu des KPIs globaux
    """
    # En-tête principal
    st.title(config.APP_TITLE)
    st.markdown(config.APP_DESCRIPTION)

    st.divider()

    # Section d'introduction
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("🎯 Objectifs de l'application")
        st.markdown("""
        Cette application d'aide à la décision marketing vous permet de :

        1. **Analyser les cohortes d'acquisition** 📊
           - Suivre l'évolution des cohortes de clients dans le temps
           - Calculer les taux de rétention par cohorte
           - Identifier les périodes d'acquisition les plus performantes

        2. **Segmenter vos clients avec RFM** 🎯
           - Segmentation basée sur Récence, Fréquence et Montant
           - Identification des clients à forte valeur
           - Priorisation des actions marketing

        3. **Calculer la Customer Lifetime Value** 💰
           - CLV empirique basée sur les cohortes
           - CLV prédictive avec formules
           - Optimisation de la valeur client

        4. **Simuler des scénarios marketing** 🔬
           - Impact de l'amélioration de la rétention
           - Effet de l'augmentation du panier moyen
           - Projection des revenus futurs

        5. **Exporter vos analyses** 📤
           - Export des données en CSV
           - Export des graphiques en PNG
           - Rapports personnalisés
        """)

    with col2:
        st.header("🗺️ Navigation")
        st.markdown("""
        Utilisez le menu de gauche pour naviguer entre les différentes pages :

        - **🏠 Overview** : Vue d'ensemble et KPIs
        - **📊 Cohortes** : Analyse des cohortes
        - **🎯 Segments** : Segmentation RFM
        - **🔬 Scénarios** : Simulations
        - **📤 Export** : Exports et rapports
        """)

        st.info("""
        **💡 Astuce**

        Utilisez les filtres globaux dans la barre latérale pour affiner
        vos analyses sur toutes les pages.
        """)

    st.divider()

    # KPIs globaux (si données chargées)
    if st.session_state.data_loaded and st.session_state.df_clean is not None:
        st.header("📈 Vue d'ensemble rapide")

        kpis = utils.calculate_kpis(st.session_state.df_clean)
        st.session_state.kpis = kpis

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="Clients totaux",
                value=f"{kpis['total_customers']:,}",
                delta=None
            )

        with col2:
            st.metric(
                label="Revenu total",
                value=utils.format_currency(kpis['total_revenue']),
                delta=None
            )

        with col3:
            st.metric(
                label="Panier moyen",
                value=utils.format_currency(kpis['avg_order_value']),
                delta=None
            )

        with col4:
            st.metric(
                label="Taux de rétention",
                value=utils.format_percentage(kpis['retention_rate']),
                delta=None
            )


    else:
        st.warning("⚠️ Chargez les données pour voir les KPIs globaux")

        # Bouton pour charger les données
        if st.button("📂 Charger les données", type="primary"):
            with st.spinner("Chargement et nettoyage des données..."):
                try:
                    df_raw = load_and_cache_data(config.RAW_DATA_CSV)
                    st.session_state.df_raw = df_raw
                    
                    df_clean = clean_and_cache_data(df_raw)
                    st.session_state.df_clean = df_clean
                    
                    st.session_state.data_loaded = True
                    st.success("Données chargées et nettoyées avec succès !")
                    st.rerun()
                except FileNotFoundError:
                    st.error(f"Erreur : Le fichier de données brutes '{config.RAW_DATA_CSV}' est introuvable.")
                except Exception as e:
                    st.error(f"Une erreur est survenue lors du chargement des données : {e}")


    st.divider()

    # Guide de démarrage rapide
    with st.expander("📖 Guide de démarrage rapide", expanded=False):
        st.markdown("""
        ### Comment utiliser cette application ?

        #### 1. Charger les données
        - Cliquez sur "Charger les données" ci-dessus
        - Les données seront automatiquement nettoyées et préparées
        - Un message de confirmation apparaîtra

        #### 2. Explorer les analyses
        - Naviguez vers la page **Overview** pour voir les KPIs détaillés
        - Consultez l'**Analyse de cohortes** pour comprendre la rétention
        - Explorez la **Segmentation RFM** pour identifier vos meilleurs clients

        #### 3. Appliquer des filtres
        - Utilisez les filtres globaux dans la barre latérale
        - Les filtres s'appliquent à toutes les pages
        - Vous pouvez réinitialiser les filtres à tout moment

        #### 4. Simuler des scénarios
        - Allez sur la page **Scénarios**
        - Ajustez les paramètres (rétention, panier moyen, etc.)
        - Visualisez l'impact sur vos KPIs

        #### 5. Exporter vos résultats
        - Sur la page **Export**, téléchargez vos analyses
        - Formats disponibles : CSV pour les données, PNG pour les graphiques
        """)

    # Footer
    st.divider()
    st.caption("""
    📊 Application d'Aide à la Décision Marketing |
    Données : Online Retail II Dataset |
    Développé avec Streamlit
    """)


# ==============================================================================
# FONCTION PRINCIPALE
# ==============================================================================

def main():
    """
    Fonction principale de l'application Streamlit.

    Cette fonction :
    - Initialise l'état de session
    - Affiche la sidebar
    - Affiche la page d'accueil
    """
    # Initialisation
    init_session_state()

    # Sidebar
    render_sidebar()

    # Page d'accueil
    render_home_page()


# ==============================================================================
# POINT D'ENTREE
# ==============================================================================

if __name__ == "__main__":
    main()
