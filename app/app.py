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
# #GESTION DE L'ETAT DE SESSION
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

    # Filtres actifs (période, pays, seuil, etc.)
    if 'active_filters' not in st.session_state:
        st.session_state.active_filters = {}

    # KPIs globaux
    if 'kpis' not in st.session_state:
        st.session_state.kpis = {}

    # Paramètres avancés globaux (pour respecter le périmètre fonctionnel)
    # - unité de temps (mois / trimestre)
    # - mode retours (inclure / exclure / neutraliser)
    # - type client (placeholder si pas dans les données)
    if 'unit_of_time' not in st.session_state:
        st.session_state.unit_of_time = "Mois"

    if 'returns_mode' not in st.session_state:
        st.session_state.returns_mode = "Inclure"

    if 'customer_type' not in st.session_state:
        st.session_state.customer_type = "Tous"


# ==============================================================================
#FONCTIONS DE CHARGEMENT DES DONNEES
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

    Notes
    -----
    Utilise le mode optimisé (strict_mode=False) pour conserver un maximum de données.
    Pour les analyses nécessitant uniquement les clients avec ID (RFM, CLV, Cohortes),
    filtrer avec df[df['HasCustomerID']].
    """
    return utils.clean_data(_df, verbose=True, strict_mode=False)


# ==============================================================================
#SIDEBAR - NAVIGATION ET FILTRES GLOBAUX
# ==============================================================================

def render_sidebar():
    """
    Affiche la barre latérale avec navigation et filtres globaux.

    Cette fonction crée :
    - Le menu de navigation entre les pages
    - Les filtres globaux applicables à toutes les analyses
    - Les paramètres avancés (unité de temps, retours, type client)
    - Les informations sur les données chargées
    """
    with st.sidebar:
        # En-tête
        st.title("Navigation")

        # Informations sur les données
        if st.session_state.data_loaded:
            st.success("Données chargées")

            if st.session_state.df_clean is not None:
                df = st.session_state.df_clean
                # Compter les clients avec ID (exclure les NaN)
                clients_with_id = df[df['HasCustomerID']]['Customer ID'].nunique()
                total_trans = len(df)
                trans_with_client = len(df[df['HasCustomerID']])

                st.info(f"""
                **Données disponibles :**
                - {total_trans:,} transactions totales
                - {trans_with_client:,} avec Customer ID ({(trans_with_client/total_trans)*100:.1f}%)
                - {clients_with_id:,} clients uniques
                - Période : {df['InvoiceDate'].min().strftime('%Y-%m-%d')}
                  à {df['InvoiceDate'].max().strftime('%Y-%m-%d')}
                """)
        else:
            st.warning("Données non chargées")

        st.divider()

        # ----------------------------------------------------------------------
        # Filtres globaux "de base"
        # ----------------------------------------------------------------------
        st.subheader("Filtres globaux")

        if st.session_state.data_loaded and st.session_state.df_clean is not None:
            df_clean = st.session_state.df_clean

            # Sélecteur de période (analyse glissante possible via les pages)
            min_date = df_clean['InvoiceDate'].min().date()
            max_date = df_clean['InvoiceDate'].max().date()
            date_range = st.date_input(
                "Période d'analyse",
                value=st.session_state.active_filters.get(
                    'date_range',
                    (min_date, max_date)
                ),
                min_value=min_date,
                max_value=max_date,
                help="Période d'analyse des transactions (fenêtre temporelle)."
            )

            # Sélecteur pays
            all_countries = sorted(df_clean['Country'].unique())
            selected_countries = st.multiselect(
                "Pays",
                options=all_countries,
                default=st.session_state.active_filters.get(
                    'countries',
                    all_countries
                ),
                help="Filtrer l'analyse sur un ou plusieurs pays."
            )

            # Seuil de commande (montant minimum)
            min_amount = st.slider(
                "Seuil de commande (montant minimum)",
                min_value=0,
                max_value=int(df_clean['TotalAmount'].max()),
                value=st.session_state.active_filters.get('min_amount', 0),
                help="Exclure les très petites commandes (bruit) sous ce montant."
            )

            # ------------------------------------------------------------------
            # Paramètres avancés globaux (pour coller au périmètre fonctionnel)
            # ------------------------------------------------------------------
            st.subheader("Paramètres avancés")

            # Unité de temps (mois / trimestre) utilisée dans les pages
            unit_of_time = st.radio(
                "Unité de temps",
                ["Mois", "Trimestre"],
                key="unit_of_time_radio",
                index=["Mois", "Trimestre"].index(st.session_state.unit_of_time),
                help="Définit la granularité temporelle des analyses (ex. courbes, cohortes)."
            )

            # Mode retours : inclure / exclure / neutraliser
            returns_mode = st.radio(
                "Mode retours",
                ["Inclure", "Exclure", "Neutraliser"],
                key="returns_mode_radio",
                index=["Inclure", "Exclure", "Neutraliser"].index(st.session_state.returns_mode),
                help=(
                    "Inclure : les retours sont comptés négativement dans le CA.\n"
                    "Exclure : les retours sont retirés du périmètre.\n"
                    "Neutraliser : CA net (achats - retours), avec badge à afficher dans les pages."
                )
            )

            # Type client (placeholder si pas encore présent dans les données)
            customer_type = st.selectbox(
                "Type de client",
                ["Tous", "B2C", "B2B"],
                index=["Tous", "B2C", "B2B"].index(st.session_state.customer_type),
                help="Filtrer selon le type de clients si disponible dans les données."
            )

            # Sauvegarde dans l'état de session (accès direct depuis les pages)
            st.session_state.unit_of_time = unit_of_time
            st.session_state.returns_mode = returns_mode
            st.session_state.customer_type = customer_type

            # Sauvegarde également dans active_filters pour utils.apply_filters() & co
            st.session_state.active_filters = {
                'date_range': date_range,
                'countries': selected_countries,
                'min_amount': min_amount,
                'unit_of_time': unit_of_time,
                'returns_mode': returns_mode,
                'customer_type': customer_type
            }

        else:
            st.info("Chargez les données pour voir et utiliser les filtres globaux.")

        st.divider()

        # Actions
        st.subheader("Actions")

        if st.button("Recharger les données", use_container_width=True):
            st.cache_data.clear()
            st.session_state.data_loaded = False
            st.rerun()

        # Informations
        st.divider()
        st.caption(f"Version 1.0.0 | {datetime.now().year}")


# ==============================================================================
#PAGE D'ACCUEIL
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
        st.header("Objectifs de l'application")
        st.markdown("""
        Cette application d'aide à la décision marketing vous permet de :

        1. **Analyser les cohortes d'acquisition**
           - Suivre l'évolution des cohortes de clients dans le temps
           - Calculer les taux de rétention par cohorte
           - Identifier les périodes d'acquisition les plus performantes

        2. **Segmenter vos clients avec RFM**
           - Segmentation basée sur Récence, Fréquence et Montant
           - Identification des clients à forte valeur
           - Priorisation des actions marketing

        3. **Calculer la Customer Lifetime Value**
           - CLV empirique basée sur les cohortes
           - CLV prédictive avec formules
           - Optimisation de la valeur client

        4. **Simuler des scénarios marketing**
           - Impact de l'amélioration de la rétention
           - Effet de l'augmentation du panier moyen
           - Projection des revenus futurs

        5. **Exporter vos analyses**
           - Export des données en CSV
           - Export des graphiques en PNG
           - Rapports personnalisés
        """)

    with col2:
        st.header("Navigation")
        st.markdown("""
        Utilisez le menu de gauche pour naviguer entre les différentes pages :

        - **Overview** : Vue d'ensemble et KPIs (North Star, CLV baseline…)
        - **Cohortes** : Rétention par cohortes d'acquisition
        - **Segments** : Segmentation RFM & priorités d'activation
        - **Scénarios** : Simulation (remise, marge, rétention…)
        - **Export** : Plan d’action & exports (listes activables, CSV)
        """)

        st.info("""
        ℹ️ **Astuce**

        Utilisez les filtres globaux dans la barre latérale pour affiner
        vos analyses sur toutes les pages (période, pays, retours, etc.).
        """)

    st.divider()

    # KPIs globaux (si données chargées)
    if st.session_state.data_loaded and st.session_state.df_clean is not None:
        st.header("Vue d'ensemble rapide")

        kpis = utils.calculate_kpis(st.session_state.df_clean)
        st.session_state.kpis = kpis

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="Clients totaux",
                value=f"{kpis['total_customers']:,}",
                delta=None,
                help=(
                    "Nombre total de clients uniques ayant au moins une transaction "
                    "sur toute la période disponible."
                )
            )

        with col2:
            st.metric(
                label="Revenu total",
                value=utils.format_currency(kpis['total_revenue']),
                delta=None,
                help="Chiffre d'affaires total (achats - retours si neutralisation)."
            )

        with col3:
            st.metric(
                label="Panier moyen",
                value=utils.format_currency(kpis['avg_order_value']),
                delta=None,
                help="Montant moyen par transaction (CA / nombre de commandes)."
            )

        with col4:
            st.metric(
                label="Taux de rétention",
                value=utils.format_percentage(kpis['retention_rate']),
                delta=None,
                help=(
                    "Proportion de clients ayant réalisé au moins 2 commandes "
                    "sur la période."
                )
            )

    else:
        st.warning("Chargez les données pour voir les KPIs globaux")

        # Bouton pour charger les données
        if st.button("Charger les données", type="primary"):
            with st.spinner("Chargement et nettoyage des données..."):
                try:
                    df_raw = load_and_cache_data(config.RAW_DATA_XLSX)
                    st.session_state.df_raw = df_raw

                    df_clean = clean_and_cache_data(df_raw)
                    st.session_state.df_clean = df_clean

                    st.session_state.data_loaded = True
                    st.success("Données chargées et nettoyées avec succès !")
                    st.rerun()
                except FileNotFoundError:
                    st.error(f"Erreur : Le fichier de données brutes '{config.RAW_DATA_XLSX}' est introuvable.")
                except Exception as e:
                    st.error(f"Une erreur est survenue lors du chargement des données : {e}")

    st.divider()

    # Guide de démarrage rapide
    with st.expander("Guide de démarrage rapide", expanded=False):
        st.markdown("""
        ### Comment utiliser cette application ?

        #### 1. Charger les données
        - Cliquez sur **« Charger les données »** ci-dessus
        - Les données seront automatiquement nettoyées et préparées
        - Un message de confirmation apparaîtra

        #### 2. Explorer les analyses
        - Naviguez vers la page **Overview** pour voir les KPIs détaillés
        - Consultez l'**Analyse de cohortes** pour comprendre la rétention
        - Explorez la **Segmentation RFM** pour identifier vos meilleurs clients

        #### 3. Appliquer des filtres
        - Utilisez les filtres globaux dans la barre latérale (période, pays, seuil de commande, retours…)
        - Les filtres s'appliquent à toutes les pages
        - Vous pouvez réinitialiser les filtres en rechargeant les données

        #### 4. Simuler des scénarios
        - Allez sur la page **Scénarios**
        - Ajustez les paramètres (rétention, panier moyen, marge, remises…)
        - Visualisez immédiatement l’impact sur CA, CLV et rétention

        #### 5. Exporter vos résultats
        - Sur la page **Export**, téléchargez vos analyses
        - Formats disponibles : CSV pour les données, PNG pour les graphiques
        """)

    # Footer
    st.divider()
    st.caption("""
    Application d'Aide à la Décision Marketing |
    Données : Online Retail II Dataset |
    Développé avec Streamlit
    """)


# ==============================================================================
#FONCTION PRINCIPALE
# ==============================================================================

def main():
    """
    Fonction principale de l'application Streamlit.

    Cette fonction :
    - Initialise l'état de session
    - Affiche la sidebar (filtres globaux + paramètres avancés)
    - Affiche la page d'accueil (explications + KPIs globaux)
    """
    # Initialisation
    init_session_state()

    # Sidebar (navigation + filtres globaux pour toute l'app)
    render_sidebar()

    # Page d'accueil (la navigation vers Overview / Cohortes / Segments / Scénarios / Export
    # est gérée par les fichiers dans le dossier `pages/`)
    render_home_page()


# ==============================================================================
#POINT D'ENTREE
# ==============================================================================

if __name__ == "__main__":
    main()
