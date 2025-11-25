"""
Application Streamlit d'aide a la decision marketing.
Analyse de cohortes, segmentation RFM, calcul CLV et simulation de scenarios.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import config
import utils


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


def init_session_state():
    """Initialise les variables de session Streamlit."""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'df_raw' not in st.session_state:
        st.session_state.df_raw = None
    if 'df_clean' not in st.session_state:
        st.session_state.df_clean = None
    if 'df_cohorts' not in st.session_state:
        st.session_state.df_cohorts = None
    if 'df_rfm' not in st.session_state:
        st.session_state.df_rfm = None
    if 'active_filters' not in st.session_state:
        st.session_state.active_filters = {}
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


@st.cache_data(show_spinner="Chargement des donnees en cours...")
def load_and_cache_data(file_path):
    """Charge et met en cache les donnees."""
    return utils.load_data(file_path)


@st.cache_data(show_spinner="Nettoyage des donnees en cours...")
def clean_and_cache_data(_df):
    """Nettoie et met en cache les donnees."""
    return utils.clean_data(_df, verbose=True, strict_mode=False)


def render_sidebar():
    """Affiche la sidebar avec navigation et filtres globaux."""
    with st.sidebar:
        st.title("Navigation")

        if st.session_state.data_loaded:
            st.success("Donnees chargees")

            if st.session_state.df_clean is not None:
                df = st.session_state.df_clean
                clients_with_id = df[df['HasCustomerID']]['Customer ID'].nunique()
                total_trans = len(df)
                trans_with_client = len(df[df['HasCustomerID']])

                st.info(f"""
                **Donnees disponibles:**
                - {total_trans:,} transactions totales
                - {trans_with_client:,} avec Customer ID ({(trans_with_client/total_trans)*100:.1f}%)
                - {clients_with_id:,} clients uniques
                - Periode: {df['InvoiceDate'].min().strftime('%Y-%m-%d')}
                  a {df['InvoiceDate'].max().strftime('%Y-%m-%d')}
                """)
        else:
            st.warning("Donnees non chargees")

        st.divider()

        # ----------------------------------------------------------------------
        # Filtres globaux "de base"
        # ----------------------------------------------------------------------
        st.subheader("Filtres globaux")

        if st.session_state.data_loaded and st.session_state.df_clean is not None:
            df_clean = st.session_state.df_clean

            min_date = df_clean['InvoiceDate'].min().date()
            max_date = df_clean['InvoiceDate'].max().date()
            date_range = st.date_input(
                "Selection de periode",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                help="Période d'analyse des transactions (fenêtre temporelle)."
            )

            all_countries = df_clean['Country'].unique()
            selected_countries = st.multiselect(
                "Selection de pays",
                options=all_countries,
                default=st.session_state.active_filters.get(
                    'countries',
                    all_countries
                ),
                help="Filtrer l'analyse sur un ou plusieurs pays."
            )

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
            st.info("Chargez les donnees pour voir les filtres.")

        st.divider()

        # Actions
        st.subheader("Actions")

        if st.button("Recharger les donnees", use_container_width=True):
            st.cache_data.clear()
            st.session_state.data_loaded = False
            st.rerun()

        st.divider()
        st.caption(f"Version 1.0.0 | {datetime.now().year}")


def render_home_page():
    """Affiche la page d'accueil de l'application."""
    st.title(config.APP_TITLE)
    st.markdown(config.APP_DESCRIPTION)

    st.divider()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Objectifs de l'application")
        st.markdown("""
        Cette application d'aide a la decision marketing vous permet de :

        1. **Analyser les cohortes d'acquisition**
           - Suivre l'evolution des cohortes de clients dans le temps
           - Calculer les taux de retention par cohorte
           - Identifier les periodes d'acquisition les plus performantes

        2. **Segmenter vos clients avec RFM**
           - Segmentation basee sur Recence, Frequence et Montant
           - Identification des clients a forte valeur
           - Priorisation des actions marketing

        3. **Calculer la Customer Lifetime Value**
           - CLV empirique basee sur les cohortes
           - CLV predictive avec formules
           - Optimisation de la valeur client

        4. **Simuler des scenarios marketing**
           - Impact de l'amelioration de la retention
           - Effet de l'augmentation du panier moyen
           - Projection des revenus futurs

        5. **Exporter vos analyses**
           - Export des donnees en CSV
           - Export des graphiques en PNG
           - Rapports personnalises
        """)

    with col2:
        st.header("Navigation")
        st.markdown("""
        Utilisez le menu de gauche pour naviguer entre les differentes pages :

        - **Overview** : Vue d'ensemble et KPIs
        - **Cohortes** : Analyse des cohortes
        - **Segments** : Segmentation RFM
        - **Scenarios** : Simulations
        - **Export** : Exports et rapports
        """)

        st.info("""
        ℹ️ **Astuce**

        Utilisez les filtres globaux dans la barre laterale pour affiner
        vos analyses sur toutes les pages.
        """)

    st.divider()

    # KPIs globaux si donnees chargees
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
                label="Taux de retention",
                value=utils.format_percentage(kpis['retention_rate']),
                delta=None,
                help=(
                    "Proportion de clients ayant réalisé au moins 2 commandes "
                    "sur la période."
                )
            )

    else:
        st.warning("Chargez les donnees pour voir les KPIs globaux")

        if st.button("Charger les donnees", type="primary"):
            with st.spinner("Chargement et nettoyage des donnees..."):
                try:
                    df_raw = load_and_cache_data(config.RAW_DATA_XLSX)
                    st.session_state.df_raw = df_raw

                    df_clean = clean_and_cache_data(df_raw)
                    st.session_state.df_clean = df_clean

                    st.session_state.data_loaded = True
                    st.success("Donnees chargees et nettoyees avec succes !")
                    st.rerun()
                except FileNotFoundError:
                    st.error(f"Erreur : Le fichier de donnees brutes '{config.RAW_DATA_XLSX}' est introuvable.")
                except Exception as e:
                    st.error(f"Une erreur est survenue lors du chargement des donnees : {e}")

    st.divider()

    # Guide de demarrage rapide
    with st.expander("Guide de demarrage rapide", expanded=False):
        st.markdown("""
        ### Comment utiliser cette application ?

        #### 1. Charger les donnees
        - Cliquez sur "Charger les donnees" ci-dessus
        - Les donnees seront automatiquement nettoyees et preparees
        - Un message de confirmation apparaitra

        #### 2. Explorer les analyses
        - Naviguez vers la page **Overview** pour voir les KPIs detailles
        - Consultez l'**Analyse de cohortes** pour comprendre la retention
        - Explorez la **Segmentation RFM** pour identifier vos meilleurs clients

        #### 3. Appliquer des filtres
        - Utilisez les filtres globaux dans la barre laterale
        - Les filtres s'appliquent a toutes les pages
        - Vous pouvez reinitialiser les filtres a tout moment

        #### 4. Simuler des scenarios
        - Allez sur la page **Scenarios**
        - Ajustez les parametres (retention, panier moyen, etc.)
        - Visualisez l'impact sur vos KPIs

        #### 5. Exporter vos resultats
        - Sur la page **Export**, telechargez vos analyses
        - Formats disponibles : CSV pour les donnees, PNG pour les graphiques
        """)

    st.divider()
    st.caption("""
    Application d'Aide a la Decision Marketing |
    Donnees : Online Retail II Dataset |
    Developpe avec Streamlit
    """)


def main():
    """Fonction principale de l'application Streamlit."""
    init_session_state()
    render_sidebar()
    render_home_page()


if __name__ == "__main__":
    main()
