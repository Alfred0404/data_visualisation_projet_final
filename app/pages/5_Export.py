"""
Page Export - Centralisation des exports de données et rapports.

Cette page permet d'exporter toutes les analyses réalisées sous différents
formats (CSV, Excel) pour une utilisation externe.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path
import zipfile
import io

# Imports locaux
sys.path.append(str(Path(__file__).parent.parent.parent))
import config
import utils


# CONFIGURATION DE LA PAGE
st.set_page_config(
    page_title="Export - Marketing Analytics",
    page_icon=":material/upload:",
    layout="wide"
)


# HELPER FUNCTIONS
def get_file_size(df):
    """Calculate the approximate file size in MB"""
    size_bytes = df.memory_usage(deep=True).sum()
    size_mb = size_bytes / (1024 * 1024)
    return size_mb


def format_file_size(size_mb):
    """Format file size for display"""
    if size_mb < 1:
        return f"{size_mb * 1024:.1f} KB"
    else:
        return f"{size_mb:.2f} MB"


def convert_df_to_csv(df):
    """Convert DataFrame to CSV bytes for download"""
    return df.to_csv(index=False, encoding='utf-8').encode('utf-8')


def convert_df_to_excel(df):
    """Convert DataFrame to Excel bytes for download"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Data')
    return output.getvalue()


def create_zip_archive(files_dict):
    """Create a ZIP archive from multiple files

    Parameters
    ----------
    files_dict : dict
        Dictionary mapping filenames to file contents (bytes)

    Returns
    -------
    bytes
        ZIP file contents
    """
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for filename, content in files_dict.items():
            zip_file.writestr(filename, content)
    return zip_buffer.getvalue()


# EN-TETE DE LA PAGE
st.title("Export et Téléchargements")
st.markdown("""
Exportez vos analyses, visualisations et rapports dans différents formats
pour les partager ou les intégrer à vos présentations.
""")

st.divider()


# VERIFICATION DES DONNEES
if not st.session_state.get('data_loaded', False):
    st.warning("Veuillez d'abord charger les données depuis la page d'accueil.")
    st.stop()


# EXPORTS DE DONNEES
st.header("Export des Données")

st.markdown("""
Exportez les datasets traités et les résultats d'analyses au format CSV ou Excel.
""")

tab1, tab2, tab3, tab4 = st.tabs(["Données nettoyées", "Analyse RFM", "Analyse Cohortes", "Export groupé"])

# TAB 1: Données nettoyées
with tab1:
    st.subheader("Données brutes nettoyées")

    # Get data from session state
    df_clean = st.session_state.get('df_clean', pd.DataFrame())

    # Appliquer les filtres globaux
    if not df_clean.empty:
        active_filters = st.session_state.get('active_filters', {})
        df_clean = utils.apply_global_filters(df_clean, active_filters)

    if not df_clean.empty:
        st.markdown(f"""
        Dataset principal après nettoyage, prêt pour analyse externe.

        **Aperçu :**
        - {len(df_clean):,} lignes
        - {len(df_clean.columns)} colonnes
        - Taille estimée : {format_file_size(get_file_size(df_clean))}
        """)

        # Show preview of data
        with st.expander("Aperçu des données (10 premières lignes)"):
            st.dataframe(df_clean.head(10), use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            # Options d'export
            available_columns = list(df_clean.columns)
            include_columns = st.multiselect(
                "Colonnes à inclure (laisser vide pour tout exporter)",
                available_columns,
                default=[],
                help="Sélectionner les colonnes à exporter"
            )

        with col2:
            export_format = st.radio(
                "Format d'export",
                ["CSV", "Excel"],
                horizontal=True,
                key="clean_data_format"
            )

        # Prepare download button
        df_to_export = df_clean[include_columns] if include_columns else df_clean
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if export_format == "CSV":
            file_data = convert_df_to_csv(df_to_export)
            filename = f"cleaned_data_{timestamp}.csv"
            mime_type = "text/csv"
        else:
            file_data = convert_df_to_excel(df_to_export)
            filename = f"cleaned_data_{timestamp}.xlsx"
            mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

        st.download_button(
            label=f"Télécharger données nettoyées ({export_format})",
            data=file_data,
            file_name=filename,
            mime=mime_type,
            use_container_width=True,
            type="primary"
        )
    else:
        st.info("Aucune donnée nettoyée disponible")


# TAB 2: Analyse RFM
with tab2:
    st.subheader("Analyse RFM")

    # Get RFM data from session state
    df_rfm = st.session_state.get('df_rfm', pd.DataFrame())

    if not df_rfm.empty:
        st.markdown(f"""
        Scores RFM calculés pour tous les clients avec segmentation.

        **Aperçu :**
        - {len(df_rfm):,} clients
        - {df_rfm['Segment'].nunique()} segments
        - Taille estimée : {format_file_size(get_file_size(df_rfm))}
        """)

        # Show preview
        with st.expander("Aperçu des scores RFM (10 premières lignes)"):
            st.dataframe(df_rfm.head(10), use_container_width=True)

        # Show segment distribution
        st.markdown("**Distribution des segments:**")
        segment_dist = df_rfm['Segment'].value_counts()
        col_a, col_b, col_c = st.columns(3)

        for i, (segment, count) in enumerate(segment_dist.head(3).items()):
            with [col_a, col_b, col_c][i]:
                st.metric(segment, f"{count:,}")

        col1, col2 = st.columns(2)

        with col1:
            export_format_rfm = st.radio(
                "Format d'export",
                ["CSV", "Excel"],
                horizontal=True,
                key="rfm_format"
            )

        with col2:
            include_scores = st.checkbox(
                "Inclure les scores individuels (R, F, M)",
                value=True,
                help="Inclure les scores R_Score, F_Score, M_Score"
            )

        # Prepare data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if export_format_rfm == "CSV":
            file_data = convert_df_to_csv(df_rfm)
            filename = f"rfm_analysis_{timestamp}.csv"
            mime_type = "text/csv"
        else:
            file_data = convert_df_to_excel(df_rfm)
            filename = f"rfm_analysis_{timestamp}.xlsx"
            mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

        st.download_button(
            label=f"Télécharger analyse RFM ({export_format_rfm})",
            data=file_data,
            file_name=filename,
            mime=mime_type,
            use_container_width=True,
            type="primary"
        )
    else:
        st.info("Aucune analyse RFM disponible. Veuillez d'abord accéder à la page Segments.")


# TAB 3: Analyse Cohortes
with tab3:
    st.subheader("Analyse des Cohortes")

    # Get cohort data from session state
    df_cohorts = st.session_state.get('df_cohorts', pd.DataFrame())

    if not df_cohorts.empty:
        st.markdown(f"""
        Données de cohortes avec indices de rétention.

        **Aperçu :**
        - {len(df_cohorts):,} transactions
        - {df_cohorts['CohortMonth'].nunique()} cohortes
        - Taille estimée : {format_file_size(get_file_size(df_cohorts))}
        """)

        # Show preview
        with st.expander("Aperçu des données de cohortes (10 premières lignes)"):
            st.dataframe(df_cohorts.head(10), use_container_width=True)

        # Show cohort summary
        st.markdown("**Cohortes identifiées:**")
        cohort_summary = df_cohorts.groupby('CohortMonth').agg({
            'Customer ID': 'nunique',
            'TotalAmount': 'sum'
        }).reset_index()
        cohort_summary.columns = ['Cohorte', 'Nb Clients', 'Revenue Total']
        cohort_summary = cohort_summary.head(5)

        st.dataframe(cohort_summary, use_container_width=True, hide_index=True)

        export_format_cohort = st.radio(
            "Format d'export",
            ["CSV", "Excel"],
            horizontal=True,
            key="cohort_format"
        )

        # Prepare data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if export_format_cohort == "CSV":
            file_data = convert_df_to_csv(df_cohorts)
            filename = f"cohort_analysis_{timestamp}.csv"
            mime_type = "text/csv"
        else:
            file_data = convert_df_to_excel(df_cohorts)
            filename = f"cohort_analysis_{timestamp}.xlsx"
            mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

        st.download_button(
            label=f"Télécharger analyse cohortes ({export_format_cohort})",
            data=file_data,
            file_name=filename,
            mime=mime_type,
            use_container_width=True,
            type="primary"
        )
    else:
        st.info("Aucune analyse de cohortes disponible. Veuillez d'abord accéder à la page Cohortes.")


# TAB 4: Export groupé
with tab4:
    st.subheader("Export groupé (ZIP)")

    st.markdown("""
    Téléchargez toutes vos analyses en un seul fichier ZIP.
    """)

    # Check which datasets are available
    available_datasets = []

    df_clean = st.session_state.get('df_clean', pd.DataFrame())
    df_rfm = st.session_state.get('df_rfm', pd.DataFrame())
    df_cohorts = st.session_state.get('df_cohorts', pd.DataFrame())
    kpis = st.session_state.get('kpis', {})

    if not df_clean.empty:
        available_datasets.append("Données nettoyées")
    if not df_rfm.empty:
        available_datasets.append("Analyse RFM")
    if not df_cohorts.empty:
        available_datasets.append("Analyse Cohortes")
    if kpis:
        available_datasets.append("KPIs globaux")

    if available_datasets:
        st.markdown("**Datasets disponibles:**")
        for dataset in available_datasets:
            st.markdown(f"- {dataset}")

        st.divider()

        # Selection
        datasets_to_export = st.multiselect(
            "Sélectionner les datasets à inclure",
            available_datasets,
            default=available_datasets,
            help="Sélectionner les analyses à inclure dans le ZIP"
        )

        export_format_zip = st.radio(
            "Format des fichiers",
            ["CSV", "Excel"],
            horizontal=True,
            key="zip_format"
        )

        if st.button("Générer archive ZIP", use_container_width=True, type="primary"):
            try:
                with st.spinner("Création de l'archive ZIP..."):
                    files_dict = {}
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                    # Add selected datasets
                    if "Données nettoyées" in datasets_to_export and not df_clean.empty:
                        if export_format_zip == "CSV":
                            files_dict[f"cleaned_data_{timestamp}.csv"] = convert_df_to_csv(df_clean)
                        else:
                            files_dict[f"cleaned_data_{timestamp}.xlsx"] = convert_df_to_excel(df_clean)

                    if "Analyse RFM" in datasets_to_export and not df_rfm.empty:
                        if export_format_zip == "CSV":
                            files_dict[f"rfm_analysis_{timestamp}.csv"] = convert_df_to_csv(df_rfm)
                        else:
                            files_dict[f"rfm_analysis_{timestamp}.xlsx"] = convert_df_to_excel(df_rfm)

                    if "Analyse Cohortes" in datasets_to_export and not df_cohorts.empty:
                        if export_format_zip == "CSV":
                            files_dict[f"cohort_analysis_{timestamp}.csv"] = convert_df_to_csv(df_cohorts)
                        else:
                            files_dict[f"cohort_analysis_{timestamp}.xlsx"] = convert_df_to_excel(df_cohorts)

                    if "KPIs globaux" in datasets_to_export and kpis:
                        kpis_df = pd.DataFrame([kpis])
                        if export_format_zip == "CSV":
                            files_dict[f"kpis_{timestamp}.csv"] = convert_df_to_csv(kpis_df)
                        else:
                            files_dict[f"kpis_{timestamp}.xlsx"] = convert_df_to_excel(kpis_df)

                    # Create ZIP
                    zip_data = create_zip_archive(files_dict)

                    # Offer download
                    st.download_button(
                        label="Télécharger l'archive ZIP",
                        data=zip_data,
                        file_name=f"marketing_analytics_{timestamp}.zip",
                        mime="application/zip",
                        use_container_width=True
                    )

                st.success(f"Archive créée avec succès ! {len(files_dict)} fichier(s) inclus.")

            except Exception as e:
                st.error(f"Erreur lors de la création du ZIP: {str(e)}")

    else:
        st.info("Aucune donnée disponible pour l'export. Veuillez d'abord effectuer des analyses.")


st.divider()


# VISUALISATIONS
st.header("Export des Visualisations")

st.markdown("""
Les graphiques Plotly peuvent être exportés directement depuis chaque page :

1. **Survolez un graphique** avec votre souris
2. Un menu apparaît en haut à droite du graphique
3. Cliquez sur **l'icône appareil photo** pour télécharger
4. Choisissez le format souhaité (PNG, SVG, etc.)
""")

st.info("""
**Astuce:** Pour une qualité optimale, utilisez le format SVG qui est vectoriel et s'adapte à toutes les tailles sans perte de qualité.
""")

st.divider()


# INFORMATIONS ET AIDE
st.header("Informations et Aide")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Formats disponibles")

    st.markdown("""
    **CSV (Comma-Separated Values)**
    - Format universel, compatible avec tous les outils
    - Idéal pour Excel, Google Sheets, Python, R
    - Taille de fichier réduite
    - Recommandé pour les grands datasets

    **Excel (XLSX)**
    - Format Microsoft Excel natif
    - Préserve le formatage
    - Compatible avec Excel, LibreOffice, Google Sheets
    - Idéal pour les présentations et rapports
    """)

with col2:
    st.subheader("Recommandations")

    st.markdown("""
    **Pour l'analyse:**
    - Utilisez CSV pour importer dans Python/R
    - Exportez les données nettoyées en premier

    **Pour les rapports:**
    - Utilisez Excel pour les présentations
    - Exportez les analyses RFM et Cohortes
    - Utilisez les graphiques en haute résolution

    **Pour partager:**
    - Utilisez l'export groupé (ZIP)
    - Inclut toutes les analyses en un seul fichier
    - Facilite le partage et l'archivage
    """)

st.divider()


# STATISTIQUES D'EXPORT
st.header("Statistiques")

col1, col2, col3 = st.columns(3)

df_clean = st.session_state.get('df_clean', pd.DataFrame())
# Appliquer les filtres globaux aux statistiques
if not df_clean.empty:
    active_filters = st.session_state.get('active_filters', {})
    df_clean = utils.apply_global_filters(df_clean, active_filters)

df_rfm = st.session_state.get('df_rfm', pd.DataFrame())
df_cohorts = st.session_state.get('df_cohorts', pd.DataFrame())

with col1:
    datasets_available = 0
    if not df_clean.empty:
        datasets_available += 1
    if not df_rfm.empty:
        datasets_available += 1
    if not df_cohorts.empty:
        datasets_available += 1

    st.metric("Datasets disponibles", datasets_available)

with col2:
    total_size = 0
    if not df_clean.empty:
        total_size += get_file_size(df_clean)
    if not df_rfm.empty:
        total_size += get_file_size(df_rfm)
    if not df_cohorts.empty:
        total_size += get_file_size(df_cohorts)

    st.metric("Taille totale estimée", format_file_size(total_size))

with col3:
    total_rows = 0
    if not df_clean.empty:
        total_rows += len(df_clean)

    st.metric("Lignes de données", f"{total_rows:,}")


# FOOTER
st.divider()
st.caption("Page Export - Dernière mise à jour : " + datetime.now().strftime("%Y-%m-%d %H:%M"))
