"""
Page Analyse de Cohortes - Etude de la retention client par cohorte d'acquisition.

Cette page permet d'analyser l'evolution des cohortes de clients dans le temps
et de calculer les taux de retention pour optimiser les strategies d'acquisition
et de fidelisation.
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


# FONCTIONS HELPERS POUR LE CACHING
@st.cache_data(show_spinner=False)
def cached_create_cohorts(df: pd.DataFrame) -> pd.DataFrame:
    """Cache la creation des cohortes pour optimiser les performances."""
    return utils.create_cohorts(df)


@st.cache_data(show_spinner=False)
def cached_calculate_retention(cohort_df: pd.DataFrame) -> pd.DataFrame:
    """Cache le calcul de la matrice de retention."""
    return utils.calculate_retention(cohort_df)


@st.cache_data(show_spinner=False)
def cached_cohort_summary(df_cohorts: pd.DataFrame) -> pd.DataFrame:
    """Cache le calcul du tableau resume des cohortes."""
    # Filtrer les ventes uniquement
    df_sales = df_cohorts[~df_cohorts['IsReturn']].copy()

    # Creer le resume par cohorte
    summary = df_sales.groupby('CohortMonth').agg({
        'Customer ID': 'nunique',
        'Invoice': 'nunique',
        'TotalAmount': 'sum'
    }).reset_index()

    summary.columns = ['CohortMonth', 'Nb_Clients', 'Nb_Transactions', 'CA_Total']

    # Calculer le panier moyen et la frequence
    summary['Panier_Moyen'] = summary['CA_Total'] / summary['Nb_Transactions']
    summary['Frequence_Moy'] = summary['Nb_Transactions'] / summary['Nb_Clients']

    # Convertir CohortMonth en string pour l'affichage
    summary['CohortMonth'] = summary['CohortMonth'].astype(str)

    return summary


# CONFIGURATION DE LA PAGE
st.set_page_config(
    page_title="Analyse de Cohortes - Marketing Analytics",
    page_icon=":material/bar_chart:",
    layout="wide"
)


# EN-TETE DE LA PAGE
st.title("Analyse de Cohortes")
st.markdown("""
L'analyse de cohortes permet de suivre l'evolution de groupes de clients acquis
durant la meme periode et d'evaluer leur comportement au fil du temps.
""")

st.divider()


# VERIFICATION DES DONNEES
if not st.session_state.get('data_loaded', False):
    st.warning("Veuillez d'abord charger les donnees depuis la page d'accueil.")
    st.stop()

df = st.session_state.get('df_clean', None)

if df is None:
    st.error("Erreur lors du chargement des donnees")
    st.stop()

# Appliquer les filtres globaux
active_filters = st.session_state.get('active_filters', {})
df = utils.apply_global_filters(df, active_filters)

# FILTRES SPECIFIQUES
# Filtrer les donnees avec CustomerID pour l'analyse de cohortes
df_with_customers = df[df['HasCustomerID']].copy()

with st.sidebar:
    st.subheader("Filtres - Cohortes")

    # Calculer les cohortes disponibles
    df_temp = cached_create_cohorts(df_with_customers)
    available_cohorts = sorted(df_temp['CohortMonth'].unique())

    # Filtrer par periode de cohortes
    st.markdown("#### Periode d'analyse")

    # Convertir en datetime pour le slider
    cohort_dates = [pd.Period(c).to_timestamp() for c in available_cohorts]

    if len(cohort_dates) > 1:
        date_range = st.date_input(
            "Selectionner la periode",
            value=(cohort_dates[0], cohort_dates[-1]),
            min_value=cohort_dates[0],
            max_value=cohort_dates[-1],
            help="Filtrer les cohortes par periode d'acquisition"
        )

        # Appliquer le filtre si une plage est selectionnee
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_period = pd.Period(date_range[0], freq='M')
            end_period = pd.Period(date_range[1], freq='M')
            df_temp = df_temp[
                (df_temp['CohortMonth'] >= start_period) &
                (df_temp['CohortMonth'] <= end_period)
            ]

    # Taille minimale de cohorte
    min_cohort_size = st.number_input(
        "Taille minimale de cohorte",
        min_value=1,
        value=config.MIN_COHORT_SIZE,
        help="Exclure les cohortes avec moins de N clients"
    )

    # Type de visualisation
    st.markdown("#### Options de visualisation")
    show_percentages = st.checkbox(
        "Afficher en pourcentage",
        value=True,
        help="Afficher les taux de retention en % ou en nombres absolus"
    )

    st.divider()


# CREATION DES COHORTES
st.header("Creation des Cohortes")

with st.spinner("Creation des cohortes en cours..."):
    df_cohorts = cached_create_cohorts(df_with_customers)
    st.session_state.df_cohorts = df_cohorts

# Filtrer par taille minimale de cohorte
cohort_sizes = df_cohorts.groupby('CohortMonth')['Customer ID'].nunique()
valid_cohorts = cohort_sizes[cohort_sizes >= min_cohort_size].index
df_cohorts_filtered = df_cohorts[df_cohorts['CohortMonth'].isin(valid_cohorts)].copy()

# Statistiques des cohortes
num_cohorts = df_cohorts_filtered['CohortMonth'].nunique()
cohort_sizes_filtered = df_cohorts_filtered.groupby('CohortMonth')['Customer ID'].nunique()
avg_cohort_size = cohort_sizes_filtered.mean()
max_cohort_size = cohort_sizes_filtered.max()
max_cohort_month = cohort_sizes_filtered.idxmax()

# Calculer la periode d'analyse
max_cohort_index = df_cohorts_filtered['CohortIndex'].max()

# Afficher les metriques
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Nombre de cohortes",
        value=f"{num_cohorts}",
        help="Nombre de mois d'acquisition differents"
    )

with col2:
    st.metric(
        label="Taille moyenne",
        value=f"{int(avg_cohort_size)}",
        help="Nombre moyen de clients par cohorte"
    )

with col3:
    st.metric(
        label="Plus grande cohorte",
        value=f"{int(max_cohort_size)}",
        help=f"Cohorte de {max_cohort_month} - {int(max_cohort_size)} clients"
    )

with col4:
    st.metric(
        label="Periode d'analyse",
        value=f"{int(max_cohort_index) + 1} mois",
        help="Duree de la periode d'analyse en mois"
    )

st.divider()


# HEATMAP DE RETENTION
st.header("Heatmap de Retention")

st.markdown("""
Cette heatmap montre le pourcentage de clients de chaque cohorte qui sont revenus
effectuer un achat lors des mois suivants leur acquisition.
""")

# Calculer la matrice de retention
with st.spinner("Calcul de la matrice de retention..."):
    retention_matrix = cached_calculate_retention(df_cohorts_filtered)

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
    # Preparer les donnees pour la heatmap
    if value_type == "Nombre absolu":
        # Calculer le nombre absolu de clients
        cohort_counts = df_cohorts_filtered.groupby(['CohortMonth', 'CohortIndex'])['Customer ID'].nunique().reset_index()
        cohort_pivot = cohort_counts.pivot_table(
            index='CohortMonth',
            columns='CohortIndex',
            values='Customer ID'
        )
        heatmap_data = cohort_pivot
        color_label = "Nombre de clients"
        text_format = '.0f'
    else:
        heatmap_data = retention_matrix
        color_label = "Retention (%)"
        text_format = '.1f'

    # Convertir l'index en string pour l'affichage
    heatmap_data.index = heatmap_data.index.astype(str)

    # Creer la heatmap avec Plotly
    fig_heatmap = px.imshow(
        heatmap_data,
        labels=dict(x="Mois depuis acquisition", y="Cohorte", color=color_label),
        color_continuous_scale=colormap,
        aspect="auto",
        text_auto=text_format
    )

    fig_heatmap.update_layout(
        title="Matrice de Retention par Cohorte",
        xaxis_title="Mois depuis acquisition (CohortIndex)",
        yaxis_title="Cohorte (mois d'acquisition)",
        height=max(400, num_cohorts * 25),  # Hauteur adaptative
        font=dict(size=10)
    )

    fig_heatmap.update_xaxes(side="bottom")

    st.plotly_chart(fig_heatmap, use_container_width=True)

st.divider()


# COURBES DE RETENTION
st.header("Courbes de Retention par Cohorte")

st.markdown("""
Ces courbes montrent l'evolution de la retention pour chaque cohorte au fil des mois.
Elles permettent d'identifier les cohortes les plus performantes.
""")

# Options d'affichage
with st.expander("Options d'affichage", expanded=False):
    col1, col2 = st.columns(2)

    with col1:
        # Selection des cohortes a afficher
        all_cohorts_list = [str(c) for c in retention_matrix.index]

        if len(all_cohorts_list) > 10:
            # Si trop de cohortes, proposer de selectionner
            selected_cohorts = st.multiselect(
                "Cohortes a afficher",
                all_cohorts_list,
                default=all_cohorts_list[-5:],  # 5 dernieres par defaut
                help="Selectionner les cohortes a inclure dans le graphique"
            )
        else:
            selected_cohorts = all_cohorts_list

    with col2:
        # Nombre de mois a afficher
        max_months_available = len(retention_matrix.columns)
        months_to_show = st.slider(
            "Mois a afficher",
            min_value=3,
            max_value=min(12, max_months_available),
            value=min(6, max_months_available),
            help="Nombre de mois depuis acquisition a afficher"
        )

# Filtrer la matrice selon les selections
if len(selected_cohorts) == 0:
    selected_cohorts = all_cohorts_list

retention_matrix_filtered = retention_matrix.loc[selected_cohorts, :months_to_show-1]

# Creer le graphique de courbes
fig_curves = go.Figure()

for cohort in retention_matrix_filtered.index:
    fig_curves.add_trace(go.Scatter(
        x=retention_matrix_filtered.columns,
        y=retention_matrix_filtered.loc[cohort],
        mode='lines+markers',
        name=str(cohort),
        line=dict(width=2),
        marker=dict(size=6)
    ))

fig_curves.update_layout(
    title="Evolution de la Retention par Cohorte",
    xaxis_title="Mois depuis acquisition",
    yaxis_title="Taux de retention (%)",
    hovermode='x unified',
    height=500,
    legend=dict(
        title="Cohorte",
        orientation="v",
        yanchor="top",
        y=1,
        xanchor="left",
        x=1.02
    )
)

fig_curves.update_yaxes(range=[0, 105])

st.plotly_chart(fig_curves, use_container_width=True)

st.divider()


# ANALYSE COMPARATIVE
st.header("Analyse Comparative des Cohortes")

# Extraire les taux de retention pour M1 et M3
if 1 in retention_matrix.columns:
    retention_m1 = retention_matrix[1].sort_values(ascending=False)
else:
    retention_m1 = pd.Series(dtype=float)

if 3 in retention_matrix.columns:
    retention_m3 = retention_matrix[3].sort_values(ascending=False)
else:
    retention_m3 = pd.Series(dtype=float)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Retention M1 par cohorte")

    if len(retention_m1) > 0:
        # Calculer la moyenne pour la ligne de reference
        avg_m1 = retention_m1.mean()

        # Creer un bar chart
        fig_m1 = go.Figure()

        colors = ['green' if val >= avg_m1 else 'red' for val in retention_m1.values]

        fig_m1.add_trace(go.Bar(
            x=[str(idx) for idx in retention_m1.index],
            y=retention_m1.values,
            marker_color=colors,
            text=retention_m1.values.round(1),
            textposition='outside',
            texttemplate='%{text}%',
            hovertemplate='<b>%{x}</b><br>Retention M1: %{y:.1f}%<extra></extra>'
        ))

        # Ajouter une ligne de moyenne
        fig_m1.add_hline(
            y=avg_m1,
            line_dash="dash",
            line_color="blue",
            annotation_text=f"Moyenne: {avg_m1:.1f}%",
            annotation_position="right"
        )

        fig_m1.update_layout(
            xaxis_title="Cohorte",
            yaxis_title="Retention M1 (%)",
            height=400,
            showlegend=False
        )

        st.plotly_chart(fig_m1, use_container_width=True)
    else:
        st.info("Donnees M1 non disponibles pour ces cohortes")

with col2:
    st.subheader("Retention M3 par cohorte")

    if len(retention_m3) > 0:
        # Calculer la moyenne pour la ligne de reference
        avg_m3 = retention_m3.mean()

        # Creer un bar chart
        fig_m3 = go.Figure()

        colors = ['green' if val >= avg_m3 else 'red' for val in retention_m3.values]

        fig_m3.add_trace(go.Bar(
            x=[str(idx) for idx in retention_m3.index],
            y=retention_m3.values,
            marker_color=colors,
            text=retention_m3.values.round(1),
            textposition='outside',
            texttemplate='%{text}%',
            hovertemplate='<b>%{x}</b><br>Retention M3: %{y:.1f}%<extra></extra>'
        ))

        # Ajouter une ligne de moyenne
        fig_m3.add_hline(
            y=avg_m3,
            line_dash="dash",
            line_color="blue",
            annotation_text=f"Moyenne: {avg_m3:.1f}%",
            annotation_position="right"
        )

        fig_m3.update_layout(
            xaxis_title="Cohorte",
            yaxis_title="Retention M3 (%)",
            height=400,
            showlegend=False
        )

        st.plotly_chart(fig_m3, use_container_width=True)
    else:
        st.info("Donnees M3 non disponibles pour ces cohortes")

st.divider()


# METRIQUES DE RETENTION
st.header("Metriques de Retention Globales")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Retention moyenne")

    # Calculer les retentions moyennes pour chaque periode
    metrics_data = []
    for period in [1, 3, 6, 12]:
        if period in retention_matrix.columns:
            avg_retention = retention_matrix[period].mean()
            metrics_data.append({
                'Periode': f'M{period}',
                'Retention': avg_retention
            })

    if metrics_data:
        metrics_df = pd.DataFrame(metrics_data)

        # Afficher sous forme de metriques
        for i, row in metrics_df.iterrows():
            if i < len(metrics_df) - 1:
                # Calculer la variation vs periode suivante
                delta = metrics_df.iloc[i+1]['Retention'] - row['Retention']
                st.metric(
                    label=row['Periode'],
                    value=f"{row['Retention']:.1f}%",
                    delta=f"{delta:.1f}%",
                    delta_color="inverse"  # Baisse = rouge (car c'est negatif pour la retention)
                )
            else:
                st.metric(
                    label=row['Periode'],
                    value=f"{row['Retention']:.1f}%"
                )
    else:
        st.info("Donnees de retention non disponibles")

with col2:
    st.subheader("Meilleure/Pire cohorte")

    if len(retention_m3) > 0:
        ref_retention = retention_m3
        ref_period = "M3"
    elif len(retention_m1) > 0:
        ref_retention = retention_m1
        ref_period = "M1"
    else:
        ref_retention = pd.Series(dtype=float)
        ref_period = "N/A"

    if len(ref_retention) > 0:
        best_cohort = ref_retention.idxmax()
        best_retention = ref_retention.max()

        worst_cohort = ref_retention.idxmin()
        worst_retention = ref_retention.min()

        gap = best_retention - worst_retention

        st.metric(
            label=f"Meilleure cohorte ({ref_period})",
            value=f"{best_retention:.1f}%",
            help=f"Cohorte {best_cohort}"
        )

        st.metric(
            label=f"Pire cohorte ({ref_period})",
            value=f"{worst_retention:.1f}%",
            help=f"Cohorte {worst_cohort}"
        )

        st.metric(
            label="Ecart",
            value=f"{gap:.1f} pts",
            help="Difference entre meilleure et pire cohorte"
        )
    else:
        st.info("Donnees non disponibles")

with col3:
    st.subheader("Tendance de retention")

    # Analyser l'evolution de la retention M1 dans le temps
    if len(retention_m1) >= 3:
        # Creer un mini graphique sparkline
        fig_trend = go.Figure()

        x_vals = [str(idx) for idx in retention_m1.index]
        y_vals = retention_m1.values

        fig_trend.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='lines+markers',
            line=dict(color='blue', width=2),
            marker=dict(size=8),
            fill='tozeroy',
            fillcolor='rgba(31, 119, 180, 0.2)'
        ))

        # Ajouter une ligne de tendance
        from numpy.polynomial import Polynomial
        x_numeric = np.arange(len(y_vals))
        p = Polynomial.fit(x_numeric, y_vals, 1)
        trend_y = p(x_numeric)

        fig_trend.add_trace(go.Scatter(
            x=x_vals,
            y=trend_y,
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name='Tendance'
        ))

        fig_trend.update_layout(
            title="Evolution M1 dans le temps",
            xaxis_title="Cohorte",
            yaxis_title="Retention M1 (%)",
            height=250,
            showlegend=False,
            margin=dict(l=20, r=20, t=40, b=20)
        )

        st.plotly_chart(fig_trend, use_container_width=True)

        # Determiner la tendance
        if trend_y[-1] > trend_y[0]:
            trend_direction = "Amelioration"
            trend_color = "green"
            trend_delta = trend_y[-1] - trend_y[0]
        else:
            trend_direction = "Degradation"
            trend_color = "red"
            trend_delta = trend_y[-1] - trend_y[0]

        st.markdown(f"**Tendance:** :{trend_color}[{trend_direction}] ({trend_delta:+.1f} pts)")
    else:
        st.info("Donnees insuffisantes pour analyser la tendance")

st.divider()


# TABLE DETAILLEE DES COHORTES
st.header("Tableau Detaille des Cohortes")

st.markdown("""
Vue tabulaire detaillee de toutes les cohortes avec leurs metriques cles.
""")

# Creer le tableau detaille
cohort_summary = cached_cohort_summary(df_cohorts_filtered)

# Ajouter les taux de retention au tableau
for period in [1, 3, 6, 12]:
    col_name = f'Retention_M{period}'
    if period in retention_matrix.columns:
        # Mapper les valeurs de retention
        retention_dict = retention_matrix[period].to_dict()
        cohort_summary[col_name] = cohort_summary['CohortMonth'].apply(
            lambda x: retention_dict.get(pd.Period(x, freq='M'), np.nan)
        )
    else:
        cohort_summary[col_name] = np.nan

# Calculer la CLV moyenne par cohorte (approximation)
cohort_summary['CLV_Moy'] = cohort_summary['CA_Total'] / cohort_summary['Nb_Clients']

# Options de tri et filtrage
col1, col2, col3 = st.columns(3)

with col1:
    sort_by = st.selectbox(
        "Trier par",
        ["CohortMonth", "Nb_Clients", "Retention_M1", "Retention_M3", "CA_Total"],
        help="Critere de tri du tableau"
    )

with col2:
    sort_order = st.selectbox(
        "Ordre",
        ["Decroissant", "Croissant"],
        help="Ordre de tri"
    )

with col3:
    filter_min_size = st.number_input(
        "Taille minimale",
        min_value=0,
        value=0,
        help="Filtrer les cohortes avec un minimum de clients"
    )

# Appliquer le tri et le filtrage
cohort_summary_display = cohort_summary[cohort_summary['Nb_Clients'] >= filter_min_size].copy()

if sort_by in cohort_summary_display.columns:
    ascending = (sort_order == "Croissant")
    cohort_summary_display = cohort_summary_display.sort_values(sort_by, ascending=ascending)

# Formater les colonnes pour l'affichage
display_df = cohort_summary_display.copy()

# Arrondir les valeurs
numeric_cols = ['Panier_Moyen', 'Frequence_Moy', 'CLV_Moy']
for col in numeric_cols:
    if col in display_df.columns:
        display_df[col] = display_df[col].round(2)

retention_cols = [c for c in display_df.columns if c.startswith('Retention_')]
for col in retention_cols:
    if col in display_df.columns:
        display_df[col] = display_df[col].round(1)

# Afficher le tableau avec style
st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "CohortMonth": st.column_config.TextColumn("Cohorte", help="Mois d'acquisition"),
        "Nb_Clients": st.column_config.NumberColumn("Clients", help="Nombre de clients dans la cohorte", format="%d"),
        "Nb_Transactions": st.column_config.NumberColumn("Transactions", help="Nombre total de transactions", format="%d"),
        "CA_Total": st.column_config.NumberColumn("CA Total", help="Chiffre d'affaires total", format="£%.2f"),
        "Panier_Moyen": st.column_config.NumberColumn("Panier Moyen", help="Valeur moyenne par transaction", format="£%.2f"),
        "Frequence_Moy": st.column_config.NumberColumn("Frequence", help="Nombre moyen de transactions par client", format="%.2f"),
        "Retention_M1": st.column_config.NumberColumn("Ret. M1", help="Taux de retention au mois 1", format="%.1f%%"),
        "Retention_M3": st.column_config.NumberColumn("Ret. M3", help="Taux de retention au mois 3", format="%.1f%%"),
        "Retention_M6": st.column_config.NumberColumn("Ret. M6", help="Taux de retention au mois 6", format="%.1f%%"),
        "Retention_M12": st.column_config.NumberColumn("Ret. M12", help="Taux de retention au mois 12", format="%.1f%%"),
        "CLV_Moy": st.column_config.NumberColumn("CLV Moyenne", help="Customer Lifetime Value moyenne", format="£%.2f")
    }
)

st.divider()


# INSIGHTS ET RECOMMANDATIONS
st.header("Insights et Recommandations")

with st.expander("Analyse des cohortes", expanded=True):
    # Generer des insights automatiques
    insights = []

    # Insight 1: Meilleure cohorte
    if len(ref_retention) > 0:
        best_cohort = ref_retention.idxmax()
        best_retention = ref_retention.max()
        insights.append(
            f"**Meilleure performance:** La cohorte de {best_cohort} affiche la meilleure retention "
            f"a {ref_period} avec {best_retention:.1f}%. Cette cohorte devrait etre etudiee pour "
            f"comprendre les facteurs de succes (campagnes marketing, saisonnalite, produits)."
        )

    # Insight 2: Tendance generale
    if len(retention_m1) >= 3:
        first_half_avg = retention_m1.iloc[:len(retention_m1)//2].mean()
        second_half_avg = retention_m1.iloc[len(retention_m1)//2:].mean()

        if second_half_avg > first_half_avg:
            insights.append(
                f"**Tendance positive:** La retention M1 s'ameliore dans le temps "
                f"(+{second_half_avg - first_half_avg:.1f} pts). Les efforts de fidelisation "
                f"semblent porter leurs fruits."
            )
        else:
            insights.append(
                f"**Tendance negative:** La retention M1 se degrade dans le temps "
                f"({second_half_avg - first_half_avg:.1f} pts). Il est recommande d'analyser "
                f"les facteurs de cette baisse et d'ajuster la strategie d'acquisition."
            )

    # Insight 3: Taille des cohortes
    if avg_cohort_size > 0:
        large_cohorts = cohort_sizes_filtered[cohort_sizes_filtered > avg_cohort_size * 1.5]
        if len(large_cohorts) > 0:
            insights.append(
                f"**Periodes d'acquisition forte:** {len(large_cohorts)} cohorte(s) depassent "
                f"significativement la taille moyenne. Ces periodes correspondent probablement "
                f"a des campagnes marketing reussies ou des pics saisonniers."
            )

    # Insight 4: Retention a M3
    if len(retention_m3) > 0:
        avg_m3 = retention_m3.mean()
        if avg_m3 < 20:
            insights.append(
                f"**Alerte retention:** Le taux de retention moyen a M3 est faible ({avg_m3:.1f}%). "
                f"Il est crucial de mettre en place des actions de reactivation rapide pour les nouveaux clients."
            )
        elif avg_m3 > 35:
            insights.append(
                f"**Retention solide:** Le taux de retention moyen a M3 est bon ({avg_m3:.1f}%). "
                f"La strategie de fidelisation est efficace."
            )

    # Insight 5: Ecart entre cohortes
    if len(ref_retention) > 0:
        gap = ref_retention.max() - ref_retention.min()
        if gap > 20:
            insights.append(
                f"**Forte variabilite:** L'ecart de retention entre la meilleure et la pire cohorte "
                f"est important ({gap:.1f} pts). Cette heterogeneite suggere que certains facteurs "
                f"(saisonnalite, source d'acquisition, qualite des leads) ont un impact significatif "
                f"sur la retention."
            )

    # Afficher les insights
    if insights:
        for i, insight in enumerate(insights, 1):
            st.markdown(f"{i}. {insight}")
            st.markdown("")
    else:
        st.info("Donnees insuffisantes pour generer des insights automatiques.")

    # Recommandations
    st.markdown("### Recommandations d'actions")

    recommendations = [
        "**Optimiser l'onboarding:** Concentrer les efforts sur les 30 premiers jours pour maximiser "
        "la retention M1, periode critique pour la fidelisation.",

        "**Segmenter les cohortes:** Analyser les caracteristiques des cohortes performantes "
        "(canaux d'acquisition, profils clients, produits achetes) pour repliquer leur succes.",

        "**Programme de reactivation:** Mettre en place des campagnes automatisees a J+30, J+60 et J+90 "
        "pour les clients inactifs afin d'ameliorer la retention.",

        "**Tester des incentives:** Proposer des offres personnalisees aux cohortes a faible retention "
        "pour evaluer l'impact sur le comportement d'achat.",

        "**Analyser la saisonnalite:** Etudier les variations de retention selon les periodes "
        "d'acquisition pour adapter les strategies marketing."
    ]

    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"{i}. {rec}")

st.divider()


# EXPORT
st.header("Export des Analyses")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Exporter matrice de retention (CSV)", use_container_width=True):
        try:
            # Preparer la matrice pour l'export
            export_matrix = retention_matrix.copy()
            export_matrix.index = export_matrix.index.astype(str)

            # Convertir en CSV
            csv = export_matrix.to_csv(sep=config.CSV_SEPARATOR)

            # Telecharger
            st.download_button(
                label="Telecharger CSV",
                data=csv,
                file_name=f"matrice_retention_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
            st.success("Matrice prete a telecharger !")
        except Exception as e:
            st.error(f"Erreur lors de l'export : {str(e)}")

with col2:
    if st.button("Exporter tableau cohortes (CSV)", use_container_width=True):
        try:
            # Convertir en CSV
            csv = cohort_summary_display.to_csv(index=False, sep=config.CSV_SEPARATOR)

            # Telecharger
            st.download_button(
                label="Telecharger CSV",
                data=csv,
                file_name=f"tableau_cohortes_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
            st.success("Tableau pret a telecharger !")
        except Exception as e:
            st.error(f"Erreur lors de l'export : {str(e)}")

with col3:
    if st.button("Exporter donnees completes (CSV)", use_container_width=True):
        try:
            # Exporter le dataframe complet des cohortes
            export_df = df_cohorts_filtered.copy()

            # Convertir les colonnes Period en string
            export_df['CohortMonth'] = export_df['CohortMonth'].astype(str)
            export_df['InvoiceMonth'] = export_df['InvoiceMonth'].astype(str)

            # Convertir en CSV
            csv = export_df.to_csv(index=False, sep=config.CSV_SEPARATOR)

            # Telecharger
            st.download_button(
                label="Telecharger CSV",
                data=csv,
                file_name=f"cohortes_complet_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
            st.success("Donnees pretes a telecharger !")
        except Exception as e:
            st.error(f"Erreur lors de l'export : {str(e)}")


# FOOTER
st.divider()
st.caption("Page Analyse de Cohortes - Derniere mise a jour : " + datetime.now().strftime("%Y-%m-%d %H:%M"))
