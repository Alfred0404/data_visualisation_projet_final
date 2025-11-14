"""
Configuration centralisée pour l'application d'aide à la décision marketing.

Ce module contient tous les paramètres de configuration, chemins de fichiers,
constantes métier et paramètres par défaut utilisés dans l'application.
"""

import os
from pathlib import Path
from datetime import datetime

# ==============================================================================
# CHEMINS DES FICHIERS ET DOSSIERS
# ==============================================================================

# Racine du projet
PROJECT_ROOT = Path(__file__).parent.absolute()

# Répertoires de données
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Répertoires de l'application
APP_DIR = PROJECT_ROOT / "app"
PAGES_DIR = APP_DIR / "pages"

# Répertoires de documentation
DOCS_DIR = PROJECT_ROOT / "docs"
PREZ_DIR = DOCS_DIR / "prez"

# Notebooks
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Fichiers de données sources
RAW_DATA_CSV = RAW_DATA_DIR / "online_retail_II.csv"
RAW_DATA_XLSX = RAW_DATA_DIR / "online_retail_II.xlsx"

# Fichiers de données traitées
PROCESSED_DATA_CSV = PROCESSED_DATA_DIR / "cleaned_data.csv"
COHORTS_DATA_CSV = PROCESSED_DATA_DIR / "cohorts.csv"
RFM_DATA_CSV = PROCESSED_DATA_DIR / "rfm_segments.csv"

# Répertoires d'export
EXPORT_DIR = PROJECT_ROOT / "exports"
EXPORT_DATA_DIR = EXPORT_DIR / "data"
EXPORT_CHARTS_DIR = EXPORT_DIR / "charts"


# ==============================================================================
# COLONNES DU DATASET
# ==============================================================================

# Colonnes attendues dans le dataset Online Retail II
REQUIRED_COLUMNS = [
    "Invoice",           # Numéro de facture
    "StockCode",         # Code produit
    "Description",       # Description du produit
    "Quantity",          # Quantité achetée
    "InvoiceDate",       # Date et heure de la transaction
    "Price",             # Prix unitaire
    "Customer ID",       # Identifiant client
    "Country"            # Pays du client
]

# Colonnes calculées
CALCULATED_COLUMNS = {
    "TotalAmount": "Quantity * Price",
    "InvoiceMonth": "Date de la facture au mois près",
    "CohortMonth": "Mois de première transaction du client",
    "CohortIndex": "Nombre de mois depuis la première transaction"
}


# ==============================================================================
# PARAMETRES DE NETTOYAGE DES DONNEES
# ==============================================================================

# Préfixe des factures d'annulation
CANCELLATION_PREFIX = "C"

# Seuils de validation
MIN_QUANTITY = 0
MIN_PRICE = 0
MIN_TOTAL_AMOUNT = 0


# ==============================================================================
# PARAMETRES RFM (Recency, Frequency, Monetary)
# ==============================================================================

# Date de référence pour le calcul de la récence (peut être modifiée)
# Par défaut : date maximale dans les données
RFM_REFERENCE_DATE = None  # Sera calculée dynamiquement

# Nombre de quartiles pour la segmentation RFM
RFM_QUANTILES = 4

# Labels des scores RFM (du meilleur au moins bon)
RFM_SCORE_LABELS = {
    4: "Excellent",
    3: "Bon",
    2: "Moyen",
    1: "Faible"
}

# Segments RFM principaux
RFM_SEGMENTS = {
    "Champions": {
        "R": [4],
        "F": [4],
        "M": [4],
        "description": "Clients les plus précieux - achètent récemment, souvent et dépensent beaucoup"
    },
    "Loyal Customers": {
        "R": [3, 4],
        "F": [3, 4],
        "M": [3, 4],
        "description": "Clients fidèles avec un bon potentiel"
    },
    "Potential Loyalists": {
        "R": [3, 4],
        "F": [1, 2],
        "M": [2, 3],
        "description": "Clients récents avec potentiel de fidélisation"
    },
    "New Customers": {
        "R": [4],
        "F": [1],
        "M": [1, 2],
        "description": "Nouveaux clients à fidéliser"
    },
    "At Risk": {
        "R": [1, 2],
        "F": [3, 4],
        "M": [3, 4],
        "description": "Bons clients qui n'ont pas acheté récemment - risque de perte"
    },
    "Cannot Lose Them": {
        "R": [1],
        "F": [4],
        "M": [4],
        "description": "Meilleurs clients en perte - action urgente requise"
    },
    "Hibernating": {
        "R": [1, 2],
        "F": [1, 2],
        "M": [1, 2],
        "description": "Clients inactifs à faible valeur"
    },
    "Lost": {
        "R": [1],
        "F": [1],
        "M": [1],
        "description": "Clients perdus"
    }
}


# ==============================================================================
# PARAMETRES CLV (Customer Lifetime Value)
# ==============================================================================

# Taux de rétention par défaut (à calculer ou ajuster selon les données)
DEFAULT_RETENTION_RATE = 0.30  # 30%

# Taux d'actualisation (discount rate) pour les calculs CLV
DEFAULT_DISCOUNT_RATE = 0.10  # 10% annuel

# Période d'analyse en mois
CLV_ANALYSIS_PERIOD_MONTHS = 12

# Horizon de prévision CLV en années
CLV_FORECAST_HORIZON_YEARS = 3


# ==============================================================================
# PARAMETRES D'ANALYSE DES COHORTES
# ==============================================================================

# Format d'affichage des cohortes
COHORT_DATE_FORMAT = "%Y-%m"

# Nombre minimum de clients par cohorte pour être significative
MIN_COHORT_SIZE = 10

# Périodes d'analyse pour les cohortes (en mois)
COHORT_ANALYSIS_PERIODS = [0, 1, 3, 6, 12]  # M0, M1, M3, M6, M12


# ==============================================================================
# PARAMETRES DE VISUALISATION
# ==============================================================================

# Palette de couleurs principale (cohérence visuelle)
COLOR_PALETTE = {
    "primary": "#1f77b4",      # Bleu
    "secondary": "#ff7f0e",    # Orange
    "success": "#2ca02c",      # Vert
    "warning": "#ff9800",      # Jaune-orange
    "danger": "#d62728",       # Rouge
    "info": "#17a2b8",         # Cyan
    "neutral": "#7f7f7f"       # Gris
}

# Palette séquentielle pour heatmaps
HEATMAP_COLORSCALE = "RdYlGn"  # Rouge-Jaune-Vert

# Palette catégorielle pour segments
SEGMENT_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"
]

# Taille par défaut des figures
DEFAULT_FIGURE_SIZE = (12, 6)

# DPI pour l'export des graphiques
EXPORT_DPI = 300


# ==============================================================================
# PARAMETRES STREAMLIT
# ==============================================================================

# Configuration de la page Streamlit
PAGE_CONFIG = {
    "page_title": "Marketing Decision Support",
    "page_icon": "📊",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Titre de l'application
APP_TITLE = "Application d'Aide à la Décision Marketing"

# Description de l'application
APP_DESCRIPTION = """
Cette application permet d'analyser les données de ventes pour:
- Analyser les cohortes d'acquisition et leur rétention
- Segmenter les clients selon la méthodologie RFM
- Calculer la Customer Lifetime Value (CLV)
- Simuler différents scénarios marketing
- Exporter les analyses et visualisations
"""


# ==============================================================================
# PARAMETRES DE FILTRAGE
# ==============================================================================

# Pays à inclure par défaut (peut être modifié dans l'interface)
DEFAULT_COUNTRIES = None  # None = tous les pays

# Plage de dates par défaut
DEFAULT_DATE_RANGE = None  # None = toute la période disponible

# Montant minimum de transaction
DEFAULT_MIN_TRANSACTION_AMOUNT = 0


# ==============================================================================
# PARAMETRES DE SIMULATION
# ==============================================================================

# Scénarios de simulation prédéfinis
SIMULATION_SCENARIOS = {
    "optimistic": {
        "retention_increase": 0.10,  # +10%
        "aov_increase": 0.15,        # +15% Average Order Value
        "frequency_increase": 0.20,  # +20%
        "label": "Scénario Optimiste"
    },
    "realistic": {
        "retention_increase": 0.05,  # +5%
        "aov_increase": 0.08,        # +8%
        "frequency_increase": 0.10,  # +10%
        "label": "Scénario Réaliste"
    },
    "conservative": {
        "retention_increase": 0.02,  # +2%
        "aov_increase": 0.03,        # +3%
        "frequency_increase": 0.05,  # +5%
        "label": "Scénario Conservateur"
    }
}


# ==============================================================================
# FORMATS D'EXPORT
# ==============================================================================

# Formats de dates pour les exports
EXPORT_DATE_FORMAT = "%Y-%m-%d"
EXPORT_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

# Séparateur CSV
CSV_SEPARATOR = ","

# Encodage des fichiers
FILE_ENCODING = "utf-8"


# ==============================================================================
# CONSTANTES METIER
# ==============================================================================

# Seuils d'alerte pour les KPIs
KPI_THRESHOLDS = {
    "churn_rate_warning": 0.30,      # 30% de churn = alerte
    "churn_rate_critical": 0.50,     # 50% de churn = critique
    "retention_rate_good": 0.40,     # 40% de rétention = bon
    "retention_rate_excellent": 0.60 # 60% de rétention = excellent
}

# Périodes de référence pour les comparaisons
COMPARISON_PERIODS = {
    "MoM": "Month over Month",
    "QoQ": "Quarter over Quarter",
    "YoY": "Year over Year"
}


# ==============================================================================
# VALIDATION DE LA CONFIGURATION
# ==============================================================================

def validate_config():
    """
    Valide que tous les répertoires nécessaires existent.
    Crée les répertoires manquants si nécessaire.
    """
    required_dirs = [
        DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
        EXPORT_DIR, EXPORT_DATA_DIR, EXPORT_CHARTS_DIR
    ]

    for directory in required_dirs:
        directory.mkdir(parents=True, exist_ok=True)

    return True


# ==============================================================================
# INITIALISATION
# ==============================================================================

if __name__ == "__main__":
    print("Configuration du projet Marketing Decision Support")
    print("=" * 70)
    print(f"Racine du projet: {PROJECT_ROOT}")
    print(f"Répertoire des données: {DATA_DIR}")
    print(f"Répertoire de l'application: {APP_DIR}")
    print("=" * 70)

    # Validation de la configuration
    if validate_config():
        print("Configuration validée avec succès !")
    else:
        print("Erreur lors de la validation de la configuration.")
