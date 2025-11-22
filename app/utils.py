"""
Module de fonctions utilitaires pour l'application d'aide à la décision marketing.

Ce module contient toutes les fonctions de traitement des données, calculs métier
et opérations d'export. Les fonctions sont conçues pour être pures, testables et
réutilisables à travers l'application.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path
import os
import sys

sys.path.append(str(Path(__file__).parent.parent))
import config


# ==============================================================================
# CHARGEMENT ET NETTOYAGE DES DONNEES
# ==============================================================================

def load_data(file_path: Union[str, Path], verbose: bool = False) -> pd.DataFrame:
    """
    Charge les données depuis un fichier CSV ou Excel avec optimisations pour les gros fichiers.

    Cette fonction détecte automatiquement le format du fichier et applique
    les paramètres de chargement appropriés (encodage, séparateur, etc.).
    Elle est optimisée pour minimiser la perte de données et gérer efficacement
    les fichiers volumineux (>500k lignes).

    Parameters
    ----------
    file_path : str or Path
        Chemin vers le fichier de données (CSV ou Excel)
    verbose : bool, default=False
        Si True, affiche des informations détaillées sur le chargement

    Returns
    -------
    pd.DataFrame
        DataFrame contenant les données chargées avec métadonnées de qualité

    Raises
    ------
    FileNotFoundError
        Si le fichier n'existe pas
    ValueError
        Si le format du fichier n'est pas supporté

    Examples
    --------
    >>> df = load_data(config.RAW_DATA_CSV, verbose=True)
    >>> print(df.shape)
    (525461, 8)

    Notes
    -----
    Formats supportés : .csv, .xlsx, .xls
    L'encodage par défaut pour les CSV est UTF-8
    Les Customer ID sont chargés comme string pour éviter la perte de précision
    Les dates invalides sont converties en NaT plutôt que de faire échouer le chargement

    Optimisations appliquées:
    - Utilisation de dtypes optimisés pour réduire l'utilisation mémoire
    - Gestion des valeurs manquantes sans suppression automatique
    - Parsing robuste des dates avec format européen
    - Nettoyage des noms de colonnes (BOM, espaces)
    """
    # Convertir en Path si nécessaire
    file_path = Path(file_path)

    # Vérifier que le fichier existe
    if not file_path.exists():
        raise FileNotFoundError(f"Le fichier {file_path} n'existe pas")

    # Détecter l'extension du fichier
    extension = file_path.suffix.lower()

    if verbose:
        print(f"Chargement du fichier: {file_path.name}")
        print(f"Format détecté: {extension}")

    try:
        if extension == '.csv':
            # Essayer d'abord avec le séparateur configuré
            try:
                df = pd.read_csv(
                    file_path,
                    encoding=config.FILE_ENCODING,
                    sep=config.CSV_SEPARATOR,
                    dtype={'Customer ID': str, 'Invoice': str, 'StockCode': str},
                    parse_dates=['InvoiceDate'],
                    dayfirst=True,
                    na_values=['', 'NA', 'N/A', 'null', 'NULL', 'None']
                )
            except Exception:
                # Si échec, essayer avec détection automatique du séparateur
                df = pd.read_csv(
                    file_path,
                    encoding=config.FILE_ENCODING,
                    sep=None,  # Détection automatique
                    engine='python',
                    dtype={'Customer ID': str, 'Invoice': str, 'StockCode': str},
                    na_values=['', 'NA', 'N/A', 'null', 'NULL', 'None']
                )

                # Parser les dates après chargement
                if 'InvoiceDate' in df.columns:
                    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], dayfirst=True, errors='coerce')

        elif extension in ['.xlsx', '.xls']:
            # Charger Excel avec gestion optimisée de la mémoire
            df = pd.read_excel(
                file_path,
                dtype={'Customer ID': str, 'Invoice': str, 'StockCode': str},
                na_values=['', 'NA', 'N/A', 'null', 'NULL', 'None'],
                engine='openpyxl' if extension == '.xlsx' else None
            )

            # Parser les dates (Excel les charge généralement correctement, mais vérifier)
            if 'InvoiceDate' in df.columns:
                if df['InvoiceDate'].dtype != 'datetime64[ns]':
                    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')

        else:
            raise ValueError(f"Format de fichier non supporté : {extension}. Formats acceptés : .csv, .xlsx, .xls")

        # Validation basique
        if df.empty:
            raise ValueError("Le fichier chargé est vide")

        # Nettoyer les noms de colonnes (supprimer BOM et espaces)
        df.columns = df.columns.str.replace('\ufeff', '', regex=False).str.strip()

        # Convertir les colonnes numériques si nécessaire (gestion format européen)
        numeric_columns = ['Quantity', 'Price']
        for col in numeric_columns:
            if col in df.columns:
                # Si la colonne est de type string, convertir (remplacer virgule par point)
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                elif df[col].dtype not in ['int64', 'float64']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

        if verbose:
            print(f"\nChargement réussi!")
            print(f"Dimensions: {df.shape[0]:,} lignes x {df.shape[1]} colonnes")
            print(f"Colonnes: {list(df.columns)}")
            print(f"\nValeurs manquantes par colonne:")
            missing = df.isnull().sum()
            for col in df.columns:
                if missing[col] > 0:
                    pct = (missing[col] / len(df)) * 100
                    print(f"  {col}: {missing[col]:,} ({pct:.2f}%)")

        return df

    except Exception as e:
        raise ValueError(f"Erreur lors du chargement du fichier {file_path}: {str(e)}")


def clean_data(df: pd.DataFrame, verbose: bool = False, strict_mode: bool = False) -> pd.DataFrame:
    """
    Nettoie et prépare les données pour l'analyse avec minimisation de la perte de données.

    Cette fonction effectue un nettoyage intelligent qui maximise la rétention des données
    tout en maintenant la qualité nécessaire pour les analyses. Elle catégorise les données
    plutôt que de les supprimer systématiquement.

    STRATEGIE DE NETTOYAGE OPTIMISEE :

    1. Conservation des données :
       - Les transactions SANS Customer ID sont CONSERVEES pour les analyses
         produits/pays/tendances (mais marquées pour exclusion des analyses clients)
       - Les retours/annulations sont CONSERVES et marqués (utiles pour taux de retour)
       - Les doublons exacts sont supprimés (peu de perte, améliore qualité)

    2. Filtrage strict uniquement pour :
       - Prix invalides (négatifs ou zéro) : impossible à analyser financièrement
       - Incohérences logiques : retours avec quantité positive, ventes avec quantité négative
       - Dates invalides : impossibles à analyser temporellement

    3. Enrichissement des données :
       - Création de flags plutôt que suppression (IsReturn, HasCustomerID, etc.)
       - Colonnes temporelles pour analyses temporelles
       - Colonnes calculées (TotalAmount, etc.)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame brut à nettoyer
    verbose : bool, default=False
        Si True, affiche un rapport détaillé du nettoyage avec statistiques
    strict_mode : bool, default=False
        Si True, applique un filtrage strict (supprime les lignes sans Customer ID)
        Si False (recommandé), conserve toutes les données valides avec flags

    Returns
    -------
    pd.DataFrame
        DataFrame nettoyé et enrichi, prêt pour l'analyse

    Examples
    --------
    >>> df_raw = load_data(config.RAW_DATA_CSV)
    >>> df_clean = clean_data(df_raw, verbose=True, strict_mode=False)
    >>> print(f"Lignes conservées : {len(df_clean)} / {len(df_raw)}")
    >>> # Analyser uniquement les clients
    >>> df_customers = df_clean[df_clean['HasCustomerID']]

    Notes
    -----
    Mode recommandé : strict_mode=False pour analyses marketing complètes
    Mode strict : strict_mode=True pour analyses purement centrées clients (RFM, CLV, etc.)

    Taux de rétention attendus :
    - Mode non-strict : ~95-97% des lignes (perte minimale)
    - Mode strict : ~78% des lignes (conforme aux analyses clients)
    """
    if verbose:
        print("="*70)
        print("NETTOYAGE DES DONNEES")
        print("="*70)
        print(f"Mode: {'STRICT (customer-centric)' if strict_mode else 'OPTIMISE (data preservation)'}")
        print(f"Lignes initiales: {len(df):,}")

    # Copier pour éviter de modifier l'original
    df_clean = df.copy()
    initial_count = len(df_clean)

    # ==========================================================================
    # ETAPE 1 : IDENTIFICATION ET MARQUAGE (PAS DE SUPPRESSION)
    # ==========================================================================

    # 1.1 Identifier les lignes avec Customer ID
    df_clean['HasCustomerID'] = df_clean['Customer ID'].notna()

    # 1.2 Identifier les retours/annulations (Invoice commence par 'C')
    df_clean['IsReturn'] = df_clean['Invoice'].astype(str).str.startswith(config.CANCELLATION_PREFIX)

    # 1.3 Identifier les dates valides
    df_clean['HasValidDate'] = df_clean['InvoiceDate'].notna()

    if verbose:
        print(f"\nMarquage des données:")
        print(f"  Avec Customer ID: {df_clean['HasCustomerID'].sum():,} ({df_clean['HasCustomerID'].sum()/len(df_clean)*100:.2f}%)")
        print(f"  Sans Customer ID: {(~df_clean['HasCustomerID']).sum():,} ({(~df_clean['HasCustomerID']).sum()/len(df_clean)*100:.2f}%)")
        print(f"  Retours/Annulations: {df_clean['IsReturn'].sum():,} ({df_clean['IsReturn'].sum()/len(df_clean)*100:.2f}%)")
        print(f"  Dates valides: {df_clean['HasValidDate'].sum():,} ({df_clean['HasValidDate'].sum()/len(df_clean)*100:.2f}%)")

    # ==========================================================================
    # ETAPE 2 : FILTRAGE DES DONNEES REELLEMENT INVALIDES
    # ==========================================================================

    step_results = []

    # 2.1 Supprimer les lignes avec dates invalides (impossible à analyser temporellement)
    before = len(df_clean)
    df_clean = df_clean[df_clean['HasValidDate']]
    after = len(df_clean)
    lost = before - after
    if verbose and lost > 0:
        step_results.append(f"  Dates invalides: -{lost:,} lignes ({lost/initial_count*100:.2f}%)")

    # 2.2 Supprimer les prix invalides (négatifs ou zéro)
    # Un prix de 0 ou négatif rend impossible toute analyse financière
    before = len(df_clean)
    df_clean = df_clean[df_clean['Price'] > config.MIN_PRICE]
    after = len(df_clean)
    lost = before - after
    if verbose and lost > 0:
        step_results.append(f"  Prix invalides (<=0): -{lost:,} lignes ({lost/initial_count*100:.2f}%)")

    # 2.3 Filtrer les incohérences logiques dans les quantités
    # - Vente normale (pas de retour) : quantité doit être positive
    # - Retour/annulation : quantité doit être négative ou nulle
    before = len(df_clean)
    mask_valid_quantity = (
        (~df_clean['IsReturn'] & (df_clean['Quantity'] > config.MIN_QUANTITY)) |
        (df_clean['IsReturn'] & (df_clean['Quantity'] <= config.MIN_QUANTITY))
    )
    df_clean = df_clean[mask_valid_quantity]
    after = len(df_clean)
    lost = before - after
    if verbose and lost > 0:
        step_results.append(f"  Incohérences quantités: -{lost:,} lignes ({lost/initial_count*100:.2f}%)")

    # 2.4 Supprimer les doublons exacts (erreurs de saisie/import)
    before = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    after = len(df_clean)
    lost = before - after
    if verbose and lost > 0:
        step_results.append(f"  Doublons exacts: -{lost:,} lignes ({lost/initial_count*100:.2f}%)")

    # 2.5 MODE STRICT : Supprimer les lignes sans Customer ID
    if strict_mode:
        before = len(df_clean)
        df_clean = df_clean[df_clean['HasCustomerID']]
        after = len(df_clean)
        lost = before - after
        if verbose and lost > 0:
            step_results.append(f"  Sans Customer ID (mode strict): -{lost:,} lignes ({lost/initial_count*100:.2f}%)")

    if verbose and step_results:
        print(f"\nSuppression des données invalides:")
        for result in step_results:
            print(result)

    # ==========================================================================
    # ETAPE 3 : ENRICHISSEMENT DES DONNEES
    # ==========================================================================

    # 3.1 Calculer le montant total de chaque ligne
    df_clean['TotalAmount'] = df_clean['Quantity'] * df_clean['Price']

    # 3.2 Créer les colonnes temporelles pour analyses
    df_clean['Year'] = df_clean['InvoiceDate'].dt.year
    df_clean['Month'] = df_clean['InvoiceDate'].dt.month
    df_clean['Quarter'] = df_clean['InvoiceDate'].dt.quarter
    df_clean['DayOfWeek'] = df_clean['InvoiceDate'].dt.dayofweek  # 0 = Lundi, 6 = Dimanche
    df_clean['Hour'] = df_clean['InvoiceDate'].dt.hour
    df_clean['Date'] = df_clean['InvoiceDate'].dt.date

    # 3.3 Créer une colonne pour le type de transaction (plus lisible)
    df_clean['TransactionType'] = df_clean['IsReturn'].map({
        True: 'Return',
        False: 'Sale'
    })

    # 3.4 Créer une colonne de catégorisation pour l'analyse
    # Permet de filtrer facilement selon les besoins d'analyse
    def categorize_transaction(row):
        """Catégorise chaque transaction pour faciliter les analyses"""
        if not row['HasCustomerID']:
            return 'NoCustomer_' + row['TransactionType']
        else:
            return 'Customer_' + row['TransactionType']

    df_clean['Category'] = df_clean.apply(categorize_transaction, axis=1)

    # ==========================================================================
    # ETAPE 4 : ORGANISATION FINALE
    # ==========================================================================

    # 4.1 Trier par date pour analyses temporelles
    df_clean = df_clean.sort_values('InvoiceDate').reset_index(drop=True)

    # 4.2 Réorganiser les colonnes pour une meilleure lisibilité
    # Colonnes de base d'abord, puis colonnes calculées, puis flags
    base_cols = ['Invoice', 'StockCode', 'Description', 'Quantity', 'Price',
                 'InvoiceDate', 'Customer ID', 'Country']
    calc_cols = ['TotalAmount', 'Year', 'Month', 'Quarter', 'DayOfWeek', 'Hour', 'Date',
                 'TransactionType', 'Category']
    flag_cols = ['IsReturn', 'HasCustomerID', 'HasValidDate']

    # Réorganiser en vérifiant que les colonnes existent
    cols_order = []
    for col in base_cols + calc_cols + flag_cols:
        if col in df_clean.columns:
            cols_order.append(col)

    # Ajouter les colonnes restantes (si nouvelles colonnes ajoutées)
    for col in df_clean.columns:
        if col not in cols_order:
            cols_order.append(col)

    df_clean = df_clean[cols_order]

    # ==========================================================================
    # ETAPE 5 : RAPPORT FINAL
    # ==========================================================================

    if verbose:
        final_count = len(df_clean)
        retained_pct = (final_count / initial_count) * 100
        lost_total = initial_count - final_count

        print(f"\n{'='*70}")
        print("RESULTAT DU NETTOYAGE")
        print(f"{'='*70}")
        print(f"Lignes initiales:     {initial_count:,}")
        print(f"Lignes conservées:    {final_count:,}")
        print(f"Lignes supprimées:    {lost_total:,}")
        print(f"Taux de rétention:    {retained_pct:.2f}%")

        print(f"\nRÉPARTITION DES DONNÉES CONSERVÉES:")
        print(f"  Ventes avec client:      {((df_clean['Category'] == 'Customer_Sale').sum()):,} ({(df_clean['Category'] == 'Customer_Sale').sum()/final_count*100:.2f}%)")
        print(f"  Retours avec client:     {((df_clean['Category'] == 'Customer_Return').sum()):,} ({(df_clean['Category'] == 'Customer_Return').sum()/final_count*100:.2f}%)")

        if not strict_mode:
            print(f"  Ventes sans client:      {((df_clean['Category'] == 'NoCustomer_Sale').sum()):,} ({(df_clean['Category'] == 'NoCustomer_Sale').sum()/final_count*100:.2f}%)")
            print(f"  Retours sans client:     {((df_clean['Category'] == 'NoCustomer_Return').sum()):,} ({(df_clean['Category'] == 'NoCustomer_Return').sum()/final_count*100:.2f}%)")

        print(f"\nCONSEILS D'UTILISATION:")
        if strict_mode:
            print("  Mode strict activé - données prêtes pour analyses RFM/CLV/Cohortes")
            print("  Toutes les lignes ont un Customer ID valide")
        else:
            print("  Mode optimisé - données complètes conservées")
            print("  Pour analyses clients uniquement: df[df['HasCustomerID']]")
            print("  Pour analyses produits/pays: utiliser toutes les données")
            print("  Pour analyses financières: utiliser df[df['Category'].str.contains('Sale')]")

        print(f"{'='*70}\n")

    return df_clean


def get_data_quality_report(df: pd.DataFrame) -> Dict:
    """
    Génère un rapport détaillé sur la qualité des données.

    Cette fonction analyse un DataFrame et retourne un dictionnaire complet
    contenant des métriques de qualité, des statistiques descriptives et
    des indicateurs de complétude.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame à analyser

    Returns
    -------
    dict
        Dictionnaire contenant :
        - basic_info : informations générales (dimensions, mémoire)
        - missing_values : valeurs manquantes par colonne
        - data_types : types de données
        - unique_counts : nombre de valeurs uniques
        - quality_flags : flags de qualité si disponibles
        - statistics : statistiques descriptives pour colonnes numériques

    Examples
    --------
    >>> df_clean = clean_data(df_raw)
    >>> report = get_data_quality_report(df_clean)
    >>> print(f"Completude globale: {report['completeness']:.2f}%")

    Notes
    -----
    Particulièrement utile après le nettoyage pour valider la qualité
    """
    report = {}

    # Informations basiques
    report['basic_info'] = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 ** 2),
        'columns': list(df.columns)
    }

    # Valeurs manquantes
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    report['missing_values'] = {
        col: {'count': int(missing[col]), 'percentage': float(missing_pct[col])}
        for col in df.columns if missing[col] > 0
    }

    # Complétude globale
    total_cells = len(df) * len(df.columns)
    non_null_cells = df.notna().sum().sum()
    report['completeness'] = (non_null_cells / total_cells * 100) if total_cells > 0 else 0

    # Types de données
    report['data_types'] = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # Nombre de valeurs uniques (pour identifier les colonnes catégorielles)
    report['unique_counts'] = {
        col: int(df[col].nunique())
        for col in df.columns
    }

    # Quality flags si disponibles (colonnes ajoutées par clean_data)
    quality_flags = ['HasCustomerID', 'IsReturn', 'HasValidDate', 'TransactionType', 'Category']
    available_flags = [flag for flag in quality_flags if flag in df.columns]

    if available_flags:
        report['quality_flags'] = {}
        for flag in available_flags:
            if df[flag].dtype == 'bool':
                report['quality_flags'][flag] = {
                    'true_count': int(df[flag].sum()),
                    'false_count': int((~df[flag]).sum()),
                    'true_percentage': float((df[flag].sum() / len(df) * 100))
                }
            else:
                report['quality_flags'][flag] = dict(df[flag].value_counts().to_dict())

    # Statistiques pour colonnes numériques
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 0:
        report['numeric_statistics'] = {}
        for col in numeric_cols:
            if col in df.columns:
                report['numeric_statistics'][col] = {
                    'mean': float(df[col].mean()),
                    'median': float(df[col].median()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'q25': float(df[col].quantile(0.25)),
                    'q75': float(df[col].quantile(0.75))
                }

    # Statistiques temporelles si InvoiceDate existe
    if 'InvoiceDate' in df.columns:
        report['temporal_info'] = {
            'start_date': str(df['InvoiceDate'].min()),
            'end_date': str(df['InvoiceDate'].max()),
            'date_range_days': (df['InvoiceDate'].max() - df['InvoiceDate'].min()).days
        }

    return report


# ==============================================================================
# ANALYSE DES COHORTES
# ==============================================================================

def create_cohorts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée les cohortes d'acquisition basées sur le mois de première transaction.

    Une cohorte regroupe tous les clients qui ont effectué leur première
    transaction durant le même mois. Cette fonction identifie le mois de
    première transaction pour chaque client et calcule l'indice de cohorte
    (nombre de mois écoulés depuis la première transaction).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenant les transactions nettoyées
        Doit contenir : 'Customer ID', 'InvoiceDate'

    Returns
    -------
    pd.DataFrame
        DataFrame enrichi avec les colonnes :
        - CohortMonth : mois de première transaction (YYYY-MM)
        - CohortIndex : nombre de mois depuis la première transaction

    Examples
    --------
    >>> df_cohorts = create_cohorts(df_clean)
    >>> print(df_cohorts[['Customer ID', 'CohortMonth', 'CohortIndex']].head())

    Notes
    -----
    Format des cohortes : YYYY-MM (année-mois)
    CohortIndex = 0 pour le mois d'acquisition
    """
    # Copier pour éviter de modifier l'original
    df_cohorts = df.copy()

    # Créer une colonne pour le mois de facture (InvoiceMonth)
    df_cohorts['InvoiceMonth'] = df_cohorts['InvoiceDate'].dt.to_period('M')

    # Identifier le mois de première transaction pour chaque client (cohorte)
    cohort_data = df_cohorts.groupby('Customer ID')['InvoiceMonth'].min().reset_index()
    cohort_data.columns = ['Customer ID', 'CohortMonth']

    # Fusionner les informations de cohorte avec le dataframe principal
    df_cohorts = df_cohorts.merge(cohort_data, on='Customer ID', how='left')

    # Calculer l'indice de cohorte (nombre de mois depuis la première transaction)
    # La différence entre deux Period donne le nombre de mois
    df_cohorts['CohortIndex'] = (df_cohorts['InvoiceMonth'] - df_cohorts['CohortMonth']).apply(lambda x: x.n)

    return df_cohorts


def calculate_retention(cohort_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les taux de rétention pour chaque cohorte.

    Cette fonction crée une matrice de rétention montrant le pourcentage de
    clients de chaque cohorte qui sont revenus effectuer un achat durant
    les mois suivants leur acquisition.

    Parameters
    ----------
    cohort_df : pd.DataFrame
        DataFrame avec les colonnes de cohorte (CohortMonth, CohortIndex)

    Returns
    -------
    pd.DataFrame
        Matrice de rétention avec :
        - Index : CohortMonth (mois de la cohorte)
        - Colonnes : CohortIndex (0, 1, 2, 3, ..., N mois)
        - Valeurs : taux de rétention (0-100%)

    Examples
    --------
    >>> retention_matrix = calculate_retention(df_cohorts)
    >>> print(retention_matrix)
                  0     1     2     3     4
    2009-12   100.0  35.2  28.4  22.1  18.5
    2010-01   100.0  38.7  31.2  25.6  20.3

    Notes
    -----
    La période 0 (acquisition) affiche toujours 100%
    Les taux sont exprimés en pourcentage du nombre initial de clients
    """
    # Compter le nombre de clients uniques par cohorte et par index
    cohort_counts = cohort_df.groupby(['CohortMonth', 'CohortIndex'])['Customer ID'].nunique().reset_index()
    cohort_counts.columns = ['CohortMonth', 'CohortIndex', 'CustomerCount']

    # Créer une table pivot : lignes = CohortMonth, colonnes = CohortIndex
    cohort_pivot = cohort_counts.pivot_table(
        index='CohortMonth',
        columns='CohortIndex',
        values='CustomerCount'
    )

    # Calculer les taux de rétention (pourcentage par rapport au M0)
    # Le M0 (index 0) représente la taille de la cohorte à l'acquisition
    cohort_size = cohort_pivot.iloc[:, 0]  # Colonne 0 = taille initiale

    # Diviser chaque colonne par la taille de la cohorte et multiplier par 100
    retention_matrix = cohort_pivot.divide(cohort_size, axis=0) * 100

    return retention_matrix


# ==============================================================================
# SEGMENTATION RFM (Recency, Frequency, Monetary)
# ==============================================================================

def calculate_rfm(df: pd.DataFrame, as_of_date: Optional[datetime] = None) -> pd.DataFrame:
    """
    Calcule les scores RFM (Recency, Frequency, Monetary) pour chaque client.

    La méthodologie RFM segmente les clients selon :
    - Recency (R) : nombre de jours depuis le dernier achat
    - Frequency (F) : nombre total de transactions
    - Monetary (M) : montant total dépensé

    Chaque dimension est divisée en quartiles (1-4, 4 étant le meilleur).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame des transactions nettoyées
    as_of_date : datetime, optional
        Date de référence pour le calcul de la récence
        Si None, utilise la date maximale dans les données

    Returns
    -------
    pd.DataFrame
        DataFrame avec une ligne par client contenant :
        - Customer ID
        - Recency : jours depuis dernier achat
        - Frequency : nombre de transactions
        - Monetary : montant total dépensé
        - R_Score : score de récence (1-4)
        - F_Score : score de fréquence (1-4)
        - M_Score : score monétaire (1-4)
        - RFM_Score : score combiné (ex: "444" = meilleur client)
        - RFM_Segment : segment marketing (ex: "Champions")

    Examples
    --------
    >>> rfm_df = calculate_rfm(df_clean)
    >>> print(rfm_df.head())
    >>> print(rfm_df['RFM_Segment'].value_counts())

    Notes
    -----
    Les segments RFM sont définis dans config.RFM_SEGMENTS
    Le score 4 représente le meilleur quartile pour chaque dimension
    """
    # Définir la date de référence
    if as_of_date is None:
        as_of_date = df['InvoiceDate'].max()

    # Filtrer les ventes (exclure les retours)
    df_sales = df[~df['IsReturn']].copy()

    # Calculer les métriques RFM par client
    rfm = df_sales.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (as_of_date - x.max()).days,  # Recency
        'Invoice': 'nunique',  # Frequency (nombre de factures uniques)
        'TotalAmount': 'sum'  # Monetary
    }).reset_index()

    rfm.columns = ['Customer ID', 'Recency', 'Frequency', 'Monetary']

    # Calculer les scores RFM en quartiles (1-4)
    # Utiliser pd.qcut avec gestion des duplicatas
    def assign_score(series, ascending=True):
        """Assigner des scores 1-4 en gérant les cas limites"""
        try:
            if ascending:
                # Score direct : valeurs élevées = score élevé
                scores = pd.qcut(series, q=4, labels=[1, 2, 3, 4], duplicates='drop')
            else:
                # Score inversé : valeurs faibles = score élevé
                scores = pd.qcut(series, q=4, labels=[4, 3, 2, 1], duplicates='drop')
        except ValueError:
            # Si qcut échoue (trop de duplicatas), utiliser une méthode alternative
            if ascending:
                scores = pd.cut(series, bins=4, labels=[1, 2, 3, 4], include_lowest=True, duplicates='drop')
            else:
                scores = pd.cut(series, bins=4, labels=[4, 3, 2, 1], include_lowest=True, duplicates='drop')

        return scores.astype(int) if scores.notna().all() else scores.fillna(2).astype(int)

    # Pour Recency : score inversé (faible récence = bon score)
    rfm['R_Score'] = assign_score(rfm['Recency'], ascending=False)
    # Pour Frequency et Monetary : score direct (élevé = bon score)
    rfm['F_Score'] = assign_score(rfm['Frequency'], ascending=True)
    rfm['M_Score'] = assign_score(rfm['Monetary'], ascending=True)

    # Créer le score RFM combiné
    rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

    # Fonction pour attribuer les segments
    def assign_segment(row):
        r, f, m = row['R_Score'], row['F_Score'], row['M_Score']

        # Champions : R=4, F=4, M=4
        if r == 4 and f == 4 and m == 4:
            return "Champions"
        # Cannot Lose Them : R=1, F=4, M=4
        elif r == 1 and f == 4 and m == 4:
            return "Cannot Lose Them"
        # Loyal Customers : R≥3, F≥3, M≥3
        elif r >= 3 and f >= 3 and m >= 3:
            return "Loyal Customers"
        # At Risk : R≤2, F≥3, M≥3
        elif r <= 2 and f >= 3 and m >= 3:
            return "At Risk"
        # New Customers : R=4, F=1
        elif r == 4 and f == 1:
            return "New Customers"
        # Potential Loyalists : R≥3, F≤2, M≥2
        elif r >= 3 and f <= 2 and m >= 2:
            return "Potential Loyalists"
        # Hibernating : R≤2, F≤2, M≤2
        elif r <= 2 and f <= 2 and m <= 2:
            return "Hibernating"
        # Lost : R=1, F=1, M=1
        elif r == 1 and f == 1 and m == 1:
            return "Lost"
        else:
            return "Others"

    rfm['RFM_Segment'] = rfm.apply(assign_segment, axis=1)
    # Ajouter aussi 'Segment' pour compatibilité
    rfm['Segment'] = rfm['RFM_Segment']

    return rfm


# ==============================================================================
# CALCUL DE LA CUSTOMER LIFETIME VALUE (CLV)
# ==============================================================================

def calculate_clv_empirical(df: pd.DataFrame, period_months: int = 12) -> pd.DataFrame:
    """
    Calcule la Customer Lifetime Value empirique basée sur les données historiques.

    Cette méthode calcule la CLV en se basant sur l'analyse des cohortes :
    - Valeur moyenne par transaction
    - Nombre moyen de transactions par période
    - Taux de rétention observé par cohorte

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame avec les données de cohortes
    period_months : int, default=12
        Période d'observation en mois

    Returns
    -------
    pd.DataFrame
        DataFrame avec la CLV par cohorte/segment contenant :
        - CohortMonth ou Segment
        - Average Order Value (AOV)
        - Purchase Frequency
        - Customer Value
        - Retention Rate
        - CLV

    Examples
    --------
    >>> clv_empirical = calculate_clv_empirical(df_cohorts, period_months=12)
    >>> print(f"CLV moyenne : {clv_empirical['CLV'].mean():.2f}")

    Notes
    -----
    Formule simplifiée : CLV = AOV * Frequency * Retention Rate
    Cette approche utilise les données historiques réelles
    """
    # Filtrer les ventes (hors retours)
    df_sales = df[~df['IsReturn']].copy()

    # Calculer les métriques par client sur la période
    clv_data = df_sales.groupby('Customer ID').agg({
        'TotalAmount': 'sum',  # CA total
        'Invoice': 'nunique',  # Nombre de transactions
        'InvoiceDate': ['min', 'max']  # Dates première et dernière transaction
    }).reset_index()

    # Aplatir les colonnes multi-index
    clv_data.columns = ['Customer ID', 'Total_Revenue', 'Num_Transactions', 'First_Purchase', 'Last_Purchase']

    # Calculer le nombre de jours depuis la dernière transaction
    max_date = df['InvoiceDate'].max()
    clv_data['Last_Purchase_Days'] = (max_date - clv_data['Last_Purchase']).dt.days

    # Calculer l'AOV (Average Order Value)
    clv_data['AOV'] = clv_data['Total_Revenue'] / clv_data['Num_Transactions']

    # Calculer la durée de vie en mois (du premier au dernier achat)
    clv_data['Lifespan_Days'] = (clv_data['Last_Purchase'] - clv_data['First_Purchase']).dt.days
    clv_data['Lifespan_Months'] = clv_data['Lifespan_Days'] / 30.44  # Moyenne jours/mois

    # Éviter division par zéro : si lifespan = 0, utiliser 1 mois
    clv_data['Lifespan_Months'] = clv_data['Lifespan_Months'].replace(0, 1)

    # Calculer la fréquence mensuelle (transactions par mois)
    clv_data['Monthly_Frequency'] = clv_data['Num_Transactions'] / clv_data['Lifespan_Months']

    # CLV empirique = CA observé sur la période
    clv_data['CLV_Empirical'] = clv_data['Total_Revenue']

    # Sélectionner les colonnes pertinentes
    result = clv_data[[
        'Customer ID',
        'CLV_Empirical',
        'Num_Transactions',
        'AOV',
        'Last_Purchase_Days'
    ]].rename(columns={
        'Num_Transactions': 'nb_transactions',
        'AOV': 'avg_basket',
        'Last_Purchase_Days': 'last_purchase_days'
    })

    return result


def calculate_clv_formula(
    df: pd.DataFrame,
    retention_rate: Optional[float] = None,
    discount_rate: Optional[float] = None,
    forecast_periods: int = 36
) -> pd.DataFrame:
    """
    Calcule la Customer Lifetime Value prédictive avec la formule classique.

    Cette méthode utilise la formule théorique de CLV :
    CLV = (AOV * Purchase Frequency * Gross Margin) / (1 + Discount Rate - Retention Rate)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenant les données clients
    retention_rate : float, optional
        Taux de rétention (0-1). Si None, utilise config.DEFAULT_RETENTION_RATE
    discount_rate : float, optional
        Taux d'actualisation (0-1). Si None, utilise config.DEFAULT_DISCOUNT_RATE
    forecast_periods : int, default=36
        Nombre de périodes (mois) pour la prévision

    Returns
    -------
    pd.DataFrame
        DataFrame avec la CLV calculée par client

    Examples
    --------
    >>> clv_formula = calculate_clv_formula(
    ...     df_clean,
    ...     retention_rate=0.35,
    ...     discount_rate=0.10
    ... )
    >>> print(f"CLV totale prévue : {clv_formula['CLV'].sum():,.2f}")

    Notes
    -----
    Cette approche est prédictive et basée sur des hypothèses
    Les taux peuvent être ajustés selon le contexte business
    """
    # Utiliser les valeurs par défaut si non spécifiées
    if retention_rate is None:
        retention_rate = config.DEFAULT_RETENTION_RATE
    if discount_rate is None:
        discount_rate = config.DEFAULT_DISCOUNT_RATE

    # Filtrer les ventes (hors retours)
    df_sales = df[~df['IsReturn']].copy()

    # Calculer les métriques par client
    clv_data = df_sales.groupby('Customer ID').agg({
        'TotalAmount': 'sum',
        'Invoice': 'nunique',
        'InvoiceDate': ['min', 'max']
    }).reset_index()

    clv_data.columns = ['Customer ID', 'Total_Revenue', 'Num_Transactions', 'First_Purchase', 'Last_Purchase']

    # Calculer la durée de vie en mois
    clv_data['Lifespan_Days'] = (clv_data['Last_Purchase'] - clv_data['First_Purchase']).dt.days
    clv_data['Lifespan_Months'] = clv_data['Lifespan_Days'] / 30.44
    clv_data['Lifespan_Months'] = clv_data['Lifespan_Months'].replace(0, 1)

    # Calculer l'AOV et la fréquence mensuelle
    clv_data['AOV'] = clv_data['Total_Revenue'] / clv_data['Num_Transactions']
    clv_data['Monthly_Frequency'] = clv_data['Num_Transactions'] / clv_data['Lifespan_Months']
    clv_data['Monthly_Revenue'] = clv_data['AOV'] * clv_data['Monthly_Frequency']

    # Formule CLV : Marge × (r / (1 + d - r))
    # Où Marge = revenu mensuel moyen, r = taux de rétention, d = taux d'actualisation
    clv_data['CLV_Formula'] = clv_data['Monthly_Revenue'] * (
        retention_rate / (1 + discount_rate - retention_rate)
    )

    # Sélectionner les colonnes pertinentes
    result = clv_data[[
        'Customer ID',
        'CLV_Formula',
        'Monthly_Revenue'
    ]].rename(columns={
        'Monthly_Revenue': 'monthly_avg_revenue'
    })

    return result


# ==============================================================================
# FILTRAGE ET TRANSFORMATION
# ==============================================================================

def apply_filters(df: pd.DataFrame, filters_dict: Dict) -> pd.DataFrame:
    """
    Applique un ensemble de filtres au DataFrame.

    Cette fonction centralise la logique de filtrage pour assurer la cohérence
    à travers l'application. Les filtres peuvent porter sur :
    - Plages de dates
    - Pays
    - Montants de transaction
    - Segments de clients
    - Produits

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame à filtrer
    filters_dict : dict
        Dictionnaire de filtres avec la structure :
        {
            'date_range': (start_date, end_date),
            'countries': ['UK', 'France'],
            'min_amount': 10.0,
            'segments': ['Champions', 'Loyal Customers'],
            'customer_ids': [12345, 67890]
        }

    Returns
    -------
    pd.DataFrame
        DataFrame filtré

    Examples
    --------
    >>> filters = {
    ...     'date_range': ('2010-01-01', '2010-12-31'),
    ...     'countries': ['United Kingdom'],
    ...     'min_amount': 50.0
    ... }
    >>> df_filtered = apply_filters(df_clean, filters)

    Notes
    -----
    Les filtres non spécifiés (None ou absents) sont ignorés
    Les filtres sont appliqués de manière cumulative (AND)
    """
    # Copier pour éviter de modifier l'original
    df_filtered = df.copy()

    # Filtre : plage de dates
    if 'start_date' in filters_dict and filters_dict['start_date'] is not None:
        start_date = pd.to_datetime(filters_dict['start_date'])
        df_filtered = df_filtered[df_filtered['InvoiceDate'] >= start_date]

    if 'end_date' in filters_dict and filters_dict['end_date'] is not None:
        end_date = pd.to_datetime(filters_dict['end_date'])
        df_filtered = df_filtered[df_filtered['InvoiceDate'] <= end_date]

    # Ancien format de date_range pour compatibilité
    if 'date_range' in filters_dict and filters_dict['date_range'] is not None:
        start_date, end_date = filters_dict['date_range']
        if start_date:
            start_date = pd.to_datetime(start_date)
            df_filtered = df_filtered[df_filtered['InvoiceDate'] >= start_date]
        if end_date:
            end_date = pd.to_datetime(end_date)
            df_filtered = df_filtered[df_filtered['InvoiceDate'] <= end_date]

    # Filtre : pays
    if 'countries' in filters_dict and filters_dict['countries'] is not None and len(filters_dict['countries']) > 0:
        df_filtered = df_filtered[df_filtered['Country'].isin(filters_dict['countries'])]

    # Filtre : type de client (B2B ou B2C) - basé sur le volume par exemple
    if 'customer_type' in filters_dict and filters_dict['customer_type'] is not None:
        # Stratégie simple : B2B = transactions > 100 unités en moyenne
        if filters_dict['customer_type'] == 'B2B':
            df_filtered = df_filtered[df_filtered['Quantity'] > 100]
        elif filters_dict['customer_type'] == 'B2C':
            df_filtered = df_filtered[df_filtered['Quantity'] <= 100]

    # Filtre : montant minimum de commande
    if 'min_order_value' in filters_dict and filters_dict['min_order_value'] is not None:
        df_filtered = df_filtered[df_filtered['TotalAmount'] >= filters_dict['min_order_value']]

    # Ancien format min_amount pour compatibilité
    if 'min_amount' in filters_dict and filters_dict['min_amount'] is not None:
        df_filtered = df_filtered[df_filtered['TotalAmount'] >= filters_dict['min_amount']]

    # Filtre : exclure les retours
    if 'exclude_returns' in filters_dict and filters_dict['exclude_returns'] is True:
        df_filtered = df_filtered[~df_filtered['IsReturn']]

    # Filtre : IDs clients spécifiques
    if 'customer_ids' in filters_dict and filters_dict['customer_ids'] is not None and len(filters_dict['customer_ids']) > 0:
        df_filtered = df_filtered[df_filtered['Customer ID'].isin(filters_dict['customer_ids'])]

    # Filtre : segments RFM (si disponible dans le df)
    if 'segments' in filters_dict and filters_dict['segments'] is not None and len(filters_dict['segments']) > 0:
        if 'RFM_Segment' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['RFM_Segment'].isin(filters_dict['segments'])]

    return df_filtered


# ==============================================================================
# SIMULATION DE SCENARIOS
# ==============================================================================

def simulate_scenario(df: pd.DataFrame, params: Dict) -> Dict:
    """
    Simule l'impact de différents scénarios marketing sur les KPIs.

    Cette fonction permet de modéliser l'effet de changements dans :
    - Taux de rétention
    - Valeur moyenne des commandes (AOV)
    - Fréquence d'achat
    - Taille de la base client

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de base pour la simulation
    params : dict
        Paramètres du scénario :
        {
            'retention_increase': 0.10,  # +10%
            'aov_increase': 0.15,        # +15%
            'frequency_increase': 0.20,  # +20%
            'customer_growth': 0.05      # +5%
        }

    Returns
    -------
    dict
        Résultats de la simulation :
        {
            'current': {
                'revenue': float,
                'customers': int,
                'clv': float,
                ...
            },
            'projected': {
                'revenue': float,
                'customers': int,
                'clv': float,
                ...
            },
            'delta': {
                'revenue': float,
                'customers': int,
                'clv': float,
                ...
            }
        }

    Examples
    --------
    >>> scenario = {
    ...     'retention_increase': 0.10,
    ...     'aov_increase': 0.05
    ... }
    >>> results = simulate_scenario(df_clean, scenario)
    >>> print(f"Augmentation revenue : {results['delta']['revenue']:,.2f}")

    Notes
    -----
    Les scénarios prédéfinis sont dans config.SIMULATION_SCENARIOS
    Les résultats sont des projections basées sur des hypothèses
    """
    # Extraire les paramètres avec valeurs par défaut
    margin_pct = params.get('margin_pct', 0.4)
    retention_delta = params.get('retention_delta', params.get('retention_increase', 0.0))
    discount_pct = params.get('discount_pct', 0.0)
    target_segment = params.get('target_segment', None)
    aov_increase = params.get('aov_increase', 0.0)
    frequency_increase = params.get('frequency_increase', 0.0)
    customer_growth = params.get('customer_growth', 0.0)

    # Filtrer les ventes (hors retours)
    df_sales = df[~df['IsReturn']].copy()

    # Si un segment est ciblé, filtrer
    if target_segment and 'RFM_Segment' in df_sales.columns:
        df_target = df_sales[df_sales['RFM_Segment'] == target_segment]
    else:
        df_target = df_sales

    # Calculer les KPIs actuels
    current_revenue = df_target['TotalAmount'].sum()
    current_customers = df_target['Customer ID'].nunique()
    current_transactions = df_target['Invoice'].nunique()
    current_aov = current_revenue / current_transactions if current_transactions > 0 else 0
    current_frequency = current_transactions / current_customers if current_customers > 0 else 0

    # Calculer la CLV actuelle
    current_retention_rate = config.DEFAULT_RETENTION_RATE
    current_discount_rate = config.DEFAULT_DISCOUNT_RATE
    current_monthly_revenue = current_revenue / 12  # Approximation
    current_clv = current_monthly_revenue * (current_retention_rate / (1 + current_discount_rate - current_retention_rate))

    # Projections avec les changements
    projected_customers = current_customers * (1 + customer_growth)
    projected_aov = current_aov * (1 + aov_increase) * (1 - discount_pct)
    projected_frequency = current_frequency * (1 + frequency_increase)
    projected_transactions = projected_customers * projected_frequency
    projected_revenue = projected_aov * projected_transactions

    # CLV projetée
    projected_retention_rate = min(current_retention_rate + retention_delta, 0.95)  # Cap à 95%
    projected_monthly_revenue = projected_revenue / 12
    projected_clv = projected_monthly_revenue * (projected_retention_rate / (1 + current_discount_rate - projected_retention_rate))

    # Calculer les deltas
    delta_revenue = projected_revenue - current_revenue
    delta_customers = projected_customers - current_customers
    delta_clv = projected_clv - current_clv
    delta_revenue_pct = (delta_revenue / current_revenue * 100) if current_revenue > 0 else 0

    # Construire le résultat
    results = {
        'current': {
            'revenue': current_revenue,
            'customers': int(current_customers),
            'transactions': int(current_transactions),
            'aov': current_aov,
            'frequency': current_frequency,
            'retention_rate': current_retention_rate,
            'clv': current_clv
        },
        'projected': {
            'revenue': projected_revenue,
            'customers': int(projected_customers),
            'transactions': int(projected_transactions),
            'aov': projected_aov,
            'frequency': projected_frequency,
            'retention_rate': projected_retention_rate,
            'clv': projected_clv
        },
        'delta': {
            'revenue': delta_revenue,
            'revenue_pct': delta_revenue_pct,
            'customers': int(delta_customers),
            'clv': delta_clv
        },
        'params': params
    }

    return results


# ==============================================================================
# EXPORT DE DONNEES
# ==============================================================================

def export_to_csv(df: pd.DataFrame, filename: str, directory: Optional[Path] = None) -> str:
    """
    Exporte un DataFrame au format CSV.

    Cette fonction gère l'export de données avec les bonnes configurations :
    - Encodage UTF-8
    - Séparateur selon config
    - Formats de dates standardisés
    - Création automatique des répertoires

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame à exporter
    filename : str
        Nom du fichier (avec ou sans extension .csv)
    directory : Path, optional
        Répertoire de destination. Si None, utilise config.EXPORT_DATA_DIR

    Returns
    -------
    str
        Chemin complet du fichier créé

    Examples
    --------
    >>> file_path = export_to_csv(rfm_df, 'rfm_analysis_2024.csv')
    >>> print(f"Fichier exporté : {file_path}")

    Notes
    -----
    Le répertoire de destination est créé s'il n'existe pas
    Les fichiers existants sont écrasés sans avertissement
    """
    # Utiliser le répertoire par défaut si non spécifié
    if directory is None:
        directory = config.PROCESSED_DATA_DIR

    # Convertir en Path si nécessaire
    directory = Path(directory)

    # Créer le répertoire s'il n'existe pas
    directory.mkdir(parents=True, exist_ok=True)

    # Ajouter l'extension .csv si absente
    if not filename.endswith('.csv'):
        filename += '.csv'

    # Construire le chemin complet
    file_path = directory / filename

    try:
        # Exporter le DataFrame
        df.to_csv(
            file_path,
            index=False,
            encoding=config.FILE_ENCODING,
            sep=config.CSV_SEPARATOR,
            date_format=config.EXPORT_DATETIME_FORMAT
        )

        return str(file_path.absolute())

    except Exception as e:
        raise IOError(f"Erreur lors de l'export vers {file_path}: {str(e)}")


def export_chart_to_png(
    fig: Union[plt.Figure, go.Figure],
    filename: str,
    directory: Optional[Path] = None,
    dpi: int = 300
) -> str:
    """
    Exporte un graphique (Matplotlib ou Plotly) au format PNG.

    Cette fonction unifie l'export de visualisations, qu'elles soient créées
    avec Matplotlib ou Plotly, en gérant automatiquement le format approprié.

    Parameters
    ----------
    fig : matplotlib.figure.Figure or plotly.graph_objects.Figure
        Figure à exporter
    filename : str
        Nom du fichier (avec ou sans extension .png)
    directory : Path, optional
        Répertoire de destination. Si None, utilise config.EXPORT_CHARTS_DIR
    dpi : int, default=300
        Résolution pour les exports Matplotlib (DPI)

    Returns
    -------
    str
        Chemin complet du fichier créé

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [1, 4, 9])
    >>> file_path = export_chart_to_png(fig, 'my_chart.png')

    Notes
    -----
    Format Plotly : utilise write_image()
    Format Matplotlib : utilise savefig() avec bbox_inches='tight'
    """
    # Utiliser le répertoire par défaut si non spécifié
    if directory is None:
        directory = config.EXPORT_CHARTS_DIR

    # Convertir en Path si nécessaire
    directory = Path(directory)

    # Créer le répertoire s'il n'existe pas
    directory.mkdir(parents=True, exist_ok=True)

    # Ajouter l'extension .png si absente
    if not filename.endswith('.png'):
        filename += '.png'

    # Construire le chemin complet
    file_path = directory / filename

    try:
        # Détecter le type de figure et exporter
        if isinstance(fig, plt.Figure):
            # Export Matplotlib
            fig.savefig(
                file_path,
                dpi=dpi,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none'
            )
        elif isinstance(fig, go.Figure):
            # Export Plotly
            fig.write_image(
                file_path,
                width=1200,
                height=600,
                scale=2  # Haute résolution
            )
        else:
            raise TypeError(f"Type de figure non supporté : {type(fig)}. Attendu : matplotlib.figure.Figure ou plotly.graph_objects.Figure")

        return str(file_path.absolute())

    except Exception as e:
        raise IOError(f"Erreur lors de l'export du graphique vers {file_path}: {str(e)}")


# ==============================================================================
# CALCULS DE METRIQUES
# ==============================================================================

def calculate_kpis(df: pd.DataFrame) -> Dict[str, Union[float, int]]:
    """
    Calcule les principaux KPIs métier.

    Cette fonction centralise le calcul de tous les indicateurs clés :
    - Nombre total de clients
    - Revenu total
    - Panier moyen (Average Order Value)
    - Fréquence d'achat moyenne
    - Taux de rétention global
    - CLV moyenne
    - Nombre de transactions

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame des transactions

    Returns
    -------
    dict
        Dictionnaire des KPIs :
        {
            'total_customers': int,
            'total_revenue': float,
            'avg_order_value': float,
            'purchase_frequency': float,
            'retention_rate': float,
            'avg_clv': float,
            'total_transactions': int
        }

    Examples
    --------
    >>> kpis = calculate_kpis(df_clean)
    >>> for key, value in kpis.items():
    ...     print(f"{key}: {value:,.2f}")
    """
    # Filtrer les ventes (hors retours)
    df_sales = df[~df['IsReturn']].copy()

    # KPIs de base
    total_customers = df_sales['Customer ID'].nunique()
    total_revenue = df_sales['TotalAmount'].sum()
    total_transactions = df_sales['Invoice'].nunique()

    # Panier moyen
    avg_order_value = total_revenue / total_transactions if total_transactions > 0 else 0

    # Fréquence d'achat moyenne (transactions par client)
    purchase_frequency = total_transactions / total_customers if total_customers > 0 else 0

    # Calculer le taux de rétention (clients qui reviennent au moins une fois)
    customer_transaction_counts = df_sales.groupby('Customer ID')['Invoice'].nunique()
    repeat_customers = (customer_transaction_counts > 1).sum()
    retention_rate = repeat_customers / total_customers if total_customers > 0 else 0

    # Taux de rétention à M+1 et M+3
    if 'CohortIndex' in df_sales.columns:
        # Pour M+1 : clients qui ont fait une transaction au moins 1 mois après leur première
        customers_m1 = df_sales[df_sales['CohortIndex'] >= 1]['Customer ID'].nunique()
        retention_rate_m1 = customers_m1 / total_customers if total_customers > 0 else 0

        # Pour M+3
        customers_m3 = df_sales[df_sales['CohortIndex'] >= 3]['Customer ID'].nunique()
        retention_rate_m3 = customers_m3 / total_customers if total_customers > 0 else 0
    else:
        retention_rate_m1 = retention_rate
        retention_rate_m3 = retention_rate * 0.7  # Approximation

    # Taux de retour
    total_returns = df[df['IsReturn']]['Invoice'].nunique()
    return_rate = total_returns / (total_transactions + total_returns) if (total_transactions + total_returns) > 0 else 0

    # Clients actifs (derniers 90 jours)
    max_date = df['InvoiceDate'].max()
    cutoff_date = max_date - timedelta(days=90)
    active_customers = df_sales[df_sales['InvoiceDate'] >= cutoff_date]['Customer ID'].nunique()

    # CLV moyenne (basée sur la formule simple)
    avg_clv = (avg_order_value * purchase_frequency) / (1 - retention_rate + 0.1) if retention_rate < 0.9 else avg_order_value * purchase_frequency * 10

    # Construire le dictionnaire de KPIs
    kpis = {
        'total_customers': int(total_customers),
        'total_revenue': float(total_revenue),
        'total_transactions': int(total_transactions),
        'avg_order_value': float(avg_order_value),
        'purchase_frequency': float(purchase_frequency),
        'retention_rate': float(retention_rate),
        'retention_rate_m1': float(retention_rate_m1),
        'retention_rate_m3': float(retention_rate_m3),
        'return_rate': float(return_rate),
        'active_customers': int(active_customers),
        'avg_clv': float(avg_clv)
    }

    return kpis


def calculate_churn_rate(df: pd.DataFrame, inactive_months: int = 6) -> float:
    """
    Calcule le taux de churn (attrition) des clients.

    Un client est considéré comme churné s'il n'a pas effectué de transaction
    depuis un nombre de mois défini.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame des transactions
    inactive_months : int, default=6
        Nombre de mois d'inactivité pour considérer un client churné

    Returns
    -------
    float
        Taux de churn (0-1)

    Examples
    --------
    >>> churn = calculate_churn_rate(df_clean, inactive_months=6)
    >>> print(f"Taux de churn : {churn:.1%}")
    """
    # Filtrer les ventes (hors retours)
    df_sales = df[~df['IsReturn']].copy()

    # Date de référence (date maximale dans les données)
    max_date = df_sales['InvoiceDate'].max()

    # Date de cutoff pour le churn
    cutoff_date = max_date - timedelta(days=inactive_months * 30)

    # Identifier la dernière transaction de chaque client
    last_purchase = df_sales.groupby('Customer ID')['InvoiceDate'].max()

    # Clients churnés : dernière transaction avant la date de cutoff
    churned_customers = (last_purchase < cutoff_date).sum()

    # Total de clients
    total_customers = df_sales['Customer ID'].nunique()

    # Taux de churn
    churn_rate = churned_customers / total_customers if total_customers > 0 else 0

    return float(churn_rate)


def get_churn_predictions(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifie les clients à risque de churn basé sur leur profil RFM.

    Cette fonction analyse les scores RFM pour identifier les clients
    susceptibles de churner et leur attribue une probabilité et un niveau de risque.

    Parameters
    ----------
    rfm_df : pd.DataFrame
        DataFrame RFM avec scores et segments

    Returns
    -------
    pd.DataFrame
        DataFrame avec colonnes :
        - Customer ID
        - churn_probability : probabilité de churn (0-1)
        - risk_level : niveau de risque ('Low', 'Medium', 'High', 'Critical')

    Examples
    --------
    >>> rfm = calculate_rfm(df_clean)
    >>> churn_pred = get_churn_predictions(rfm)
    >>> high_risk = churn_pred[churn_pred['risk_level'] == 'High']
    >>> print(f"Clients à haut risque : {len(high_risk)}")

    Notes
    -----
    Les règles sont basées sur les scores RFM :
    - R_score faible = risque élevé
    - Segments "At Risk", "Lost", "Cannot Lose Them" = priorité haute
    """
    # Copier pour éviter de modifier l'original
    churn_df = rfm_df.copy()

    # Calculer la probabilité de churn basée sur les scores RFM
    # Formule : plus R est faible, plus la probabilité est élevée
    # Ajuster avec F et M pour affiner
    churn_df['churn_probability'] = (
        (5 - churn_df['R_Score']) * 0.6 +  # Recency pèse 60%
        (5 - churn_df['F_Score']) * 0.2 +  # Frequency pèse 20%
        (5 - churn_df['M_Score']) * 0.2    # Monetary pèse 20%
    ) / 4  # Normaliser entre 0 et 1

    # Définir le niveau de risque
    def assign_risk_level(row):
        prob = row['churn_probability']
        segment = row['RFM_Segment']

        # Segments critiques
        if segment in ['Lost', 'Cannot Lose Them']:
            return 'Critical'
        # Segments à haut risque
        elif segment in ['At Risk', 'Hibernating'] or prob >= 0.7:
            return 'High'
        # Risque moyen
        elif prob >= 0.4:
            return 'Medium'
        # Faible risque
        else:
            return 'Low'

    churn_df['risk_level'] = churn_df.apply(assign_risk_level, axis=1)

    # Sélectionner les colonnes pertinentes
    result = churn_df[['Customer ID', 'churn_probability', 'risk_level']]

    # Trier par probabilité décroissante
    result = result.sort_values('churn_probability', ascending=False).reset_index(drop=True)

    return result


# ==============================================================================
# UTILITAIRES DE VALIDATION
# ==============================================================================

def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> Tuple[bool, List[str]]:
    """
    Valide qu'un DataFrame contient toutes les colonnes requises.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame à valider
    required_columns : list of str
        Liste des colonnes requises

    Returns
    -------
    tuple of (bool, list)
        (validation_ok, missing_columns)

    Examples
    --------
    >>> valid, missing = validate_dataframe(df, config.REQUIRED_COLUMNS)
    >>> if not valid:
    ...     print(f"Colonnes manquantes : {missing}")
    """
    # Identifier les colonnes manquantes
    missing_columns = [col for col in required_columns if col not in df.columns]

    # Validation OK si aucune colonne manquante
    is_valid = len(missing_columns) == 0

    return is_valid, missing_columns


# ==============================================================================
# UTILITAIRES DE FORMATAGE
# ==============================================================================

def format_currency(amount: float, currency: str = "GBP") -> str:
    """
    Formate un montant en devise.

    Parameters
    ----------
    amount : float
        Montant à formater
    currency : str, default="GBP"
        Code de la devise (GBP, EUR, USD, etc.)

    Returns
    -------
    str
        Montant formaté (ex: "£1,234.56")

    Examples
    --------
    >>> print(format_currency(1234.56))
    £1,234.56
    """
    # Dictionnaire des symboles de devises
    currency_symbols = {
        'GBP': '£',
        'EUR': '€',
        'USD': '$',
        'JPY': '¥',
        'CHF': 'CHF '
    }

    # Obtenir le symbole (par défaut, utiliser le code de devise)
    symbol = currency_symbols.get(currency.upper(), currency + ' ')

    # Formater avec séparateurs de milliers
    formatted = f"{symbol}{amount:,.2f}"

    return formatted


def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Formate un nombre en pourcentage.

    Parameters
    ----------
    value : float
        Valeur à formater (0-1)
    decimals : int, default=1
        Nombre de décimales

    Returns
    -------
    str
        Pourcentage formaté (ex: "25.5%")

    Examples
    --------
    >>> print(format_percentage(0.255))
    25.5%
    """
    # Convertir en pourcentage et formater
    percentage = value * 100
    formatted = f"{percentage:.{decimals}f}%"

    return formatted


# ==============================================================================
# MODULE INFO
# ==============================================================================

if __name__ == "__main__":
    print("Module utils.py - Fonctions utilitaires")
    print("=" * 70)
    print("Ce module contient toutes les fonctions métier de l'application")
    print("Pour utiliser ces fonctions, importez-les dans vos scripts ou pages")
    print("=" * 70)
