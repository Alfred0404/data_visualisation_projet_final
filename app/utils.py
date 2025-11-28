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


def load_data(file_path: Union[str, Path], verbose: bool = False) -> pd.DataFrame:
    """
    Charge les données depuis un fichier Excel.

    Parameters
    ----------
    file_path : str or Path
        Chemin vers le fichier Excel (.xlsx ou .xls)
    verbose : bool, default=False
        Si True, affiche des informations sur le chargement

    Returns
    -------
    pd.DataFrame
        DataFrame contenant les données chargées

    Raises
    ------
    FileNotFoundError
        Si le fichier n'existe pas
    ValueError
        Si le format du fichier n'est pas Excel
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Le fichier {file_path} n'existe pas")

    extension = file_path.suffix.lower()

    if extension not in ['.xlsx', '.xls']:
        raise ValueError(f"Format non supporté : {extension}. Utilisez un fichier Excel (.xlsx ou .xls)")

    if verbose:
        print(f"Chargement du fichier: {file_path.name}")

    try:
        df = pd.read_excel(
            file_path,
            dtype={'Customer ID': str, 'Invoice': str, 'StockCode': str},
            engine='openpyxl' if extension == '.xlsx' else None
        )

        if 'InvoiceDate' in df.columns and df['InvoiceDate'].dtype != 'datetime64[ns]':
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')

        if df.empty:
            raise ValueError("Le fichier chargé est vide")

        df.columns = df.columns.str.strip()

        for col in ['Quantity', 'Price']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if verbose:
            print(f"Chargement réussi: {df.shape[0]:,} lignes x {df.shape[1]} colonnes")

        return df

    except Exception as e:
        raise ValueError(f"Erreur lors du chargement: {str(e)}")


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
    return df_clean


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


def calculate_rfm(df: pd.DataFrame, as_of_date: Optional[datetime] = None) -> pd.DataFrame:
    """
    Calcule les scores RFM (Recency, Frequency, Monetary) pour chaque client.

    La méthodologie RFM segmente les clients selon :
    - Recency (R) : nombre de jours depuis le dernier achat
    - Frequency (F) : nombre total de transactions
    - Monetary (M) : montant total dépensé

    Chaque dimension est divisée en quintiles (1-5, 5 étant le meilleur).

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
        - R_Score : score de récence (1-5)
        - F_Score : score de fréquence (1-5)
        - M_Score : score monétaire (1-5)
        - RFM_Score : score combiné (ex: "555" = meilleur client)
        - RFM_Segment : segment marketing (ex: "Champions")

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

    # Calculer les scores RFM en quintiles (1-5)
    # Utiliser pd.qcut avec gestion des duplicatas
    def assign_score(series, ascending=True):
        """Assigner des scores 1-5 en gérant les cas limites"""
        try:
            if ascending:
                # Score direct : valeurs élevées = score élevé
                scores = pd.qcut(series, q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
            else:
                # Score inversé : valeurs faibles = score élevé
                scores = pd.qcut(series, q=5, labels=[5, 4, 3, 2, 1], duplicates='drop')
        except ValueError:
            # Si qcut échoue (trop de duplicatas), utiliser une méthode alternative
            if ascending:
                scores = pd.cut(series, bins=5, labels=[1, 2, 3, 4, 5], include_lowest=True, duplicates='drop')
            else:
                scores = pd.cut(series, bins=5, labels=[5, 4, 3, 2, 1], include_lowest=True, duplicates='drop')

        return scores.astype(int) if scores.notna().all() else scores.fillna(3).astype(int)

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

        # Champions : R=5, F=5, M=5
        if r == 5 and f == 5 and m == 5:
            return "Champions"
        # Cannot Lose Them : R=1, F=5, M=5
        elif r == 1 and f == 5 and m == 5:
            return "Cannot Lose Them"
        # Loyal Customers : R≥4, F≥4, M≥4
        elif r >= 4 and f >= 4 and m >= 4:
            return "Loyal Customers"
        # At Risk : R≤2, F≥4, M≥4
        elif r <= 2 and f >= 4 and m >= 4:
            return "At Risk"
        # New Customers : R=5, F=1
        elif r == 5 and f == 1:
            return "New Customers"
        # Potential Loyalists : R≥4, F≤2, M≥3
        elif r >= 4 and f <= 2 and m >= 3:
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

    # Calculer le taux de rétention réel (clients qui reviennent au moins une fois)
    customer_transaction_counts = df_target.groupby('Customer ID')['Invoice'].nunique()
    repeat_customers = (customer_transaction_counts > 1).sum()
    current_retention_rate = repeat_customers / current_customers if current_customers > 0 else config.DEFAULT_RETENTION_RATE

    # Calculer la CLV actuelle
    current_discount_rate = config.DEFAULT_DISCOUNT_RATE
    current_monthly_revenue = current_revenue / 12  # Approximation
    current_clv = current_monthly_revenue * (current_retention_rate / (1 + current_discount_rate - current_retention_rate))

    # Calculer le taux de rétention projeté d'abord (nécessaire pour les autres calculs)
    projected_retention_rate = min(current_retention_rate + retention_delta, 0.95)  # Cap à 95%

    # Impact de la rétention sur la fréquence d'achat
    # Une meilleure rétention signifie que les clients achètent plus fréquemment
    # Multiplieur basé sur le changement de rétention : si rétention +10%, fréquence +5%
    retention_impact_on_frequency = 1 + (retention_delta * 0.5)

    # Projections avec les changements
    projected_customers = current_customers * (1 + customer_growth)
    projected_aov = current_aov * (1 + aov_increase) * (1 - discount_pct)
    projected_frequency = current_frequency * (1 + frequency_increase) * retention_impact_on_frequency
    projected_transactions = projected_customers * projected_frequency
    projected_revenue = projected_aov * projected_transactions
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

    # NORTH STAR METRIC : Revenu à 90 jours par nouveau client

    # Calculer la valeur moyenne par client sur 90 jours (proxy pour nouveaux clients)
    if 'CohortIndex' in df_sales.columns and 'CohortMonth' in df_sales.columns:
        # Filtrer les transactions des 3 premiers mois (≈90 jours)
        early_customers = df_sales[df_sales['CohortIndex'] <= 2].copy()
        if len(early_customers) > 0:
            # Revenu moyen par client sur leurs 90 premiers jours
            revenue_by_customer = early_customers.groupby('Customer ID')['TotalAmount'].sum()
            north_star_metric = revenue_by_customer.mean()
        else:
            # Fallback si pas de données cohortes
            north_star_metric = total_revenue / total_customers if total_customers > 0 else 0
    else:
        north_star_metric = avg_clv * 0.25 if avg_clv > 0 else (total_revenue / total_customers if total_customers > 0 else 0)

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
        'avg_clv': float(avg_clv),
        'north_star_metric': float(north_star_metric)  # North Star : Revenu à 90j par nouveau client
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
    """
    # Convertir en pourcentage et formater
    percentage = value * 100
    formatted = f"{percentage:.{decimals}f}%"

    return formatted


def apply_global_filters(df: pd.DataFrame, filters: dict = None) -> pd.DataFrame:
    """
    Applique les filtres globaux aux donnees.

    Cette fonction permet d'appliquer de maniere coherente les filtres
    definis dans la sidebar globale a travers toutes les pages de l'application.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a filtrer
    filters : dict, optional
        Dictionnaire contenant les filtres a appliquer.
        Cles attendues:
        - 'date_range': tuple de deux dates (start, end)
        - 'countries': list de pays a inclure
        - 'min_amount': montant minimum de transaction

    Returns
    -------
    pd.DataFrame
        DataFrame filtre selon les criteres
    """
    if filters is None or df.empty:
        return df

    df_filtered = df.copy()

    # Filtre par periode
    if 'date_range' in filters and filters['date_range'] is not None:
        date_range = filters['date_range']
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
            # Convertir en datetime si necessaire
            if 'InvoiceDate' in df_filtered.columns:
                start_dt = pd.Timestamp(start_date)
                end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                df_filtered = df_filtered[
                    (df_filtered['InvoiceDate'] >= start_dt) &
                    (df_filtered['InvoiceDate'] <= end_dt)
                ]

    # Filtre par pays
    if 'countries' in filters and filters['countries'] is not None:
        countries = filters['countries']
        if isinstance(countries, list) and len(countries) > 0 and 'Country' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['Country'].isin(countries)]

    # Filtre par montant minimum
    if 'min_amount' in filters and filters['min_amount'] is not None:
        min_amount = filters['min_amount']
        if min_amount > 0 and 'TotalAmount' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['TotalAmount'] >= min_amount]

    return df_filtered


def prepare_df_for_export(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare un DataFrame pour l'export en convertissant les types problematiques.

    Cette fonction gere:
    - Les colonnes Period (convertit en string)
    - Les colonnes datetime (convertit en format ISO)
    - Les colonnes timedelta (convertit en string)
    - Les types numpy (convertit en types Python natifs)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a preparer pour l'export

    Returns
    -------
    pd.DataFrame
        DataFrame pret pour l'export
    """
    df_export = df.copy()

    for col in df_export.columns:
        # Convertir les colonnes Period en string
        if pd.api.types.is_period_dtype(df_export[col]):
            df_export[col] = df_export[col].astype(str)

        # Convertir les colonnes datetime en string ISO
        elif pd.api.types.is_datetime64_any_dtype(df_export[col]):
            df_export[col] = df_export[col].dt.strftime('%Y-%m-%d %H:%M:%S')

        # Convertir les colonnes timedelta en string
        elif pd.api.types.is_timedelta64_dtype(df_export[col]):
            df_export[col] = df_export[col].astype(str)

    return df_export


def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    """
    Convertit un DataFrame en bytes CSV pour telechargement.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a convertir

    Returns
    -------
    bytes
        Contenu CSV encode en UTF-8
    """
    df_export = prepare_df_for_export(df)
    return df_export.to_csv(index=False, encoding='utf-8').encode('utf-8')


def convert_df_to_excel(df: pd.DataFrame) -> bytes:
    """
    Convertit un DataFrame en bytes Excel pour telechargement.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a convertir

    Returns
    -------
    bytes
        Contenu Excel (XLSX)
    """
    df_export = prepare_df_for_export(df)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_export.to_excel(writer, index=False, sheet_name='Data')

    return output.getvalue()


if __name__ == "__main__":
    print("Module utils.py - Fonctions utilitaires")
    print("=" * 70)
    print("Ce module contient toutes les fonctions métier de l'application")
    print("Pour utiliser ces fonctions, importez-les dans vos scripts ou pages")
    print("=" * 70)
