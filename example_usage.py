"""
Exemple d'Utilisation des Fonctions utils.py

Ce script démontre comment utiliser toutes les fonctions implémentées
dans utils.py pour un pipeline complet d'analyse marketing.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from app import utils
import config
import pandas as pd
from datetime import datetime

def exemple_complet():
    """
    Exemple d'utilisation complète du pipeline d'analyse.

    Note : Ce code suppose que le fichier de données existe.
    Adaptez le chemin selon votre configuration.
    """

    print("=" * 80)
    print("EXEMPLE D'UTILISATION COMPLÈTE - PIPELINE D'ANALYSE MARKETING")
    print("=" * 80)
    print()

    # =========================================================================
    # ÉTAPE 1 : CHARGEMENT DES DONNÉES
    # =========================================================================
    print("📁 ÉTAPE 1 : Chargement des données")
    print("-" * 80)

    # Vérifier si le fichier existe
    if config.RAW_DATA_CSV.exists():
        print(f"Chargement depuis : {config.RAW_DATA_CSV}")
        df_raw = utils.load_data(config.RAW_DATA_CSV)
        print(f"✓ Données chargées : {df_raw.shape[0]:,} lignes × {df_raw.shape[1]} colonnes")
        print(f"✓ Colonnes : {list(df_raw.columns)}")
    else:
        print(f"⚠ Fichier non trouvé : {config.RAW_DATA_CSV}")
        print("Création d'un DataFrame d'exemple pour la démonstration...")

        # Créer un DataFrame d'exemple
        df_raw = pd.DataFrame({
            'Invoice': ['536365', '536365', '536366', '536367', '536368'],
            'StockCode': ['85123A', '71053', '84406B', '84029G', '84029E'],
            'Description': ['ITEM A', 'ITEM B', 'ITEM C', 'ITEM D', 'ITEM E'],
            'Quantity': [6, 6, 8, 6, 6],
            'InvoiceDate': pd.to_datetime([
                '2009-12-01 08:26:00',
                '2009-12-01 08:26:00',
                '2009-12-01 08:28:00',
                '2009-12-01 08:34:00',
                '2009-12-01 08:34:00'
            ]),
            'Price': [2.55, 3.39, 2.75, 3.39, 3.39],
            'Customer ID': ['17850', '17850', '17851', '13047', '13047'],
            'Country': ['United Kingdom', 'United Kingdom', 'United Kingdom', 'United Kingdom', 'United Kingdom']
        })
        print(f"✓ DataFrame d'exemple créé : {df_raw.shape[0]} lignes")

    print()

    # =========================================================================
    # ÉTAPE 2 : NETTOYAGE DES DONNÉES
    # =========================================================================
    print("🧹 ÉTAPE 2 : Nettoyage des données")
    print("-" * 80)

    df_clean = utils.clean_data(df_raw)
    print(f"✓ Données nettoyées : {df_clean.shape[0]:,} lignes")
    print(f"✓ Lignes supprimées : {df_raw.shape[0] - df_clean.shape[0]:,}")
    print(f"✓ Nouvelles colonnes créées :")
    print(f"  - TotalAmount")
    print(f"  - IsReturn")
    print(f"  - Year, Month, Quarter, DayOfWeek, Hour")
    print()

    # =========================================================================
    # ÉTAPE 3 : CRÉATION DES COHORTES
    # =========================================================================
    print("👥 ÉTAPE 3 : Analyse des cohortes")
    print("-" * 80)

    df_cohorts = utils.create_cohorts(df_clean)
    print(f"✓ Cohortes créées")
    print(f"✓ Colonnes ajoutées : CohortMonth, CohortIndex, InvoiceMonth")
    print(f"✓ Nombre de cohortes : {df_cohorts['CohortMonth'].nunique()}")

    # Matrice de rétention
    retention_matrix = utils.calculate_retention(df_cohorts)
    print(f"✓ Matrice de rétention calculée : {retention_matrix.shape}")
    print(f"  - {retention_matrix.shape[0]} cohortes × {retention_matrix.shape[1]} périodes")
    print()

    # =========================================================================
    # ÉTAPE 4 : SEGMENTATION RFM
    # =========================================================================
    print("🎯 ÉTAPE 4 : Segmentation RFM")
    print("-" * 80)

    rfm_df = utils.calculate_rfm(df_clean)
    print(f"✓ Analyse RFM effectuée pour {len(rfm_df):,} clients")
    print(f"✓ Distribution des segments :")
    for segment, count in rfm_df['RFM_Segment'].value_counts().head(5).items():
        print(f"  - {segment}: {count} clients ({count/len(rfm_df)*100:.1f}%)")
    print()

    # =========================================================================
    # ÉTAPE 5 : CALCUL DE LA CLV
    # =========================================================================
    print("💰 ÉTAPE 5 : Calcul de la Customer Lifetime Value")
    print("-" * 80)

    # CLV empirique
    clv_empirical = utils.calculate_clv_empirical(df_clean, period_months=12)
    print(f"✓ CLV empirique calculée pour {len(clv_empirical):,} clients")
    print(f"  - CLV moyenne : {utils.format_currency(clv_empirical['CLV_Empirical'].mean())}")
    print(f"  - CLV médiane : {utils.format_currency(clv_empirical['CLV_Empirical'].median())}")

    # CLV formule
    clv_formula = utils.calculate_clv_formula(df_clean, retention_rate=0.3, discount_rate=0.1)
    print(f"✓ CLV prédictive calculée pour {len(clv_formula):,} clients")
    print(f"  - CLV moyenne : {utils.format_currency(clv_formula['CLV_Formula'].mean())}")
    print(f"  - CLV médiane : {utils.format_currency(clv_formula['CLV_Formula'].median())}")
    print()

    # =========================================================================
    # ÉTAPE 6 : CALCUL DES KPIs
    # =========================================================================
    print("📊 ÉTAPE 6 : Calcul des KPIs principaux")
    print("-" * 80)

    kpis = utils.calculate_kpis(df_clean)
    print(f"✓ KPIs calculés :")
    print(f"  - Clients totaux : {kpis['total_customers']:,}")
    print(f"  - Revenu total : {utils.format_currency(kpis['total_revenue'])}")
    print(f"  - Transactions : {kpis['total_transactions']:,}")
    print(f"  - Panier moyen : {utils.format_currency(kpis['avg_order_value'])}")
    print(f"  - Fréquence d'achat : {kpis['purchase_frequency']:.2f}")
    print(f"  - Taux de rétention : {utils.format_percentage(kpis['retention_rate'])}")
    print(f"  - Clients actifs (90j) : {kpis['active_customers']:,}")

    # Taux de churn
    churn_rate = utils.calculate_churn_rate(df_clean, inactive_months=6)
    print(f"  - Taux de churn (6 mois) : {utils.format_percentage(churn_rate)}")
    print()

    # =========================================================================
    # ÉTAPE 7 : PRÉDICTION DU CHURN
    # =========================================================================
    print("⚠️  ÉTAPE 7 : Identification des clients à risque")
    print("-" * 80)

    churn_predictions = utils.get_churn_predictions(rfm_df)
    print(f"✓ Prédictions de churn effectuées")
    print(f"✓ Distribution par niveau de risque :")
    for risk_level, count in churn_predictions['risk_level'].value_counts().items():
        print(f"  - {risk_level}: {count} clients ({count/len(churn_predictions)*100:.1f}%)")

    # Top 5 clients à risque
    print(f"✓ Top 5 clients à risque :")
    top_risk = churn_predictions.head(5)
    for idx, row in top_risk.iterrows():
        print(f"  - Client {row['Customer ID']}: {utils.format_percentage(row['churn_probability'])} ({row['risk_level']})")
    print()

    # =========================================================================
    # ÉTAPE 8 : FILTRAGE DES DONNÉES
    # =========================================================================
    print("🔍 ÉTAPE 8 : Exemple de filtrage")
    print("-" * 80)

    filters = {
        'exclude_returns': True,
        'min_order_value': 10.0
    }
    df_filtered = utils.apply_filters(df_clean, filters)
    print(f"✓ Données filtrées : {df_filtered.shape[0]:,} lignes")
    print(f"✓ Filtres appliqués :")
    print(f"  - Exclusion des retours")
    print(f"  - Montant minimum : {utils.format_currency(10.0)}")
    print()

    # =========================================================================
    # ÉTAPE 9 : SIMULATION DE SCÉNARIO
    # =========================================================================
    print("🎲 ÉTAPE 9 : Simulation de scénario marketing")
    print("-" * 80)

    scenario_params = {
        'retention_increase': 0.10,  # +10% rétention
        'aov_increase': 0.05,        # +5% panier moyen
        'frequency_increase': 0.08   # +8% fréquence
    }

    simulation_results = utils.simulate_scenario(df_clean, scenario_params)
    print(f"✓ Scénario simulé : +10% rétention, +5% AOV, +8% fréquence")
    print(f"✓ Résultats actuels :")
    print(f"  - Revenu : {utils.format_currency(simulation_results['current']['revenue'])}")
    print(f"  - Clients : {simulation_results['current']['customers']:,}")
    print(f"  - AOV : {utils.format_currency(simulation_results['current']['aov'])}")
    print(f"✓ Résultats projetés :")
    print(f"  - Revenu : {utils.format_currency(simulation_results['projected']['revenue'])}")
    print(f"  - Clients : {simulation_results['projected']['customers']:,}")
    print(f"  - AOV : {utils.format_currency(simulation_results['projected']['aov'])}")
    print(f"✓ Impact :")
    print(f"  - Δ Revenu : {utils.format_currency(simulation_results['delta']['revenue'])} ({simulation_results['delta']['revenue_pct']:.1f}%)")
    print()

    # =========================================================================
    # ÉTAPE 10 : EXPORT DES RÉSULTATS
    # =========================================================================
    print("💾 ÉTAPE 10 : Export des résultats")
    print("-" * 80)

    # Export CSV
    try:
        rfm_path = utils.export_to_csv(rfm_df, 'rfm_analysis_example.csv')
        print(f"✓ Analyse RFM exportée : {rfm_path}")
    except Exception as e:
        print(f"⚠ Erreur export RFM : {e}")

    try:
        clv_path = utils.export_to_csv(clv_empirical, 'clv_empirical_example.csv')
        print(f"✓ CLV empirique exportée : {clv_path}")
    except Exception as e:
        print(f"⚠ Erreur export CLV : {e}")

    print()

    # =========================================================================
    # RÉSUMÉ FINAL
    # =========================================================================
    print("=" * 80)
    print("✅ PIPELINE COMPLET EXÉCUTÉ AVEC SUCCÈS")
    print("=" * 80)
    print()
    print("Fonctions utilisées :")
    print("  1. load_data")
    print("  2. clean_data")
    print("  3. create_cohorts")
    print("  4. calculate_retention")
    print("  5. calculate_rfm")
    print("  6. calculate_clv_empirical")
    print("  7. calculate_clv_formula")
    print("  8. calculate_kpis")
    print("  9. calculate_churn_rate")
    print(" 10. get_churn_predictions")
    print(" 11. apply_filters")
    print(" 12. simulate_scenario")
    print(" 13. export_to_csv")
    print(" 14. format_currency")
    print(" 15. format_percentage")
    print()
    print("Toutes les fonctions de utils.py sont opérationnelles ! 🎉")
    print("=" * 80)


def exemple_formatage():
    """Exemple d'utilisation des fonctions de formatage"""
    print("\n" + "=" * 80)
    print("EXEMPLE : FONCTIONS DE FORMATAGE")
    print("=" * 80)

    # Format monétaire
    montants = [1234.56, 9876543.21, 42.0]
    devises = ['GBP', 'EUR', 'USD']

    print("\n💵 Format monétaire :")
    for montant, devise in zip(montants, devises):
        formatted = utils.format_currency(montant, devise)
        print(f"  {montant:>12,.2f} {devise} → {formatted}")

    # Format pourcentage
    print("\n📊 Format pourcentage :")
    valeurs = [0.255, 0.8765, 0.042, 0.999]
    for val in valeurs:
        formatted = utils.format_percentage(val, decimals=1)
        print(f"  {val:.4f} → {formatted}")

    print()


def exemple_validation():
    """Exemple d'utilisation de la validation"""
    print("\n" + "=" * 80)
    print("EXEMPLE : VALIDATION DE DATAFRAME")
    print("=" * 80)

    # Créer un DataFrame de test
    df_test = pd.DataFrame({
        'Invoice': ['001', '002'],
        'Customer ID': ['12345', '67890'],
        'Price': [10.0, 20.0]
    })

    print(f"\nDataFrame de test : {list(df_test.columns)}")

    # Test 1 : Colonnes présentes
    colonnes_requises = ['Invoice', 'Customer ID', 'Price']
    valid, missing = utils.validate_dataframe(df_test, colonnes_requises)
    print(f"\n✓ Test 1 - Colonnes requises : {colonnes_requises}")
    print(f"  Validation : {'✓ OK' if valid else '✗ ÉCHEC'}")
    if missing:
        print(f"  Colonnes manquantes : {missing}")

    # Test 2 : Colonnes manquantes
    colonnes_requises = ['Invoice', 'Quantity', 'Country']
    valid, missing = utils.validate_dataframe(df_test, colonnes_requises)
    print(f"\n✓ Test 2 - Colonnes requises : {colonnes_requises}")
    print(f"  Validation : {'✓ OK' if valid else '✗ ÉCHEC'}")
    if missing:
        print(f"  Colonnes manquantes : {missing}")

    print()


if __name__ == "__main__":
    # Exécuter l'exemple complet
    exemple_complet()

    # Exemples additionnels
    exemple_formatage()
    exemple_validation()
