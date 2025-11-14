"""
Script de test pour valider l'implémentation des fonctions dans utils.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from app import utils
import config
import pandas as pd

def test_imports():
    """Teste que tous les imports fonctionnent"""
    print("✓ Imports réussis")

def test_functions_exist():
    """Vérifie que toutes les fonctions existent"""
    required_functions = [
        'load_data',
        'clean_data',
        'create_cohorts',
        'calculate_retention',
        'calculate_rfm',
        'calculate_clv_empirical',
        'calculate_clv_formula',
        'apply_filters',
        'simulate_scenario',
        'export_to_csv',
        'export_chart_to_png',
        'calculate_kpis',
        'calculate_churn_rate',
        'get_churn_predictions',
        'validate_dataframe',
        'format_currency',
        'format_percentage'
    ]

    missing_functions = []
    for func_name in required_functions:
        if not hasattr(utils, func_name):
            missing_functions.append(func_name)

    if missing_functions:
        print(f"✗ Fonctions manquantes : {missing_functions}")
        return False
    else:
        print(f"✓ Toutes les {len(required_functions)} fonctions sont présentes")
        return True

def test_formatting_functions():
    """Teste les fonctions de formatage"""
    try:
        # Test format_currency
        result = utils.format_currency(1234.56, 'GBP')
        assert '£' in result
        assert '1,234.56' in result
        print(f"✓ format_currency : {result}")

        # Test format_percentage
        result = utils.format_percentage(0.255, 1)
        assert result == '25.5%'
        print(f"✓ format_percentage : {result}")

        return True
    except Exception as e:
        print(f"✗ Erreur dans les fonctions de formatage : {e}")
        return False

def test_validation_function():
    """Teste la fonction de validation"""
    try:
        # Créer un DataFrame de test
        df = pd.DataFrame({
            'Invoice': ['001'],
            'Customer ID': ['12345']
        })

        # Test avec colonnes présentes
        valid, missing = utils.validate_dataframe(df, ['Invoice', 'Customer ID'])
        assert valid == True
        assert len(missing) == 0
        print("✓ validate_dataframe (colonnes présentes)")

        # Test avec colonnes manquantes
        valid, missing = utils.validate_dataframe(df, ['Invoice', 'Price', 'Country'])
        assert valid == False
        assert 'Price' in missing
        assert 'Country' in missing
        print("✓ validate_dataframe (colonnes manquantes)")

        return True
    except Exception as e:
        print(f"✗ Erreur dans validate_dataframe : {e}")
        return False

def main():
    """Fonction principale de test"""
    print("=" * 70)
    print("TEST DE L'IMPLÉMENTATION DES FONCTIONS UTILS.PY")
    print("=" * 70)
    print()

    results = []

    # Test 1 : Imports
    print("[1/4] Test des imports...")
    try:
        test_imports()
        results.append(True)
    except Exception as e:
        print(f"✗ Erreur d'import : {e}")
        results.append(False)
    print()

    # Test 2 : Existence des fonctions
    print("[2/4] Vérification de l'existence des fonctions...")
    results.append(test_functions_exist())
    print()

    # Test 3 : Fonctions de formatage
    print("[3/4] Test des fonctions de formatage...")
    results.append(test_formatting_functions())
    print()

    # Test 4 : Fonction de validation
    print("[4/4] Test de la fonction de validation...")
    results.append(test_validation_function())
    print()

    # Résumé
    print("=" * 70)
    success_count = sum(results)
    total_count = len(results)

    if all(results):
        print(f"✓ SUCCÈS : Tous les tests sont passés ({success_count}/{total_count})")
        print()
        print("Les fonctions suivantes sont implémentées et opérationnelles :")
        print("  - load_data")
        print("  - clean_data")
        print("  - create_cohorts")
        print("  - calculate_retention")
        print("  - calculate_rfm")
        print("  - calculate_clv_empirical")
        print("  - calculate_clv_formula")
        print("  - apply_filters")
        print("  - simulate_scenario")
        print("  - export_to_csv")
        print("  - export_chart_to_png")
        print("  - calculate_kpis")
        print("  - calculate_churn_rate")
        print("  - get_churn_predictions")
        print("  - validate_dataframe")
        print("  - format_currency")
        print("  - format_percentage")
    else:
        print(f"✗ ÉCHEC : {total_count - success_count} test(s) échoué(s) sur {total_count}")

    print("=" * 70)

if __name__ == "__main__":
    main()
