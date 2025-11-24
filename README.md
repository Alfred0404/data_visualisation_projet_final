# Application d'Aide a la Decision Marketing

## Description

Application d'analyse de donnees marketing developpee avec Streamlit, permettant d'exploiter le dataset Online Retail II pour optimiser les strategies marketing et maximiser la valeur client. Cette solution propose des analyses approfondies de cohortes, une segmentation client basee sur la methodologie RFM, le calcul de la Customer Lifetime Value (CLV) et la simulation de scenarios marketing.

## Fonctionnalites

### Analyse de Cohortes
- Suivi de l'evolution des cohortes d'acquisition dans le temps
- Calcul des taux de retention par cohorte
- Identification des periodes d'acquisition les plus performantes
- Visualisations interactives des comportements de retention

### Segmentation RFM
- Segmentation des clients selon trois criteres : Recence, Frequence et Montant
- Identification automatique des segments strategiques (Champions, Clients Fideles, Clients a Risque, etc.)
- Priorisation des actions marketing par segment
- Tableaux de bord dedies par segment client

### Customer Lifetime Value (CLV)
- CLV empirique basee sur l'analyse des cohortes historiques
- CLV predictive utilisant des modeles mathematiques
- Projections de revenus futurs par segment
- Optimisation du retour sur investissement des campagnes marketing

### Simulation de Scenarios
- Modelisation de l'impact de l'amelioration du taux de retention
- Simulation de l'effet de l'augmentation du panier moyen
- Projection des revenus selon differents scenarios (optimiste, realiste, conservateur)
- Aide a la decision strategique basee sur des donnees

### Export et Reporting
- Export des donnees analysees au format CSV
- Export des visualisations au format PNG haute resolution
- Generation de rapports personnalises pour le partage avec les equipes

## Technologies Utilisees

### Backend & Data Science
- **Python 3.8+** : Langage de programmation principal
- **Pandas** : Manipulation et analyse de donnees
- **NumPy** : Calculs numeriques et operations matricielles
- **Scikit-learn** : Algorithmes de machine learning et segmentation

### Visualisation
- **Streamlit** : Framework d'application web interactive
- **Plotly** : Graphiques interactifs et tableaux de bord
- **Matplotlib** : Visualisations statiques
- **Seaborn** : Visualisations statistiques avancees

### Traitement de Donnees
- **openpyxl** : Lecture et ecriture de fichiers Excel

## Installation

### Prerequis
- Python 3.8 ou superieur
- pip (gestionnaire de paquets Python)

### Etapes d'installation

1. Cloner le depot :
```bash
git clone https://github.com/Alfred0404/data_visualisation_projet_final.git
cd projet_final
```

2. Installer les dependances :
```bash
pip install -r requirements.txt
```

3. Verifier la structure des repertoires :
```bash
python config.py
```

Cette commande creera automatiquement les repertoires necessaires s'ils n'existent pas.

## Utilisation

### Lancement de l'application

```bash
streamlit run app/app.py
```

L'application sera accessible dans votre navigateur a l'adresse `http://localhost:8501`.

### Workflow recommande

1. **Chargement des donnees** : Au demarrage, cliquez sur "Charger les donnees" pour importer et nettoyer automatiquement le dataset
2. **Application des filtres** : Utilisez les filtres globaux dans la barre laterale pour affiner vos analyses (periode, pays, montant minimum)
3. **Navigation entre les pages** : Explorez les differentes analyses via le menu de navigation
4. **Export des resultats** : Telechargez vos analyses et visualisations depuis la page Export

### Pages disponibles

- **Overview** : Vue d'ensemble et KPIs globaux
- **Cohortes** : Analyse detaillee des cohortes d'acquisition
- **Segments** : Segmentation RFM et profils clients
- **Scenarios** : Simulation de scenarios marketing
- **Export** : Telechargement des donnees et graphiques

## Structure du Projet

```
projet_final/
├── app/
│   ├── app.py              # Point d'entree de l'application Streamlit
│   ├── pages/              # Pages multi-pages de Streamlit
│   │   ├── 1_Overview.py
│   │   ├── 2_Cohortes.py
│   │   ├── 3_Segments.py
│   │   ├── 4_Scenarios.py
│   │   └── 5_Export.py
│   └── utils.py            # Fonctions utilitaires
├── data/
│   ├── raw/                # Donnees brutes (Online Retail II)
│   └── processed/          # Donnees traitees et analyses
├── notebooks/              # Notebooks Jupyter pour exploration
├── docs/                   # Documentation du projet
├── exports/                # Exports de donnees et graphiques
├── config.py               # Configuration centralisee
├── requirements.txt        # Dependances Python
└── README.md              # Ce fichier
```

## Methodologie

### Nettoyage des Donnees
- Suppression des transactions annulees
- Filtrage des valeurs aberrantes
- Gestion des donnees manquantes
- Validation des montants et quantites

### Analyse RFM
Segmentation basee sur trois dimensions :
- **Recence (R)** : Nombre de jours depuis le dernier achat
- **Frequence (F)** : Nombre total de transactions
- **Montant (M)** : Chiffre d'affaires total genere

Chaque dimension est scoree de 1 a 4 (quartiles), permettant d'identifier 8 segments principaux.

### Calcul de la CLV
Deux approches complementaires :
- **CLV empirique** : Basee sur les comportements reels des cohortes historiques
- **CLV predictive** : Utilisant des formules mathematiques avec taux de retention et valeur moyenne

### Analyse de Cohortes
Regroupement des clients par mois de premiere transaction et suivi de leur comportement dans le temps pour mesurer la retention et la valeur generee par periode.

## Dataset

Le projet utilise le dataset **Online Retail II** contenant des transactions de vente en ligne entre 2009 et 2011 :
- Plus de 500 000 transactions
- Informations sur les produits, clients, montants et dates
- Couverture geographique internationale

## Configuration

Les parametres de l'application peuvent etre ajustes dans `config.py` :
- Seuils de segmentation RFM
- Parametres de calcul CLV (taux de retention, taux d'actualisation)
- Palettes de couleurs pour les visualisations
- Scenarios de simulation predefinis

## Auteur

Projet realise dans le cadre du cours de Data Visualisation - ECE Paris

[Alfred de Vulpian](https://github.com/alfred0404)

[Nicolas DONIER](https://github.com/reinod15)

[Thibault GAREL](https://github.com/Thibault-GAREL)

[Maxime DUTERTRE](https://github.com/madmax0978)

[Alexandre GARREAU](https://github.com/AlexDreams)

[Kimarjie LUCENARODRIGO](https://github.com/kimarjie)
