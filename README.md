# Application d'Aide a la Decision Marketing

## Description

Application web interactive d'analyse marketing basee sur 525K transactions e-commerce (Online Retail II). Offre des outils d'aide a la decision pour optimiser la retention client et maximiser le ROI marketing.

## Fonctionnalites

### Analyse de Cohortes
Suivi de la retention client par cohorte d'acquisition avec visualisations interactives.

### Segmentation RFM
Classification des clients en 8 segments strategiques (Champions, Fideles, A Risque, Perdus...) selon Recence, Frequence et Montant.

### Customer Lifetime Value
Calcul de la valeur vie client (empirique et predictive) pour optimiser les investissements marketing.

### Simulation de Scenarios
Modelisation de l'impact des actions marketing (retention, panier moyen, frequence) sur le CA futur.

### Export et Reporting
Export CSV/Excel des analyses et telechargement des visualisations.

## Technologies Utilisees

- **Python 3.8+** | **Pandas** | **NumPy** : Traitement et analyse de donnees
- **Streamlit** : Application web interactive
- **Plotly** : Visualisations interactives
- **openpyxl** : Import/export Excel

## Installation

### Prerequis
- Python 3.8 ou superieur
- pip (gestionnaire de paquets Python)

### Etapes d'installation

1. Cloner le depot :
```bash
git clone https://github.com/Alfred0404/data_visualisation_projet_final.git
cd data_visualisation_projet_final
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

**Nettoyage** : Filtrage des valeurs aberrantes, gestion des donnees manquantes, validation des montants.

**RFM** : Scores 1-5 (quintiles) sur Recence, Frequence et Montant → 8 segments strategiques.

**CLV** : Calcul empirique (historique reel) et predictif (formule mathematique).

**Cohortes** : Regroupement par mois d'acquisition, suivi de retention dans le temps.

## Dataset

**Online Retail II** (2009-2011) : 525K transactions e-commerce, 4.3K clients, 37 pays.

## Configuration

Parametres ajustables dans `config.py` : seuils RFM, taux CLV, scenarios predefinis, palettes de couleurs.

## Auteur

Projet realise dans le cadre du cours de Data Visualisation - ECE Paris

[Alfred de Vulpian](https://github.com/alfred0404)

[Nicolas DONIER](https://github.com/reinod15)

[Thibault GAREL](https://github.com/Thibault-GAREL)

[Maxime DUTERTRE](https://github.com/madmax0978)

[Alexandre GARREAU](https://github.com/AlexDreams)

[Kimarjie LUCENARODRIGO](https://github.com/kimarjie)
