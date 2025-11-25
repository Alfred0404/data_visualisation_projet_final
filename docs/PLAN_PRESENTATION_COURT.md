# Plan de Presentation Court - Application Marketing Analytics

## Informations
- **Duree** : 10-12 minutes
- **Slides** : 8 slides
- **Format** : Direct et concret avec explications des concepts

---

## SLIDE 1 : Titre + Contexte

### Contenu
**Titre** : Application d'Aide a la Decision Marketing

**Problematique** :
- Comment identifier les meilleurs clients ?
- Comment optimiser la retention ?
- Comment simuler l'impact des investissements marketing ?

**Solution** : Application Python/Streamlit pour analyse de 525K transactions

### Script (1 min)
"Bonjour. Les entreprises e-commerce ont des millions de transactions mais peinent a les exploiter. Mon application repond a trois questions : qui sont nos meilleurs clients, comment les fideliser, et quel sera le ROI - le Retour sur Investissement - de nos actions marketing. J'ai developpe une solution complete en Python avec Streamlit pour analyser 525,000 transactions."

---

## SLIDE 2 : Architecture + Dataset

### Contenu
**Stack** :
- Backend : Python + Pandas + NumPy
- Frontend : Streamlit + Plotly
- Structure : app.py (308 lignes) + utils.py (1025 lignes) + 5 pages

**Dataset** :
- Online Retail II (2009-2011)
- 525K transactions, 5000+ clients, 38 pays

**Pipeline** :
1. Chargement automatique CSV/Excel
2. Nettoyage (85% donnees conservees)
3. Enrichissement (TotalAmount, flags, dates)

### Script (1 min 30)
"Architecture Python en 3 couches. Le backend utilise Pandas pour la manipulation de donnees, Streamlit pour l'interface web. J'ai structure le projet professionnellement : app.py pour l'interface, utils.py avec 1025 lignes de fonctions metier, et 5 pages specialisees. Le dataset Online Retail II contient 525,000 transactions sur 2 ans. Mon pipeline en 3 etapes charge, nettoie et enrichit les donnees automatiquement, conservant 85% des transactions apres filtrage des valeurs aberrantes - c'est-a-dire les valeurs extremes qui fausseraient les analyses."

---

## SLIDE 3 : Navigation et Filtres Globaux

### Contenu
**5 Pages de l'application** :
1. **Overview** : KPIs globaux + graphiques
2. **Cohortes** : Analyse de retention
3. **Segments** : Segmentation RFM
4. **Scenarios** : Simulations marketing
5. **Export** : Telechargements CSV/Excel

**Innovation : Filtres Globaux**
```
Sidebar (app.py)
  ↓ session_state
utils.apply_global_filters()
  ↓
Applique sur TOUTES les pages
```

**Filtres** : Periode, Pays, Montant minimum

### Script (1 min 30)
"L'application a 5 pages specialisees. Overview pour les KPIs - les indicateurs de performance cles comme le chiffre d'affaires et le nombre de clients. Cohortes pour analyser la retention - c'est-a-dire combien de clients reviennent acheter. Segments pour la segmentation RFM que j'expliquerai. Scenarios pour simuler l'impact d'actions marketing. Et Export pour telecharger les resultats. Une innovation technique : les filtres globaux. L'utilisateur definit ses filtres une seule fois dans la sidebar, et ils s'appliquent automatiquement sur toutes les pages grace a la fonction apply_global_filters. Si je filtre sur UK uniquement, toutes mes analyses portent sur UK. C'est transparent et coherent."

---

## SLIDE 4 : Analyse de Cohortes

### Contenu
**Principe** :
- Regrouper clients par mois de premiere transaction
- Suivre leur comportement dans le temps

**Vocabulaire explique** :
- **Cohorte** : Groupe de clients ayant effectue leur premier achat le meme mois
- **Retention** : Pourcentage de clients qui reviennent acheter
- **M1, M3, M6, M12** : Retention a 1, 3, 6 et 12 mois apres le premier achat

**Visualisation** :
- Heatmap de retention (vert = bon, rouge = faible)
- Metriques : Taille, Retention M1/M3/M6/M12, CA

**Code** :
```python
df_cohorts = utils.create_cohorts(df)          # ligne 525
retention = utils.calculate_retention_matrix() # ligne 595
```

**Insight** : Retention moyenne M12 = 25%, chute importante apres M3

### Script (1 min 30)
"L'analyse de cohortes regroupe les clients par mois de premiere transaction - on appelle ca une cohorte - et suit leur comportement. Par exemple, tous les clients qui ont achete pour la premiere fois en janvier 2010 forment la cohorte janvier 2010. La fonction create_cohorts identifie chaque cohorte, puis calculate_retention_matrix genere une matrice de retention. La retention, c'est le pourcentage de clients qui reviennent acheter. Par exemple, si 100 clients achetent en janvier et que 30 reviennent en mars, la retention a M3 est de 30%. La heatmap permet d'identifier instantanement les cohortes performantes en vert et les problematiques en rouge. Insight cle : le taux de retention moyen a 12 mois est de 25%, avec une chute importante apres le 3eme mois - c'est la que les efforts de fidelisation doivent se concentrer."

---

## SLIDE 5 : Segmentation RFM + CLV

### Contenu
**RFM explique** :

**R - Recency (Recence)** :
- Definition : Nombre de jours depuis le dernier achat
- Logique : Un client qui a achete recemment est plus susceptible de racheter
- Score : 4 = tres recent (moins de 30 jours), 1 = ancien (plus de 180 jours)

**F - Frequency (Frequence)** :
- Definition : Nombre total de transactions
- Logique : Un client qui achete souvent est plus engage et fidele
- Score : 4 = tres frequent (20+ achats), 1 = occasionnel (1-2 achats)

**M - Monetary (Montant)** :
- Definition : Chiffre d'affaires total genere par le client
- Logique : Un client qui depense beaucoup a plus de valeur
- Score : 4 = gros depensier (1000£+), 1 = petit panier (moins de 100£)

```python
calculate_rfm(df)  # utils.py ligne 636
```

**8 Segments** :
- **Champions** (R=5, F=5, M=5) : 12% clients → 38% CA
  - Definition : Clients recents, frequents et gros depensiers
- **Loyal Customers** : Clients fideles reguliers
- **Potential Loyalists** : Nouveaux clients prometteurs
- **At Risk** : Bons clients devenus inactifs
- **Lost** : Clients perdus a reconquerir

**CLV (Customer Lifetime Value)** expliquee :
- **Definition** : Valeur totale qu'un client generera durant toute sa relation avec l'entreprise
- **Utilite** : Savoir combien on peut investir pour acquerir ou fideliser un client
- **Exemple** : Si un Champion a une CLV de 2000£, on peut depenser jusqu'a 500£ en marketing pour le garder

**Deux methodes** :
- **Empirique** : Somme des revenus reels historiques (factuel)
- **Predictive** : Formule mathematique pour projeter la valeur future
- **Ratio Champions/Lost** : 20x (2000£ vs 100£)

### Script (2 min)
"La segmentation RFM classe chaque client selon trois dimensions. R pour Recency, la recence : combien de jours depuis le dernier achat. Un client qui a achete il y a 5 jours est plus susceptible de racheter qu'un client inactif depuis 6 mois. F pour Frequency, la frequence : combien de fois le client a achete. Plus il achete souvent, plus il est engage. M pour Monetary, le montant : combien le client a depense au total. Un client a 1000 livres de CA est plus precieux qu'un client a 50 livres.

La fonction calculate_rfm attribue un score de 1 a 5 pour chaque dimension. Par exemple, R=5 signifie achat tres recent, F=5 signifie tres frequent, M=5 signifie gros depensier. On combine ces scores pour identifier 8 segments strategiques.

Les Champions ont R=5, F=5, M=5 - ce sont nos meilleurs clients : recents, frequents et qui depensent beaucoup. Resultat cle : 12% de Champions generent 38% du chiffre d'affaires - c'est le principe de Pareto.

J'ai aussi calcule la CLV - Customer Lifetime Value - c'est la valeur vie client. Ca represente le revenu total qu'un client generera durant toute sa relation avec nous. J'ai implemente deux methodes : empirique basee sur l'historique reel, et predictive avec une formule mathematique. Un Champion a une CLV 20 fois superieure a un client Lost - 2000 livres contre 100 livres. Ca justifie des budgets marketing tres differents : je peux depenser 500 livres pour fideliser un Champion, mais pas pour un client Lost."

---

## SLIDE 6 : Simulations de Scenarios

### Contenu
**Concepts expliques** :

**Parametres ajustables** :
- **Retention** : Taux de clients qui reviennent acheter (+5% = 5% de clients en plus qui reviennent)
- **Panier moyen (AOV)** : Montant moyen par commande (+10% = chaque client depense 10% de plus)
- **Frequence** : Nombre de commandes par an (+15% = les clients achetent 15% plus souvent)

**3 Scenarios predefinis** :
- **Conservateur** : +2% retention, +3% AOV, +5% frequence (objectifs prudents)
- **Realiste** : +5% retention, +8% AOV, +10% frequence (objectifs atteignables)
- **Optimiste** : +10% retention, +15% AOV, +20% frequence (objectifs ambitieux)

**ROI (Return On Investment)** explique :
- Definition : Retour sur Investissement
- Formule : ROI = (Gain - Cout) / Cout × 100
- Exemple : Si je depense 100K£ et gagne 500K£, ROI = 400%
- Interpretation : Pour chaque livre investie, je gagne 4 livres

**Fonction** :
```python
results = simulate_scenario(df, params)  # ligne 1052
# Retourne : revenue, CLV, ROI
```

**3 Visualisations** :
- **Bar chart** : Compare la situation actuelle vs projetee
- **Line chart** : Montre l'evolution mois par mois sur 1-3 ans
- **Tornado chart** : Identifie quel levier a le plus d'impact

**Insight** : +5% retention = +300K£ sur 3 ans (ROI 3-5x)

### Script (2 min)
"La page Scenarios permet de simuler l'impact de strategies marketing. Je vais expliquer les parametres.

La retention, c'est le pourcentage de clients qui reviennent. Si j'ameliore la retention de 5%, ca veut dire que 5% de clients en plus vont revenir acheter une deuxieme fois.

Le panier moyen, ou AOV pour Average Order Value, c'est le montant moyen par commande. Si j'augmente le panier moyen de 10%, chaque client depense 10% de plus a chaque achat, par exemple via de l'upselling.

La frequence, c'est le nombre de fois qu'un client achete par an. Si j'augmente la frequence de 15%, les clients achetent 15% plus souvent, par exemple grace a des emails marketing reguliers.

L'utilisateur ajuste ces curseurs. J'ai predefini trois scenarios realistes dans le fichier config : conservateur avec des objectifs prudents, realiste avec des objectifs atteignables, et optimiste pour des objectifs ambitieux.

La fonction simulate_scenario calcule instantanement l'impact sur le revenu, la CLV et le ROI. Le ROI, c'est le Retour sur Investissement : pour chaque livre investie en marketing, combien je gagne. Un ROI de 300% signifie que je gagne 3 livres pour chaque livre investie.

Trois visualisations complementaires : un bar chart compare avant/apres, un line chart montre l'evolution mensuelle sur 1 a 3 ans - on voit la croissance progressive, et un tornado chart identifie le levier le plus impactant - dans notre cas c'est la retention.

Resultat majeur : ameliorer la retention de seulement 5% - faire revenir 5% de clients en plus - genere 300,000 livres de chiffre d'affaires supplementaire sur 3 ans, avec un ROI estime entre 3 et 5 fois l'investissement. Ca veut dire que si je depense 100,000 livres en actions de fidelisation, je gagne entre 300,000 et 500,000 livres."

---

## SLIDE 7 : Demo Live

### Contenu
**Parcours de demonstration** :

1. Charger les donnees (bouton)
2. Appliquer filtre UK uniquement
3. Observer KPIs Overview
4. Visualiser heatmap Cohortes
5. Analyser segment Champions
6. Lancer simulation scenario Realiste
7. Exporter resultats en CSV

**Points a montrer** :
- Interactivite des graphiques (hover)
- Filtres temps reel
- Export en un clic

### Script (2 min)
"Demonstration rapide. Je clique sur Charger les donnees - le systeme charge et nettoie automatiquement 525,000 transactions en quelques secondes grace au cache Streamlit.

Je filtre sur le Royaume-Uni uniquement - vous voyez, les KPIs s'actualisent instantanement. On passe de 5000 clients a 3900 clients britanniques. Le chiffre d'affaires, le panier moyen, tout se recalcule automatiquement.

Sur la page Cohortes, voici la heatmap de retention - c'est une carte de chaleur. Le vert indique une bonne retention, le rouge une mauvaise retention. Je survole la cohorte de decembre 2010 - vous voyez, 45% de retention a M1, ca descend a 32% a M3, puis 25% a M12. C'est une bonne cohorte.

Sur la page Segments, j'analyse le profil des Champions. Ils sont 450 clients, generent 1.2 millions de livres de CA, avec une CLV moyenne de 2600 livres. L'application me donne des recommandations marketing concretes : programme VIP, acces anticipe aux nouveaux produits, personnalisation maximale.

Je lance maintenant une simulation scenario Realiste : +5% de retention, +8% de panier moyen, +10% de frequence. Le systeme calcule instantanement - le revenu projete passe de 3.8M£ a 4.5M£, soit +700K£. Le graphique d'evolution montre la croissance progressive sur 3 ans. Le tornado chart indique que la retention est le levier le plus impactant.

Enfin, j'exporte tous les resultats en CSV en un clic - le fichier contient les scores RFM de tous les clients, pret a etre importe dans un CRM ou Excel. L'interface est fluide, intuitive et professionnelle."

---

## SLIDE 8 : Conclusion

### Contenu
**Realisations** :
- 1334 lignes de code (app.py + utils.py + pages)
- 15+ fonctions metier
- Filtres globaux coherents
- Cache Streamlit pour performances

**Competences** :
- Python avance (Pandas, NumPy, Plotly)
- Data science (RFM, cohortes, CLV)
- Developpement web (Streamlit)
- Architecture logicielle

**Impact Business concret** :
- **Identification** : 12% de Champions = 38% du CA → Prioriser ces clients
- **Valorisation** : CLV Champions 20x superieure → Justifie budgets differencies
- **Optimisation** : +5% retention = +300K£ sur 3 ans → ROI 3-5x
- **Decision** : Simulations avant investissement → Reduit le risque

**Prochaines etapes** :
- ML pour prediction de churn (identifier clients a risque avant qu'ils partent)
- Integration CRM pour actions directes (envoyer emails automatiquement)
- A/B testing integre (mesurer l'impact reel des campagnes)

### Script (1 min)
"Pour conclure, application complete de 1334 lignes avec 15 fonctions metier. Architecture professionnelle avec filtres globaux - un seul filtre s'applique partout - et caching pour les performances.

J'ai developpe des competences en Python data science, Streamlit pour les interfaces web, et architecture logicielle.

L'impact business est concret et mesurable. Premier point : identification. On sait maintenant que 12% de Champions generent 38% du chiffre d'affaires - on peut concentrer nos efforts sur eux. Deuxieme point : valorisation. Un Champion vaut 20 fois plus qu'un client Lost - 2000 livres contre 100 livres - donc on peut justifier un budget marketing 20 fois superieur pour les fideliser. Troisieme point : optimisation. Ameliorer la retention de seulement 5% genere 300,000 livres supplementaires sur 3 ans avec un ROI de 3 a 5 fois - c'est enorme. Quatrieme point : decision. On peut simuler avant d'investir - ca reduit enormement le risque.

Les prochaines etapes seraient d'integrer du machine learning pour predire quels clients vont partir avant qu'ils ne partent, connecter un CRM pour envoyer automatiquement des emails cibles, et implementer de l'A/B testing pour mesurer l'impact reel de chaque campagne.

Merci pour votre attention, je suis pret pour vos questions."

---

## QUESTIONS FREQUENTES (Preparation)

**Q : Pourquoi Streamlit ?**
R : "Streamlit permet un developpement 5 fois plus rapide qu'avec Flask ou Dash. Il gere automatiquement le state - l'etat de l'application - et le rerendering. L'interface est suffisamment professionnelle pour une vraie entreprise. Parfait pour un projet academique avec deadline serree."

**Q : Performances avec plus de donnees ?**
R : "Actuellement optimise pour 500,000 lignes - ca charge en 2-3 secondes. Pour 10 millions de lignes et plus, j'implementerais du chunking - traiter les donnees par morceaux, une base de donnees SQL pour le stockage au lieu de CSV, et du sampling pour les visualisations - afficher un echantillon representatif plutot que tout. Le cache Streamlit aide deja enormement - les donnees ne sont chargees qu'une seule fois."

**Q : Comment validez-vous vos calculs ?**
R : "Trois niveaux de validation. Un : tests sur des sous-ensembles de donnees dont je connais le resultat attendu - par exemple, si je prends 10 clients et calcule leur RFM a la main, je dois trouver le meme resultat. Deux : verification manuelle de 10 clients aleatoires pour le RFM. Trois : les formules de CLV sont issues de la litterature academique, notamment les travaux de Fader et Hardie de l'Universite de Wharton qui sont la reference en Customer Analytics."

**Q : Deploiement en production ?**
R : "Deux options. Pour un deploiement rapide et gratuit, Streamlit Cloud - je pousse le code sur GitHub et l'app est en ligne en 5 minutes. Pour une vraie entreprise, Docker plus AWS ou Azure avec authentification OAuth - connexion securisee - et encryption des donnees sensibles. J'ajouterais aussi des logs d'audit pour tracer qui accede a quoi."

**Q : Cout et maintenance ?**
R : "Streamlit Cloud est gratuit jusqu'a 1 gigabyte de donnees et 3 applications. Pour une entreprise, compter 50-100 euros par mois sur AWS selon le trafic. La maintenance est simple grace a l'architecture modulaire - si je veux ajouter une nouvelle analyse, je cree juste un nouveau fichier dans le dossier pages. L'ajout de nouvelles fonctionnalites est facile grace a la separation claire entre utils.py qui contient la logique metier et les pages qui gerent l'affichage."

**Q : Pourquoi 85% de donnees conservees apres nettoyage ?**
R : "J'ai adopte une approche intelligente de nettoyage. Je supprime uniquement ce qui est vraiment inutilisable : les transactions annulees - identifiees par le prefix C dans le numero de facture - les prix negatifs ou zero, et les dates invalides. Par contre, je CONSERVE les transactions sans Customer ID pour les analyses globales de CA ou de produits. Je marque simplement ces transactions avec un flag 'HasCustomerID' a False. Comme ca, pour l'analyse RFM qui necessite un Customer ID, je filtre, mais pour analyser les ventes par pays, je garde tout. C'est pour ca que je conserve 85% au lieu de 60-70% avec une approche naive qui supprimerait tout ce qui n'est pas parfait."

---

## GLOSSAIRE RAPIDE (A avoir en tete)

**Marketing Analytics** :
- **Cohorte** : Groupe de clients acquis la meme periode
- **Retention** : % de clients qui reviennent acheter
- **Churn** : % de clients qui partent (inverse de retention)
- **AOV** : Average Order Value = Panier moyen
- **CLV** : Customer Lifetime Value = Valeur vie client
- **ROI** : Return On Investment = Retour sur investissement

**RFM** :
- **R** : Recency = Jours depuis dernier achat
- **F** : Frequency = Nombre d'achats
- **M** : Monetary = Montant total depense

**Technique** :
- **Cache** : Stockage temporaire pour accelerer
- **State** : Etat de l'application (filtres, donnees chargees, etc.)
- **Pipeline** : Suite d'etapes de traitement
- **API** : Interface pour connecter des systemes

---

## CONSEILS RAPIDES

### Avant
- Repetez 2-3 fois en chronometrant
- Testez que l'app fonctionne
- Preparez screenshots si demo echoue
- Relisez le glossaire

### Pendant
- **Parlez clairement, pas trop vite**
- **Expliquez TOUS les acronymes la premiere fois**
- Montrez votre passion
- Donnez des exemples concrets (2000£ vs 100£)
- Gardez le contact visuel
- Utilisez des analogies si besoin

### Analogies utiles si questions
- **Cohorte** = "C'est comme une promotion a l'ecole - tous ceux qui sont entres en 2020"
- **Retention** = "C'est comme le taux de personnes qui reviennent dans un restaurant"
- **CLV** = "C'est comme estimer combien un abonne Netflix va rapporter sur 5 ans"
- **ROI** = "Si j'investis 100€ et gagne 400€, mon ROI est de 300%"

### Timing
- Intro : 1 min
- Technique : 3 min
- Fonctionnalites : 5 min (avec explications)
- Demo : 2 min
- Conclusion : 1 min
- **TOTAL : 12 min**

---

**Presentation claire, pedagogique et professionnelle !**
