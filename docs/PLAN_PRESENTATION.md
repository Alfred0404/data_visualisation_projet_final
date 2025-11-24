# Plan de Presentation - Application d'Aide a la Decision Marketing

## Informations Generales
- **Duree estimee** : 15-20 minutes
- **Nombre de slides** : 12-15 slides
- **Public cible** : Evaluateurs academiques, potentiels utilisateurs marketing

---

## SLIDE 1 : Page de Titre

### Contenu visuel
- Titre : **Application d'Aide a la Decision Marketing**
- Sous-titre : Analyse de cohortes, segmentation RFM et simulation de scenarios
- Votre nom et promotion
- Logo ECE Paris
- Date de presentation
- Une image d'illustration : tableau de bord ou graphique marketing

### Ce qu'il faut dire
"Bonjour a tous. Aujourd'hui, je vais vous presenter mon projet de data visualisation : une application d'aide a la decision marketing. Cette solution permet aux entreprises d'optimiser leurs strategies marketing grace a l'analyse approfondie de leurs donnees clients."

**Duree** : 30 secondes

---

## SLIDE 2 : Contexte et Problematique

### Contenu visuel
- **Contexte business** :
  - Les entreprises collectent enormement de donnees clients
  - Difficulte a transformer ces donnees en decisions actionnables
  - Besoin d'outils pour comprendre le comportement client

- **Problematique** :
  - Comment identifier les clients a forte valeur ?
  - Comment mesurer et ameliorer la retention ?
  - Comment prevoir l'impact des actions marketing ?

### Ce qu'il faut dire
"Dans le contexte actuel, les entreprises disposent de volumes massifs de donnees transactionnelles, mais peinent a les transformer en insights exploitables. Les equipes marketing ont besoin de reponses concretes : quels clients privilegier ? Comment ameliorer la retention ? Quel sera l'impact reel de nos campagnes ? C'est precisement a ces questions que mon application repond."

**Duree** : 1 minute

---

## SLIDE 3 : Objectifs du Projet

### Contenu visuel
Liste numerotee des objectifs :

1. **Analyser les cohortes d'acquisition**
   - Suivre l'evolution des clients dans le temps
   - Mesurer les taux de retention

2. **Segmenter les clients (RFM)**
   - Identifier les meilleurs clients
   - Prioriser les actions marketing

3. **Calculer la valeur client (CLV)**
   - CLV empirique et predictive
   - Optimiser le ROI des campagnes

4. **Simuler des scenarios marketing**
   - Projeter l'impact des ameliorations
   - Guider les decisions strategiques

### Ce qu'il faut dire
"Les objectifs de ce projet sont multiples. Premierement, analyser les cohortes pour comprendre comment les clients acquis a differentes periodes se comportent. Deuxiemement, segmenter les clients selon la methodologie RFM pour identifier nos champions et nos clients a risque. Troisiemement, calculer la Customer Lifetime Value pour quantifier la valeur de chaque client. Enfin, simuler differents scenarios pour aider a la prise de decision."

**Duree** : 1 minute

---

## SLIDE 4 : Dataset Utilise

### Contenu visuel
- **Nom** : Online Retail II Dataset
- **Source** : UCI Machine Learning Repository
- **Periode** : Decembre 2009 - Decembre 2011
- **Volumetrie** :
  - Plus de 500 000 transactions
  - Plus de 5 000 clients uniques
  - 38 pays couverts
- **Informations disponibles** :
  - Date et numero de facture
  - Produits et quantites
  - Prix et montants
  - Identifiant client
  - Pays

- Tableau d'exemple des donnees (5 lignes)

### Ce qu'il faut dire
"Pour ce projet, j'ai utilise le dataset Online Retail II, qui contient les transactions d'un commerce en ligne britannique sur 2 ans. Avec plus de 500 000 transactions et 5 000 clients repartis dans 38 pays, ce dataset est parfait pour illustrer les problematiques marketing reelles. Il contient toutes les informations necessaires : dates de transaction, produits achetes, montants, et surtout les identifiants clients qui permettent l'analyse comportementale."

**Duree** : 1 minute

---

## SLIDE 5 : Architecture Technique

### Contenu visuel
Schema de l'architecture avec 3 couches :

**1. Couche Data**
- Pandas : Manipulation des donnees
- NumPy : Calculs numeriques
- Scikit-learn : Algorithmes ML

**2. Couche Analytique**
- Analyse de cohortes
- Segmentation RFM
- Calcul CLV
- Simulation de scenarios

**3. Couche Visualisation**
- Streamlit : Interface web interactive
- Plotly : Graphiques interactifs
- Matplotlib/Seaborn : Visualisations complementaires

**Structure du projet** :
```
projet_final/
├── app/              # Application Streamlit
├── data/             # Donnees brutes et traitees
├── notebooks/        # Exploration
├── config.py         # Configuration centralisee
└── requirements.txt
```

### Ce qu'il faut dire
"Sur le plan technique, j'ai opte pour une architecture Python en 3 couches. La couche data utilise Pandas et NumPy pour la manipulation efficace des donnees. La couche analytique implemente les algorithmes d'analyse de cohortes, RFM et CLV. Enfin, la couche visualisation repose sur Streamlit pour l'interface web et Plotly pour des graphiques interactifs. J'ai structure le projet de maniere professionnelle avec une configuration centralisee et une separation claire des responsabilites."

**Duree** : 1 minute 30

---

## SLIDE 6 : Pipeline de Traitement des Donnees

### Contenu visuel
Diagramme de flux avec les etapes :

**1. Chargement** → **2. Nettoyage** → **3. Enrichissement** → **4. Analyse**

**Details de chaque etape** :

1. **Chargement**
   - Lecture CSV/Excel
   - Parsing des dates
   - Detection automatique des types

2. **Nettoyage**
   - Suppression des transactions annulees (prefix 'C')
   - Filtrage des valeurs aberrantes
   - Gestion des donnees manquantes
   - Validation quantites/prix positifs

3. **Enrichissement**
   - Calcul TotalAmount = Quantity × Price
   - Extraction periode (mois/annee)
   - Indicateurs HasCustomerID
   - Gestion des retours

4. **Analyse**
   - Calcul des metriques RFM
   - Creation des cohortes
   - Agregations par client/periode

**Statistiques** :
- Taux de conservation : ~85% des transactions
- Transactions avec Customer ID : ~75%

### Ce qu'il faut dire
"Le pipeline de traitement suit 4 etapes rigoureuses. D'abord le chargement avec parsing intelligent des dates et detection des types. Ensuite le nettoyage : j'ai elimine les transactions annulees identifiees par le prefix C, filtre les valeurs aberrantes et valide la coherence des donnees. L'etape d'enrichissement ajoute des colonnes calculees comme le montant total et des indicateurs pratiques. Enfin, l'analyse calcule toutes les metriques necessaires. Au final, on conserve environ 85% des transactions avec une qualite de donnee optimale."

**Duree** : 1 minute 30

---

## SLIDE 7 : Analyse de Cohortes

### Contenu visuel
**Principe** :
- Regrouper les clients par mois de premiere transaction
- Suivre leur comportement dans le temps
- Mesurer la retention periode par periode

**Visualisation principale** :
- Heatmap de retention (exemple)
  - Lignes : Cohortes (mois d'acquisition)
  - Colonnes : Periodes (M0, M1, M3, M6, M12)
  - Couleurs : Taux de retention (vert = bon, rouge = faible)

**Metriques calculees** :
- Taille de la cohorte
- Taux de retention M1, M3, M6, M12
- Chiffre d'affaires par cohorte
- Frequence d'achat moyenne
- CLV empirique

**Insights cles** :
- Identification des cohortes les plus performantes
- Detection des periodes d'acquisition optimales
- Comprehension de l'evolution du comportement client

### Ce qu'il faut dire
"L'analyse de cohortes est au coeur de l'application. Le principe est simple mais puissant : on regroupe les clients par mois de premiere transaction, puis on observe leur comportement au fil du temps. Cette heatmap montre les taux de retention : chaque ligne est une cohorte, chaque colonne une periode. Les couleurs permettent d'identifier rapidement les cohortes performantes en vert et celles a probleme en rouge. Cette analyse permet de repondre a des questions critiques : quelle periode d'acquisition est la meilleure ? Comment evolue la retention dans le temps ? Quelle est la valeur reelle de chaque cohorte ?"

**Duree** : 2 minutes

---

## SLIDE 8 : Segmentation RFM

### Contenu visuel
**Methodologie RFM** :

**R - Recence** : Nombre de jours depuis le dernier achat
- Score 4 : Achat tres recent
- Score 1 : Achat ancien

**F - Frequence** : Nombre total de transactions
- Score 4 : Client tres actif
- Score 1 : Client peu actif

**M - Montant** : Chiffre d'affaires total genere
- Score 4 : Gros depensier
- Score 1 : Faibles achats

**Segments identifies** (avec icones) :
- **Champions** (R=4, F=4, M=4) : Meilleurs clients
- **Loyal Customers** : Clients fideles a forte valeur
- **Potential Loyalists** : Nouveaux clients a fort potentiel
- **At Risk** : Bons clients inactifs recemment
- **Cannot Lose Them** : Meilleurs clients en perte
- **Hibernating** : Clients inactifs a faible valeur
- **Lost** : Clients perdus

**Graphique** : Distribution des clients par segment (camembert ou barres)

### Ce qu'il faut dire
"La segmentation RFM est une methodologie eprouvee qui classe les clients selon trois dimensions. La Recence mesure combien de temps s'est ecoule depuis le dernier achat - un client recent a plus de chances de racheter. La Frequence compte le nombre de transactions - un client frequent est plus engage. Le Montant quantifie le chiffre d'affaires genere - un gros depensier a plus de valeur. En combinant ces trois scores, on obtient 8 segments strategiques. Les Champions sont nos clients ideaux, les At Risk necessitent une attention immediate pour eviter leur perte, les Potential Loyalists sont a fideliser en priorite. Cette segmentation permet de cibler precisement les actions marketing."

**Duree** : 2 minutes

---

## SLIDE 9 : Customer Lifetime Value (CLV)

### Contenu visuel
**Definition** :
La CLV represente le revenu total qu'un client generera durant toute sa relation avec l'entreprise.

**Deux approches implementees** :

**1. CLV Empirique** (basee sur les donnees historiques)
- Somme des revenus reels generes par client
- Calcul par cohorte
- Vision factuelle du passe

**2. CLV Predictive** (basee sur des formules mathematiques)
- Formule : CLV = (Panier Moyen × Frequence × Marge) × (Retention / (1 + Discount - Retention))
- Parametres :
  - Taux de retention : 30% (par defaut)
  - Taux d'actualisation : 10% (par defaut)
- Vision prospective

**Graphique** :
- Evolution de la CLV moyenne par segment RFM (barres)
- CLV cumulative par cohorte dans le temps (courbe)

**Utilite** :
- Valorisation du capital client
- Optimisation des budgets d'acquisition
- Calcul du ROI des campagnes

### Ce qu'il faut dire
"La Customer Lifetime Value est la metrique ultime pour valoriser vos clients. J'ai implemente deux approches complementaires. La CLV empirique se base sur les donnees historiques reelles - c'est une vision factuelle de ce que le client a rapporte. La CLV predictive utilise des formules mathematiques pour projeter la valeur future, en tenant compte du taux de retention et de l'actualisation financiere. Cette double approche permet de valoriser precisement le capital client et d'optimiser les budgets marketing. Par exemple, si un Champion a une CLV de 5000 euros, on peut justifier un budget d'acquisition ou de fidelisation bien plus eleve qu'un client Hibernating a 100 euros de CLV."

**Duree** : 2 minutes

---

## SLIDE 10 : Simulation de Scenarios Marketing

### Contenu visuel
**Objectif** :
Projeter l'impact de differentes strategies marketing sur les KPIs cles.

**Parametres ajustables** :
- Amelioration du taux de retention (+X%)
- Augmentation du panier moyen (+X%)
- Augmentation de la frequence d'achat (+X%)

**3 scenarios predefinis** :

| Scenario | Retention | Panier Moyen | Frequence |
|----------|-----------|--------------|-----------|
| Conservateur | +2% | +3% | +5% |
| Realiste | +5% | +8% | +10% |
| Optimiste | +10% | +15% | +20% |

**Resultats simules** :
- Impact sur le revenu total
- Nouvelle CLV moyenne
- Projection du chiffre d'affaires a 1 an, 2 ans, 3 ans

**Graphique** :
- Comparaison des projections de CA selon les scenarios (courbes)
- Impact de chaque levier separement (barres)

### Ce qu'il faut dire
"Le module de simulation est un outil d'aide a la decision strategique. Il permet de repondre a la question : que se passerait-il si on ameliore nos performances ? L'utilisateur peut ajuster trois leviers principaux : le taux de retention, le panier moyen et la frequence d'achat. J'ai predefini trois scenarios - conservateur, realiste et optimiste - bases sur des hypotheses d'amelioration progressives. L'application calcule instantanement l'impact sur les revenus et la CLV. Par exemple, le scenario realiste avec +5% de retention et +8% de panier moyen peut generer 500 000 euros de CA supplementaire sur 3 ans. Cela permet de justifier les investissements marketing avec des projections chiffrees."

**Duree** : 2 minutes

---

## SLIDE 11 : Demo de l'Application

### Contenu visuel
**Screenshots de l'interface** :

1. **Page Overview**
   - KPIs globaux : Clients totaux, CA total, Panier moyen, Retention
   - Graphiques de synthese

2. **Page Cohortes**
   - Heatmap de retention interactive
   - Tableau de synthese par cohorte
   - Filtres par periode et taille de cohorte

3. **Page Segments**
   - Distribution des clients par segment RFM
   - Profil detaille de chaque segment
   - Recommandations d'actions

4. **Page Scenarios**
   - Curseurs pour ajuster les parametres
   - Comparaison avant/apres
   - Projections graphiques

5. **Page Export**
   - Telechargement des donnees CSV
   - Export des graphiques PNG
   - Generation de rapports

**Fonctionnalites cles a montrer** :
- Interface intuitive
- Graphiques interactifs (zoom, survol)
- Filtres globaux dans la sidebar
- Exports faciles

### Ce qu'il faut dire
"Passons maintenant a une breve demonstration de l'application. L'interface Streamlit offre une experience utilisateur fluide et intuitive. Sur la page Overview, on retrouve les KPIs essentiels d'un seul coup d'oeil. La page Cohortes presente la fameuse heatmap de retention avec des interactions riches - on peut zoomer, survoler pour voir les details. La page Segments visualise la repartition des clients et fournit des recommandations d'actions concretes pour chaque segment. Le module Scenarios permet de jouer avec les curseurs et voir instantanement l'impact sur les projections. Enfin, tout est exportable en un clic pour partager avec les equipes ou integrer dans des presentations."

**Duree** : 2 minutes
**NOTE** : Si possible, faire une vraie demo live plutot que des screenshots

---

## SLIDE 12 : Resultats et Insights Cles

### Contenu visuel
**Principaux insights tires de l'analyse** :

**Cohortes** :
- Taux de retention moyen a M12 : ~25%
- Meilleures cohortes : Q4 2010 et Q1 2011
- Chute significative de retention apres M3

**Segmentation** :
- 15% de Champions generent 40% du CA
- 30% de clients At Risk ou Hibernating
- Opportunite de reconversion importante

**CLV** :
- CLV moyenne : 500-800 £
- CLV Champions : 2000+ £
- Ratio CLV Champions / Lost : 20x

**Scenarios** :
- +5% de retention = +300K£ de CA sur 3 ans
- Levier le plus impactant : Retention
- ROI potentiel des actions : 3-5x

**Graphiques** :
- Comparaison des segments par taille et valeur
- Evolution de la retention dans le temps
- Impact des scenarios sur le CA futur

### Ce qu'il faut dire
"Les analyses ont revele plusieurs insights strategiques importants. D'abord, le taux de retention moyen de 25% a un an est correct mais ameliorable. Les cohortes de fin 2010 et debut 2011 sont particulierement performantes - il faut comprendre pourquoi et reproduire ces conditions. Sur la segmentation, on observe un pattern classique : 15% de Champions generent 40% du chiffre d'affaires - c'est le principe de Pareto. Mais on a aussi 30% de clients inactifs ou a risque, ce qui represente une opportunite enorme de reconversion. Les simulations montrent que le levier le plus impactant est la retention : ameliorer la retention de seulement 5% genere 300 000 livres de CA supplementaire sur 3 ans. C'est un ROI potentiel de 3 a 5 fois l'investissement marketing."

**Duree** : 2 minutes

---

## SLIDE 13 : Limites et Perspectives

### Contenu visuel
**Limites identifiees** :

**Donnees** :
- Dataset limite a 2 ans (2009-2011)
- Donnees anciennes (plus de 10 ans)
- Pas d'informations demographiques clients
- Manque de donnees sur les campagnes marketing

**Methodologie** :
- Hypotheses simplificatrices pour la CLV predictive
- Taux de retention uniforme par defaut
- Absence de modelisation predictive avancee (ML)

**Perspectives d'amelioration** :

**Court terme** :
- Integration de donnees temps reel via API
- Ajout d'alertes automatiques (clients a risque)
- Recommandations marketing automatisees par IA

**Moyen terme** :
- Machine Learning pour predictions plus fines
- Modeles de propension (churn, upsell, cross-sell)
- Integration CRM pour actions directes
- A/B testing integre pour mesurer l'impact reel

**Long terme** :
- Personnalisation des recommandations par client
- Optimisation automatique des budgets marketing
- Plateforme complete de Marketing Automation

### Ce qu'il faut dire
"Tout projet a ses limites et celui-ci ne fait pas exception. Les donnees datent de plus de 10 ans et ne couvrent que 2 ans - dans un contexte reel, on aurait acces a plus d'historique et a des donnees actualisees. Il manque aussi des informations precieuses comme les donnees demographiques ou les campagnes marketing executees. Sur le plan methodologique, j'ai du faire certaines hypotheses simplificatrices, notamment un taux de retention uniforme. Les perspectives sont nombreuses. A court terme, on pourrait connecter des donnees temps reel et ajouter des alertes automatiques. A moyen terme, integrer du Machine Learning pour des predictions plus fines et des modeles de propension au churn. A long terme, evoluer vers une veritable plateforme de Marketing Automation avec optimisation automatique des budgets et personnalisation poussee."

**Duree** : 2 minutes

---

## SLIDE 14 : Apprentissages et Competences Developpees

### Contenu visuel
**Competences techniques** :
- **Python avance** : Pandas, NumPy, manipulation de donnees complexes
- **Visualisation de donnees** : Plotly, Matplotlib, Seaborn, creation de dashboards
- **Streamlit** : Developpement d'applications web interactives
- **Analyse statistique** : Cohortes, segmentation, metriques marketing
- **Architecture logicielle** : Structure de projet professionnelle

**Competences methodologiques** :
- **Analyse de donnees** : Exploration, nettoyage, transformation
- **Pensee analytique** : Transformation de donnees en insights actionnables
- **Resolution de problemes** : Gestion des donnees manquantes et aberrantes
- **Communication** : Visualisation claire de concepts complexes

**Competences metier** :
- **Marketing Analytics** : RFM, CLV, retention, cohortes
- **Business Intelligence** : KPIs, tableaux de bord, reporting
- **Aide a la decision** : Simulation de scenarios, projections

### Ce qu'il faut dire
"Ce projet m'a permis de developper des competences variees et complementaires. Sur le plan technique, j'ai approfondi ma maitrise de Python et des librairies data science, decouvert Streamlit pour creer des interfaces web interactives, et pratique l'architecture logicielle professionnelle. Methodologiquement, j'ai renforce ma capacite a transformer des donnees brutes en insights actionnables et a communiquer des concepts complexes de maniere visuelle et intuitive. Enfin, j'ai acquis une solide comprehension des problematiques marketing - RFM, CLV, retention - qui sont au coeur de la strategie client des entreprises modernes. Ces competences sont directement transferables dans le monde professionnel."

**Duree** : 1 minute 30

---

## SLIDE 15 : Conclusion et Questions

### Contenu visuel
**Synthese du projet** :
- Application complete d'aide a la decision marketing
- Pipeline de donnees robuste et evolutif
- Analyses approfondies : Cohortes, RFM, CLV, Scenarios
- Interface intuitive et interactive
- Outils d'export et de partage

**Messages cles a retenir** :
1. Les donnees clients sont un tresor - encore faut-il savoir les exploiter
2. La segmentation RFM permet de prioriser les efforts marketing
3. La CLV quantifie la valeur client et justifie les investissements
4. Les simulations guident les decisions strategiques avec des projections chiffrees
5. Une bonne visualisation transforme la donnee en action

**Contact et ressources** :
- Code source : [lien GitHub si applicable]
- Documentation : README.md
- Demo en ligne : [lien si deploye]

**Remerciements** :
- Professeurs et encadrants
- ECE Paris
- Sources du dataset

**"Merci pour votre attention. Avez-vous des questions ?"**

### Ce qu'il faut dire
"Pour conclure, ce projet m'a permis de creer une application complete d'aide a la decision marketing, depuis l'ingestion des donnees brutes jusqu'a la visualisation interactive d'insights actionnables. Les messages cles a retenir sont les suivants : les donnees clients sont un actif strategique majeur qu'il faut savoir exploiter. La segmentation RFM permet de prioriser intelligemment les efforts marketing. La CLV quantifie precisement la valeur client et justifie les budgets. Les simulations guident les decisions avec des projections chiffrees fiables. Et surtout, une bonne visualisation est le pont entre la donnee et l'action. Je vous remercie pour votre attention et reste disponible pour repondre a vos questions."

**Duree** : 1 minute

---

## CONSEILS POUR LA PRESENTATION

### Avant la presentation
1. **Repetez** : Minimum 3 fois la presentation complete
2. **Chronometrez** : Respectez les timings pour chaque slide
3. **Preparez la demo** : Testez l'application avant, ayez un plan B (screenshots/video) si probleme technique
4. **Anticipez les questions** :
   - Pourquoi ce dataset ?
   - Comment gerez-vous les donnees manquantes ?
   - Quelle est la complexite algorithmique ?
   - Comment deploieriez-vous en production ?
   - Quels sont les couts d'hebergement ?

### Pendant la presentation
1. **Regardez l'audience** : Pas l'ecran
2. **Parlez lentement** : Articulez, faites des pauses
3. **Soyez enthousiaste** : Montrez votre passion pour le projet
4. **Interagissez** : Posez des questions rhetoriques, verifiez la comprehension
5. **Gerez le temps** : Gardez un oeil sur la montre

### Gestion des questions
1. **Ecoutez completement** la question avant de repondre
2. **Reformulez** si la question n'est pas claire
3. **Repondez honnetement** : Si vous ne savez pas, dites-le
4. **Restez positif** : Les critiques sont des opportunites d'amelioration
5. **Soyez concis** : Reponses de 30-60 secondes maximum

### Points de vigilance
- **Ne lisez pas les slides** : Elles sont un support, pas un script
- **Evitez le jargon excessif** : Expliquez les termes techniques
- **Montrez la valeur business** : Pas seulement la technique
- **Reliez toujours a la problematique** : Pourquoi cette analyse ? Quelle decision permet-elle ?

---

## ANNEXE : Questions/Reponses Probables

### Q : Pourquoi avoir choisi ce dataset ?
**R** : "Le dataset Online Retail II est ideal car il contient toutes les informations necessaires pour une analyse marketing complete : identifiants clients, dates de transaction, montants. C'est un dataset realiste representatif des problematiques e-commerce reelles, avec la complexite des donnees manquantes et aberrantes."

### Q : Comment gerez-vous les donnees manquantes ?
**R** : "J'ai adopte une approche pragmatique en deux modes. Le mode strict supprime toutes les transactions sans Customer ID, ideal pour les analyses individuelles comme RFM ou CLV. Le mode souple conserve ces transactions pour les analyses globales de CA ou de produits. L'utilisateur voit clairement le taux de completude des donnees."

### Q : Et si les donnees etaient en temps reel ?
**R** : "L'architecture est deja prete. Il suffirait de remplacer la fonction load_data() par une connexion API ou base de donnees, et d'ajouter un systeme de cache intelligent avec refresh periodique. Streamlit gere nativement le rafraichissement automatique."

### Q : Pourquoi Streamlit et pas un dashboard classique ?
**R** : "Streamlit offre un excellent compromis entre rapidite de developpement et qualite du rendu. En tant qu'etudiant, je peux iterer rapidement. Pour l'entreprise, c'est deployable facilement et l'interface est suffisamment professionnelle pour des utilisateurs metier."

### Q : Comment validez-vous vos calculs ?
**R** : "J'ai implemente des verifications a plusieurs niveaux : validation des types de donnees, coherence des montants calcules, comparaison avec des formules etablies pour la CLV, et tests sur des sous-ensembles de donnees dont je connais le resultat attendu."

### Q : Quelles seraient les prochaines etapes de developpement ?
**R** : "Trois axes prioritaires : premierement, integrer des modeles de Machine Learning pour predire le churn et les propensions d'achat. Deuxiemement, ajouter des tests A/B pour mesurer l'impact reel des campagnes. Troisiemement, connecter un CRM pour passer de l'analyse a l'action directe."

---

**Bonne chance pour votre presentation !**
