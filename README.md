# Pyck-Win
Projet Datascientest : Paris sportifs - Battre les bookmakers

Ce projet a pour objectif de créer un algorithme de machine learning ou Deep learning visant à prédire l'issue d'un match avec une plus haute précision que les bookmakers à l'origine des côtes pouvant être observées sur les sites ou applications dédiés aux paris sportifs.

Les données étudiées proviennent de la *Premier League* de football (Grande-Bretagne) sur les saisons allant de 2013-2014 à 2017-2018.



## Bases de données :
La base de données finale est composée de 3 fichiers dont les sources sont :
  - https://datahub.io/sports-data/english-premier-league#data-cli
  - https://www.kaggle.com/shubhmamp/english-premier-league-match-data


Les 3 fichiers sont pour chaque saison :
  - 'season_match_stats.json' : Contient tous les matchs de la saison (id match, id/noms d'équipes, date, score  à mi-temps, score final).
  - 'season_stats.json' : Contient de nombreuses statistiques de jeux (possession, passes réussies, etc.).
  - 'bookmakers.csv' : Contient les cotes générées par différents bookmakers.



## Traitement et nettoyage de la base de données :
1) Fusion des season_match_stats.json et season_stats.json,
2) Ajout d'une colonne 'winner' contenant le gagnant du match (équipe à domicile, égalité, équipe extérieure),
3) Réorganisation des colonnes.
4) Date mise sous format datetime<br/><br/>

Les statistiques de matchs gardées sont :
  - La possession,
  - Le nombre de buts marqués,
  - Le nombre de points marqués,
  - Le nombre de buts encaissés,
  - Le nombre de tirs cadrés,
  - Le nombre de tirs hors cadres,
  - Le nombre de tirs bloqués,
  - La note moyenne attribuée.<br/><br/>

5) Fusion sur les colonnes d'identification des matchs de bookmakers.csv (Date, noms d'équipe),
6) Suppression des colonnes considérées non influentes sur les modèles (Division),
7) Suppression des colonnes redondantes de bookmakers (score final, score à mi-temps),
9) Détermination des meilleurs bookmakers pour chaque type de paris (équipe à domicile gagnante, égalité, équipe extérieure gagnante),
10) Dichotomisation des noms des équipes.

## Visualisation des données :

Les visualisations suivantes permettent de comprendre au mieux le jeu de données étudié.


### Statistiques de matchs :
![Correlation entre possession et taux de victoire](https://user-images.githubusercontent.com/84863172/126036613-4966131a-144c-41bf-b895-ee541e70a1ac.png)
\
On remarque que la possession est un facteur important dans la prédiction de l'issu d'un match de football.

![repartition victoire à domicile-extérieur2](https://user-images.githubusercontent.com/84863172/126036616-d2e4f492-a09f-4d39-b953-e519c3e7277b.png)
\
De même le fait de jouer à domicile augmentent les chances de gagner un match.


### Cotes de bookmakers :
![repartition bookmakers](https://user-images.githubusercontent.com/84863172/126036620-20680ee6-e411-4d8b-b6b1-3d32173fb9fc.png)
\

Le bookmaker effectuant les meilleures prédictions est le bookmaker nommé PSC.

![repartition cotes H,D,A](https://user-images.githubusercontent.com/84863172/126036623-b95f22ef-5f5e-4537-855b-5c577f7774bd.png)
\
On remarque à bon escient que les bookmakers ont tendances à mettre des côtes moins élevées sur les victoires à domiciles.


## Séparation du jeu de données :

Les variables explicatives contiennent les statistiques choisies ainsi que les noms d'équipes dichotomisés.
La variable cible est l'issue du match à prédire.


Le jeu de données est séparé de manière à garder l'ordre des matchs simulant ainsi la condition réelle du parieur.



## Standardisation :
Les données sont normalisées transformant chaque statistique en des valeurs comprises entre 0 et 1 permettant ainsi de traiter des données de même ordre de grandeur.


## Modèles étudiés :
### SVM :

Paramètres d'optimisation :

![parameters_svm](https://user-images.githubusercontent.com/84863172/126036461-8e985a19-b321-4ae3-9373-b42c5d8e0381.PNG)

La métrique choisit est la précision car nous nous intéressons principalement à la classe des vrais positifs.
Avant optimisation :<br/>
Précision sur le jeu de train :  0.671<br/>
Précision sur le jeu de test :  0.592

Après optimisation : <br/>
Précision sur le jeu de train :  0.671<br/>
Précision sur le jeu de test :  0.592


### Random Forest Classifier


### XGBoost



### Simulation de paris
Simuler un pari permet de voir s'il y a gain ou perte à l'issue un match sur lequel l'algorithme a effectué une prédiction.
La simulation mise sur l'issue ayant la plus grande probabilité et avec un seuil minimum de 60%.

![res-simus](https://user-images.githubusercontent.com/84863172/126037150-f82e3ce5-0c2f-4cbc-afd3-3fae0dfa23b5.PNG)



