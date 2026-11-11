# QRT Data Challenge : prédiction de rendements d'actions

Projet réalisé en M2 ISIFAR (Université Paris Cité) à partir du data challenge proposé par QRT et l'ENS. Il s'agit de prédire les rendements journaliers de 50 actions à partir de leur historique. J'ai comparé trois familles de méthodes, ce qui permet aussi de voir à quel point le problème est difficile.

Tout le code est dans `Final_projet_code.ipynb`.

## Données

- `X_train` : rendements journaliers de 50 actions sur 754 jours.
- `Y_train` : rendements à prédire, sur 504 jours.
- Pour prédire le rendement du jour $t$, on utilise les 250 jours précédents.

Pour les méthodes 2 et 3, les 504 jours à prédire sont coupés en deux : 403 jours d'entraînement et 101 jours de test.

## Analyse exploratoire

- Les rendements ont une distribution proche de la normale, avec des queues un peu plus épaisses.
- D'après les tests ADF et KPSS, 49 séries sur 50 sont stationnaires.
- 82 paires d'actions ont une corrélation supérieure à 0.5 en valeur absolue.

Des séries stationnaires et proches d'un bruit blanc laissent peu de place aux modèles de séries temporelles classiques, ce que la suite confirme.

## Méthode 1 : facteurs orthonormaux (variété de Stiefel)

On cherche une matrice $A$ de taille 250 x 10, à colonnes orthonormées ($A^\top A = I$), qui résume la fenêtre des 250 derniers rendements en 10 facteurs. On régresse ensuite le rendement du lendemain sur ces facteurs.

La qualité d'une prédiction est mesurée par la similarité cosinus entre le vecteur des rendements prédits et celui des rendements réels, moyennée sur les jours :

$$M = \frac{1}{T} \sum_{t=1}^{T} \frac{\langle S_t, R_t \rangle}{\lVert S_t \rVert\, \lVert R_t \rVert}$$

Elle est comprise entre -1 et 1, et elle est positive quand le modèle trouve en moyenne la bonne direction.

Les matrices $A$ sont tirées au hasard (100 tirages, orthonormalisés par Gram-Schmidt) et on garde la meilleure. Deux variantes :

- V1 : les mêmes coefficients pour toutes les actions.
- V2 : des coefficients propres à chaque action.

Ces scores sont calculés sur les données d'entraînement, et V2 a beaucoup plus de paramètres que V1. Ils ne sont donc pas comparables aux scores de test des autres méthodes.

## Méthode 2 : ARMA et VAR

- ARMA : un modèle par action, avec des ordres $p$ et $q$ entre 0 et 2 choisis par AIC. La seule série non stationnaire est différenciée une fois.
- VAR sur ACP : on réduit les 50 séries à 19 composantes principales (80 % de la variance), puis on estime un VAR sur ces composantes.

Pour le VAR, les critères AIC, BIC et HQ choisissent tous un retard de 0 : les rendements passés n'apportent rien. On impose un retard de 1 pour pouvoir prédire. Les deux modèles donnent une métrique cosinus proche de zéro sur le test.

## Méthode 3 : forêt aléatoire

Pour chaque action, on résume la fenêtre de 250 jours en 6 variables :

| Variable | Description |
|---|---|
| Moyenne 250 j | tendance longue |
| Volatilité 250 j | niveau de risque |
| Moyenne 20 j | tendance récente |
| Moyenne 5 j | très court terme |
| Rendement de la veille | dernier mouvement |
| Corrélation moyenne | lien avec les autres actions |

Cela donne 300 variables (6 x 50). Une forêt de 300 arbres (profondeur 6) est entraînée pour chaque action. Les variables les plus utilisées sont le rendement de la veille (33 %) et la moyenne sur 5 jours (24 %).

Sur le test, la forêt n'a cependant aucun pouvoir prédictif : la métrique cosinus vaut -0.001 et le R² moyen est négatif.

## Résultats

| Méthode | Métrique cosinus | Évaluée sur |
|---|---|---|
| Stiefel V1 | 0.042 | entraînement |
| Stiefel V2 | 0.137 | entraînement |
| ARMA | -0.006 | test |
| VAR sur ACP | -0.034 | test |
| Forêt aléatoire | -0.001 | test |

Aucune méthode ne bat le hasard sur le test. Les scores de la méthode Stiefel sont positifs, mais ils sont calculés sur l'entraînement et sont donc optimistes. Ces résultats sont cohérents avec ce qu'on sait des rendements journaliers, qui sont très proches d'un bruit blanc.

## Pistes

- Évaluer la méthode Stiefel sur le même jeu de test que les autres.
- Remplacer la recherche aléatoire par une optimisation par gradient sur la variété de Stiefel.
- Ajouter d'autres variables (volatilité réalisée, bêta de marché).

Bibliothèques : numpy, pandas, scikit-learn, statsmodels, matplotlib, seaborn.
