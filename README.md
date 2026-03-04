# 📈 QRT Data Challenge — Prédiction de Rendements Boursiers

**M2 ISIFAR · Université Paris Cité · 2025**

Ce projet s'inscrit dans le cadre du QRT/ENS Data Challenge. L'objectif est de prédire les rendements futurs de 50 actions à partir de leur historique sur 753 jours de trading. Plutôt que d'appliquer une seule méthode, j'ai choisi de comparer trois familles d'approches très différentes — ce qui permet de mieux comprendre pourquoi la prédiction de rendements est un problème difficile.



## 📦 Données

- `X_train` : rendements journaliers de 50 actions sur 754 jours (features)
- `Y_train` : rendements cibles sur 504 jours (labels)
- La fenêtre de prédiction est de 250 jours : pour prédire le rendement au jour *t*, on utilise les 250 jours précédents.



## 🔍 Analyse exploratoire

Avant de modéliser, j'ai d'abord regardé les données de près :

- Les rendements suivent une distribution quasi-normale, légèrement leptokurtique (queues épaisses)
- Les tests ADF montrent que **49 actions sur 50 sont stationnaires** (I(0)) — les séries se comportent comme un bruit blanc
- La matrice de corrélation révèle quelques clusters d'actions fortement liées

Ce constat initial est important : si les séries sont proches d'un bruit blanc, les modèles de séries temporelles classiques vont avoir du mal.



## ⚙️ Méthodes comparées

### 1. Facteurs orthonormaux sur la variété de Stiefel

L'idée centrale est de chercher une matrice *A* orthonormale (A⊤A = I) qui projette la fenêtre de 250 jours de rendements en un petit vecteur de facteurs décorrélés. Ces facteurs sont ensuite utilisés pour prédire le rendement futur.

On a testé deux variantes :

- **V1 — beta vecteur partagé** : un seul vecteur β pour toutes les actions. Simple mais peu flexible.
- **V2 — beta matrice par action** : chaque action a ses propres coefficients. Plus expressif, gain significatif sur la métrique.

La recherche de la meilleure matrice A se fait par tirage aléatoire (Monte Carlo sur la variété de Stiefel), ce qui donne de bons résultats même sans gradient.

### 2. ARIMA / VAR-PCA

Approche classique des séries temporelles :

- **ARIMA** : un modèle par action, avec sélection automatique des ordres (p, d, q) par AIC
- **VAR-PCA** : on réduit d'abord la dimension à k composantes principales (80% de variance expliquée), puis on estime un VAR sur ces composantes

Résultat : les deux modèles produisent une métrique cosinus proche de zéro. Le critère AIC sélectionne un lag de 0 pour le VAR, ce qui confirme qu'il n'y a pas de structure autocorrélée exploitable dans les rendements. C'est cohérent avec l'hypothèse d'efficience des marchés.

### 3. Random Forest avec features de momentum

Seul modèle non linéaire du projet. Plutôt que d'utiliser les rendements bruts, on construit des features résumant chaque fenêtre de 250 jours :

| Feature | Description |
|||
| Moyenne 250j | Tendance longue période |
| Volatilité 250j | Niveau de risque |
| Moyenne 20j | Tendance récente |
| Moyenne 5j | Court terme |
| Moyenne 1j | Signal immédiat |
| Corrélation moyenne | Dépendance avec les autres actions |

Le Random Forest est entraîné en multi-output (50 actions simultanément) et évalué sur un test set indépendant.



## 📊 Résultats

| Méthode | Métrique cosinus | Évaluation |
||||
| Stiefel V1 | ~0.043 | Train |
| Stiefel V2 | ~0.145 | Train |
| ARMA/ARIMA | -0.006 | Test |
| VAR-PCA | -0.034 | Test |
| Random Forest | > 0 | Test |

La métrique cosinus mesure l'alignement directionnel entre les rendements prédits et réels. Une valeur positive signifie que le modèle prédit la bonne direction en moyenne.

Les métriques faibles ne signifient pas que les modèles sont mal construits — elles reflètent la difficulté intrinsèque de la prédiction de rendements financiers. Le Random Forest est le seul modèle évalué sur test avec un signal positif réel, porté principalement par les features de momentum court terme (1j, 5j).



## 🚀 Pistes d'amélioration

- Optimisation guidée sur la variété de Stiefel par **gradient riemannien** plutôt que Monte Carlo
- Modèle **LSTM** pour capturer des dépendances temporelles longues
- Features supplémentaires : volatilité réalisée, ratio de Sharpe glissant, bêta de marché



## 🛠️ Stack technique

```
Python · NumPy · pandas · scikit-learn · statsmodels · matplotlib · seaborn
```

## 📁 Structure

```
qrt-challenge/
├── Final_projet_code.ipynb   # Notebook principal (analyse complète)
└── README.md
```
