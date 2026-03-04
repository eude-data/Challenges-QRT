# 📈 QRT Data Challenge — Prédiction de Rendements Boursiers

> M2 ISIFAR · Université Paris Cité · 2025

## Objectif

Prédire les rendements futurs de 50 actions à partir de leur historique (753 jours), en comparant trois familles de méthodes quantitatives.

## Méthodologie

| Méthode | Approche | Évaluation |
|---|---|---|
| **Stiefel V1** | Facteurs orthonormaux — beta vecteur partagé | Train |
| **Stiefel V2** | Facteurs orthonormaux — beta matrice par action | Train |
| **ARMA/ARIMA** | Modèles séries temporelles par action | Test |
| **VAR-PCA** | VAR multivarié avec réduction dimensionnelle | Test |
| **Random Forest** | Approche non linéaire avec features de momentum | Test |

## Résultats

| Méthode | Métrique cosinus |
|---|---|
| Stiefel V1 | ~0.043 |
| Stiefel V2 | ~0.145 |
| ARMA | -0.006 |
| VAR-PCA | -0.034 |
| Random Forest | > 0 |

La méthode Stiefel V2 obtient les meilleurs résultats. Le Random Forest est le seul modèle non linéaire évalué sur un test set indépendant avec un signal directionnel positif.

## Features construites (Random Forest)

- Moyenne 250j, Volatilité 250j
- Momentum 20j, 5j, 1j
- Corrélation moyenne avec les autres actions

## Stack

![Python](https://img.shields.io/badge/Python-3.10-blue)
![NumPy](https://img.shields.io/badge/NumPy-orange)
![scikit--learn](https://img.shields.io/badge/scikit--learn-green)
![statsmodels](https://img.shields.io/badge/statsmodels-yellow)

```python
numpy · pandas · scikit-learn · statsmodels · matplotlib · seaborn
```

## Structure

```
📁 qrt-challenge/
├── Final_projet_code.ipynb   # Notebook principal
└── README.md
```
