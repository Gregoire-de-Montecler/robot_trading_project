# Dossier Features

Ce dossier regroupe l'ensemble des modules responsables de la création, du traitement et de la combinaison des features (caractéristiques) issues des données financières. Ces features servent à enrichir les données pour l'entraînement des modèles de Machine Learning et l'analyse technique.

---

## Modules Présents

- **feature_combination.py**  
  Ce module orchestre l'ajout et la combinaison des features. Il intègre les résultats de plusieurs fonctions de feature engineering (issues de *feature_engineering.py*) et combine également des informations provenant d'indicateurs techniques et d'autres sources (via *technical_indicators.py*).  
  **Rôle principal :** Fusionner et harmoniser les différentes features calculées sur plusieurs timeframes.

- **feature_engineering.py**  
  Ce module se charge d'ajouter et de nettoyer les features issues des indicateurs techniques standards. Il fait appel à plusieurs fonctions (par exemple, le calcul du RSI, MACD, stochastiques, etc.) afin d'enrichir le DataFrame de données financières.  
  **Rôle principal :** Générer et préparer des features pertinentes à partir des données brutes.

- **technical_indicators.py**  
  Ce module implémente le calcul d'indicateurs techniques communs (RSI, MACD, stochastiques, moyennes mobiles, pivots, etc.).  
  **Rôle principal :** Fournir des indicateurs techniques standards pour l'analyse des tendances et des patterns du marché. 

- **strategie.py**  
  Ce module contient la logique propriétaire de ma stratégie de trading et a été volontairement masqué pour protéger mon savoir-faire spécifique.  
  **Rôle principal :** Appliquer des règles personnalisées pour générer des signaux ou ajuster les features en fonction de la stratégie de trading personnelle.

---

## Remarques

- **Sécurité et Confidentialité :**  
  Pour protéger la valeur stratégique de mon savoir-faire, le module *strategie.py* est exclu du dépôt public.  

- **Intégration des Modules :**  
  Les modules de ce dossier sont interconnectés :  
  - *feature_engineering.py* et *technical_indicators.py* fournissent des fonctions de calcul qui sont ensuite utilisées par *feature_combination.py*.  
  - *feature_combination.py* peut également faire appel à des parties de *strategie.py* (lorsque le mode de fonctionnement le permet) pour appliquer des ajustements spécifiques à la stratégie.

---



