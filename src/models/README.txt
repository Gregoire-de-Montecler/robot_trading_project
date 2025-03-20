# Dossier Models

Ce dossier regroupe l'ensemble des modules responsables de la modélisation, de l'optimisation des hyperparamètres et de la sélection des modèles pour le projet de trading automatisé. Ces modules exploitent Optuna pour optimiser la performance des modèles et orchestrer l'entraînement, la validation croisée et l'évaluation des résultats.

---

## Modules Présents

- **ml_modeling.py**  
  Ce module constitue le cœur de la modélisation. Il contient :
  - Une **classe principale de modélisation** (*Modelisation*) qui définit les méthodes communes pour l'entraînement, la prédiction et l'évaluation des modèles.
  - Des **classes dérivées** pour la régression (*Regressor*) et la classification (*Classifier*), qui préparent les données cibles et ajustent les spécificités de chaque type de modèle.
  - Des fonctions utilitaires pour la **normalisation** des données (via un scaler), la **validation croisée** et la gestion de l'optimisation multi-pli.
  - Une intégration avec Optuna pour effectuer des recherches d'hyperparamètres et sélectionner les meilleures configurations en combinant la sélection des features et des modèles.
  
  **Rôle principal :**  
  Orchestrer l'ensemble du processus de modélisation et d'évaluation des modèles de Machine Learning, depuis le prétraitement des données jusqu'à l'obtention d'un score d'évaluation global sur des validations croisées. Ce module assure également le suivi et la log des performances lors des expérimentations.

- **optuna_feature.py**  
  Ce module propose des fonctions pour la sélection dynamique de features via Optuna.  
  **Rôle principal :** Optimiser la sélection des features à utiliser dans la modélisation en testant différentes combinaisons définies dans des groupes prédéfinis.

- **optuna_hyperparametre.py**  
  Ce module gère la suggestion et l'optimisation des hyperparamètres des modèles à l'aide d'Optuna.  
  **Rôle principal :** Générer et valider des combinaisons d'hyperparamètres pour améliorer les performances des modèles, tout en éliminant les configurations incohérentes grâce à des règles de pruning.

- **optuna_model.py**  
  Ce module contient des fonctions pour la sélection du modèle à utiliser ainsi qu'un dictionnaire associant des noms de modèles aux classes correspondantes (par exemple, CatBoostClassifier, XGBClassifier, etc.).  
  **Rôle principal :** Permettre la sélection et l'optimisation du modèle le plus adapté via Optuna.

---

## Remarques


- **Intégration des Modules :**  
  Les modules de ce dossier sont interconnectés :
  - *ml_modeling.py* orchestre l'entraînement, la validation et la prédiction des modèles en se basant sur les paramètres optimisés.
  - *optuna_feature.py*, *optuna_hyperparametre.py* et *optuna_model.py* collaborent pour optimiser globalement le pipeline de modélisation, en sélectionnant les meilleures features, hyperparamètres et modèles.

