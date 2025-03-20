# Modules Principaux du Projet

Ce dossier regroupe les deux modules essentiels qui orchestrent l'exécution du projet de trading automatisé. Ces modules pilotent, d'une part, l'ensemble du pipeline de backtesting et d'optimisation des modèles, et d'autre part, la prise de trades en temps réel.

---

## Modules Présents

- **main.py**  
  Ce module constitue le point d'entrée principal pour le backtesting et l'optimisation des modèles. Il orchestre l'ensemble du pipeline, qui inclut :
  - La création des répertoires pour les données brutes, prétraitées, évolutives, et les modèles.
  - Le téléchargement et le prétraitement des données (via les modules du dossier *data*).
  - L'ajout des features et l'application de la stratégie (via les modules du dossier *features*).
  - La préparation finale des données évolutives pour l'entraînement.
  - La recherche de la meilleure combinaison de features et de modèles, ainsi que l'optimisation des hyperparamètres (via les modules du dossier *models*).
  
  **Rôle principal :**  
  Piloter l'ensemble du processus de traitement des données, de création de features, et d'optimisation de la modélisation, en réalisant notamment du backtesting sur des périodes historiques.

- **automatique_trade.py**  
  Ce module gère l'exécution automatique des trades en temps réel. Il se connecte à l'API de trading et orchestre :
  - La normalisation et l'encodage des données en temps réel.
  - La prise de décision pour l'ouverture ou la fermeture d'ordres, en s'appuyant sur les signaux générés par les modèles.
  - La communication avec l'API (via WebSocket) pour envoyer les commandes de trading.
  
  **Rôle principal :**  
  Exécuter les trades en temps réel en appliquant la logique de prise de décision basée sur les prédictions et signaux des modèles, tout en assurant la gestion des positions ouvertes.

---

## Remarques

- **Intégration et Flux d'Exécution :**  
  - *main.py* est utilisé pour exécuter l'ensemble du pipeline de backtesting et d'optimisation, en passant par le téléchargement des données, l'ajout des features, la préparation des données évolutives, et l'optimisation des modèles via Optuna.
  - *automatique_trade.py* intervient en temps réel pour surveiller les signaux du marché et exécuter les ordres de trading (ou les fermer) en fonction des conditions définies.


- **Utilisation et Tests :**  
  - Pour tester le pipeline complet, lancez *main.py* qui exécutera l'ensemble du processus de préparation et d'optimisation sur des données historiques.
  - Pour passer en production ou en simulation de trading en temps réel, utilisez *automatique_trade.py* qui est conçu pour interagir directement avec l'API de trading.

---

