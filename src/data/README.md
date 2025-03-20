# Dossier de Gestion des Données

Ce dossier regroupe les modules chargés de récupérer, transformer et préparer les données de trading pour le projet. Il permet de constituer la base des données nécessaires à l'entraînement des modèles de Machine Learning et à l'analyse.

---

## Contenu du dossier

- **data_downloader.py**  
  Ce module se charge du téléchargement des données via l’API XTB.  
  **Responsabilités principales :**
  - Établir une connexion WebSocket et s'authentifier avec les identifiants sécurisés.
  - Envoyer des commandes pour récupérer des chandeliers (candles) sur plusieurs timeframes (15min, h, 4h, D).
  - Transformer les données JSON reçues en DataFrame.
  - Fusionner les données téléchargées avec d’éventuelles données existantes.
  - Gérer les erreurs et sauvegarder temporairement les réponses en format JSON.

- **prepare_evolving_data.py**  
  Ce module prépare les données brutes pour qu'elles puissent être utilisées dans la phase d'entraînement.  
  **Responsabilités principales :**
  - Charger et concaténer plusieurs fichiers CSV contenant les données brutes.
  - Trier les données de manière chronologique.
  - Réaliser un encodage one-hot sur la colonne `asset` pour extraire les devises.
  - Ajuster les timestamps pour garantir leur unicité (via une fonction de décalage).
  - Sauvegarder le résultat final sous forme de fichier CSV pour la suite du traitement.

- **time_transformations.py**  
  *Note : Ce module contient des fonctions spécifiques de transformation temporelle qui font partie de la stratégie confidentielle. Pour des raisons de confidentialité, son contenu n’est pas exposé publiquement.*

---

## Vue d'ensemble du processus

1. **Téléchargement des données**  
   Le module `data_downloader.py` se connecte à l’API XTB, récupère les données sous différents timeframes et les transforme en DataFrame. Les données brutes sont ensuite sauvegardées temporairement au format JSON.

2. **Prétraitement des données**  
   Le module `prepare_evolving_data.py` charge ces fichiers, les concatène, applique des transformations (encodage, ajustement des timestamps) et sauvegarde le résultat final sous forme de CSV. Ce fichier est ensuite utilisé pour l'entraînement et l'évaluation des modèles.

3. **Gestion des transformations spécifiques**  
   Le module `time_transformations.py` intervient dans le processus de prétraitement pour effectuer des ajustements temporels particuliers. Son contenu a été volontairement masqué pour protéger le cœur de la stratégie de trading.

---

Ce dossier illustre l'intégration des opérations de téléchargement et de transformation des données, essentielles pour alimenter le pipeline de trading automatisé.

