# Dossier Utils

Ce dossier regroupe l'ensemble des modules utilitaires utilisés dans le projet. Ces modules fournissent des fonctions communes pour le logging, la gestion des fichiers, les calculs mathématiques spécifiques, le parallélisme, la gestion du temps, la configuration du projet et la manipulation des données. Ils servent de base aux opérations partagées par l'ensemble du code.

---

## Modules Présents

- **file_utils.py**  
  Ce module contient des fonctions pour la gestion des fichiers et des dossiers, telles que la création de répertoires et la manipulation de fichiers temporaires.  
  **Rôle principal :** Faciliter la gestion des opérations sur le système de fichiers (création, suppression, déplacement de dossiers, etc.).

- **logger.py**  
  Ce module fournit une configuration centralisée pour le logging du projet.  
  **Rôle principal :** Configurer et retourner des loggers avec des handlers adaptés pour la console et les fichiers, afin d'assurer un suivi détaillé des opérations.

- **math_utils.py**  
  Ce module contient des fonctions pour effectuer des calculs mathématiques spécifiques aux séries temporelles, comme le calcul de la pente et la différence en pourcentage.  
  **Rôle principal :** Fournir des outils de calcul pour l'analyse et l'évaluation des données.

- **mutliprocessing_utils.py**  
  Ce module offre des fonctions utilitaires pour la gestion du parallélisme via multiprocessing et multithreading.  
  **Rôle principal :** Permettre l'exécution parallèle de fonctions sur des listes d'arguments pour accélérer les traitements.

- **time_utils.py**  
  Ce module fournit des fonctions pour la gestion et le formatage du temps, la conversion de timestamps Unix en datetime, le calcul du temps écoulé, et le découpage de séries temporelles.  
  **Rôle principal :** Manipuler et formater les informations temporelles pour une utilisation cohérente dans le projet.

- **config.py**  
  Ce module charge et expose la configuration du projet à partir de fichiers YAML, en définissant notamment les chemins, les paramètres généraux et les constantes.  
  **Rôle principal :** Centraliser et standardiser la configuration du projet pour faciliter son adaptation et sa maintenance.

- **data_utils.py**  
  Ce module contient des fonctions utilitaires pour la manipulation et la transformation des données, comme la conversion de fichiers JSON en DataFrame, l'ajustement des colonnes de prix et la concaténation de DataFrames.  
  **Rôle principal :** Simplifier la préparation et la manipulation des données brutes pour l'ensemble du projet.

---

## Remarques

- **Centralisation et Réutilisation :**  
  Les modules de ce dossier sont conçus pour être largement réutilisables dans l'ensemble du projet. Ils fournissent des fonctionnalités communes qui facilitent le développement et la maintenance du code.

- **Interconnexion :**  
  Certains modules, comme `logger.py` et `config.py`, sont essentiels pour la configuration et le suivi du projet et sont utilisés par la majorité des autres modules.

