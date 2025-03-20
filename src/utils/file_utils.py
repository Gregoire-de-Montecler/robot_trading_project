
"""
Module Description:
-------------------
Décrire brièvement le but du fichier ou des fonctions qu'il contient.
Exemple : Ce module contient des fonctions pour [prétraitement, visualisation, etc.].
"""

# === Imports ===
# 1. Modules natifs Python
import os
import logging
import shutil
import tempfile
# 2. Bibliothèques externes installées via pip
import json
import pandas as pd



# 3. Modules internes du projet
from utils.logger import setup_logger
from utils.config import  Config

# === Constantes globales ===
seed = Config.seed
# === Configuration du logger ===
logger = setup_logger("utils","file_utils")
logger.setLevel(logging.ERROR)

# === Classes ===



# === Fonctions ===

def create_directory(directory_path: str) -> None:
    """
    Crée un dossier si il n'existe pas déjà. Enregistre les informations dans les logs.

    Paramètres :
    ----------
    directory_path : str
        Le chemin du dossier à créer.

    Retour :
    --------
    None
        Cette fonction ne retourne rien. Elle se contente de créer le dossier et d'enregistrer des informations dans les logs.
    
    Enregistre dans les logs :
    --------------------------
    - Si le dossier est créé avec succès.
    - Si le dossier existe déjà.
    - En cas d'erreur lors de la création du dossier.
    """
    try:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            logger.info(f"Dossier '{directory_path}' créé avec succès.")
        else:
            logger.info(f"Dossier '{directory_path}' déjà existant.")
    except Exception as e:
        logger.error(f"Erreur lors de la création du dossier '{directory_path}': {e}")
        raise RuntimeError(f"Impossible de créer le dossier '{directory_path}'.") from e


    
# === Point d'entrée pour les tests ===
if __name__ == "__main__":
    import shutil
    import tempfile

    temp_dir = tempfile.gettempdir()
    joblib_temp_dir = [d for d in os.listdir(temp_dir) if "joblib_memmapping_folder" in d]

    for d in joblib_temp_dir:
        shutil.rmtree(os.path.join(temp_dir, d), ignore_errors=True)
    pass
