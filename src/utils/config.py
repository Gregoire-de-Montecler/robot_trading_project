
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
import random

# 2. Bibliothèques externes installées via pip

import numpy as np

from ruamel.yaml import YAML
# 3. Modules internes du projet
from utils.logger import setup_logger

# === Constantes globales ===
CONFIG_PATH = os.path.join(os.getcwd(),"config","config.yaml")
MODEL_PATH = os.path.join(os.getcwd(),"config","model.yaml")

yaml = YAML()
# === Configuration du logger ===
logger = setup_logger("utils", "config")
logger.setLevel(logging.DEBUG)

# === Classes ===



# === Fonctions ===

def load_config(file_path):
    """Charge le fichier YAML et retourne son contenu."""
    with open(file_path, "r") as file:
        return yaml.load(file)

class Config():
    # Charger les configurations
    config = load_config(CONFIG_PATH)

    # Exposer les constantes comme variables globales
    asset_type = config["settings"]["asset_type"]
    path = os.path.expanduser(config["settings"]["base_path"])
    raw_path = os.path.join(path, "data", "raw", asset_type)
    processed_path = os.path.join(path, "data", "processed", asset_type)
    evolving_path = os.path.join(path, "data", "evolving", asset_type)
    scaler_path = os.path.join(path, "scalers", asset_type)
    downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
    models_path = os.path.join(path, "models", asset_type)
    results_csv_path = os.path.join(path, "results", "csv",asset_type)
    results_image_path = os.path.join(path, "results", "image",asset_type)
    results_json_path = os.path.join(path,"results","json",asset_type)
    storage_path = os.path.join(path,"results","optnua",asset_type)
    
    tlb_time_frames = config["settings"]["tlb_time_frame"]
    time_frames = tuple(config["settings"]["time_frames"])
    max_worker = config["settings"]["max_worker"]
    seed = config["settings"]["seed"]
    ws_url = config["settings"]["ws_url"]
    risque = config["settings"]["risque"]
    random.seed(seed)
    np.random.seed(seed)

class Model():
    # Charger les configurations
    config = load_config(MODEL_PATH)
    test_size = config["settings"]["test_size"]
    score_threshold = config["settings"]["score_threshold"]
    seed = Config.seed
    random.seed(seed)
    np.random.seed(seed)

# === Fonctions ===




# === Point d'entrée pour les tests ===
if __name__ == "__main__":
    # Exemple de test rapide du fichier

    pass
