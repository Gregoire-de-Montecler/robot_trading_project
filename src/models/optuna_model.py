
"""
Module Description:
-------------------
Décrire brièvement le but du fichier ou des fonctions qu'il contient.
Exemple : Ce module contient des fonctions pour [prétraitement, visualisation, etc.].
"""

# === Imports ===
# 1. Modules natifs Python
import os
import random
import logging
import threading
import tempfile

import optuna
# 2. Bibliothèques externes installées via pip
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import make_scorer

import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor, BaggingRegressor
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier

from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.linear_model import LogisticRegression
# 3. Modules internes du projet
from utils.logger import setup_logger


from utils.config import Config, Model 

# === Constantes globales ===
test_size = Model.test_size
path = Config.path
asset_type = Config.asset_type
score_threshold = Model.score_threshold
dir_json = Config.results_json_path
dir_results = Config.results_csv_path
dir_processed = Config.processed_path
dir_raw = Config.raw_path
seed = Config.seed
dir_models = Config.models_path
dir_scaler = Config.scaler_path
save_lock = threading.Lock()
storage = Config.storage_path
storage = f"sqlite:///{os.path.join(storage,'optuna_storage.db')}"
joblib_temp_dir = tempfile.mkdtemp()
os.environ["JOBLIB_TEMP_FOLDER"] = joblib_temp_dir

NOM = "kumo_4h"
# === Configuration du logger ===
logger = setup_logger("models","ml_modeling")
logger.setLevel(logging.ERROR)
optuna.logging.get_logger("optuna").setLevel(logging.DEBUG)
MODELS = {
        "GradientBoostingRegressor": GradientBoostingRegressor,
        "ExtraTreesRegressor":ExtraTreesRegressor,
        "BaggingRegressor":BaggingRegressor,
        "XGBRegressor":XGBRegressor,
        "LGBMRegressor":LGBMRegressor,
        "CatBoostRegressor": CatBoostRegressor,
        
        "GradientBoostingClassifier":GradientBoostingClassifier,
        "AdaBoostClassifier":AdaBoostClassifier,
        "XGBClassifier":XGBClassifier,
        "LGBMClassifier":LGBMClassifier,
        "CatBoostClassifier":CatBoostClassifier,
        "LogisticRegression":LogisticRegression,
        "BaggingClassifier":BaggingClassifier,
         
}

# === Classes ===


   

# === Fonctions ===


def suggest_models(trial):
    # model_name = trial.suggest_categorical("model",MODELS.keys())
    model_name = trial.suggest_categorical("model",["CatBoostClassifier"])
    return model_name
         


# === Point d'entrée pour les tests ===
if __name__ == "__main__":
    # Exemple de test rapide du fichier
    print(MODELS.keys())
    pass
