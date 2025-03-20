
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

# === Classes ===


   

# === Fonctions ===
def suggest_params(config, trial):
    params = {}
    for param, settings in config.items():
        param_type = settings.get("type",{})
        obligatory = False
        
        if param_type == "conditional":
            condition = settings["condition"]
            if eval(condition,{},params):
                param = param.split("__")[-1]
                obligatory = settings.get("obligatory", False)
                param_type = settings.get("type_param",{})
            else:
                continue
        
            
        if param_type == "int":
            value = trial.suggest_int(param,
                                            settings["min"],
                                            settings["max"],
                                            step=settings.get("step",1))
            
        elif param_type == "float":
            value = trial.suggest_float(param,
                                                settings["min"],
                                                settings["max"],
                                                step=settings.get("step",None),
                                                log=settings.get("log",False))
        
        elif param_type == "categorical":
            value = trial.suggest_categorical(param,
                                                    settings["values"])
        
        if obligatory:
            str_contidition = settings["condition"]
            str_param = str_contidition.split("==")[0].strip(" ")
            str_value = str_contidition.split("==")[1].strip(" ").strip('"')
            params[str_param] = str_value + ":" +param + "=" + str(value)
        else:
            params[param] = value
    logger.info(params)
    return params

def prune_inconsistent_parameters(params):
    """
    Prune les combinaisons de paramètres incohérentes avant d'entraîner un modèle.

    Args:
        params (dict): Dictionnaire contenant les paramètres sélectionnés.

    Raises:
        optuna.exceptions.TrialPruned: Si les paramètres sont jugés incohérents.
    """

    # Règle 1 : Low learning rate + Low iterations
    if params["learning_rate"] < 0.02 and params["iterations"] < 1000:
        raise optuna.exceptions.TrialPruned()

    # Règle 2 : High depth + Low regularization or small leaves
    if params["depth"] == 6 and (params["l2_leaf_reg"] < 4 or params["min_data_in_leaf"] < 20):
        raise optuna.exceptions.TrialPruned()

    # Règle 3 : High learning rate + Low iterations
    if params["learning_rate"] > 0.05 and params["iterations"] < 1000:
        raise optuna.exceptions.TrialPruned()

    # Règle 4 : Bootstrap-specific conditions
    if params["bootstrap_type"] == "Bayesian" and params.get("bagging_temperature", 0) > 1.2 and params["min_data_in_leaf"] < 20:
        raise optuna.exceptions.TrialPruned()
    if params["bootstrap_type"] == "Bernoulli" and params.get("subsample", 1) < 0.8 and params["depth"] > 5:
        raise optuna.exceptions.TrialPruned()

    # Règle 5 : Small leaves with high complexity
    if params["min_data_in_leaf"] < 20 and (params["depth"] == 6 or params.get("bagging_temperature", 0) > 1.2):
        raise optuna.exceptions.TrialPruned()  
    return params 




# === Point d'entrée pour les tests ===
if __name__ == "__main__":
    # Exemple de test rapide du fichier
    pass
