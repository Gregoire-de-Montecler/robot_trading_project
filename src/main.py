
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
import time
import random
import numpy as np
from datetime import timedelta
import itertools
# 2. Bibliothèques externes installées via pip

import pandas as pd
import optuna
# 3. Modules internes du projet
from utils.logger import setup_logger
from utils.file_utils import create_directory
from utils.config import Config,Model
from utils.time_utils import format_elapsed_time, split_time_series
from utils.mutliprocessing_utils import multithreading_wrapper ,multiprocessing_wrapper
from models.ml_modeling import search_features_models, predict_test, search_hyperparametre
from data.data_downloader import download_and_process_timeframes
from data.prepare_evolving_data import prepare_evolving_data

from features.feature_combination import add_features_and_strategie

from models.optuna_feature import GROUPS


# === Constantes globales ===
ASSETS = ["EURCHF","EURGBP","EURJPY","EURUSD","EURAUD","EURCAD","EURSEK","EURNOK","EURNZD",
          "GBPCHF","GBPJPY","GBPUSD","GBPAUD","GBPNOK","GBPNZD","GBPSEK",
          "NZDUSD","NZDCHF","NZDJPY",
          "USDCAD","USDJPY","USDNOK","USDSEK","USDSGD",
          "AUDCAD","AUDCHF","AUDJPY",
          "CADCHF","CADJPY",
          ]

dir_raw = Config.raw_path
dir_processed = Config.processed_path
dir_evolving = Config.evolving_path
dir_models = Config.models_path
dir_results_csv = Config.results_csv_path
dir_results_image = Config.results_image_path
dir_results_json = Config.results_json_path
dir_storage = Config.storage_path
tlb_time_frame =Config.tlb_time_frames
time_frames = Config.time_frames
test_size = Model.test_size
seed =Config.seed
# === Configuration du logger ===
logger = setup_logger("general","cat_boost_decembre_mono_feature")
logger.setLevel(logging.INFO)

# === Classes ===


# === Fonctions ===

def info_data(data, args_list):
    debut = args_list[0][1].index[0]
    fin = args_list[-1][1].index[-1]
    logger.info(f"{debut=} , {fin=}")
    df_results = data[(data.index>debut) & (data.index<fin)]
    reel = round(df_results["gain_resultat"].sum(),2)
    max = round(df_results[df_results["resultat"]==True]["gain_resultat"].sum(),2)
    min = round(df_results[df_results["resultat"]==False]["gain_resultat"].sum(),2)
    logger.info(f"{reel=} , {max=} , {min=}")

def process_model(train_fold,test_fold,step):
    last_trade = test_fold.index[0]
    train_time_idx = train_fold[train_fold["time_close"]< last_trade].index
    train_for_fit = train_fold.loc[train_time_idx]

    best_model, best_features = search_features_models(train_for_fit,step)
    
    # score =  predict_test(train_for_fit,test_fold, best_model, best_features)
    # logger.info(f" avant params {step=} {score=}")
    
    # best_params = search_hyperparametre(train_for_fit,step,best_model, best_features)
    best_params = {}
    if step == 0:
        logger.info(f"{best_model=}")
        logger.info(f"{best_features=}")
        logger.info(f"{best_params=}")
        
    if best_model and best_features:
        score =  predict_test(train_for_fit,test_fold, best_model, best_features,best_params,step)
        return (step,score)
    return (step,0)
    
        

    



    study = optuna.create_study(direction="maximize")
    study.optimize(optimize,n_trials=50)
    best_model_feature = study.best_params
    pass





def scoring_metric():
    pass
    

# === Point d'entrée pour les tests ==
if __name__ == "__main__":
    # Exemple de test rapide du fichier
    start_time = time.time()
    
#===================================     Exporter les donnéex XTB       ===================================

    create_directory(dir_raw)
    list_raw_assets = [asset.strip(".csv") for asset in os.listdir(dir_raw)]
    args_asset = [(asset,dir_raw) for asset in ASSETS if asset not in list_raw_assets]

    # download_and_process_timeframes(*args_asset[0])
    # exit()

    multithreading_wrapper(download_and_process_timeframes, args_asset)

#===================================        Ajouts des features         ===================================

    create_directory(dir_processed)
    list_processed_asset = [asset.strip(".csv") for asset in os.listdir(dir_processed)]
    list_raw_assets = [asset.strip(".csv") for asset in os.listdir(dir_raw)]
    args_asset = [(dir_raw, dir_processed, asset, tlb_time_frame, time_frames ) for asset in list_raw_assets if asset not in list_processed_asset]

   
    # add_features_and_strategie(*args_asset[0])
    # exit()
    
    multiprocessing_wrapper(add_features_and_strategie,args_asset)
 
#===================================        Préparation des données         ===================================

    create_directory(dir_evolving)
    create_directory(dir_models)
    if not os.listdir(dir_evolving):

        prepare_evolving_data(dir_processed,dir_evolving)
    
#===================================        recherche combinaison best feature best modele     ===================================
  

    df = pd.read_csv(os.path.join(dir_evolving,"xtb_data_train.csv"),index_col="time")
    df.index = pd.to_datetime(df.index)
    df["time_close"] = pd.to_datetime(df["time_close"])
    
    
    df = df[df.index.month == 12]

    #il faut supprimer les trades a 0 lors du test il faudra juste garder ceux dans la date 
    df = df[~(df["gain_resultat"]==0)]
    
    pct_test = 5/df.shape[0]
    nb_split = int((df.shape[0] - 200 )/5)
    tscv = split_time_series(total_length=df.shape[0],nb_split= nb_split, pct_test=pct_test)# , nb_train=200 ,nb_test=5)

    scores = 0
    liste_score = []
    args_list = [(df.iloc[train_idx],df.iloc[test_idx],step)for step,(train_idx, test_idx) in enumerate(tscv)]
    info_data(df, args_list)
    resultat = []
    resultat.append(process_model(*args_list[0]))
    args_list.pop(0)
    resultat.extend(multiprocessing_wrapper(process_model,args_list,10))
    resultat_triee = sorted(resultat, key=lambda x: x[0])

    for step,score in resultat_triee:
        scores += score
        logger.info(f"{step=} {scores}")
                        
                    


        #===================================        recherche hyperparametre       ===================================
    format_elapsed_time(start_time)
