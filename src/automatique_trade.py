
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
import joblib
import json
# 2. Bibliothèques externes installées via pip
import pandas as pd
from datetime import datetime
import websocket


# 3. Modules internes du projet
from utils.logger import setup_logger
from utils.config import Config
from utils.file_utils import create_directory
from data.data_downloader import download_and_process_timeframes
from utils.mutliprocessing_utils import multithreading_wrapper ,multiprocessing_wrapper
from features.feature_combination import add_features_and_strategie, add_feature_prediction
from utils.time_utils import format_elapsed_time, veficication_intervalle_heure
from data.prepare_evolving_data import encoding_asset
from models.ml_modeling import normalize_data
from secret import PASSWORD,USERID

# === Constantes globales ===
raw_folder = Config.raw_path
ASSETS = ["EURCHF","EURGBP","EURJPY","EURUSD","EURAUD","EURCAD","EURSEK","EURNOK","EURNZD",
          "GBPCHF","GBPJPY","GBPUSD","GBPAUD","GBPNOK","GBPNZD","GBPSEK",
          "NZDUSD","NZDCHF","NZDJPY",
          "USDCAD","USDJPY","USDNOK","USDSEK","USDSGD",
          "AUDCAD","AUDCHF","AUDJPY",
          "CADCHF","CADJPY",
          ]
dir_raw = Config.raw_path
dir_models = Config.models_path
dir_processed = Config.processed_path
tlb_time_frame =Config.tlb_time_frames
time_frames = Config.time_frames
ws_url_ = Config.ws_url
# === Configuration du logger ===
logger = setup_logger("general", "automatique_trade")
logger.setLevel(logging.INFO)

# === Classes ===



# === Fonctions ===

def process_ml(dir_processed: str, dir_models: str,asset: str, trade_en_cours: list):
    """ 
    Processus de machine learning pour un actif donné.
    :param dir_processed: str, chemin du dossier contenant les données prétraitées.
    :param dir_models: str, chemin du dossier contenant les modèles.
    :param asset: str, nom de l'actif à traiter.""" 
    #===================================        Standardisation       ===================================  
    df = pd.read_csv(os.path.join(dir_processed,f"{asset}.csv"),index_col="time")
    df = encoding_asset(df,dir_models,mode="test")
    try:
        X_test, y_test = normalize_data(df,["resultat","gain_resultat","pos_stop_loss","volume"],dir_models,"data")
    except:
        os.remove(os.path.join(dir_processed,f"{asset}.csv"))
        os.remove(os.path.join(dir_raw,f"{asset}.csv"))
        logger.info(f"Suppression de l'actif {asset} car il y a un probleme dans la normalisation des données")
        return
    y_test["long"] = X_test["long"]

    #===================================        Analyse trade  Sortie     ===================================  
    results = [pair for pair in trade_en_cours if pair[1] == asset]   
    
    for result in results:
        
        if result[0] == 0:
            if y_test[y_test["long"]== True]["gain_resultat"].iloc[-1] != 0:
                close_trade(result[0],result[1],result[2],result[3])
                logger.info(f"sortie de trade long pour l'actif {asset}")
        elif result[0] == 1:
            if y_test[y_test["long"]== False]["gain_resultat"].iloc[-1] != 0:
                close_trade(result[0],result[1],result[2],result[3])
                logger.info(f"sortie de trade short pour l'actif {asset}")
    #===================================        Analyse trade  Entrée    ===================================  
    
    if y_test["gain_resultat"].iloc[-1] == 0:
        if veficication_intervalle_heure(y_test.index[-1]):
    #===================================        ML sur data       ===================================  
            models = [model for model in os.listdir(dir_models) if model.startswith("data")]
            data_predict = pd.DataFrame()
            for model in models:
                model = joblib.load(os.path.join(dir_models,model))
                name_model = model.__class__.__name__
                try:
                    array_classe = model.classes_
                    index_true = [i for i, x in enumerate(array_classe) if x >= 1][0]
                    y_pred = model.predict_proba(X_test.iloc[-1:])[:,index_true]
                except:
                    y_pred = model.predict(X_test.iloc[-1:])
                data_predict["prediction_"+name_model] = y_pred
                
            data_predict.index = X_test.index[-1:]
    #===================================        LinearRegression sur prevision      ===================================    
            data_predict = add_feature_prediction(data_predict)
            data_predict, y = normalize_data(data_predict,[],dir_models,"prediction")
            model = joblib.load(os.path.join(dir_models,"prediction_LinearRegression.pkl"))
            y_pred = model.predict(data_predict)
            logger.debug(f"Prediction pour l'actif {asset} : {y_pred}")
            if y_pred[0] >0:
                prise_trade(asset, y_test["long"].iloc[-1], y_test["volume"].iloc[-1], y_test["pos_stop_loss"].iloc[-1])
                logger.info(f"Prise de trade pour {"long" if y_test["long"].iloc[-1] else "short"} l'actif {asset} avec une prediction de {y_pred[0]}")
            elif y_pred[0] < 0:
                logger.info(f"Ne prend pas de trade pour l'actif {asset} avec une prediction de {y_pred[0]}")
        
def prise_trade(asset: str, trade: str, volume: float, stop_loss: float):
    """Prend un trade pour un actif donné.
    :param asset: str, nom de l'actif.
    :param trade: str, type de trade ("long" ou "short").
    :param volume: float, volume de l'ordre.
    :param stop_loss: float, stop loss de l'ordre.
    """
    direction_trade = "0" if trade else "1"
    def opening_trade(ws):
        trade_command = {
            "command": "tradeTransaction",
            "arguments": {
                "tradeTransInfo": {
                    "cmd": direction_trade,  # '0' correspond à l'achat (long)
                    "customComment": f"Trade {trade} {asset}",  # Commentaire pour identifier la transaction
                    "expiration": 0,  # Pas de délai d'expiration
                    "offset": 0,  # Décalage non utilisé
                    "order": 0,  # 0 car il s'agit d'un ordre d'ouverture
                    "price": 0.1,  # Pour un ordre au marché, laissez à 0
                    "sl": stop_loss,  # Stop loss à 1,05
                    "symbol": asset,  # Symbole de l'instrument
                    "tp": 0.0,  # Pas de take profit spécifié
                    "type": 0,  # Type '0' pour l'ouverture
                    "volume": volume  # Volume de l'ordre
                }
            }
        }
        ws.send(json.dumps(trade_command))
        
    def on_message(ws,message):
        response = json.loads(message)
        if "streamSessionId" in response:
            logger.debug("Connexion au compte Reussi")
            opening_trade(ws)

        elif "returnData" in response:
            logger.debug("Commande de transaction commerciale envoyée.")
            ws.close()

        
    
    def on_error(ws, error):
        print("Erreur :", error)

    def on_close(ws, close_status_code, close_msg):
        print("Connexion fermée")

    def on_open(ws):
        # Étape 1 : Connexion
        login_command = {
            "command": "login",
            "arguments": {
                "userId": USERID,
                "password": PASSWORD
            }
        }
        # Envoi de la commande de connexion
        ws.send(json.dumps(login_command))

  

    # Initialisation de la connexion WebSocket
    ws_url = ws_url_
    ws = websocket.WebSocketApp(ws_url,
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)

    # Lancer la connexion WebSocket
    ws.run_forever()

def close_trade(trade: int, asset: str, order: float, volume: float):
    """Ferme un trade en cours. 
    :param trade: int, 0 pour long et 1 pour short.
    :param asset: str, nom de l'actif.
    :param order: float, numéro de l'ordre.
    :param volume: float, volume de l'ordre.
    

    """
    trade = 1 if trade == 0 else 0

    # Crée une fonction de connexion
    def on_open(ws):
        # Envoie la commande de connexion avec ID utilisateur, mot de passe, appId et appName
        login_command = {
            "command": "login",
            "arguments": {
                "userId": USERID,         # Remplace par ton ID utilisateur
                "password": PASSWORD,     # Remplace par ton mot de passe
                "appId": "test",                 # Champ appId (optionnel, tu peux remplacer si nécessaire)
                "appName": "test"                # Nom de l'application (optionnel)
            }
        }
        
        ws.send(json.dumps(login_command))
        
        

         # Attendre avant de faire d'autres requêtes
   

    def closing_trade(ws):
        trade_command = {
            "command": "tradeTransaction",
            "arguments": {
                "tradeTransInfo": {
                    "cmd": trade,  # '0' correspond à l'achat (long)
                    "customComment": f"Trade {trade} {asset}",  # Commentaire pour identifier la transaction
                    "expiration": 0,  # Pas de délai d'expiration
                    "offset": 0,  # Décalage non utilisé
                    "order": order,  # 0 car il s'agit d'un ordre d'ouverture
                    "price": 0.1,  # Pour un ordre au marché, laisse à 0
                    "sl": 0,  # Stop loss
                    "symbol": asset,  # Symbole de l'instrument
                    "tp": 0.0,  # Pas de take profit spécifié
                    "type": 2,  # Type '2' pour l'ouverture
                    "volume": volume  # Volume de l'ordre
                }
            }
        }

        ws.send(json.dumps(trade_command))
        logger.info("Commande de transaction commerciale envoyée.")


    def on_message(ws, message):
        response = json.loads(message)
        if "streamSessionId" in response:
            logger.info("Connexion au compte réussie")
            closing_trade(ws)
        if "returnData" in response:
            ws.close()
        
        
        

                    
    
            
    # Fonction pour gérer les erreurs
    def on_error(ws, error):
        print("Erreur:", error)
        ws.close()

    # Fonction pour fermer la connexion
    def on_close(ws, close_status_code, close_msg):
        print("### closed ###", close_status_code, close_msg)

    ws_url = ws_url_
    # Initialise la connexion WebSocket
    ws = websocket.WebSocketApp(ws_url,
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)

    # Démarre la connexion
    ws.run_forever()
    
def get_trade():
    def on_open(ws):
        print("Connexion établie")
        
        # Envoie la commande de connexion avec ID utilisateur, mot de passe, appId et appName
        login_command = {
            "command": "login",
            "arguments": {
                "userId": USERID,         # Remplace par ton ID utilisateur
                "password": PASSWORD,     # Remplace par ton mot de passe
                "appId": "test",                 # Champ appId (optionnel, tu peux remplacer si nécessaire)
                "appName": "test"                # Nom de l'application (optionnel)
            }
        }
        
        ws.send(json.dumps(login_command))
         # Attendre avant de faire d'autres requêtes
        get_position(ws)  # Appel pour obtenir les positions

    def get_position(ws):
        get_traderecords = {
            "command": "getTrades",
            "arguments": {
                "openedOnly": True
            }
        }
        
        ws.send(json.dumps(get_traderecords))


    def on_message(ws, message):
        response = json.loads(message)
        if "returnData" in response:
            with open("trade.json","w") as f:
                json.dump(response,f)

            ws.close()
            
    # Fonction pour gérer les erreurs
    def on_error(ws, error):
        print("Erreur:", error)
        ws.close()

    # Fonction pour fermer la connexion
    def on_close(ws, close_status_code, close_msg):
        print("### closed ###", close_status_code, close_msg)

    ws_url = ws_url_
    # Initialise la connexion WebSocket
    ws = websocket.WebSocketApp(ws_url,
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)

    # Démarre la connexion
    ws.run_forever()
    if os.path.exists("trade.json"):
        with open("trade.json","r") as f:
            response = json.load(f)
            returnData = response.get("returnData","")
        if returnData:
            trade_en_cours = [(trade.get("cmd",""),trade.get("symbol",""), trade.get("position",""), trade.get("volume","")) for trade in response["returnData"] ]
            trade_en_cours= set(trade_en_cours)
            return trade_en_cours
    return []
    
    
# === Point d'entrée pour les tests ===
if __name__ == "__main__":
    # Exemple de test rapide du fichier
    start_time = time.time()


#===================================        Recuperer bougie manquante        ===================================    
    create_directory(dir_raw)
    args_asset = [(asset,dir_raw) for asset in ASSETS]
    # download_and_process_timeframes(*args_asset[16])
    multithreading_wrapper(download_and_process_timeframes, args_asset)
    logger.debug("Fin de la récupération des données")
   
#===================================        Ajouts features manquante et strategie       ===================================
    create_directory(dir_processed)
    list_raw_assets = [asset.strip(".csv") for asset in os.listdir(dir_raw)]
    args_asset = [(dir_raw, dir_processed, asset, tlb_time_frame, time_frames, "trade") for asset in list_raw_assets]
    # add_features_and_strategie(*args_asset[0])
    multiprocessing_wrapper(add_features_and_strategie,args_asset)
    logger.debug("Fin de l'ajout des features et de la stratégie")
#===================================        Sauvegarde trade en cours       ===================================
    trade_en_cours = get_trade()
    logger.info(f"trade en cours {trade_en_cours}")
#===================================        Machine Learning       ===================================  
    list_raw_assets = [asset.strip(".csv") for asset in os.listdir(dir_processed)]
    args_asset = [(dir_processed,dir_models,asset,trade_en_cours) for asset in list_raw_assets]
    
    # process_ml(*args_asset[0])
    multithreading_wrapper(process_ml,args_asset)

 
    
    format_elapsed_time(start_time)