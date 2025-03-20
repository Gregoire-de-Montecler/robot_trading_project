
"""
Module Description:
-------------------
export les données de XTB
"""

# === Imports ===
# 1. Modules natifs Python
import os
import logging
import math
import time
# 2. Bibliothèques externes installées via pip
import websocket
import json
from datetime import datetime
import pandas as pd
# 3. Modules internes du projet
from utils.logger import setup_logger
from utils.config import Config
from utils.data_utils import transform_json_to_dataframe
from utils.time_utils import calculate_missing_candles
from secret import PASSWORD,USERID
from data.time_transformations import three_line_break



# === Constantes globales ===
ws_url_ = Config.ws_url
# === Configuration du logger ===
logger = setup_logger("data", "data_downloader")
logger.setLevel(logging.INFO)

# === Classes ===



# === Fonctions ===
def get_data(asset:str, freq: str,n: int, period:int=15)->bool:
    """
    Récupère les données pour un actif donné et une fréquence spécifique via l'API XTB.

    Cette fonction ouvre une connexion WebSocket, s'authentifie et envoie une requête pour récupérer
    les chandeliers (candles) correspondants. Elle gère également les réponses et les erreurs.
    
    :param asset: Le symbole de l'actif (ex. "EURUSD").
    :param freq: La fréquence souhaitée (ex. "15min", "h", "4h", "D").
    :param n: Nombre de bougies manquantes à télécharger.
    :param period: Période en minutes associée à la fréquence.
    :return: True si le téléchargement s'est bien déroulé, False sinon.
    """
    # Fonction interne pour envoyer la commande de récupération des chandeliers
    def get_candles(ws, date, period, symbol):
        candles_params = {
            "command": "getChartLastRequest",
            "arguments": {
                "info": {
                        "period": period,
                        "start": date,
                        "symbol": symbol.upper()
                    }
        }}

        ws.send(json.dumps(candles_params))
        
    # Fonction interne pour gérer l'authentification
    def login(ws):
        login_command = {
            "command": "login",
            "arguments": {
                "userId": USERID,
                "password": PASSWORD,
            }
        }

        ws.send(json.dumps(login_command))

    # Fonction interne pour traiter la réponse du serveur
    def get_response(ws, message, asset,period,last_date,freq):
        response = json.loads(message)
       # Vérifier si la connexion est établie grâce au 'streamSessionId'
        if "streamSessionId" in response:
            logger.info("Connexion au compte Reussi")
            get_candles(ws=ws, date=last_date, period=period, symbol=asset)
        # Gestion des erreurs en fonction du code d'erreur retourné
        elif "errorCode" in response:
            logger.error(f"Erreur reçue : {response}")
            ws.close()
            if response.get("errorCode","") == "BE115":
                asset_exist.append(False)
            else:
                exit()
        # Traitement des données retournées
        elif "returnData" in response and period == period:

            with open(f"{asset}_{freq}.json","w") as f:
                json.dump(response, f, indent=4, ensure_ascii=False)
                logger.info("fichier json correctement sauvegardé")
        
            ws.close()
        response = ""


    def on_error(ws, error):
        logger.error("Erreur:", error)
        ws.close()

    def on_close(ws, close_status_code, close_msg):
        logger.info(f"### closed ### {close_status_code=} {close_msg=}")
    
    # Initialiser une liste pour vérifier l'existence de l'actif
    asset_exist=[]
    ws_url = ws_url_
    # Déterminer la date de début en fonction du nombre de bougies manquantes
    end_date = math.trunc(datetime.now().timestamp()*1000)
    last_date = end_date - n*period*60*1000
    # Créer une instance de connexion WebSocket avec les fonctions de rappel définies
    ws = websocket.WebSocketApp(ws_url,
                                on_open= login,
                                on_message=lambda ws, message: get_response(ws, message, asset,period,last_date,freq),
                                on_error= on_error,
                                on_close= on_close
    )


    ws.run_forever()
    if asset_exist:
        return False
    return True
    
def download_and_process_timeframes(asset: str, dir_path: str) ->pd.DataFrame:
    """
    Télécharge et transforme les données pour plusieurs timeframes.

    Args:
        asset (str): Symbole de l'actif.
        dir_path (str): path du dossier de sauvegarde des données
    Returns:
        pd.DataFrame: Données fusionnées pour tous les timeframes.
    """
    logger.info(f"Downloading {asset=} from XTB.")
    
    # Définir les fréquences et leur période associée (en minutes)
    frequencies = {"15min": 15, "h": 60, "4h": 240, "D": 1440}
    # Charger les données précédentes si elles existent, sinon initialiser un DataFrame vide
    if not os.path.exists(os.path.join(dir_path,f"{asset}.csv")):
        previous_data = pd.DataFrame()
    else:
        previous_data = pd.read_csv(os.path.join(dir_path,f"{asset}.csv"),index_col="time")
        previous_data.index = pd.to_datetime(previous_data.index)     
    # Pour chaque fréquence, télécharger les données manquantes et les transformer
    for freq, number in frequencies.items():
        n = calculate_missing_candles(dir_path,asset,freq)
        logger.info(f"Nombre de bougie manquante pour {asset} : {n}")
        asset_exist = get_data(asset=asset,freq=freq, n=n, period=number)
        if not asset_exist:
            logger.error(f"{asset=} selectionner n'existe pas")
            return pd.DataFrame()
       
       # Transformer le JSON téléchargé en DataFrame
        df_period = transform_json_to_dataframe(f"{asset}_{freq}.json")
        df_period = df_period[~df_period.index.duplicated()]
        if freq != "15min":
            df_period.columns = [f"open_{freq}", f"high_{freq}", f"low_{freq}", f"close_{freq}"]
            df_period = df_period.shift(1)
        elif freq == "15min":
            if previous_data.empty:
                df_period = three_line_break(df_period, timeframe=freq)
            else:
                previous_data_15 = previous_data[["open_15min", "high_15min", "low_15min", "close_15min"]].dropna().iloc[-3:]
                df_period = three_line_break(df_period, timeframe=freq, previous_data=previous_data_15)
                # df_period = df_period.iloc[:-1]
               
        # Fusionner les nouvelles données avec les données existantes
        try:
            df = pd.merge(df, df_period, left_index=True, right_index=True, how="outer")
        except Exception as e:
            logger.error(f"ERROR : {e}")
            df = df_period.copy()
        # Nettoyer le fichier JSON temporaire
        os.remove(f"{asset}_{freq}.json")
    # Combiner les nouvelles données avec les données précédentes et supprimer les doublons   
    df = df.combine_first(previous_data)
    df = df[~df.index.duplicated()]
    # Sauvegarder le DataFrame final
    df.to_csv(os.path.join(dir_path,f"{asset}.csv"))

def get_tick_and_pip(asset: str) -> tuple:
    """
    Récupère les informations de tick et calcule la valeur en pip pour un actif donné via une connexion WebSocket.

    La fonction récupère les données du symbole, enregistre temporairement des fichiers JSON,
    et extrait les valeurs nécessaires pour déterminer le tick size, la valeur du pip, et le tick step.

    :param asset: Le symbole de l'actif.
    :return: Un tuple contenant (tick_size_currency, pip_eur_value, tick_step_currency).
    """
    def on_message(ws, message, asset, processed_assets):
        """
        Gère la réception de messages WebSocket. 
        Vérifie si le message est une réponse de `getSymbol` et traite la donnée reçue.
        
        Parameters:
            ws (WebSocketApp): Instance WebSocket en cours de connexion.
            message (str): Message reçu en JSON.
            processed_assets (set): Ensemble des symboles déjà traités.
        """
    

        response = json.loads(message)

        if "streamSessionId" in response:
            logger.info("vous êtes bien connecté")

            # Étape 2 : Récupération des informations du symbole, si pas encore traité

            if asset not in processed_assets:
                processed_assets.add(asset) 
                get_asset_command = {
                    "command": "getSymbol",
                    "arguments": {
                        "symbol": asset
                    }
                }
        
                ws.send(json.dumps(get_asset_command))
            
        
        if "errorCode" in response:
            logger.info(response)
            ws.close()
            exit()
    
                
        if "returnData" in response:
            currency_code = response.get("returnData", {}).get("currency", "")
            
            # Enregistrement de la réponse dans un fichier JSON nommé d'après la devise
            json_file = f"{currency_code.lower()}_forex_{asset}.json"
            with open(json_file, "w") as file:
                json.dump(response, file, indent=4)
            
            logger.info(f"JSON pour {currency_code} correctement sauvegardé dans {json_file}.")
            
            # Vérifie que le fichier a bien été créé et contient des données
            if os.path.isfile(json_file) and os.path.getsize(json_file) > 0:
                logger.info("Sauvegarde JSON réussie.")
            else:
                logger.info("Erreur lors de la sauvegarde du fichier JSON.")

            # Envoie une nouvelle commande si la devise est différente de l'euro
            if currency_code != "EUR" and currency_code not in processed_assets:
                eur_currency_pair = f"EUR{currency_code}"
                get_eur_currency_command = {
                    "command": "getSymbol",
                    "arguments": {
                        "symbol": eur_currency_pair
                    }
                }
                ws.send(json.dumps(get_eur_currency_command))
                processed_assets.add(currency_code)  # Marque cette devise comme traitée
            else:
                logger.info("Réponse pour la devise EUR reçue, fermeture de la connexion.")
                ws.close()


    def on_error(ws, error):
        logger.info(f"Erreur : {error}")


    def on_close(ws, close_status_code, close_msg):
        logger.info("Connexion WebSocket fermée")


    def on_open(ws):
        login_command = {
            "command": "login",
            "arguments": {
                "userId": USERID,
                "password": PASSWORD
            }
        }
        ws.send(json.dumps(login_command))
        


    def load_and_remove_json(file_name):
        with open(file_name, "r") as file:
            data = json.load(file)
        os.remove(file_name)
        return data
    
    ws_url = ws_url_
    processed_assets = set()

    ws = websocket.WebSocketApp(
        ws_url,
        on_open=on_open,
        on_message=lambda ws, message: on_message(ws, message, asset, processed_assets),
        on_error=on_error,
        on_close=on_close
    )

    ws.run_forever()

    eur_forex_file = f'eur_forex_{asset}.json'
    response_eur = load_and_remove_json(eur_forex_file)

    currency_files = [f for f in os.listdir(os.getcwd()) if f.endswith(f"forex_{asset}.json") and f != eur_forex_file]
    logger.info(f"Liste des fichiers de devises : {currency_files}")
    asset_currency_file = currency_files[0] if currency_files else None
    is_currency_eur = False
    if asset_currency_file:
        response_currency = load_and_remove_json(asset_currency_file)

    elif response_eur.get("returnData", {}).get("symbol", 0) == asset:
        response_currency = response_eur
        is_currency_eur = True
    else:
        logger.info("Erreur : Aucun fichier JSON pour la devise trouvée.")
        exit()

    ask_eur_currency = response_eur.get("returnData", {}).get("ask", 0)
    ask_currency = response_currency.get("returnData", {}).get("ask", 0)
    tick_step_currency = response_currency.get("returnData", {}).get("tickValue", 0)
    tick_size_currency = response_currency.get("returnData", {}).get("tickSize", 0)

    pip_value_currency = round(tick_size_currency * 10000 / ask_currency, 4) if ask_currency else 0

    pip_eur_value = pip_value_currency if is_currency_eur else round(pip_value_currency / ask_eur_currency, 4) if ask_eur_currency else 0

    return tick_size_currency, pip_eur_value, tick_step_currency

# === Point d'entrée pour les tests ===
if __name__ == "__main__":
    # Exemple de test rapide du fichier
    
    pass
