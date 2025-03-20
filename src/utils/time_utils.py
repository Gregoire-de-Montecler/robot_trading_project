
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
import pytz
# 2. Bibliothèques externes installées via pip



# 3. Modules internes du projet
from utils.logger import setup_logger
import pandas as pd
from datetime import datetime ,timezone, timedelta
import numpy as np
# === Constantes globales ===

# === Configuration du logger ===
logger = setup_logger("utils", "time_utils")
logger.setLevel(logging.INFO)

# === Classes ===



# === Fonctions ===

def compute_time_unit(time: float, unit_limit: int) -> tuple:
    """
    Calcule les unités de temps principales (secondes, minutes, heures) et retourne leurs parties majeures et mineures.

    Args:
        time (float): Temps en secondes à convertir.
        unit_limit (int): Limite supérieure de l'unité (60 pour minutes, 24 pour heures, etc.).

    Returns:
        tuple: (int, int) Partie majeure et mineure de l'unité de temps.
    """
    time_split = str(time / unit_limit).split(".")
    major_unit = int(time_split[0])
    minor_unit = int(time_split[1][:2])  # Capture uniquement les deux premiers décimaux
    minor_unit = int(minor_unit / 100 * unit_limit)
    return major_unit, minor_unit

def format_elapsed_time(start_time: float):
    """
    Affiche le temps écoulé depuis un instant donné, formaté en heures, minutes et secondes.

    Args:
        start_time (float): Timestamp de début, généralement `time.time()`.
    """
    import time

    elapsed_time = (time.time() - start_time)
    time_units = {
        "sec": {"limit": 60},
        "min": {"limit": 60},
        "hours": {"limit": 24},
    }

    for unit, config in time_units.items():
        unit_limit = config["limit"]
        if elapsed_time > unit_limit:
            elapsed_time, remainder = compute_time_unit(elapsed_time, unit_limit)
            config.update({"remainder": remainder})
        else:
            config.update({"remainder": elapsed_time})
            break

    message = "Elapsed time is: "
    for unit, config in reversed(time_units.items()):
        remainder = config.get("remainder", False)
        if remainder:
            message += f"{int(remainder)} {unit} "
    logger.info(message)
   
def calculate_missing_candles(raw_folder: str,asset: str, timeframe: str) -> int:
    """
    Récupère la dernière bougie du dossier raw pour les intervalles :
        - 15 min
        - 1h
        - 4h
        - 1 jour (D)
        - 1 semaine (W)
    
    Formule pour trouver le nombre de bougies :
    La date est au format timestamp Unix.
    last_date = dernière date au format datetime
    nb = (datetime.now().timestamp() * 1000 - last_date.timestamp() * 1000) // (tf * 60 * 1000)
    
    Parameters:
    raw_folder (str): Path to the raw data folder.
    timeframe (str): Timeframe for the candles (e.g., '15min', '1h', '4h', 'D', 'W').
    
    Returns:
    int: Number of candles to retrieve.
    """
    # Define the file path based on the timeframe
    file_path = os.path.join(raw_folder, f"{asset}.csv")
    if os.path.exists(file_path) == False:
        logger.info(f"File {file_path} not found.")
        return 40000
    # Read the last row of the CSV file
    data = pd.read_csv(file_path,index_col="time")
    data.index = pd.to_datetime(data.index)
    last_date = data[f"close_{timeframe}"].index[-1]
    logger.info(f"Last date for {asset} : {last_date}")
        
    # Calculate the number of candles to retrieve
    tf_in_minutes = {"15min": 15, "h": 60, "4h": 240, "D": 1440, "W": 10080}
    tf = tf_in_minutes[timeframe]
    nb_candles = (datetime.now().timestamp() * 1000 - last_date.timestamp() * 1000) // (tf * 60 * 1000)
    
    return int(nb_candles) if nb_candles > 0 else 2

# def split_time_series(len_df:float , pct_test: float,nb_split: int)-> tuple:
#     nb_test = int(len_df*pct_test)
#     d = int(nb_test * nb_split)
#     nb_train = len_df - d
#     liste_index_train_test =[]
#     for i in reversed(range(nb_split)):

#             train_index = np.arange((len_df-nb_train-(i+1)*nb_test),(len_df-nb_test-i*nb_test))
#             test_index = np.arange((len_df-nb_test-i*nb_test),(len_df-i*nb_test))
#             liste_index_train_test.append((train_index,test_index))
        
#     return liste_index_train_test

# def split_time_series(len_df: int,nb_train: int,nb_test: float,nb_split: int)-> tuple:
#     if nb_split * nb_test + nb_train > len_df:
#         logger.error("Erreur sur le nombre de split")
#         raise  "Erreur sur le nombre de split"
#     liste_index_train_test =[]
#     nb_train = int(nb_train)
    
#     if nb_test <1:
#         nb_test = int(nb_train * nb_test)
        
#     if (nb_split * nb_test + nb_train) > len_df:
#         nb_split = int((len_df - nb_train*4)/nb_test)
#         logger.error(f"Ajustement nb de split {nb_split=}")
        
#     for i in reversed(range(nb_split)):

#         train_index = np.arange((len_df-nb_train-(i+1)*nb_test),(len_df-nb_test-i*nb_test))
#         test_index = np.arange((len_df-nb_test-i*nb_test),(len_df-i*nb_test))
#         liste_index_train_test.append((train_index,test_index))
    
#     return liste_index_train_test

def split_time_series(total_length: int, nb_split: int, pct_test: float = None, nb_train: int = None, nb_test: int = None) -> list:
    """
    Découpe une série temporelle en ensembles d'indices pour l'entraînement et le test.

    La fonction propose deux modes de fonctionnement :

    1. Mode pct_test :
       - Si 'pct_test' est fourni (non None), alors nb_test est calculé comme int(total_length * pct_test).
       - nb_train est alors défini par : nb_train = total_length - nb_split * nb_test.
       - Si nb_train est négatif, nb_split est automatiquement ajusté au maximum autorisé.

    2. Mode nb_train/nb_test :
       - Si 'pct_test' est None et que nb_train et nb_test sont fournis, la fonction vérifie que :
             nb_split * nb_test + nb_train <= total_length.
       - En cas d'incohérence, nb_split est ajusté automatiquement au maximum autorisé.

    Paramètres :
      - total_length (int) : Longueur totale de la série temporelle (doit être > 0).
      - nb_split (int) : Nombre initial de découpages souhaités (doit être > 0).
      - pct_test (float, optionnel) : Pourcentage (entre 0 et 1) de la série à utiliser pour le test.
      - nb_train (int, optionnel) : Nombre d'indices d'entraînement (mode nb_train/nb_test).
      - nb_test (int, optionnel) : Nombre d'indices de test (mode nb_train/nb_test).

    Retourne :
      - list : Une liste de tuples (train_index, test_index), où chaque élément est un tableau numpy d'indices.

    Exceptions :
      - ValueError : Si les paramètres sont insuffisants ou incohérents.
    """
    
    # Vérifications de base sur total_length et nb_split
    if total_length <= 0:
        raise ValueError("total_length doit être strictement positif.")
    if nb_split <= 0:
        raise ValueError("nb_split doit être strictement positif.")

    # Mode 1 : utilisation de pct_test
    if pct_test is not None:
        if not (0 < pct_test < 1):
            raise ValueError("pct_test doit être compris entre 0 et 1.")
        nb_test_calc = int(total_length * pct_test)
        if nb_test_calc < 1:
            raise ValueError("Le calcul de nb_test donne une valeur inférieure à 1, veuillez augmenter pct_test ou total_length.")
        nb_test = nb_test_calc
        nb_train = total_length - nb_split * nb_test
        if nb_train < 0:
            max_nb_split = total_length // nb_test
            logger.error(f"Adjustment: nb_split ajusté de {nb_split} à {max_nb_split} car nb_split * nb_test dépasse total_length.")
            nb_split = max_nb_split
            nb_train = total_length - nb_split * nb_test

    # Mode 2 : utilisation explicite de nb_train et nb_test
    elif nb_train is not None and nb_test is not None:
        if nb_test < 1:
            raise ValueError("nb_test doit être supérieur ou égal à 1.")
        if nb_train < 1:
            raise ValueError("nb_train doit être supérieur ou égal à 1.")
        if nb_split * nb_test + nb_train > total_length:
            max_nb_split = (total_length - nb_train) // nb_test
            logger.error(f"Adjustment: nb_split ajusté de {nb_split} à {max_nb_split} car nb_split * nb_test + nb_train dépasse total_length.")
            nb_split = max_nb_split
    else:
        raise ValueError("Paramètres insuffisants. Fournir soit pct_test, soit nb_train et nb_test.")

    liste_index_train_test = []
    # Création des découpages en parcourant nb_split en sens inverse
    for i in reversed(range(nb_split)):
        # Indices d'entraînement
        train_start = total_length - nb_train - (i + 1) * nb_test
        train_end = total_length - nb_test - i * nb_test
        train_index = np.arange(train_start, train_end)
        # Indices de test
        test_start = total_length - nb_test - i * nb_test
        test_end = total_length - i * nb_test
        test_index = np.arange(test_start, test_end)
        liste_index_train_test.append((train_index, test_index))
    
    return liste_index_train_test

def convert_unix_to_datetime(heure: int) -> str:
    """
    Convertit un timestamp Unix (en millisecondes) en format datetime lisible.

    Args:
        heure (int): Timestamp Unix en millisecondes.

    Returns:
        str: Date et heure au format classique '%Y-%m-%d %H:%M:%S'.
    """
    try:
        timestamp_s = heure / 1000
        date_time = datetime.fromtimestamp(timestamp_s)
        return date_time.strftime('%Y-%m-%d %H:%M:%S')
    except Exception as e:
        logger.error(f"Erreur lors de la conversion du timestamp Unix : {e}")
        raise
# === Point d'entrée pour les tests ===
if __name__ == "__main__":
    # Exemple de test rapide du fichier
    format_elapsed_time(1731009323)
    pass
