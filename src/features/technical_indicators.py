"""
Module pour l'ajout de features pour le Machine Learning à partir de données financières.

Ce script ajoute des indicateurs techniques aux données historiques de trading, tels que les bougies, RSI, stochastiques, moyennes mobiles et pivots.
Il est conçu pour être utilisé dans des modèles prédictifs pour la prise de décision en trading.

Fonctions Principales:
- bougie : Ajoute des features liées aux bougies et patterns de renversement.
- calcul_rsi : Calcule l'indicateur RSI et sa pente.
- stochastique : Calcule les indicateurs stochastiques K et D et leurs pentes.
- moyenne_mobile : Calcule des moyennes mobiles exponentielles et d'autres indicateurs basés sur les EMA.
- pivot_rsi_stoch : Calcule les pivots basés sur les RSI et stochastiques.

"""
# === Imports ===
# 1. Modules natifs Python
import os
import logging
import pandas as pd
import numpy as np
# 2. Bibliothèques externes installées via pip
# import seaborn as sns
# import matplotlib.pyplot as plt

# 3. Modules internes du projet
from utils.logger import setup_logger
from utils.math_utils import calcul_pente, calcul_difference
from utils.config import Config
from features.strategie import costume_
# === Constantes globales ===
risque = Config.config["settings"]["risque"]
# === Configuration du logger ===
logger = setup_logger("features", "technical_indicators")
logger.setLevel(logging.ERROR)

# === Classes ===



# === Fonctions ===
def bougie(data: pd.DataFrame, time_frame: str, three_line_break: list[str] = ["15min"]) -> pd.DataFrame:
    """
    Ajoute des features liées aux bougies japonaises et identifie les patterns de renversement.

    Args:
        data (pd.DataFrame): Le DataFrame contenant les colonnes `open_{tf}`, `close_{tf}`, correspondant aux prix d'ouverture et de fermeture du time frame spécifié.
        time_frame (str): Le time frame utilisé pour les calculs, par exemple '15min', '1h', etc.

    Returns:
        pd.DataFrame: Le DataFrame d'entrée avec les nouvelles colonnes ajoutées:
            - bull_{tf}, bear_{tf}: Booléens indiquant si la bougie est haussière ou baissière.
            - costume_blanc, costume_noir: Indicateurs de pattern de renversement haussier et baissier.
            - proche_resistance: Booléen indiquant si le prix est proche d'un point de pivot.
            - retournement_sur_polarite_bull, retournement_sur_polarite_bear: Indicateurs de retournement haussier et baissier.
    """
    logger.info(f"Calcul des Bougie pour le timeframe {time_frame}.")
    tf = time_frame
    
    data[f"bull_{tf}"] = data[f"open_{tf}"] < data[f"close_{tf}"]
    data[f"bear_{tf}"] = data[f"open_{tf}"] > data[f"close_{tf}"]
    
    #calcul du range de chaque bougie
    data[f"bougie_range_{tf}"] = abs(data[f"close_{tf}"] - data[f"open_{tf}"])
    
    #calcul du la moyenne des ranges sur 21 periodes
    data[f"ema_bougie_range_{tf}"] = data[f"bougie_range_{tf}"].ewm(span=21, adjust=False,min_periods=21).mean().fillna(data[f"bougie_range_{tf}"].mean())
        
    
    if tf in three_line_break:
        # creation de la colonne evolution
        data[f"group_change_{tf}"] = data[f"bull_{tf}"] != data[f"bull_{tf}"].shift(1)
        data[f"group_id_{tf}"] = data[f"group_change_{tf}"].cumsum()
        data[f"evolution_{tf}"] = data.groupby(f"group_id_{tf}").cumcount() + 1
        data = data.drop(columns=[f"group_change_{tf}", f"group_id_{tf}"])
        data = costume_(data,tf)
        
    data = data.drop(columns = [f"ema_bougie_range_{tf}",f"bougie_range_{tf}"])
    data = data.infer_objects(copy=False)
    return data
    
def calcul_rsi(data: pd.DataFrame, time_frame: str,calcul: bool=True) -> pd.DataFrame:
    """
    Calcule l'indicateur RSI (Relative Strength Index) et sa pente.

    Args:
        data (pd.DataFrame): Le DataFrame contenant la colonne `close_{tf}` pour les prix de clôture du time frame spécifié.
        time_frame (str): Le time frame utilisé pour les calculs, par exemple '15min', '1h', etc.

    Returns:
        pd.DataFrame: Le DataFrame d'entrée avec les colonnes suivantes ajoutées:
            - rsi_{tf}: Valeur de l'indicateur RSI.
            - pente_rsi_{tf}: Pente du RSI calculée sur trois périodes.
    """
    tf = time_frame
    if not calcul:
        logger.info(f"RSI non calculé pour le timeframe {time_frame}.")
    
    else:
        logger.info(f"Calcul du RSI pour le timeframe {time_frame}.")
        longueur =21
        src = data[f"close_{tf}"].diff()
        up = src.where(src>0,0)
        down = -src.where(src<0,0)
        sma_h = up.ewm(longueur,adjust = False,min_periods=longueur).mean()
        sma_b = down.ewm(longueur,adjust= False,min_periods=longueur).mean()
        rsi = 100-(100/(1+sma_h/sma_b))
        data[f"rsi_{tf}"] = rsi.bfill()

    conditions = [
        (data[f"rsi_{tf}"] >= 70),
        (data[f"rsi_{tf}"] >= 55),
        (data[f"rsi_{tf}"] >= 45),
        (data[f"rsi_{tf}"] >= 30)
    ]
    choix = [2, 1, 0, -1]
    data[f"zone_rsi_{time_frame}"] = np.select(conditions, choix, default=-2)
    data[f"suite_position_rsi_{tf}"] = data.groupby((data[f"zone_rsi_{tf}"] != data[f"zone_rsi_{tf}"].shift()).cumsum()).cumcount() + 1
    
    
    # 2. Pente du MACD (première dérivée)
    data[f"slope_rsi_{tf}"] = data[f"rsi_{tf}"].diff()
    
    # 3. Accélération du MACD (seconde dérivée)
    data[f"acceleration_rsi_{tf}"] = data[f"slope_rsi_{tf}"].diff()
    # data = data.drop(f"rsi_{tf}",axis=1)
    

    data = data.infer_objects(copy=False)
    return data

def stochastique(data: pd.DataFrame, time_frame: str, calcul: bool = True) -> pd.DataFrame:
    """
    Calcule les indicateurs stochastiques K et D, leur écart, et la pente de D.

    Args:
        data (pd.DataFrame): Le DataFrame contenant la colonne `close_{tf}` pour les prix de clôture du time frame spécifié.
        time_frame (str): Le time frame utilisé pour les calculs, par exemple '15min', '1h', etc.

    Returns:
        pd.DataFrame: Le DataFrame d'entrée avec les colonnes suivantes ajoutées:
            - d_stoch_{tf}: Moyenne de l'indicateur stochastique D.
            - ecart_stoch_{tf}: Différence entre K et D.
            - pente_d_stoch_{tf}: Pente de l'indicateur D stochastique sur trois périodes.
    """
    tf = time_frame
    if not calcul:
        logger.info(f"Stochastique non calculé pour le timeframe {time_frame}.")
        

        
        
    else:
        if tf == "15min":
            low_series = data[f"close_{tf}"]
            high_series = data[f"close_{tf}"]
        else:
            low_series = data[f"low_{tf}"]
            high_series = data[f"high_{tf}"]
            
        n = 34
        n1 = 3
        haut = data[f"close_{tf}"] - low_series.rolling(window=n).min()
        bas = high_series.rolling(window=n).max() - low_series.rolling(window=n).min()
        k = (haut / bas) * 100
        k1 = k.rolling(window=n1).mean()
        d = k1.rolling(window=n1).mean()
        data[f"k_stoch_{tf}"] = k1.bfill()
        data[f"d_stoch_{tf}"] = d.bfill()
    
    max_stoch = data[[f"k_stoch_{time_frame}", f"d_stoch_{time_frame}"]].max(axis=1)
    conditions = [
        (max_stoch >= 80),
        (max_stoch >= 60),
        (max_stoch >= 40),
        (max_stoch >= 20)
    ]
    choix = [2, 1, 0, -1]
    data[f"zone_stoch_{time_frame}"] = np.select(conditions, choix, default=-2)
    
    
    data[f"suite_position_stoch_{tf}"] = data.groupby((data[f"zone_stoch_{tf}"] != data[f"zone_stoch_{tf}"].shift()).cumsum()).cumcount() + 1
    data[f"croisement_stoch_{tf}"] = np.where(
        data[f"k_stoch_{tf}"]>data[f"d_stoch_{tf}"],
        1,
        -1
    )

    # 2. Pente du MACD (première dérivée)
    data[f"slope_stoch_{tf}"] = data[f"d_stoch_{tf}"].diff()
    
    # 3. Accélération du MACD (seconde dérivée)
    data[f"acceleration_stoch_{tf}"] = data[f"slope_stoch_{tf}"].diff()
    data = data.drop([f"k_stoch_{tf}"],axis=1)
    

    return data

def moyenne_mobile(data: pd.DataFrame, time_frame: str, calcul: bool = True) -> pd.DataFrame:
    """
    Calcule des moyennes mobiles exponentielles (EMA) et des indicateurs dérivés tels que la pente et la différence entre la clôture et l'EMA.

    Args:
        data (pd.DataFrame): Le DataFrame contenant la colonne `close_{tf}` pour les prix de clôture du time frame spécifié.
        time_frame (str): Le time frame utilisé pour les calculs, par exemple '15min', '1h', etc.

    Returns:
        pd.DataFrame: Le DataFrame d'entrée avec les colonnes suivantes ajoutées:
            - pente_EMA_{i}_{tf}: Pente de l'EMA sur trois périodes.
            - diff_close_ema_{i}_{tf}: Pourcentage de différence entre le prix de clôture et l'EMA.
    """
    tf = time_frame
    permutation_to_score = {
    # Groupe A (close > 3 EMA)
    ("close","ema13","ema26","ema100"):  +5,
    ("close","ema13","ema100","ema26"):  +4,
    ("close","ema26","ema13","ema100"):  +3,
    ("close","ema26","ema100","ema13"):   +3,
    ("close","ema100","ema13","ema26"):   +4,
    ("close","ema100","ema26","ema13"):   +3,

    # Groupe B (close > 2 EMA & < 1 EMA)
    ("ema13","close","ema26","ema100"):   +2,
    ("ema13","close","ema100","ema26"):   +2,
    ("ema26","close","ema13","ema100"):   +1,
    ("ema26","close","ema100","ema13"):   +1,
    ("ema100","close","ema13","ema26"):   +2,
    ("ema100","close","ema26","ema13"):   +1,

    # Groupe C (close > 1 EMA & < 2 EMA)
    ("ema13","ema26","close","ema100"):   -1,
    ("ema13","ema100","close","ema26"):   -1,
    ("ema26","ema13","close","ema100"):   -2,
    ("ema26","ema100","close","ema13"):   -2,
    ("ema100","ema13","close","ema26"):   -1,
    ("ema100","ema26","close","ema13"):   -2,

    # Groupe D (close < 3 EMA)
    ("ema13","ema26","ema100","close"):   -3,
    ("ema13","ema100","ema26","close"):   -3,
    ("ema26","ema13","ema100","close"):   -4,
    ("ema26","ema100","ema13","close"):  -4,
    ("ema100","ema13","ema26","close"):  -2,
    ("ema100","ema26","ema13","close"):  -5,
    }
    def row_to_zone(row):
        """
        Pour UNE ligne du DataFrame : 
        - on récupère close, ema13, ema26, ema100,
        - on détermine l'ordre décroissant,
        - on renvoie le score (de +12 à -12).
        """
        # Récupération des valeurs
        c     = row[f"close_{tf}"]
        e13   = row[f"EMA_13_{tf}"]
        e26   = row[f"EMA_26_{tf}"]
        e100  = row[f"EMA_100_{tf}"]
        
        # On fabrique la liste [("close", c), ("ema13", e13), ...]
        arr = [
            ("close", c),
            ("ema13", e13),
            ("ema26", e26),
            ("ema100", e100),
        ]
    
        
        # On trie par valeur décroissante
        arr_sorted = sorted(arr, key=lambda x: x[1], reverse=True)
        # Exemple: arr_sorted = [("close",1.234),("ema13",1.22),("ema26",1.20),("ema100",1.19)]
        
        # On ne garde que les noms, dans l'ordre trié
        permutation = tuple(name for (name, val) in arr_sorted)
        # Exemple: permutation = ("close","ema13","ema26","ema100")
        
        # On va chercher le score associé dans notre dico
        # (Attention si jamais des ex-aequo ou manquant => KeyError)
        score = permutation_to_score.get(permutation, np.nan)
        
        return score
    
    def compute_zone_classification(df):
        """
        Applique row_to_zone() à chaque ligne du DataFrame, et renvoie la Série de scores.
        """
        return df.apply(row_to_zone, axis=1)
    
    
    logger.info(f"Calcul des moyennes mobiles pour le timeframe {time_frame}.")
    
    for i in [13,26,100]:
        if calcul:
            data[f"EMA_{i}_{tf}"] = data[f"close_{tf}"].ewm(
                span=i, adjust=False,min_periods=i
            ).mean().bfill()
        data[f"pente_EMA_{i}_{tf}"] = calcul_pente(data[f"EMA_{i}_{tf}"])
        
        
        data[f"diff_close_ema_{i}_{tf}"] = calcul_difference(
            data[f"close_{tf}"],data[f"EMA_{i}_{tf}"])
    data[f"diff_ema_13_26_{tf}"] = calcul_difference(
        data[f"EMA_13_{tf}"],data[f"EMA_26_{tf}"])

    
    data[f"zone_MM_{tf}"] = compute_zone_classification(data)
    data[f"suite_position_MM_{tf}"] = data.groupby((data[f"zone_MM_{tf}"] != data[f"zone_MM_{tf}"].shift()).cumsum()).cumcount() + 1
    




    for i in [13,26,100]:
        data = data.drop((f"EMA_{i}_{tf}"),axis=1)
    data = data.infer_objects(copy=False)
    return data

def pivot_rsi_stoch(data: pd.DataFrame, time_frame: str) -> pd.DataFrame:
    """
    Calcule les pivots basés sur les indicateurs RSI et stochastiques, et ajoute des colonnes de différences par rapport à ces pivots.

    Args:
        data (pd.DataFrame): Le DataFrame contenant les colonnes `rsi_{tf}` et `d_stoch_{tf}` pour le time frame spécifié.
        time_frame (str): Le time frame utilisé pour les calculs, par exemple '15min', '1h', etc.

    Returns:
        pd.DataFrame: Le DataFrame d'entrée avec les colonnes suivantes ajoutées:
            - diff_pivot_low_rsi_{tf}: Différence entre le RSI actuel et son pivot bas.
            - diff_pivot_high_rsi_{tf}: Différence entre le RSI actuel et son pivot haut.
            - diff_pivot_low_d_stoch_{tf}: Différence entre D stochastique et son pivot bas.
            - diff_pivot_high_d_stoch_{tf}: Différence entre D stochastique et son pivot haut.
    """
    logger.info(f"Calcul des point pivot pour le timeframe {time_frame}.")
    tf = time_frame
    dict = {"low": np.less ,"high" : np.greater}
    for c in ["rsi","d_stoch"]:
        for clef,valeur in dict.items():
            #point le plus et plus bas 3 bougie avant et 3 bougie après
            data[f"pivot_{clef}_{c}_{tf}"] = np.where((valeur(data[f"{c}_{tf}"].shift(3), data[f"{c}_{tf}"].shift(6))) &
                                               (valeur(data[f"{c}_{tf}"].shift(3), data[f"{c}_{tf}"].shift(5))) &
                                               (valeur(data[f"{c}_{tf}"].shift(3), data[f"{c}_{tf}"].shift(4))) &
                                               (valeur(data[f"{c}_{tf}"].shift(3), data[f"{c}_{tf}"].shift(2))) &
                                               (valeur(data[f"{c}_{tf}"].shift(3), data[f"{c}_{tf}"].shift(1))) &
                                               (valeur(data[f"{c}_{tf}"].shift(3), data[f"{c}_{tf}"])),
                                                data[f"{c}_{tf}"].shift(3),np.nan)
            data[f"pivot_{clef}_{c}_{tf}"] = data[f"pivot_{clef}_{c}_{tf}"].ffill()
            data[f"diff_pivot_{clef}_{c}_{tf}"] = 100*(data[f"{c}_{tf}"] - data[f"pivot_{clef}_{c}_{tf}"])/(data[f"pivot_{clef}_{c}_{tf}"]+1e-6)
            data = data.drop((f"pivot_{clef}_{c}_{tf}"),axis=1)
            
     
    # data = data.drop(f"rsi_{tf}",axis=1)        
    data = data.infer_objects(copy=False)
    return data  

def macd_indicator(data: pd.DataFrame, time_frame: str,calcul:bool = True) -> pd.DataFrame:
    tf = time_frame
    if not calcul:
        logger.info(f"macd_indicator non calculé pour le timeframe {time_frame}.")
    
    else:
        logger.info(f"Calcul du macd_indicator pour le timeframe {time_frame}.")
        # Calcul de l'EMA rapide (souvent 12 périodes) et de l'EMA lente (souvent 26 périodes)
        data[f"ema_fast_{tf}"] = data[f"close_{tf}"].ewm(span=12, adjust=False).mean()
        data[f"ema_slow_{tf}"] = data[f"close_{tf}"].ewm(span=26, adjust=False).mean()
        
        # Calcul de la ligne MACD : différence entre l'EMA rapide et l'EMA lente
        data[f"macd_line_{tf}"] = data[f"ema_fast_{tf}"] - data[f"ema_slow_{tf}"]
        
        # Calcul de la ligne de signal : EMA (souvent sur 9 périodes) de la ligne MACD
        data[f"signal_line_{tf}"] = data[f"macd_line_{tf}"].ewm(span=9, adjust=False).mean()
        
        # Calcul de l'histogramme MACD : différence entre la ligne MACD et la ligne de signal
        data[f"macd_histogram_{tf}"] = data[f"macd_line_{tf}"] - data[f"signal_line_{tf}"]
        

    data[f"crossover_macd_{tf}"] = np.where(
    data[f"macd_line_{tf}"] > data[f"signal_line_{tf}"],
    1,
    -1
    )
    data[f"zone_macd_{tf}"] = np.where(
    data[f"macd_line_{tf}"] > 0,
    1,
    -1
    )
    # 2. Pente du MACD (première dérivée)
    data[f"slope_macd_{tf}"] = data[f"macd_line_{tf}"].diff()
    
    # 3. Accélération du MACD (seconde dérivée)
    data[f"acceleration_macd_{tf}"] = data[f"slope_macd_{tf}"].diff()
    
    
    data[f"duration_crossover_macd_{tf}"] = data.groupby(
        (data[f"crossover_macd_{tf}"] != data[f"crossover_macd_{tf}"].shift()).cumsum()
    ).cumcount() + 1
    

    
    data[f"histogram_ratio_{tf}"] = data[f"macd_histogram_{tf}"] / (data[f"close_{tf}"] + 1e-6)
    data = data.drop(columns=[f"ema_fast_{tf}",f"ema_slow_{tf}",f"signal_line_{tf}"])
    return data

def ichimoku(data: pd.DataFrame, time_frame: str) -> pd.DataFrame:
    tf = time_frame
    def donchian(data, len, columns):
        return (data[columns].max(axis=1).rolling(window=len).max()+ data[columns].min(axis=1).rolling(window=len).min())/2
    
    def feature_ichimoku(data,d_comparaison,droite_1,droite_2,name,tf):
        data[f"position_prix_{name}_{tf}"] = np.where(
            (data[f"{d_comparaison}_{tf}"] > data[[f"{droite_1}_{tf}", f"{droite_2}_{tf}"]].max(axis=1)),
            1,
            np.where(
                (data[f"{d_comparaison}_{tf}"] < data[[f"{droite_1}_{tf}", f"{droite_2}_{tf}"]].min(axis=1)),
                -1,
                0    
            )
        )
        # one_hot_encoded = pd.get_dummies(data[f"position_prix_{name}_{tf}"], prefix=f"position{name}_{tf}", dtype=int)
        # data = pd.concat([data, one_hot_encoded], axis=1)
        
        data[f"suite_position_{name}_{tf}"] = data.groupby((data[f"position_prix_{name}_{tf}"] != data[f"position_prix_{name}_{tf}"].shift()).cumsum()).cumcount() + 1
        
        data[f"epaisseur_{name}_{tf}"] = np.abs(data[f"{droite_1}_{tf}"] - data[f"{droite_2}_{tf}"])/data[f"{d_comparaison}_{tf}"]
        data[f"ecart_relative_{name}_{tf}"] = (data[f"{d_comparaison}_{tf}"] - data[[f"{droite_1}_{tf}", f"{droite_2}_{tf}"]].min(axis=1)) /(data[[f"{droite_1}_{tf}", f"{droite_2}_{tf}"]].max(axis=1) - data[[f"{droite_1}_{tf}", f"{droite_2}_{tf}"]].min(axis=1)+ 1e-6)
        # data= data.drop(f"position_prix_{name}_{tf}",axis=1)
        return data
    
    columns = [f"open_{tf}",f"close_{tf}" if tf == "15min" else f"high_{tf}",f"low_{tf}"]
    data[f"tenkan_sen_{tf}"] =donchian(data,9,columns)
    data[f"kijun_sen_{tf}"] =donchian(data,26,columns)
    data[f"ssa_{tf}"] = ((data[f"tenkan_sen_{tf}"] + data[f"kijun_sen_{tf}"]) / 2).shift(26)
    data[f"ssb_{tf}"] =donchian(data,52,columns).shift(26)
    data[f"chikou_Span_{tf}"] = data[f"close_{tf}"].shift(-26)
    
    data = feature_ichimoku(data,"close","ssa","ssb","kumo",tf)
    data = feature_ichimoku(data,"close","tenkan_sen","kijun_sen","sen",tf)
    data = feature_ichimoku(data,"chikou_Span","ssa","ssb","chikou_kumo",tf)
    data = feature_ichimoku(data,"chikou_Span","tenkan_sen","kijun_sen","chikou_sen",tf)
    data[f"ecart_chikou_close_{tf}"] = np.abs(data[f"chikou_Span_{tf}"] - data[f"close_{tf}"])/data[f"close_{tf}"]
    feature_chikou = [col for col in data.columns if "chikou" in col]
    data[feature_chikou] = data[feature_chikou].shift(26)
    
    data = data.drop(columns=[f"tenkan_sen_{tf}",f"kijun_sen_{tf}",f"ssa_{tf}",f"ssb_{tf}",f"chikou_Span_{tf}"])#,f"position_prix_chikou_kumo_{tf}",f"position_prix_chikou_sen_{tf}",f"suite_position_chikou_kumo_{tf}",f"suite_position_chikou_sen_{tf}"])

    return data

def get_volume_and_stop_loss(data: pd.DataFrame,tick_size: float, pip_value: float, tick_step: float,montant_risque: int = None) -> pd.DataFrame:
    if montant_risque == None:
        montant_risque = risque
    distance_stop = np.where(
        (data["costume_noir"] == True) | (data["costume_blanc"] == True),
        np.round(abs(data["close_15min"]-data["close_15min"].shift(2)),6),
        False
    )
    data["pos_stop_loss"] = np.where(
        distance_stop != False,
        np.where(
            data["costume_noir"] == True,
            np.round(data["close_15min"] + 3*distance_stop,6),
            np.round(data["close_15min"] - 3*distance_stop,6)
        ),
        False
    )
    tick_stop = np.where(
        distance_stop != False,
        np.round(distance_stop/(tick_size*10)*tick_step,1),
        False
    )
    data["volume"] = np.where(
        tick_stop != False,
        np.round(montant_risque/(tick_stop*pip_value)/100*tick_step,2),
        False
    )
    data = data.infer_objects(copy=False)
    return data
# === Point d'entrée pour les tests ===
if __name__ == "__main__":
    # Exemple de test rapide du fichier
    logger.info("Technical indicators module testé avec succès.")
