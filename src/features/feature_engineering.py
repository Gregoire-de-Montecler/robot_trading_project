"""
Module : Feature Engineering
----------------------------
Ajoute et nettoie les features nécessaires pour l'analyse des données financières.
"""

# === Imports ===
# 1. Modules natifs Python
import os
import logging
import pandas as pd
import numpy as np
# 2. Bibliothèques externes installées via pip



# 3. Modules internes du projet
from utils.logger import setup_logger
from features.technical_indicators import (
    bougie,
    calcul_rsi,
    stochastique,
    moyenne_mobile,
    pivot_rsi_stoch,
    ichimoku,
    macd_indicator
)

# === Constantes globales ===

# === Configuration du logger ===
logger = setup_logger("data", "feature_engineering")
logger.setLevel(logging.ERROR)

# === Classes ===



# === Fonctions ===




def add_feature(data: pd.DataFrame, time_frame: str, calcul: bool=True) -> pd.DataFrame:
    """
    Ajoute une série d'indicateurs techniques au DataFrame pour un timeframe donné.

    L'enrichissement se fait en appliquant successivement plusieurs fonctions d'indicateurs :
      1. bougie : Calcul des patterns de bougies.
      2. ichimoku : Ajout des lignes Ichimoku.
      3. calcul_rsi : Calcul du RSI et de sa dérivée.
      4. stochastique : Calcul des indicateurs stochastiques.
      5. macd_indicator : Calcul de l'indicateur MACD et de ses dérivées.
      6. moyenne_mobile : Calcul des moyennes mobiles et leurs dérivées.
      7. pivot_rsi_stoch : Calcul des pivots basés sur RSI et stochastiques.

    Parameters:
        data (pd.DataFrame): DataFrame contenant les données financières.
        time_frame (str): Timeframe pour les calculs (ex: "15min").
        calcul (bool): Si True, effectue les calculs d'indicateurs.

    Returns:
        pd.DataFrame: Le DataFrame enrichi avec les nouvelles colonnes d'indicateurs.
    """
    logger.info(f"Ajout des features pour le timeframe {time_frame}.")
    data = bougie(data, time_frame)
    data = ichimoku(data, time_frame)
    data = calcul_rsi(data, time_frame, calcul)
    data = stochastique(data, time_frame, calcul)
    data = macd_indicator(data,time_frame,calcul)
    data = moyenne_mobile(data, time_frame, calcul)
    data = pivot_rsi_stoch(data, time_frame)
    
    return data

def cleaning_feature(data: pd.DataFrame, lower_tlb_timeframe: str="15min" ) -> pd.DataFrame:
    """
    Nettoie le DataFrame en supprimant les colonnes redondantes ou inutiles.

    Les colonnes liées aux prix de base (open, close) et certains indicateurs intermédiaires 
    (par exemple 'bull', 'bear', ou certains indicateurs de pattern) sont supprimées pour alléger le DataFrame.

    Parameters:
        data (pd.DataFrame): DataFrame contenant les features.
        lower_tlb_timeframe (str): Timeframe de référence pour certaines suppressions, par défaut "15min".

    Returns:
        pd.DataFrame: DataFrame nettoyé.
    """
    tlb = lower_tlb_timeframe
    logger.info("Nettoyage des features inutiles ou redondantes.")
    data = data.sort_index(axis=1)
    # Identifier les colonnes débutant par 'open', 'close', 'bull' ou 'bear'
    colonnes_a_supprimer  = [colonne for colonne in data.columns if colonne.startswith("open") or colonne.startswith("close") or colonne.startswith("bull") or colonne.startswith("bear") ]
    # Supprimer ces colonnes ainsi que d'autres colonnes spécifiques
    data = data.drop(
        columns=[
            "lowest_costume_blanc",
            "highest_costume_noir",
            "lowest_pivot",
            "highest_pivot",
            "costume_blanc",
            "costume_noir",
            *colonnes_a_supprimer ,
        ],errors="ignore"
    )
    # Supprimer les colonnes 'low' et 'high' pour le timeframe de référence s'ils existent
    if f"low_{tlb}" in data.columns:
        data.drop(columns=[f"low_{tlb}", f"high_{tlb}"], inplace=True)
    data = data.dropna()
    logger.info("nettoyage des données terminé")
    return data


# === Point d'entrée pour les tests ===
if __name__ == "__main__":
    # Exemple de test rapide du fichier
    logger.info("Feature engineering module testé avec succès.")
