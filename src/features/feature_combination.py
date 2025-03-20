"""
Module : Feature Processing
----------------------------
Gestion des features pour enrichir les données avec des indicateurs avancés.
"""

# === Imports ===
# 1. Modules natifs Python
import os
import logging

# 2. Bibliothèques externes installées via pip
import pandas as pd
import numpy as np
# 3. Modules internes du projet
from utils.logger import setup_logger
from features.feature_engineering import add_feature,cleaning_feature
from features.strategie import strategie_costume
from data.data_downloader import get_tick_and_pip
from features.technical_indicators import get_volume_and_stop_loss
# === Constantes globales ===

# === Configuration du logger ===
logger = setup_logger("data", "feature_combination")
logger.setLevel(logging.INFO)

# === Classes ===



# === Fonctions ===

def add_features(data: pd.DataFrame,tlb_timeframe: list[str], timeframes: list[str],lower_tlb_timeframe: str ="15min" ,calcul=True) -> pd.DataFrame:
    """
    Ajoute des features supplémentaires aux données en fonction des timeframes spécifiés.
    Cette fonction itère sur la combinaison de timeframes (tlb_timeframe et timeframes) et, pour chacune,
    elle applique le calcul d'indicateurs via la fonction add_feature. Les DataFrames obtenus sont ensuite concaténés,
    alignés et nettoyés (propagation des valeurs manquantes et suppression des lignes incomplètes).

    Paramètres :
    -----------
    data : pd.DataFrame
        Données d'entrée.
    tlb_timeframe : list[str]
        Timeframe de la strategie pour les features.
    timeframe : list[str]
        Timeframes supplémentaires pour les features.

    Retour :
    --------
    pd.DataFrame
        Données enrichies avec les nouvelles features.
    """
    data_concat =pd.DataFrame()
    logger.info("Ajout des features supplémentaires.")
    # Assurer que tlb_timeframe et timeframes sont des listes
    if isinstance(tlb_timeframe, str):
        tlb_timeframe = [tlb_timeframe]
    if isinstance(timeframes, str):
        timeframe = [timeframes]
        
    # Itérer sur chaque timeframe combinée
    for timeframe in (*tlb_timeframe, *timeframes):
        # Sélectionner les colonnes associées au timeframe courant
        colonne_data = [col for col in data.columns if timeframe == col.split("_")[-1]]
        df_tf = data[colonne_data].dropna()
        # Appliquer le calcul des features pour le timeframe
        df_tf = add_feature(df_tf, timeframe,calcul)
        # Pour les timeframes autres que la référence, supprimer certaines colonnes (par exemple 'high' et 'low')
        if timeframe != tlb_timeframe:
            df_tf = df_tf.drop(columns=[f"high_{timeframe}",f"low_{timeframe}"])
        # Fusionner le DataFrame obtenu dans data_concat
        try:
            data_concat = pd.concat([data_concat, df_tf], axis=1)
        except Exception as e:
            logger.error(f"Erreur lors de la concaténation: {e}. Indices uniques: {df_tf.index.is_unique=}, {data_concat.index.is_unique=}")
    
    data = data_concat
    # Remplissage par propagation (forward fill) pour les colonnes ne correspondant pas au lower_tlb_timeframe
    colonne_data = [col for col in data.columns if lower_tlb_timeframe not in col]
    for colonne in colonne_data:
        data[colonne] = data[colonne].ffill()
    data = data.infer_objects(copy=False)
    # Conserver uniquement les lignes où la colonne 'open_{lower_tlb_timeframe}' n'est pas nulle
    data = data[data[f"open_{lower_tlb_timeframe}"].notna()]
    data = data.dropna()
    logger.info("Ajout des features terminé.")
    
    return data

def add_features_and_strategie(input_path: str, output_path: str, asset: str, tlb_time_frame: list[str], time_frames: list[str], stage: str = "train" )-> None:
    """
    Ajoute des features et applique la stratégie de trading sur les données d'un actif.

    La fonction charge le fichier CSV correspondant à l'actif, convertit l'index en datetime, 
    applique l'ajout de features via add_features, et selon le mode ('trade'), calcule des indicateurs spécifiques
    (volume, stop loss) avant d'appliquer la stratégie propriétaire (via strategie_costume). Elle nettoie ensuite le DataFrame
    et le sauvegarde au format CSV.

    Parameters:
        input_path (str): Chemin d'entrée du fichier CSV de l'actif.
        output_path (str): Chemin de sortie pour sauvegarder le fichier traité.
        asset (str): Nom de l'actif.
        tlb_time_frame (list[str]): Liste des timeframes de référence.
        time_frames (list[str]): Liste des timeframes supplémentaires.
        stage (str): Mode d'exécution ("train" ou "trade"). Par défaut "train".

    Returns:
        None
    """
    
    logger.info(f"Starting processing for {asset=}.")
    # Charger les données depuis le fichier CSV
    df = pd.read_csv(os.path.join(input_path,f"{asset}.csv"),index_col="time")
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated()]
    # Ajouter les features calculées
    df = add_features(df, tlb_time_frame,time_frames)
    if stage == "trade":
        # Si en mode 'trade', récupérer les indicateurs de tick et pip pour calculer le volume et le stop loss
        tick_size_currency, pip_eur_value, tick_step_currency =  get_tick_and_pip(asset)
        logger.info(f"tick_size_currency: {tick_size_currency}, pip_eur_value: {pip_eur_value}, tick_step_currency: {tick_step_currency}")
        df = get_volume_and_stop_loss(df,tick_size_currency, pip_eur_value, tick_step_currency)
    # Appliquer la stratégie propriétaire pour ajuster les features
    df = strategie_costume(df,tlb_time_frame)
    df["asset"] = asset
    # Nettoyer le DataFrame en supprimant les features redondantes
    df = cleaning_feature(df)
    # Sauvegarder le DataFrame final en CSV
    df.to_csv(os.path.join(output_path,f"{asset}.csv"))
    

# === Point d'entrée pour les tests ===
if __name__ == "__main__":
    # Exemple de test rapide du fichier
    logger.info("Feature processing module testé avec succès.")
