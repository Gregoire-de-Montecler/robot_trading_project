"""
Module : Data Utilities
------------------------
Fournit des fonctions utilitaires pour manipuler et transformer les données.
"""

# === Imports ===
# 1. Modules natifs Python
import os
import logging
import json
# 2. Bibliothèques externes installées via pip
import pandas as pd
import numpy as np


# 3. Modules internes du projet
from utils.logger import setup_logger
from utils.time_utils import convert_unix_to_datetime
from utils.math_utils import calculate_evaluation

# === Constantes globales ===

# === Configuration du logger ===
logger = setup_logger("utils", "data_utils")
logger.setLevel(logging.ERROR)

# === Classes ===



# === Fonctions ===

def adjust_price_column(data: pd.DataFrame, column: str, open_column: str = "open", digits: int = 1) -> pd.Series:
    """
    Ajuste une colonne de prix en fonction d'une colonne d'ouverture et d'un multiplicateur de digits.

    Args:
        data (pd.DataFrame): DataFrame contenant les colonnes nécessaires.
        column (str): Nom de la colonne à ajuster (ex: 'high', 'low', 'close').
        open_column (str): Nom de la colonne d'ouverture (par défaut 'open').
        digits (int): Facteur d'ajustement des prix.

    Returns:
        pd.Series: Colonne ajustée des prix.
    """
    if column not in data.columns or open_column not in data.columns:
        raise ValueError(f"Les colonnes '{column}' ou '{open_column}' sont manquantes dans le DataFrame.")
    return (data[open_column] + data[column]) / digits

def transform_json_to_dataframe(path_fichier_json: str) -> pd.DataFrame:
    """
    Lit un fichier JSON et le transforme en un DataFrame formaté avec des colonnes temporelles et des prix ajustés.

    Args:
        path_fichier_json (str): Chemin du fichier JSON à lire.

    Returns:
        pd.DataFrame: DataFrame contenant les colonnes ['time', 'open', 'high', 'low', 'close'] indexé par la colonne 'time'.

    Raises:
        ValueError: Si le fichier spécifié n'est pas un JSON valide.
        FileNotFoundError: Si le fichier n'existe pas.
    """
    if not path_fichier_json.endswith(".json"):
        logger.error("Le chemin spécifié ne pointe pas vers un fichier JSON.")
        raise ValueError("Veuillez fournir un fichier au format JSON.")

    if not os.path.exists(path_fichier_json):
        logger.error(f"Le fichier JSON {path_fichier_json} est introuvable.")
        raise FileNotFoundError(f"Fichier JSON non trouvé : {path_fichier_json}")

    try:
        with open(path_fichier_json, "r") as f:
            data = json.load(f)

        digits = int("1" + data.get("returnData", {}).get("digits", 0) * "0")
        rates = data.get("returnData", {}).get("rateInfos", {})
        data_df = pd.DataFrame(rates)
        # Conversion de l'heure Unix en datetime
        data_df["time"] = pd.to_datetime(data_df["ctm"].apply(convert_unix_to_datetime))

        # Ajustement des prix en fonction des digits

        for col in ["high", "low", "close"]:
            data_df[col] = adjust_price_column(data_df, col, open_column="open", digits=digits)
        data_df["open"] = data_df["open"] / digits

        # Format final
        data_df = data_df[["time", "open", "high", "low", "close"]].set_index("time")

        return data_df

    except Exception as e:
        logger.error(f"Erreur lors de la lecture ou de la transformation du fichier JSON : {e}")
        raise

def concat_data(df_1: pd.DataFrame, df_2: pd.DataFrame, axis: int = 1) -> pd.DataFrame:
    """
    Concatène deux DataFrames le long d'un axe donné.

    Args:
        df_1 (pd.DataFrame): Premier DataFrame.
        df_2 (pd.DataFrame): Deuxième DataFrame à concaténer.
        axis (int): Axe de concaténation (1 pour les colonnes, 0 pour les lignes).

    Returns:
        pd.DataFrame: DataFrame concaténé.

    Logs:
        - Avertissement si l'un des DataFrames est vide.
        - Erreur si la concaténation échoue.
    """
    try:
        # Vérification si l'un des DataFrames est vide
        if df_1.empty or df_2.empty:
            if df_1.empty and df_2.empty:
                logger.warning("Les deux DataFrames sont vides. Retour d'un DataFrame vide.")
                return pd.DataFrame()

            df_vide = "deuxième" if df_2.empty else "premier"
            logger.warning(f"Le {df_vide} DataFrame est vide. Retour de l'autre DataFrame.")
            return df_1 if df_2.empty else df_2

        # Concaténation des DataFrames
        result = pd.concat([df_1, df_2], axis=axis)
        logger.info("Concaténation réussie entre les deux DataFrames.")
        return result

    except Exception as e:
        logger.error(f"Erreur lors de la concaténation des DataFrames : {e}")
        raise RuntimeError("Impossible de concaténer les DataFrames.") from e  

def evaluate_model(test_data: pd.DataFrame,target_data: pd.Series, model)-> pd.DataFrame:
    """
    Teste un modèle de machine learning sur un ensemble de validation.

    Args:
        test_data (pd.DataFrame): Données d'entrée normalisées pour le test.
        target_data (pd.Series): Valeurs cibles pour le test.
        model: Modèle de machine learning à tester.

    Returns:
        pd.DataFrame: DataFrame contenant les prédictions du modèle et les valeurs réelles.
    """
    
    try:
        array_classe = model.classes_
        index_true = [i for i, x in enumerate(array_classe) if x >= 1][0]
        predictions = model.predict_proba(test_data)[:,index_true]        
    except:
        predictions  = model.predict(test_data)
    predictions  = pd.Series(predictions , index= test_data.index, name="prediction")
    results  = pd.concat([predictions , target_data], axis=1)
    return results 
# === Point d'entrée pour les tests ===
if __name__ == "__main__":
    # Exemple de test rapide du fichier
    logger.info("Data utilities module testé avec succès.")
