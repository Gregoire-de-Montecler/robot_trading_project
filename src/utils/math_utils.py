
"""
Module : Math Utilities
------------------------
Fournit des fonctions pour effectuer des calculs mathématiques spécifiques aux séries temporelles.
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

# === Constantes globales ===

# === Configuration du logger ===
logger = setup_logger("utils", "math_utils")
logger.setLevel(logging.ERROR)

# === Classes ===



# === Fonctions ===
def calcul_pente(series: pd.Series, periode: int = 3) -> pd.Series:
    """
    Calcule la pente d'une série temporelle.

    Args:
        series (pd.Series): Série temporelle d'entrée.
        periode (int): Intervalle utilisé pour le calcul de la pente. Par défaut 3.

    Returns:
        pd.Series: Série contenant les pentes calculées.
    """
    logger.debug("Calcul des pentes de la série temporelle.")
    return (series - series.shift(periode)) / periode


def calcul_difference(series_1: pd.Series, series_2: pd.Series) -> pd.Series:
    """
    Calcule la différence en pourcentage entre deux séries temporelles.

    Args:
        series_1 (pd.Series): Première série temporelle.
        series_2 (pd.Series): Deuxième série temporelle.

    Returns:
        pd.Series: Différence en pourcentage entre les deux séries.
    """
    logger.debug("Calcul des différences en pourcentage entre deux séries.")
    return 100 * (series_1 - series_2) / (series_2 + 1e-6)

def score_evaluation(data_results: pd.DataFrame)-> float:
    """
    Calcul de l'évaluation avec α = 0.5 et β = 3.
    
    Arguments :
    - data (pd.DataFrame) : datafram results
    
    Retourne :
    - L'évaluation calculée
    """
    if data_results["prediction"].max() > 1 or data_results["prediction"].min() <0:
        value = 0
    else:
        value = 0.5
    
    valeur_debut = data_results["gain_resultat"].sum()
    max_val = data_results[data_results[f"gain_resultat"]>0][f"gain_resultat"].sum()
    min_val = data_results[data_results[f"gain_resultat"]<0][f"gain_resultat"].sum()
    valeur_finale = data_results[data_results["prediction"]>=value][f"gain_resultat"].sum()
    
    return valeur_finale


def calculate_evaluation(data_results: pd.DataFrame)-> float:
    """
    Calcul de l'évaluation avec α = 0.5 et β = 3.
    
    Arguments :
    - data (pd.DataFrame) : datafram results
    
    Retourne :
    - L'évaluation calculée
    """
    if data_results["prediction"].max() > 1 or data_results["prediction"].min() <0:
        value = 0
    else:
        value = 0.5
    
    valeur_debut = data_results["gain_resultat"].sum()
    max_val = data_results[data_results[f"gain_resultat"]>0][f"gain_resultat"].sum()
    min_val = data_results[data_results[f"gain_resultat"]<0][f"gain_resultat"].sum()
    valeur_finale = data_results[data_results["prediction"]>=value][f"gain_resultat"].sum()
    
    return valeur_finale

    # Normalisation des valeurs
    valeur_debut_norm = (valeur_debut - min_val) / (max_val - min_val)
    valeur_finale_norm = (valeur_finale - min_val) / (max_val - min_val)
    
    # Calcul de g(x) et h(x)
    g = (1 - valeur_debut_norm) ** 0.5  # α = 0.5
    h = valeur_finale_norm ** 3        # β = 3
    
    # Calcul de l'évaluation
    evaluation = g * h
    
    #calcul acc
    data_prediction = data_results[data_results["prediction"]>=value]
    data_non_pred = data_results[data_results["prediction"]<value]
    tp_fp = data_prediction.shape[0]
    tp = data_prediction[data_prediction["resultat"] == True].shape[0]
    fn = data_non_pred[data_non_pred["resultat"] == True].shape[0]
    nb_true = data_results[data_results["resultat"]==True].shape[0]
    if nb_true ==0 and tp_fp ==0:
        return 1 
    
    if tp_fp == 0 or tp + fn == 0:
        return 0 
    precision = (tp/tp_fp)
    return precision

    recall = tp/ (tp + fn)
    if precision + recall == 0:
        return 0
    f1 = 2 * (precision * recall) / (precision + recall)
    
    
    return f1


# === Point d'entrée pour les tests ===
if __name__ == "__main__":
    # Exemple de test rapide du fichier
    logger.info("Math utilities module testé avec succès.")
