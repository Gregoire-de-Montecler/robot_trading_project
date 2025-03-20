
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
from datetime import timedelta
# 2. Bibliothèques externes installées via pip
import pandas as pd
import numpy as np
import joblib

# 3. Modules internes du projet
from utils.logger import setup_logger
from utils.config import Config

# === Constantes globales ===
dir_models = Config.models_path
# === Configuration du logger ===
logger = setup_logger("data","prepare_evolving_data")
logger.setLevel(logging.ERROR)

# === Classes ===



# === Fonctions ===


def load_and_concat_processed_data(processed_path:str)-> pd.DataFrame:
    """
    Charge et concatène toutes les données brutes présentes dans le dossier spécifié.

    :param processed_path: Chemin du dossier contenant les fichiers CSV bruts.
    :return: DataFrame contenant l'ensemble des données concaténées.
    """
    data = pd.DataFrame()
    # Parcourir chaque fichier CSV dans le dossier de données brutes
    for file in os.listdir(processed_path):
        # Lire le fichier CSV en DataFrame
        df = pd.read_csv(os.path.join(processed_path, file))
        # Convertir la colonne 'time' en format datetime pour faciliter les traitements ultérieurs
        df["time"] = pd.to_datetime(df["time"])
        # Ajouter les données lues à l'ensemble du DataFrame
        data = pd.concat([data, df], axis=0, ignore_index=True)

    return data

def encoding_asset(data: pd.DataFrame,pkl_path: str,mode: str = "train") -> pd.DataFrame:
    """
    Effectue un encodage one-hot sur la colonne 'asset' pour extraire les devises,
    avec la possibilité de sauvegarder ou charger les colonnes de référence.

    :param data: pd.DataFrame, contient une colonne 'asset'.
    :param pkl_path: str, chemin pour sauvegarder/charger les colonnes de référence.
    :param mode: str, "train" pour sauvegarder les colonnes, "test" pour charger les colonnes.
    :return: pd.DataFrame avec les devises encodées et les colonnes inutiles supprimées.
    """
    # En mode test, charger les colonnes de référence préalablement sauvegardées
    if mode == "test":
        reference_columns = joblib.load(os.path.join(pkl_path,"dummies_asset.pkl"))
    else:
        reference_columns = None

    # Extraire la première et la deuxième devise à partir du symbole de l'actif
    data['currency_1'] = data['asset'].str[:3]
    data['currency_2'] = data['asset'].str[3:]

    # Appliquer un one-hot encoding pour chaque devise
    one_hot_1 = pd.get_dummies(data['currency_1'])
    one_hot_2 = pd.get_dummies(data['currency_2'])

    # Fusionner les deux encodages en additionnant les valeurs (remplissage avec 0)
    one_hot_encoded = one_hot_1.add(one_hot_2, fill_value=0).astype(bool)

    # Si en mode test, aligner l'encodage sur les colonnes de référence
    if mode == "test":
        for col in reference_columns:
            if col not in one_hot_encoded.columns:
                one_hot_encoded[col] = False
        one_hot_encoded = one_hot_encoded[reference_columns]
    else:
        # En mode train, sauvegarder la liste des colonnes obtenues pour une utilisation future
        reference_columns = one_hot_encoded.columns.tolist()
        joblib.dump(reference_columns, os.path.join(pkl_path,"dummies_asset.pkl"))
    
    # Intégrer les colonnes encodées au DataFrame et supprimer les colonnes temporaires
    data = pd.concat([data.drop(columns=one_hot_encoded.columns, errors='ignore'), one_hot_encoded], axis=1)
    data.drop(["asset", "currency_1", "currency_2"], axis=1, inplace=True)
    return data

def make_unique_time_index(
    data: pd.DataFrame,
    time_column: str = "time",
    offset_unit: str = "s",
    set_as_index=True,
    keep_original: bool = False,
    sort: bool = False
) -> pd.DataFrame:
    """
    Ajuste la colonne de temps du DataFrame pour garantir que tous les timestamps sont uniques
    en ajoutant un décalage basé sur une unité de temps spécifiée aux doublons.

    Parameters:
    -----------
    data : pd.DataFrame
        Le DataFrame d'entrée contenant une colonne de temps.
    
    time_column : str, default="time"
        Le nom de la colonne contenant les timestamps à rendre uniques.
    
    set_as_index : bool, default=True
        Si True, la colonne de temps sera définie comme index du DataFrame.
    
    offset_unit : str, default="s"
        L'unité de temps pour le décalage à ajouter aux doublons. Exemples : 's' pour secondes, 'ms' pour millisecondes.
    
    keep_original : bool, default=False
        Si True, conserve la colonne de temps originale et crée une nouvelle colonne avec les timestamps ajustés.
    
    sort : bool, default=False
        Si True, trie le DataFrame par la colonne de temps avant de traiter les doublons.

    Returns:
    --------
    pd.DataFrame
        Un DataFrame avec des timestamps uniques selon les paramètres spécifiés.
    """

    # Vérification de l'existence et du type de la colonne 'time'
    if time_column not in data.columns:
        raise ValueError(f"La colonne '{time_column}' n'existe pas dans le DataFrame.")


    if not pd.api.types.is_datetime64_any_dtype(data[time_column]):
        data[time_column] = pd.to_datetime(data[time_column])

   # Optionnel : trier le DataFrame pour un traitement plus fiable des doublons
    if sort:
        data = data.sort_values(by=time_column).reset_index(drop=True)
    
    # Créer une colonne 'offset' pour différencier les timestamps identiques
    data = data.assign(offset=pd.to_timedelta(data[time_column].groupby(data[time_column]).cumcount(), unit=offset_unit))
 
    # Ajuster les timestamps en ajoutant l'offset
    adjusted_time_column = f"{time_column}_unique"

    if keep_original:
        data[adjusted_time_column] = data[time_column] + data['offset']
    else:
        # Modifier directement la colonne de temps
        data[time_column] = data[time_column] + data['offset']
        adjusted_time_column = time_column
        
    # Optionnel : définir la colonne de temps ajustée comme index
    if set_as_index:
        data = data.set_index(adjusted_time_column)

    # Nettoyer en supprimant la colonne d'offset utilisée temporairement
    data = data.drop(columns='offset')
    return data

def prepare_evolving_data(processed_path:str, evolving_path:str)->None:
    """
    Prépare les données brutes pour être enregistrées dans evolving data.
    
    :param processed_path: Chemin du dossier contenant les fichiers bruts.
    :param evolving_path: Chemin du dossier où enregistrer les fichiers traités.
    """
    # Charger et concaténer les données brutes depuis le dossier spécifié
    data = load_and_concat_processed_data(processed_path)
    # Trier les données par ordre chronologique pour garantir une cohérence temporelle
    data = data.sort_values(["time"])
    # Appliquer l'encodage one-hot sur la colonne 'asset'
    data = encoding_asset(data,dir_models,"train")
    # Ajuster les timestamps pour s'assurer qu'ils soient uniques
    training_data =  make_unique_time_index(data)
    # Sauvegarder le DataFrame final dans un fichier CSV pour une utilisation ultérieure
    training_data.to_csv(os.path.join(evolving_path, "xtb_data_train.csv"))
    logger.info("Données préparées et sauvegardées avec succès.")

# === Point d'entrée pour les tests ===
if __name__ == "__main__":
    # Exemple de test rapide du fichier
    pass
