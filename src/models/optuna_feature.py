
"""
Module Description:
-------------------
Décrire brièvement le but du fichier ou des fonctions qu'il contient.
Exemple : Ce module contient des fonctions pour [prétraitement, visualisation, etc.].
"""

# === Imports ===
# 1. Modules natifs Python

import logging


import optuna
# 2. Bibliothèques externes installées via pip

# 3. Modules internes du projet
from utils.logger import setup_logger


# === Constantes globales ===
# GROUPS =  {
#     "croisement_stoch" : ["croisement_stoch_15min","croisement_stoch_4h","croisement_stoch_D"],#,"croisement_stoch_h"],
#     # "crossover_macd" : ["crossover_macd_4h","crossover_macd_15min","crossover_macd_D","crossover_macd_h"],
#     # "diff_ema_100" : ["diff_close_ema_100_15min","diff_close_ema_100_4h","diff_close_ema_100_D","diff_close_ema_100_h"],
#     "diff_ema_13" : ["diff_close_ema_13_15min"],#,"diff_close_ema_13_4h","diff_close_ema_13_D","diff_close_ema_13_h"],
#     # "diff_ema_26" : ['diff_close_ema_26_15min','diff_close_ema_26_4h','diff_close_ema_26_D','diff_close_ema_26_h'],
#     # "diff_stop" : ['diff_close_stop'],
#     # "diff_ema_13_26" : ['diff_ema_13_26_15min','diff_ema_13_26_4h',"diff_ema_13_26_D","diff_ema_13_26_h"],
#     # "diff_pivot_high_stoch" : ["diff_pivot_high_d_stoch_15min","diff_pivot_high_d_stoch_4h","diff_pivot_high_d_stoch_D","diff_pivot_high_d_stoch_h"],
#     # "diff_pivot_high_rsi" : ["diff_pivot_high_rsi_15min","diff_pivot_high_rsi_4h","diff_pivot_high_rsi_h"],#,"diff_pivot_high_rsi_D"
#     "diff_pivot_low_stoch" : ["diff_pivot_low_d_stoch_15min","diff_pivot_low_d_stoch_h",],#"diff_pivot_low_d_stoch_4h","diff_pivot_low_d_stoch_D","diff_pivot_low_d_stoch_h"],
#     "diff_pivot_low_rsi" : ["diff_pivot_low_rsi_15min","diff_pivot_low_rsi_4h"],#,"diff_pivot_low_rsi_D","diff_pivot_low_rsi_h"],
#     # "evolution" : ["evolution_15min"],
#     # "ecart_chikou_close" : ["ecart_chikou_close_15min","ecart_chikou_close_4h","ecart_chikou_close_D","ecart_chikou_close_h"],
#     # "e_r_chikou_kumo" : ["ecart_relative_chikou_kumo_15min","ecart_relative_chikou_kumo_4h","ecart_relative_chikou_kumo_D","ecart_relative_chikou_kumo_h"],
#     # "e_r_chikou_sen" : ["ecart_relative_chikou_sen_15min","ecart_relative_chikou_sen_4h","ecart_relative_chikou_sen_D","ecart_relative_chikou_sen_h"],
#     "e_r_kumo" : ["ecart_relative_kumo_4h","ecart_relative_kumo_15min","ecart_relative_kumo_h"],#,"ecart_relative_kumo_D"
#     # "e_r_sen" : ["ecart_relative_sen_15min","ecart_relative_sen_4h","ecart_relative_sen_D","ecart_relative_sen_h"],
#     "ep_chikou_kumo" : ["epaisseur_chikou_kumo_h",],#"epaisseur_chikou_kumo_15min","epaisseur_chikou_kumo_4h","epaisseur_chikou_kumo_D",],
#     # "ep_chikou_sen" : ["epaisseur_chikou_sen_15min","epaisseur_chikou_sen_4h","epaisseur_chikou_sen_D","epaisseur_chikou_sen_h"],
#     # "ep_kumo" : ["epaisseur_kumo_15min","epaisseur_kumo_4h","epaisseur_kumo_D","epaisseur_kumo_h"],
#     # "ep_sen" : ["epaisseur_sen_15min","epaisseur_sen_4h","epaisseur_sen_D","epaisseur_sen_h"],
#     # "macd_line":["macd_line_15min","macd_line_4h","macd_line_D","macd_line_h"],
#     # "pos_chikou_kumo" : ["position_prix_chikou_kumo_15min","position_prix_chikou_kumo_4h","position_prix_chikou_kumo_D","position_prix_chikou_kumo_h"],
#     # "pos_chikou_sen" : ["position_prix_chikou_sen_15min","position_prix_chikou_sen_4h","position_prix_chikou_sen_D","position_prix_chikou_sen_h"],
#     # "position_prix_kumo_4h" : ["position_prix_kumo_4h","position_prix_kumo_15min","position_prix_kumo_D","position_prix_kumo_h"],
#     # "pos_sen" : ["position_prix_sen_15min","position_prix_sen_4h","position_prix_sen_D","position_prix_sen_h"],
#     "rsi_h" : ["rsi_h",],#"rsi_4h","rsi_15min","rsi_D",],
#     # "val_stoch" : ["d_stoch_4h","d_stoch_15min","d_stoch_D","d_stoch_h"],
#     # "s_pos_MM" : ["suite_position_MM_15min","suite_position_MM_4h","suite_position_MM_D","suite_position_MM_h"],
#     "s_pos_rsi" : ["suite_position_rsi_15min"],#,"suite_position_rsi_4h","suite_position_rsi_D","suite_position_rsi_h"],
#     # "s_pos_stoch" : ["suite_position_stoch_15min","suite_position_stoch_4h","suite_position_stoch_D","suite_position_stoch_h"],
#     "s_pos_chikou_kumo" : ["suite_position_chikou_kumo_15min"],#,"suite_position_chikou_kumo_4h","suite_position_chikou_kumo_D","suite_position_chikou_kumo_h"],
#     # "s_pos_chikou_sen" : ["suite_position_chikou_sen_15min","suite_position_chikou_sen_4h","suite_position_chikou_sen_D","suite_position_chikou_sen_h"],
#     # "s_pos_kumo" : ["suite_position_kumo_15min","suite_position_kumo_4h","suite_position_kumo_D","suite_position_kumo_h"],
#     "s_pos_sen" : ["suite_position_sen_15min","suite_position_sen_h"],#,"suite_position_sen_4h","suite_position_sen_D"],
#     # "zone_MM" : ["zone_MM_15min","zone_MM_4h","zone_MM_D","zone_MM_h"],
#     "zone_rsi" : ["zone_rsi_h","zone_rsi_15min"],#,"zone_rsi_D","zone_rsi_4h"],
#     # "zone_stoch" : ["zone_stoch_4h","zone_stoch_15min","zone_stoch_D","zone_stoch_h"],
#     "mandatory" :["gain_resultat","long","resultat","time_close","AUD","CAD","CHF","EUR","GBP","JPY","NOK","NZD","SEK","SGD","USD"],
#     }


GROUPS =  {
    "croisement_stoch_15min" : ["croisement_stoch_15min"],
    "croisement_stoch_4h" : ["croisement_stoch_4h"],
    "croisement_stoch_D" : ["croisement_stoch_D"],
    "croisement_stoch_h" : ["croisement_stoch_h"],
    "diff_pivot_low_d_stoch_15min" : ["diff_pivot_low_d_stoch_15min"],
    "diff_pivot_low_d_stoch_h" : ["diff_pivot_low_d_stoch_h"],
    "diff_pivot_low_d_stoch_4h" : ["diff_pivot_low_d_stoch_4h"],
    "diff_pivot_low_d_stoch_D" : ["diff_pivot_low_d_stoch_D"],
    "diff_pivot_low_rsi_15min" : ["diff_pivot_low_rsi_15min"],
    "diff_pivot_low_rsi_4h" : ["diff_pivot_low_rsi_4h"],
    "diff_pivot_low_rsi_D" : ["diff_pivot_low_rsi_D"],
    "diff_pivot_low_rsi_h" : ["diff_pivot_low_rsi_h"],
    "ecart_relative_kumo_4h" : ["ecart_relative_kumo_4h"],
    "ecart_relative_kumo_15min" : ["ecart_relative_kumo_15min"],
    "ecart_relative_kumo_h" : ["ecart_relative_kumo_h"],
    "ecart_relative_kumo_D" : ["ecart_relative_kumo_D"],
    "epaisseur_chikou_kumo_h" : ["epaisseur_chikou_kumo_h"],
    "epaisseur_chikou_kumo_15min" : ["epaisseur_chikou_kumo_15min"],
    "epaisseur_chikou_kumo_4h" : ["epaisseur_chikou_kumo_4h"],
    "epaisseur_chikou_kumo_D" : ["epaisseur_chikou_kumo_D"],
    "rsi_h" : ["rsi_h"],
    "rsi_4h" : ["rsi_4h"],
    "rsi_15min" : ["rsi_15min"],
    "rsi_D" : ["rsi_D"],
    "suite_position_sen_15min" : ["suite_position_sen_15min"],
    "suite_position_sen_h" : ["suite_position_sen_h"],
    "suite_position_sen_4h" : ["suite_position_sen_4h"],
    "suite_position_sen_D" : ["suite_position_sen_D"],
    "diff_pivot_high_rsi_15min" : ["diff_pivot_high_rsi_15min"],
    "diff_pivot_high_rsi_4h" : ["diff_pivot_high_rsi_4h"],
    "diff_pivot_high_rsi_h" : ["diff_pivot_high_rsi_h"],
    "diff_pivot_high_rsi_D" : ["diff_pivot_high_rsi_D"],
    "mandatory" :["gain_resultat","long","resultat","time_close","AUD","CAD","CHF","EUR","GBP","JPY","NOK","NZD","SEK","SGD","USD"],
    }

# GROUPS =  {
#     "feature" : ["croisement_stoch_15min","croisement_stoch_4h","croisement_stoch_D",
#                  "crossover_macd_4h","crossover_macd_h",
#                  "diff_close_ema_13_15min",
#                  "diff_pivot_low_rsi_15min","diff_pivot_low_rsi_4h",
#                  "diff_pivot_high_rsi_15min","diff_pivot_high_rsi_4h","diff_pivot_high_rsi_h",
#                  "ecart_relative_kumo_4h","ecart_relative_kumo_15min","ecart_relative_kumo_h",
#                  "epaisseur_chikou_kumo_h",
#                  "epaisseur_kumo_h","epaisseur_kumo_4h",
#                  "epaisseur_sen_15min","epaisseur_sen_4h","epaisseur_sen_D",
#                  "rsi_h","rsi_D","rsi_15min",
            
#                 ],
#     "mandatory" :["gain_resultat","long","resultat","time_close","AUD","CAD","CHF","EUR","GBP","JPY","NOK","NZD","SEK","SGD","USD"],
#     }
# === Configuration du logger ===
logger = setup_logger("models","optuna_feature")
logger.setLevel(logging.ERROR)
optuna.logging.get_logger("optuna").setLevel(logging.DEBUG)


# === Classes ===


   

# === Fonctions ===
def suggest_features(trial):
    selected_features = []    
    for group_name, group_features in GROUPS.items():
        if group_name == "mandatory":
                include_groupe = trial.suggest_categorical(f"{group_name}",[True])
        else:
                include_groupe = trial.suggest_categorical(f"{group_name}",[False,True])
        if include_groupe:
            selected_features.extend(group_features)
    return selected_features

# def suggest_features(trial):
#     selected_features = GROUPS["mandatory"]    
#     for i in range(3):
#         selected_features.append(trial.suggest_categorical(f"feature_{i}",GROUPS["feature"]))
#     return selected_features

# === Point d'entrée pour les tests ===
if __name__ == "__main__":
    # Exemple de test rapide du fichier
    
    pass
