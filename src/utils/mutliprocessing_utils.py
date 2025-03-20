"""
Module Description:
-------------------
Ce module contient des fonctions utilitaires pour la gestion du parallélisme avec multiprocessing.
"""

# === Imports ===
# 1. Modules natifs Python
from multiprocessing import Pool, cpu_count
import logging
import os
from multiprocessing.pool import ThreadPool

# 2. Bibliothèques externes installées via pip
# Aucune

# 3. Modules internes du projet

from utils.logger import setup_logger

# === Configuration du logger ===
logger = setup_logger("utils", "multiprocessing")
logger.setLevel(logging.INFO)
# === Constantes globales ===
MAX_WORKERS = cpu_count()

# === Fonctions ===
def multiprocessing_wrapper(func, args_list, num_processes=16):
    """
    Exécute une fonction en parallèle avec des threads.
    
    Args:
        func (callable): La fonction à exécuter.
        args_list (list): Une liste d'arguments (chaque élément sera passé à func).
        num_processes (int): Nombre de processus à utiliser.
    
    Returns:
        list: Résultats de chaque exécution de func.
    """
    try:
        with Pool(processes=num_processes) as pool:
            results = pool.starmap(func, args_list)  # Utiliser starmap pour déballer les tuples
        return results
    except Exception as e:
        logger.error(f"Error in multiprocessing wrapper: {e}")
        raise


def multithreading_wrapper(func, args_list, num_threads=16):
    """
    Exécute une fonction en parallèle avec des threads.
    
    Args:
        func (callable): La fonction à exécuter.
        args_list (list): Une liste d'arguments (chaque élément sera passé à func).
        num_threads (int): Nombre de threads à utiliser.
    
    Returns:
        None
    """
    def worker(arg):
        try:
            return func(*arg)
        except Exception as e:
            logger.error(f"Error while processing {arg}: {e}")
            raise

    with ThreadPool(processes=num_threads) as pool:
       results = pool.map(worker, args_list)
    return results

# === Point d'entrée pour les tests ===
if __name__ == "__main__":
    # Exemple d'utilisation
    pass
