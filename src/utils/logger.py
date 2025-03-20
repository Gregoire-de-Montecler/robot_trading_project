
"""
Module : Logger Utility
------------------------
Fournit une configuration centralisée pour le logging du projet.
"""

# === Imports ===
# 1. Modules natifs Python
import os
import logging
from logging.handlers import RotatingFileHandler

# 2. Bibliothèques externes installées via pip



# 3. Modules internes du projet

# === Constantes globales ===

# === Configuration du logger ===
LOG_PATH = os.path.join(os.path.expanduser("~"), "Desktop", "auto_ml", "logs")
# === Classes ===



# === Fonctions ===
def setup_logger(category: str = "general", name: str ="Trading_project" , log_dir: str = LOG_PATH ) -> logging.Logger :
    """
    Configure et retourne un logger avec des handlers pour les logs de fichier et de console.

    Args:
        name (str): Le nom du logger. Par défaut, "Trading_project".
        log_dir (str, optional): Le répertoire où les fichiers de logs seront stockés. 
                                 Si None, utilise un chemin par défaut sur le bureau.

    Returns:
        logging.Logger: Une instance de logger configurée avec des handlers.

    Raises:
        OSError: Si le répertoire de logs ne peut pas être créé.

    Example:
        >>> logger = setup_logger(name="MyApp", log_dir="./logs")
        >>> logger.info("Logger configuré avec succès.")
    """

    
        # Définir le chemin du sous-dossier de catégorie
    category_dir = os.path.join(log_dir, category)
    os.makedirs(category_dir, exist_ok=True)  # Créer le dossier si nécessaire
    
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger  # Évite une double configuration

    logger.setLevel(logging.DEBUG)

    log_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler = RotatingFileHandler(
        os.path.join(category_dir, f"{name}.log"),
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(log_format)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(log_format)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
    


# === Point d'entrée pour les tests ===
if __name__ == "__main__":
    # Exemple de test rapide du fichier
    logger = setup_logger()
    logger.info("Logger configuré avec succès dans logger.py.")

