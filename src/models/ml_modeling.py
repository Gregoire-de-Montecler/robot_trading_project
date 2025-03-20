
"""
Module Description:
-------------------
Décrire brièvement le but du fichier ou des fonctions qu'il contient.
Exemple : Ce module contient des fonctions pour [prétraitement, visualisation, etc.].
"""

# === Imports ===
# 1. Modules natifs Python
import os
import random
import logging
import threading
import tempfile
from datetime import timedelta
import optuna
import gc
# 2. Bibliothèques externes installées via pip
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import make_scorer
from inspect import signature
import joblib
import matplotlib.pyplot as plt
from sklearn.utils.validation import check_is_fitted
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor, BaggingRegressor
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.preprocessing import  MinMaxScaler, StandardScaler
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.linear_model import LogisticRegression
# 3. Modules internes du projet
from utils.logger import setup_logger
from utils.config import load_config
from utils.time_utils import split_time_series
from models.optuna_feature import suggest_features, GROUPS
from models.optuna_model import suggest_models , MODELS
from models.optuna_hyperparametre import prune_inconsistent_parameters, suggest_params
from utils.data_utils import evaluate_model
from utils.math_utils import calculate_evaluation, score_evaluation

from utils.config import Config, Model 
from multiprocessing import Lock
# === Constantes globales ===
test_size = Model.test_size
path = Config.path
asset_type = Config.asset_type
score_threshold = Model.score_threshold
dir_json = Config.results_json_path
dir_results = Config.results_csv_path
dir_processed = Config.processed_path
dir_raw = Config.raw_path
seed = Config.seed
dir_models = Config.models_path
dir_scaler = Config.scaler_path
save_lock = threading.Lock()
storage = Config.storage_path
storage = f"sqlite:///{os.path.join(storage,'optuna_storage.db')}"
joblib_temp_dir = tempfile.mkdtemp()
os.environ["JOBLIB_TEMP_FOLDER"] = joblib_temp_dir

NOM = "kumo_4h"
# === Configuration du logger ===
logger = setup_logger("models","ml_modeling")
logger.setLevel(logging.ERROR)
optuna.logging.get_logger("optuna").setLevel(logging.ERROR)

lock = Lock()
# === Classes ===


class Modelisation:
	"""
	Classe principale pour la modélisation des données. Gère le prétraitement, 
	l'entraînement des modèles, et l'analyse des résultats.
	"""


	def fit_predict(self, model, params,step=0):
		self.name_model = model.__name__
		logger.debug(f"training model {self.name_model}")
		# Gestion des paramètres comme 'verbose' et 'random_state'
		if 'verbose' in model.__init__.__code__.co_varnames:
			params["verbose"] = 0
		if 'random_state' in model.__init__.__code__.co_varnames:
			params["random_state"] = self.seed
		model_params = model(**params)
   
		weight = self.compute_weights(model_params, self.y_train)
		X_train, X_test = standard(self.X_train, self.X_test)
		X_train = X_train.drop(columns= "time_close")
		X_test = X_test.drop(columns= "time_close")
		model_fitted = model_params.fit(X_train, self.y_train,**weight)

		results = evaluate_model(X_test, self.y_test, model_fitted)

		reel = round(results["gain_resultat"].sum(),2)
		max = round(results[results["resultat"]==True]["gain_resultat"].sum(),2)
		min = round(results[results["resultat"]==False]["gain_resultat"].sum(),2)
		
  
		score = calculate_evaluation(results)
		logger.error(f"{step=} ,{score=} , {reel=} , {max=} , {min=}")
		return score

	def training_model(self,model,params,trial=None,save=False,n=0):
		"""
		Crée et entraîne un modèle avec des paramètres spécifiques (ex: `random_state`).
		"""

		self.name_model = model.__name__
		logger.debug(f"training model {self.name_model}")
		# Gestion des paramètres comme 'verbose' et 'random_state'
		if 'verbose' in model.__init__.__code__.co_varnames:
			params["verbose"] = 0
		if 'random_state' in model.__init__.__code__.co_varnames:
			params["random_state"] = self.seed


		scores = []

		tscv = split_time_series(total_length=self.X_train.shape[0], nb_split=5 ,pct_test= 0.05)
		reel = 0
		max = 0
		min = 0
		for step,(train_idx, test_idx) in enumerate(tscv):
			logger.debug(f"{train_idx=}")
			logger.debug(f"{test_idx=}")
			self_model = model(**params)
			
			last_trade = self.X_train.iloc[test_idx].index[0]
			train = self.X_train.iloc[train_idx]
			train_idx = train[train["time_close"]< last_trade].index
			
			
			X_train_fold, X_test_fold = self.X_train.loc[train_idx], self.X_train.iloc[test_idx]
			y_train_fold, y_test_fold, v_test_fold = self.y_train.loc[train_idx], self.y_train.iloc[test_idx], self.v_train.iloc[test_idx]
			weight = self.compute_weights(self_model, y_train_fold)
			
			X_train_fold = X_train_fold.drop(columns= "time_close")
			X_test_fold = X_test_fold.drop(columns= "time_close")

			X_train_fold, X_test_fold = standard(X_train_fold, X_test_fold)
	
   
		

			try:
				# Entraînez le modèle sur X_train_fold
				self_model = self_model.fit(X_train_fold, y_train_fold,**weight)
				logger.info(f"{self_model.get_params()=} for {self.name_model=}")
			except Exception as e:
				logger.error(f"Erreur lors de l'entraînement {e}")
				raise
			
			y_test_fold= pd.concat([y_test_fold,v_test_fold], axis=1)
			reel += round(y_test_fold["gain_resultat"].sum(),2)
			max += round(y_test_fold[y_test_fold["resultat"]==True]["gain_resultat"].sum(),2)
			min += round(y_test_fold[y_test_fold["resultat"]==False]["gain_resultat"].sum(),2)
			

			try:
				check_is_fitted(self_model)
			except:
				logger.error(f"{trial.number=} Pruned pour modèle non fit")
				raise optuna.exceptions.TrialPruned()
				
			else:
				results = evaluate_model(X_test_fold, y_test_fold, self_model)
				



			score = float(calculate_evaluation(results))
			if trial is not None :
				trial.report(score, step=step)
				# if trial.should_prune() and not save:# Vérification du pruning
				# 	logger.info(f"{trial.number=} Pruned")
				# 	raise optuna.exceptions.TrialPruned()# Rapport du score à Optuna
				
			scores.append(score)
	
		# Moyenne des scores sur tous les plis
		cv_train = np.array(scores)
		# self.score = np.median(cv_train[~np.isnan(cv_train)])

		resultat = cv_train[~np.isnan(cv_train)].sum()
		if trial.number == 299:
			logger.error(f"{n=}, {reel=} , {max=} , {min=}")
		return resultat
		
	def compute_weights(self, model,  y_train):
		"""
		Calcule les poids des classes et des échantillons pour l'entraînement.

		Parameters:
			y_train (pd.Series or np.ndarray): Labels du fold d'entraînement.
		Returns:
			dict: Un dictionnaire contenant "sample_weight" si les classes sont équilibrables, sinon {}.
		"""
		
		if y_train.dtype not in [np.bool_, np.int_, np.float_] or len(np.unique(y_train)) != 2:
			# Pas de calcul de poids si les conditions ne sont pas remplies
			return {}

		if np.sum(y_train == 0) == 0 or np.sum(y_train == 1) == 0:
			# Cas limite : une classe est absente
			logger.warning("Une classe est absente dans y_train_fold, pas de calcul des poids.")
			return {}
		pct_false = np.sum(y_train == 0) / y_train.shape[0]
		pct_true = np.sum(y_train == 1) / y_train.shape[0]
		class_weight = {0: 1 / pct_false, 1: 1 / pct_true}

		logger.debug(f"{pct_false=}, {pct_true=}" )
		if 'class_weight' in model.__init__.__code__.co_varnames:
			logger.debug(f"ajouts de class_weight dans les parametres de {self.name_model=} avec param = {model.get_params()}")
			model.set_params(class_weight =class_weight)
			
			
		elif 'scale_pos_weight' in model.__init__.__code__.co_varnames:
			logger.debug(f"ajouts de scale_pos_weight dans les parametres de {self.name_model=} avec param = {model.get_params()}")
			model.set_params(scale_pos_weight = (pct_false)/ (pct_true))
			
			
		elif self.name_model == 'BaggingClassifier':
			logger.debug(f"ajouts de base_est dans les parametres de {self.name_model=} avec param = {model.get_params()}")
			base_est = LogisticRegression(class_weight=class_weight)
			model.set_params(estimator = base_est)
			

		elif 'sample_weight' in signature(model.fit).parameters :
			logger.debug(f"ajouts de poids dans fit pour le model {self.name_model=}")
			sample_weights = np.array([class_weight[label] for label in y_train])
			return {"sample_weight" :sample_weights}

		logger.debug("wieght correctement calculé")
		return {}
 
class Regressor(Modelisation):
	"""
	Classe pour les modèles de régression.
	"""

	model_params = load_config(os.path.join(path,"config",'model_params.yaml')).get("regressor",{})


	def __init__(self,data_train: pd.DataFrame,data_test: pd.DataFrame = pd.DataFrame(), seed:int = 12, stage:str= ""):
		"""
		Initialise le Regressor avec des données et les traite pour le modèle.
		"""
		self.stage = stage
		self.seed = seed
		self.model_category = "regressor"
		self.train = data_train
		self.test = data_test
		self.X_train = data_train.drop(["resultat","gain_resultat"],axis=1)
		self.y_train = data_train["gain_resultat"]
		self.v_train = data_train["resultat"]
		self.val_train = data_train[["gain_resultat","resultat"]]
		if not data_test.empty:
			self.X_test = data_test.drop(["resultat","gain_resultat"],axis=1)
			self.y_test = data_test[["resultat","gain_resultat"]]
  
class Classifier(Modelisation):
	"""
	Classe pour les modèles de classification.
	"""
	model_params = load_config(os.path.join(path,"config",'model_params.yaml')).get("classifier",{})

	def __init__(self,data_train: pd.DataFrame,data_test: pd.DataFrame = pd.DataFrame(), seed:int = 12, stage:str= ""):
		"""
		Initialise le Classifier avec des données et les traite pour le modèle.
		"""
		self.stage = stage
		self.model_category = "classifier"
		self.seed = seed
		self.train = data_train
		self.test = data_test
		self.X_train = data_train.drop(["resultat","gain_resultat"],axis=1)
		self.y_train = data_train["resultat"]
		self.v_train = data_train["gain_resultat"]
		self.val_train = data_train[["gain_resultat","resultat"]]
		if not data_test.empty:
			self.X_test = data_test.drop(["resultat","gain_resultat"],axis=1)
			self.y_test = data_test[["resultat","gain_resultat"]]

   

# === Fonctions ===



def standard(X_train, X_test):
	"""
	Applique la normalisation (standardisation) sur les colonnes numériques.
	Sauvegarde également le scaler pour une réutilisation future.
	"""
	list_asset = [asset.strip(".csv") for asset in os.listdir(dir_raw)]
	numeric_columns  = list(X_train.select_dtypes(include=["number"]).columns)
	if not numeric_columns:
		return X_train, X_test


	
	last_date = X_train.index[0]
	
	first_date = X_train.index[0] - timedelta(days=365)

	for asset in list_asset:

		asset_1 = asset[:3]
		asset_2 = asset[3:]
		asset_train = X_train[(X_train[asset_1] == True) & (X_train[asset_2] == True)]
		
		asset_test = X_test[(X_test[asset_1] == True) & (X_test[asset_2] == True)]
		asset_train_index = asset_train.index
		asset_test_index = asset_test.index
		scaler = MinMaxScaler()
		df_asset = pd.read_csv(os.path.join(dir_processed,f"{asset}.csv"))
		df_asset["time"] = pd.to_datetime(df_asset["time"])
		df_asset = df_asset[(df_asset["time"]> first_date)& (df_asset["time"]< last_date)]
		scaler.fit(df_asset.loc[:,numeric_columns])
		if asset_train_index.to_list():
			X_train.loc[asset_train_index,numeric_columns] = scaler.transform(X_train.loc[asset_train_index,numeric_columns])
		if asset_test_index.to_list():
			X_test.loc[asset_test_index,numeric_columns] = scaler.transform(X_test.loc[asset_test_index,numeric_columns])


	return X_train, X_test

def search_features_models(data:pd.DataFrame, step:int):
	"""_summary_

		Args:
			data (pd.DataFrame): Dataframe
			step (int): etape pour le nom dans optuna
	"""
	stage = "features_models"
	
	def objective(trial):
		model_name = suggest_models(trial)
		feature = suggest_features(trial)
		data_feature = data[feature]
		model = MODELS[model_name]
		if "Regressor" in model_name:
			model_instance = Regressor(data_train=data_feature,seed=seed,stage=stage)
		else:
			model_instance = Classifier(data_train=data_feature,seed=seed,stage=stage)
		# params = model_instance.model_params.get(model_name,{}).get("param",{})
		params = {}
		resultat = model_instance.training_model(model,params,trial,n=step)

		return resultat

	
	storage = optuna.storages.RDBStorage(
		url="postgresql://optuna_user:optuna@localhost:5432/cat_boost_decembre_mono_feature",
		# Vous pouvez ajuster des paramètres supplémentaires
		engine_kwargs={
			"pool_size": 20,       
			"max_overflow": 10,
			"pool_recycle": 1800,
			"pool_pre_ping": True,
			"connect_args": {"options": "-c client_encoding=UTF8"}
		}
	)

	study = optuna.create_study(
		study_name=f"{step}_optimization",
		storage=storage,#f"sqlite:///dbp.sqlite3",# f"sqlite:///{step}.db",
		pruner=optuna.pruners.SuccessiveHalvingPruner(),
		sampler=optuna.samplers.TPESampler(seed=seed),
		direction="maximize",
		load_if_exists=True
			)   
	
	n_trial = 150 - len(study.trials)
 
	if n_trial == 0:
		best_model_feature = study.best_params
	else:
		study.optimize(objective,n_trials=n_trial,n_jobs=1)
		best_model_feature = study.best_params
	logger.error(f"{step=}, {study.best_trial.value=}")
	
	storage.engine.dispose()
	# best_model_feature = study.best_params
	best_model = best_model_feature.get("model","")
	features = []
	
	group_features = []
	# features = GROUPS["mandatory"]
	# for cle,value in best_model_feature.items():
	# 	features.append(value)
 
	for cle, valeur in best_model_feature.items():
		
		if valeur is True:
			group_features.append(cle)
			features.extend(GROUPS.get(cle, []))
   
	if study.best_trial.value <=0:
		return None, None
 
	return best_model , features

def search_hyperparametre(data:pd.DataFrame,step:int,model_name:str,features:list):
	stage = "search_hyperparametre"

	data = data[features]
	model = MODELS[model_name]
	def objective(trial):

		
		if "Regressor" in model_name:
			model_instance = Regressor(data_train=data,seed=seed,stage=stage)
		else:
			model_instance = Classifier(data_train=data,seed=seed,stage=stage)
		dict_params = model_instance.model_params.get(model_name,{}).get("param",{})
		params = suggest_params(dict_params,trial)
		prune_inconsistent_parameters(params)
		resultat = model_instance.training_model(model,params,trial)

		return resultat
    
	storage = optuna.storages.RDBStorage(
		url="postgresql://optuna_user:optuna@localhost:5432/optuna_fevrier_params",
		# Vous pouvez ajuster des paramètres supplémentaires
		engine_kwargs={
			"pool_size": 20,       
			"max_overflow": 10,
			"pool_recycle": 1800,
			"pool_pre_ping": True,
			"connect_args": {"options": "-c client_encoding=UTF8"}
		}
	)

	study = optuna.create_study(
		study_name=f"{step}_optimization",
		storage=storage,
		pruner=optuna.pruners.SuccessiveHalvingPruner(),
		sampler=optuna.samplers.TPESampler(seed=seed),
		direction="maximize",
		load_if_exists=True
			)   


	n_trial = 120 - len(study.trials)
 
	if n_trial == 0:
		best_params = study.best_params
	else:
		study.optimize(objective,n_trials=n_trial,n_jobs=1)
		best_params = study.best_params
	# try:
	# 	best_params = study.best_params
	# except :
	# study.optimize(objective,n_trials=120,n_jobs=1)
	# best_params = study.best_params

		
	storage.engine.dispose()
	return best_params

def predict_test(X_train:pd.DataFrame,X_test:pd.DataFrame,model_name:str,feature:list,params:dict= {},step:int=0):
	stage = "predict"
	X_train = X_train[feature]
	X_test = X_test[feature]
	model = MODELS[model_name]
	if "Regressor" in model_name:
		model_instance = Regressor(X_train, X_test,seed,stage)
	else:
		model_instance = Classifier(X_train, X_test,seed,stage)
	# params = model_instance.model_params.get(model_name,{}).get("param",{})
	resultat = model_instance.fit_predict(model, params,step)

	return resultat
    
    


# === Point d'entrée pour les tests ===
if __name__ == "__main__":
    # Exemple de test rapide du fichier
    print(MODELS.keys())
    pass
