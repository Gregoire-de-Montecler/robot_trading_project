# Dossier de Configuration

Ce dossier contient les fichiers de configuration utilisés pour paramétrer le projet. Il permet de définir les réglages globaux ainsi que les hyperparamètres pour le Machine Learning, tout en gardant certaines informations stratégiques confidentielles.

---

## Fichiers Présents

### 1. config.yaml

**Objectif :**  
Ce fichier définit les paramètres globaux du projet.

**Principaux paramètres :**

- **asset_type** :  
  Type d'actif traité (ici, `"forex"`).

- **base_path** :  
  Chemin de base pour le projet (exemple : `"~/Desktop/auto_ml"`).

- **tlb_time_frame** :  
  Intervalle de temps principal utilisé pour l'analyse (ici, `"15min"`).  
  *(Note : Ce paramètre détermine le timeframe de référence pour certaines opérations de trading.)*

- **time_frames** :  
  Liste complémentaire de time frames (par exemple : `"h"`, `"4h"`, `"D"`).

- **max_worker** :  
  Nombre maximal de threads ou processus à utiliser (ici, `16`).

- **seed** :  
  Graine pour assurer la reproductibilité (exemple : `12`).

- **ws_url** :  
  URL du WebSocket pour la connexion à l’API XTB (ici, `"wss://ws.xtb.com/demo"`).

- **risque** :  
  Niveau de risque configuré (ici, `10`).

---

### 2. model_params.yaml

**Objectif :**  
Ce fichier définit les hyperparamètres pour l’optimisation des modèles de Machine Learning, et est ici présenté uniquement avec l’exemple du **CatBoostClassifier** pour protéger l’ensemble de ton optimisation.

**Exemple de configuration pour CatBoostClassifier :**

- **eval_metric** :  
  Métrique d’évaluation, ici une valeur catégorielle avec par exemple `['AUC']`.

- **iterations** :  
  Nombre d’itérations pour l’entraînement (par exemple, un intervalle de `100` à `1500` avec un pas de `100`).

- **depth** :  
  Profondeur de l’arbre (de `4` à `12`).

- **learning_rate** :  
  Taux d’apprentissage (plage de `0.01` à `0.1`, avec échelle logarithmique).

- **l2_leaf_reg** :  
  Paramètre de régularisation (valeurs entre `1` et `10`).

- **bootstrap_type** :  
  Type de bootstrap, conditionnant certains autres paramètres.

- **bagging_temperature** :  
  Paramètre conditionnel lorsque `bootstrap_type` vaut `"Bayesian"` (valeur de `0.1` à `10`, par exemple).

- **subsample** :  
  Paramètre conditionnel lorsque `bootstrap_type` vaut `"Bernoulli"` (valeur de `0.5` à `1`).

- **leaf_estimation_iterations** et **min_data_in_leaf** :  
  Paramètres supplémentaires pour affiner la configuration du modèle.

*Note : Les configurations pour d’autres modèles ont été volontairement omises afin de ne pas divulguer l’intégralité de mon optimisation d’hyperparamètres.*

---

## Modification et Utilisation

- **Personnalisation :**  
  Vous pouvez adapter les paramètres dans **config.yaml** en fonction de vos besoins ou de votre environnement de travail (par exemple, modifier le `base_path` ou ajuster le `tlb_time_frame`).

- **Adaptation des hyperparamètres :**  
  Dans **model_params.yaml**, l’exemple de configuration pour **CatBoostClassifier** peut être ajusté selon les expérimentations menées, tout en gardant la structure conditionnelle pour certains paramètres.

---

Ce dossier centralise la configuration du projet et démontre l’attention portée à la flexibilité et à la sécurité des réglages.  
