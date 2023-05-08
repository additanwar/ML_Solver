from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb

from .modeling import ModelExecutor

# The search space for the algorithm
# we run multiple trials to find the best value to use for each hyperparameter

class ModelAssembler:
    """
    Generates the list of model classes (based on user choice or defaults)
    """
    def __init__(self, 
                task, 
                x_train,
                y_train,
                model_list:list[str]=["lgb","rf","xgb"],
                metric:str="",
                normalize:bool=True,
                class_weights={0:1,1:1},
                tune_flag:bool=True,
                cv:str="basic",
                n_fold:int=5,
                internal_val:bool=False,
                final_train_flag:bool=True) -> None:
        self.task = task
        self.model_list = model_list
        self.x_train = x_train
        self.y_train = y_train
        self.normalize = normalize
        self.metric = metric
        self.task = task
        self.class_weights = class_weights
        self.tune_flag = tune_flag
        self.cv = cv
        self.n_fold = n_fold
        self.internal_val = internal_val
        self.final_train_flag = final_train_flag
        
    def get_model_config(self):
        MODEL_CLASSES = {
            "regression": {
            "lgb": lgb.LGBMRegressor,
            "svm": SVR,
            "nn": MLPRegressor,
            "lr": LinearRegression,
            "rf": RandomForestRegressor,
            "xgb": xgb.XGBRegressor,
            "knn": KNeighborsRegressor},
        "classification": {
            "lgb": lgb.LGBMClassifier,
            "svm": SVC,
            "nn": MLPClassifier,
            "lr": LogisticRegression,
            "rf": RandomForestClassifier,
            "xgb": xgb.XGBClassifier,
            "knn": KNeighborsClassifier
        }}
        SEARCH_SPACES = {"lgb": {'num_leaves': (5, 500, 5),
                    'max_depth': (2, 100, 2),
                    'learning_rate': (0.001, 0.2, 0.001),
                    'n_estimators': (20, 500, 20),
                    'min_child_samples': (5, 100, 5), 
                    'reg_lambda': (0.1, 1, 0.1),
                    'reg_alpha': (0.1, 1, 0.1),
                    "random_state": 42,
                    'metric': "auc"},
        "svm": {"C": (100, 1000, 100),
                "kernel": ['poly','rbf','linear'],
                "gamma": ['scale', 'auto'],
                "degree": (2,10,2),
                "max_iter": 10000},
        "nn": {"activation": ["relu"],
                "alpha": (0.001, 1, 0.01),
                "learning_rate": ['adaptive'],
                "hidden_layer_sizes": [(20,),(20,10),(20,10,5),
                                        (50,),(50,20),(50,20,10),(50,20,10,5),(50,10),
                                        (10,),(10,5,2),(100,50,20),(100,50,20,10),(100,20,10,5)],
                "learning_rate_init": (0.001, 0.1, 0.001),
                "early_stopping": True,
                "max_iter": (100, 1000, 100),
                "solver": ["sgd","adam"],
                "random_state": 42},
        "lr": {"penalty": ["l2","none"],
                "C": [0.01,0.1,0.25,0.5,1,10,100,1000],
                "tol": [2e-1,1.5e-1,1e-1,1e-2,1e-3,1e-4,1e-5]},
        "rf": {'max_depth': (2, 50, 2),
                    'n_estimators': (20, 500, 20),
                    'min_samples_leaf': (5, 100, 5),
                    'min_samples_split': (2, 20, 2),
                    "max_features": [None,"log2","sqrt"],
                    "random_state": 42},
        "xgb": {
            "n_estimators": (100, 1000, 100),
            "learning_rate": (0.01, 0.1, 0.01),
            "subsample": (0.5, 1, 0.1),
            "gamma": (0, 0.5, 0.1),
            "min_child_weight": (1, 6, 2),
            "max_depth": (2,100,2),
            "lambda": (0.25,1,0.25),
            "alpha": (0.25,1,0.25),
            "random_state": 42,
            "eval_metric": "auc"},
        "knn": {
            "n_neighbors": (5,50,5),
            "weights": ["uniform","distance"]}
            }
        model_configs = {}
        for model_name in self.model_list:
            model_configs[model_name] = {"model_class": MODEL_CLASSES[self.task][model_name],
                                        "param_grid": SEARCH_SPACES[model_name]}
            if self.task=="regression":
                if model_name=="xgb":
                    model_configs[model_name]["param_grid"]["eval_metric"] = "rmse"
                elif model_name=="lgb":
                    model_configs[model_name]["param_grid"]["metric"] = "rmse"
        return model_configs

    def run_tuning(self, max_evals):
        # get model configs (model classes and parameter search grid)
        model_configs = self.get_model_config()
        self.model_dict = {}
        # performing tuning and model training
        for model_name, model_config in model_configs.items():
            # Tuning and finding the best configured model
            tuner = ModelExecutor(model_config["model_class"],
                                    self.x_train,
                                    self.y_train,
                                    normalize=self.normalize,
                                    param_grid=model_config["param_grid"],
                                    metric=self.metric,
                                    task=self.task,
                                    class_weights=self.class_weights,
                                    tune_flag=self.tune_flag,
                                    cv=self.cv,
                                    n_fold=self.n_fold,
                                    internal_val=self.internal_val,
                                    final_train_flag=self.final_train_flag)
            tuner.execute(max_evals)
            self.model_dict[model_name] = {"model": tuner.model,
                                        "val_results": tuner.best_val_results}