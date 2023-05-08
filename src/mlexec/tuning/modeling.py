import warnings
from typing import Any

import numpy as np
import pandas as pd

from sklearn.model_selection import cross_validate, KFold
import sklearn.metrics as skmetrics
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from hyperopt import fmin, hp, tpe, STATUS_OK, Trials
from hyperopt.pyll.base import scope

warnings.filterwarnings('ignore')

class ModelExecutor:
    def __init__(self,
                model_class,
                x_train: pd.DataFrame,
                y_train: pd.Series,
                param_grid: dict[str, Any]={},
                metric:str="",
                normalize:bool=True,
                model_type:str="sklearn",
                task:str="classification",
                class_weights:dict[int, int]={0:1,1:1},
                tune_flag:bool=True,
                cv:str="basic",
                n_fold:int=5,
                internal_val:bool=False,
                x_val=None,
                y_val=None,
                fit_params:dict[str, Any]={},
                final_train_flag:bool=True) -> None:
        """
        ### MODEL OPTIONS
        By default the normalize option is set to true and the class weights are 1:1.
        fit_params are used in the `fit` method of a model
        ### VALIDATION OPTIONS
        x_val, y_val are only used if cv is None or False.
        self_val indicates if the model can run validation internally and perform early stopping.
        (For instance xgboost or LightGBM)
        """
        self.best_params: dict[str, Any] = {}
        self.x_train = x_train
        self.y_train = y_train
        self._task = task

        self.primary_metric = metric
        self.class_weights = class_weights
        self._create_scorers()

        self.tune_flag = tune_flag
        self.model_type = model_type

        if self.model_type == "sklearn":
            if normalize:
                self.base_model = Pipeline([('norm', StandardScaler()),
                            ('clf', model_class())])
            else:
                self.base_model = Pipeline([('clf', model_class())])
        else:
            self.model_class = model_class

        self.cv = cv
        if self.primary_metric in ["rmse","mae","mape","misclassification_cost"]:
            self.objective = "min"
        elif self.primary_metric in ["r2","matthews_corrcoef","accuracy",
                                "recall","precision","f1_score","auc_roc","auc_pr"]:
            self.objective = "max"

        if self.tune_flag:
            self.n_fold = n_fold
            self.adapt_grid(param_grid)

        self.internal_val = internal_val
        if internal_val:
            self.x_val = x_val
            self.y_val = y_val

        self.fit_params = fit_params
        self.final_train_flag = final_train_flag

    def adapt_grid(self, param_grid: dict[str, Any]):
        """
        Adapt the paramter grid to fit the needs of bayesian optimizer
        """
        self.param_grid = {}
        # Saving the original param space so that we can adapt int and choice
        self.choice_params = {}
        self.int_params = []

        for k, v in param_grid.items():
            modified_key = "clf__"+k
            if type(v) in [str, bool, int, float, dict]:
                self.param_grid[modified_key] = v
            elif isinstance(v, list):
                # Lists are considered as choice metrics
                # Tuples are used for others
                self.choice_params[modified_key] = v
                self.param_grid[modified_key] = hp.choice(modified_key, v)
            elif len(v)==3:
                # Step size is provided
                if isinstance(v[0], int):
                    self.int_params.append(modified_key)
                    #quniform returns float, some parameters require int; use this to force int
                    self.param_grid[modified_key] = scope.int(hp.quniform(modified_key, *v))
                else:
                    self.param_grid[modified_key] = hp.quniform(modified_key, *v)
            elif isinstance(v[0], int):   
                self.param_grid[modified_key] = hp.randint(modified_key, *v)
            elif isinstance(v[0], float):
                self.param_grid[modified_key] = hp.uniform(modified_key, *v)

    def execute(self, max_evals=40):
        if self.tune_flag:
            if self.cv:
                self.inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)
                if self.cv=="nested":
                    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
            self.tune_model(max_evals)

        if self.final_train_flag:
            self.train_best_model()

        if self.cv=="nested":
            # self.tune_flag should be true in this scenario
            self.best_val_results = cross_validate(self.model,
                                            X=self.x_train,
                                            y=self.y_train,
                                            cv=outer_cv,
                                            scoring=self.metric_collection)

    def tune_model(self, max_evals):
        trials = Trials()
        self.best_params = fmin(fn=self.run_model_iter,
                space=self.param_grid,
                max_evals=max_evals,
                algo=tpe.suggest,
                verbose=False,
                trials=trials)

        self.best_val_results = trials.results[np.argmin(
                            [t['loss'] for t in trials.results]
                            )]['val_results']

        # Replacing index with values
        for i in self.choice_params:
            best_index = self.best_params[i]
            self.best_params[i] = self.choice_params[i][best_index]
        for i in self.int_params:
            self.best_params[i] = round(self.best_params[i])

    def get_val_results(self, model):
        val_results = {}
        for metric_name, metric_scorer in self.metric_collection.items():
            val_results["test_"+metric_name] = metric_scorer(model,
                                                self.x_val,
                                                self.y_val)
        return val_results

    def run_model_iter(self, params):
        if self.model_type == "sklearn":
            model = self.base_model.set_params(**params)
        else:
            model = self.model_class(**params)

        if self.cv:
            val_results = cross_validate(model,
                                        self.x_train,
                                        self.y_train,
                                        cv=self.inner_cv,
                                        scoring=self.metric_collection,
                                        error_score='raise')
        else:
            model.fit(X=self.x_train,
                    y=self.y_train,
                    **self.fit_params)
            val_results = self.get_val_results(model)
        loss = val_results["test_"+self.primary_metric].mean()

        if self.objective=="max":
            # since this is a minimization opt fn
            loss = loss*-1

        return {"loss": loss,
                'val_results': val_results,
                'params': params,
                'status': STATUS_OK}

    def train_best_model(self):
        if self.model_type=="sklearn":
            self.model = self.base_model.set_params(**self.best_params)
        else:
            self.model = self.model_class(**self.best_params)
        if not self.cv and self.tune_flag:
            self.x_train = pd.concat([self.x_train,self.x_val])
            self.y_train = pd.concat([self.y_train,self.y_val])
        self.model.fit(self.x_train, self.y_train, **self.fit_params)

    def __repr__(self) -> str:
        if self.model:
            return str(self.model[-1].__class__).split('.')[-1].split("'")[0]
        return str(self.__class__)

    @staticmethod
    def convert_metrics_to_scorers(metrics: dict[str, Any]):
        scorers = {}
        for k,v in metrics.items():
            scorers[k] = skmetrics.make_scorer(v)
        return scorers

    @staticmethod
    def get_weighted_predictions(y_test, y_pred, class_weights):
        weights = list(class_weights.values())
        tn, fp, fn, tp = skmetrics.confusion_matrix(y_test, y_pred).ravel()
        return tn*weights[0], fp*weights[0], fn*weights[1], tp*weights[1]

    def misclassification_cost(self, y_test, y_pred):
        _, w_fp, w_fn, _ = ModelExecutor.get_weighted_predictions(
            y_test, y_pred, self.class_weights)
        return (w_fp + w_fn)/len(y_test)

    def weighted_accuracy(self, y_test, y_pred):
        w_tn, w_fp, w_fn, w_tp = ModelExecutor.get_weighted_predictions(
            y_test, y_pred, self.class_weights)
        return (w_tp+w_tn)/(w_tn+ w_fp+ w_fn+ w_tp)

    def weighted_f1_score(self, y_test, y_pred):
        _, w_fp, w_fn, w_tp = ModelExecutor.get_weighted_predictions(
            y_test, y_pred, self.class_weights)
        return w_tp / (w_tp+0.5*(w_fp+w_fn))

    def _create_scorers(self):
        if self._task=="regression":
            _REGRESSION_METRICS = {
            "rmse": lambda y_test,y_pred: skmetrics.mean_squared_error(y_test,y_pred)**0.5,
            "r2": skmetrics.r2_score,
            "mae": skmetrics.mean_absolute_error,
            "mape": skmetrics.mean_absolute_percentage_error
            }
            self.metric_collection = ModelExecutor.convert_metrics_to_scorers(_REGRESSION_METRICS)
        elif self._task=="classification":
            _CLASSIFICATION_METRICS = {
            "matthews_corrcoef": skmetrics.matthews_corrcoef,
            "misclassification_cost": self.misclassification_cost,
            "accuracy": self.weighted_accuracy,
            "recall": skmetrics.recall_score,
            "precision": skmetrics.precision_score,
            "f1_score": self.weighted_f1_score,
            "auc_roc": skmetrics.roc_auc_score,
            "auc_pr": skmetrics.average_precision_score,
            }
            self.metric_collection = ModelExecutor.convert_metrics_to_scorers(
                _CLASSIFICATION_METRICS)

        # We need to set the primary metric if its none
        if not self.primary_metric:
            self.primary_metric = list(self.metric_collection.keys())[0]
