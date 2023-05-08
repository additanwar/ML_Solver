from typing import Any

import sklearn.metrics as skmetrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class ModelEvaluator:
    """
    Takes a list of model executors and creates a dataframe of validation results and 
    evaluates the models to create the test metrics dataframe as well.
    """
    def __init__(self,
                x_test,
                y_test,
                model_dict:dict[str, Any]={},
                metric_list:list[str]=[],
                class_weights:dict[int,int]={0:1, 1:1},
                task:str="classification",
                optimization_criterion:str="f1_score"):
        if not hasattr(self, "model_dict"):
            self.model_dict = model_dict
            # Weights should have negative class key first
            self.class_weights = class_weights
            # tasks supported: Classification, Regression
            self.task = task 
        
        self.x_test = x_test
        self.y_test = y_test
        self.optimization_criterion = optimization_criterion
        
        if self.task=="regression":
            self.metric_collection_func_map = {
                "rmse": lambda y_test,y_pred: skmetrics.mean_squared_error(y_test,y_pred)**0.5,
                "r2": skmetrics.r2_score,
                "mae": skmetrics.mean_absolute_error,
                "mape": skmetrics.mean_absolute_percentage_error
                }
        elif self.task=="classification":
            self.metric_collection_func_map = {
                "accuracy": skmetrics.accuracy_score,
                "recall": skmetrics.recall_score,
                "precision": skmetrics.precision_score,
                "f1_score": skmetrics.f1_score,
                "matthews_corrcoef": skmetrics.matthews_corrcoef,
                "misclassification_cost": self.get_misclassification_cost,
                "auc_roc": skmetrics.roc_auc_score,
                "auc_pr": ModelEvaluator.get_aucpr,
            }
        self.metric_collection_list = metric_list if metric_list else list(self.metric_collection_func_map.keys())
        self.results:dict[str, dict[str, float]] = {}
        self.predictions = None

    @staticmethod
    def get_optimal_threshold(y_test, y_prob, class_weights, criterion):
        if criterion=="balanced_roc":
            fpr, tpr, thresholds = skmetrics.roc_curve(y_test, y_prob)
            optimal_idx = np.argmax(tpr*class_weights[0] - fpr*class_weights[1])
            optimal_threshold = thresholds[optimal_idx]
        else:
            precision, recall, thresholds = skmetrics.precision_recall_curve(y_test, y_prob)
            if criterion=="auc_pr":
                optimal_threshold = thresholds[np.argmax(recall+precision)]
            elif criterion=="balanced_pr":
                optimal_threshold = thresholds[np.argmax(recall*class_weights[0]+precision*class_weights[1])]
            elif criterion=="f1_score":
                optimal_threshold = thresholds[np.argmax(2*precision*recall/(precision+recall))]
        return optimal_threshold

    @staticmethod
    def get_optimal_predictions(y_test, y_prob, class_weights, criterion):
        optimal_threshold = ModelEvaluator.get_optimal_threshold(y_test, y_prob, class_weights, criterion)
        print("Threshold value is:", optimal_threshold)
        y_pred = (y_prob > optimal_threshold).astype(int)
        return y_pred
    
    def get_val_scores(self):
        """
        Only used for validation results
        """
        val_scores:dict[str, dict[str, float]] = {}
        for model_name, model_artifacts in self.model_dict.items():
            val_scores[model_name] = {}
            for k,v in model_artifacts["val_results"].items():
                if "test_" in k:
                    val_scores[model_name][k.replace("test_","")] = v.mean()
        self.val_results = pd.DataFrame(val_scores).transpose()
    
    def get_best_model(self):
        self.get_val_scores()
        primary_metric = self.metric_collection_list[0]
        if primary_metric in ["rmse","mae","mape","misclassification_cost"]:
            self.best_model_name = self.val_results[primary_metric].idxmin()
        else:
            self.best_model_name = self.val_results[primary_metric].idxmax()
        self.best_model = self.model_dict[self.best_model_name]["model"]
    
    def evaluate(self, x_test=None, y_test=None, find_best_threshold=False):
        """
        Executes evaluation for each model present in the attributes.
        """
        if x_test and y_test:
            self.x_test = x_test
            self.y_test = y_test
        self.get_best_model()
        
        y_pred = self.best_model.predict(self.x_test)

        if "auc_roc" in self.metric_collection_list or "auc_pr" in self.metric_collection_list:
            y_prob = self.best_model.predict_proba(self.x_test)[:,1]
            if find_best_threshold:
                y_pred = ModelEvaluator.get_optimal_predictions(self.y_test, 
                                                                y_prob,
                                                                list(self.class_weights.values()),
                                                                criterion=self.optimization_criterion)
        self.results[self.best_model_name] = {}
        for metric in self.metric_collection_list:
            metric_func = self.metric_collection_func_map[metric]
            self.results[self.best_model_name][metric] = metric_func(self.y_test, y_pred)
            self.predictions = y_pred
        return pd.DataFrame(self.results).transpose()

    @staticmethod
    def get_aucpr(y_test, y_pred_prob):
        precision, recall, _ = skmetrics.precision_recall_curve(y_test, y_pred_prob)
        ## Gives the same result with ascending sorting
        return skmetrics.auc(np.sort(recall), precision[np.argsort(recall)])

    def get_misclassification_cost(self, y_test, y_pred):
        labels = list(self.class_weights.keys())
        weights = list(self.class_weights.values())
        _, fp, fn, _ = skmetrics.confusion_matrix(y_test, y_pred, 
                                labels=labels).ravel()
        return (fp*weights[0] + fn*weights[1])/len(y_test)

    def plot_feat_importance(self, model_name, top_n=10, figsize=(6,6)):
        plt.figure(figsize=figsize)
        model = self.model_dict[model_name]
        pd.Series(dict(
            zip(model[-1].feature_name_,
            model[-1].feature_importances_))
            ).sort_values(ascending=False)[:top_n].plot.barh()