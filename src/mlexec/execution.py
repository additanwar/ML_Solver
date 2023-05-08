import gc

import pandas as pd

from .preprocessing.prep_train_data import DFPreprocessor
from .tuning.model_assembly import ModelAssembler
from .evaluation.evaluate import ModelEvaluator

class MLExecutor(DFPreprocessor, ModelAssembler, ModelEvaluator):
    """
    Master Executor class to perform end to end ML
    """
    def __init__(self,
                df:pd.DataFrame,
                target_col: str,
                task:str="classification",
                encode_opt:str="one-hot",
                continuous_cols:list[str]=[],
                high_cardinality_cols:list[str]=[],
                low_cardinality_cols:list[str]=[],
                exclude_cols:list[str]=[],
                categorical_threshold=0.05,
                cardinality_threshold:int=5,
                continuous_impute_method="mean",
                model_list=["lgb","rf","xgb"],
                metric:str="",
                model_save_path:str=".",
                normalize:bool=True,
                class_weights:dict[int,int]={0:1, 1:1},
                tune_flag:bool=True,
                cv:str="basic",
                n_fold:int=5,
                internal_val:bool=False,
                final_train_flag:bool=True,
                max_evals:int=10):
        """Main executor object that inherits from the three building blocks of the mlexec classes.
        These three parent classes are initiated so that we can use processed

        Args:
            df (pd.DataFrame): The dataframe containing the dependent and independent variables.
            target_col (str): The column which contains the value for the dependent variable. \
            All other columns (unless excluded) are considered a predictor.
            task (str, optional): The task: `"regression"` or `"classification"`. \
            Defaults to "classification". \n
            encode_opt (str, optional): Method to encode the low caridnality columns. \
            Currently supported: `one-hot` and `label`. Defaults to "one-hot". \n
            continuous_cols (list[str], optional): Continuous columns in the dataset, \
            if unspecified it is determined using the following logic: \
            If number of Unique values in column > `categorical_threshold*len(df)` \
            --> column is continuous. Defaults to []. \n
            high_cardinality_cols (list[str], optional): High Cardinality columns \
            in the dataset, if unspecified it is determined using the following logic: \
            If number of Unique values in column is between `cardinality_threshold` \
            and `categorical_threshold*len(df)` --> column is continuous.\
            Currently these columns are embedded into a lower dimensional space \
            and then used. Defaults to []. \n
            low_cardinality_cols (list[str], optional): Low Cardinality Categorical \
            columns in the dataset. if unspecified it is determined using the following logic: \
            If number of Unique values in column <= `cardinality_threshold`. Defaults to [].\n
            exclude_cols (list[str], optional): Columns which should not be used \
            in the modelling process. Defaults to [].\n
            categorical_threshold (float, optional): Max number of unique values \
            in a column upto which it is considered a categorical column. \
            Defaults to 0.05. \n
            cardinality_threshold (int, optional): Maximum number of unique values \
            in a categorical column above which it is considered a high cardinality \
            column. Defaults to 5. \n
            continuous_impute_method (str, optional): The method to be used to impute \
            the continuous columns. In addition to `mean` and `median` Any method which \
            can be used in `pandas.DataFrame.interpolate` or `pandas.DataFrame.fillna` \
            can be used. Defaults to "mean".\n
            model_list (list, optional): List of models to tune. Based on the valdation \
            results only the best one is used to train the final model. \
            Available models: {"lgb","svm","nn","lr","rf","xgb","knn"}. \
            Defaults to ["lgb","rf","xgb"]. \n
            metric (str, optional): The metric with which the model should \
            be compared. Available Metrics: {"rmse","r2","mae","mape", \
            "matthews_corrcoef","misclassification_cost","accuracy","recall",\
            "precision","f1_score","auc_roc","auc_pr"}. Defaults to "". \n
            model_save_path (str, optional):  Path to save the model. \
            The embeddings layers for the high cardinality columns are \
            also saved here. Defaults to ".". \n
            normalize (bool, optional): Whether to normalize the data or not. \
            Defaults to True. \n
            class_weights (dict[int,int], optional): The weights to assign various \
            classes during training and calcucation of the metrics. It is expected \
            that the negative class will be the first key. Defaults to {0:1, 1:1}.\n
            tune_flag (bool, optional): Whether to tune the model or not. If false \
            the model is trained with default parameters.
            Defaults to True. \n
            cv (str, optional): Type of cross validation to perform. \
            Currently supported: `basic`, `nested` and `""` (no CV is performed). \
            Defaults to "basic". \n
            n_fold (int, optional): Number of folds to be used in k-fold cross validation. \
            Defaults to 5.\n
            final_train_flag (bool, optional): Whether to train the final model based on the \
            total data (training + validation) Defaults to True.\n
            max_evals (int, optional): Number of model configurations to run during tuning. \
            Defaults to 10.
        """
        # preprocessing data
        DFPreprocessor.__init__(self,
                                df,
                                target_col,
                                task,
                                encode_opt,
                                model_save_path=model_save_path,
                                continuous_cols=continuous_cols,
                                high_cardinality_cols=high_cardinality_cols,
                                low_cardinality_cols=low_cardinality_cols,
                                exclude_cols=exclude_cols,
                                categorical_threshold=categorical_threshold,
                                cardinality_threshold=cardinality_threshold,
                                continuous_impute_method=continuous_impute_method)
        x_train, x_test, y_train, y_test = self.prepare_data()

        # performing tuning and model training
        ModelAssembler.__init__(self,
                                x_train=x_train,
                                y_train=y_train,
                                model_list=model_list,
                                metric=metric,
                                task=task,
                                normalize=normalize,
                                class_weights=class_weights,
                                tune_flag=tune_flag,
                                cv=cv,
                                n_fold=n_fold,
                                internal_val=internal_val,
                                final_train_flag=final_train_flag)
        self.run_tuning(max_evals=max_evals)

        ModelEvaluator.__init__(self,
                        x_test=x_test,
                        y_test=y_test,
                        optimization_criterion="f1_score")

        self.get_val_scores()
        self.test_results = self.evaluate(find_best_threshold=False)
