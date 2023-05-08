## Purpose
The `mlexec` package is used to run scikit-learn type models with high abstraction, automating repetative tasks in training data preparation such as imputation, encoding etc. and during model tuning such as running CV, tuning and training.

## Steps to run
1. Install the package (distribution available on pypi)
```{sh}
pip install mlexec
```
2. Create MLExecutor objects with the following:
    1. Required Arguments:
        1. `df`: The dataframe containing the dependent and independent variables.
        2. `target_col`: The column which contains the value for the dependent variable. All other columns (unless excluded) are considered a predictor.
        3. The task: `"regression"` or `"classification"`
    2. Optional Arguments:
        1. Data Preprocessing Arguments:
            - `encode_opt`: Method to encode the low caridnality columns. Currently supported: `one-hot` and `label`.
            - `continuous_cols`: Continuous columns in the dataset, if unspecified it is determined using the following logic: If number of Unique values in column > `categorical_threshold*len(df)` --> column is continuous.
            - `high_cardinality_cols`: High Cardinality columns in the dataset, if unspecified it is determined using the following logic: If number of Unique values in column is between `cardinality_threshold` and `categorical_threshold*len(df)` --> column is continuous. Currently these columns are embedded into a lower dimensional space and then used.
            - `low_cardinality_cols`: Low Cardinality Categorical columns in the dataset. if unspecified it is determined using the following logic: If number of Unique values in column <= `cardinality_threshold`
            - `exclude_cols`: Columns which should not be used in the modelling process.
            - `categorical_threshold`: Max number of unique values in a column upto which it is considered a categorical column.
            - `cardinality_threshold`: Maximum number of unique values in a categorical column above which it is considered a high cardinality column.
            - `continuous_impute_method`: The method to be used to impute the continuous columns. In addition to `mean` and `median` Any method which can be used in `pandas.DataFrame.interpolate` or `pandas.DataFrame.fillna` can be used.
                - Low cardinality categorical columns are filled with the mode and for high cardinality columns a new category level named `MISSING` is created.
        2. Modelling Arguments:
            - `model_list`: List of models to tune. Based on the valdation results only the best one is used to train the final model.
                - Currently supported model list: `["lgb","svm","nn","lr","rf","xgb","knn"]`
            - `metric`: The metric with which the model should be compared. Based on the task there are following choices:
                - Regression:
                - Classification:
            - `model_save_path`: Path to save the model. The embeddings layers for the high cardinality columns are also saved here.
            - `normalize`: Whether to normalize the data or not.
            - `class_weights`: The weights to assign various classes during training and calculation of the metrics. It is expected that the negative class will be the first key.
            - `tune_flag`: Whether to tune the model or not. If false the model is trained with default parameters.
            - `cv`: Type of cross validation to perform. Currently supported: `basic`, `nested` and `""` (no CV is performed). CV gives a robust estimation of the model performance but also takes time.
            - `n_fold`: Number of folds to be used in k-fold cross validation. Only used if cv is not null.
            - `final_train_flag`: Whether to train the final model based on the total data (training + validation)
            - `max_evals`: Number of model configurations to run during tuning. For each model in the model list `max_evals` number of models will be trained and then compared.

```python
from mlexec import MLExecutor
import pandas as pd

df = pd.read_excel("test.xlsx")
mle_regression = MLExecutor(
        test_df,
        target_col="target_reg",
        task="regression",
        metric="r2",
        exclude_cols=["target_class"],
    )
```