import random

import numpy as np
import pandas as pd
# import pytest

from mlexec import MLExecutor

NUM_ROWS = 4000
NUM_CONTINUOUS_COLS = 5
NUM_OHE_COLS = 3
OHE_MAX_VAL = 2
NUM_EMBEDDING_COLS = 2
EMBEDDING_MAX_VAL = 20

SEED = 42

np.random.seed(SEED)
random.seed(SEED)

def map_cat_values(df:pd.DataFrame, cat_cols:list, max_val=5):
    """
    Generate random values corresponding for the various categories.
    Iterates through all the categorical columns.
    """
    for cat_col in cat_cols:
        category_value_map = {category: np.random.randint(low=0, high=max_val)
            for category in df[cat_col].unique()}
        df[cat_col.replace("embedding_","")+"_val"] = df[cat_col].map(category_value_map)
    return df

def test_dummy_df():
    # initial df with continuous columns
    continuous_cols = list(map(str,range(NUM_CONTINUOUS_COLS)))
    test_df = pd.DataFrame(np.random.random((NUM_ROWS, NUM_CONTINUOUS_COLS)),
                            columns=continuous_cols)

    # categorical columns (three levels 0,1,2)
    ohe_col_names = [f"ohe_cat_{i}" for i in range(NUM_OHE_COLS)]
    ohe_value_list = list(range(OHE_MAX_VAL))
    test_df[ohe_col_names] = np.random.choice(ohe_value_list, (NUM_ROWS, NUM_OHE_COLS))

    # embedding columns (20 levels 0...19)
    embedding_col_names = [f"embedding_cat_{i}" for i in range(NUM_EMBEDDING_COLS)]
    embedding_value_list = list(range(EMBEDDING_MAX_VAL))
    test_df[embedding_col_names] = np.random.choice(
        embedding_value_list, (NUM_ROWS, NUM_EMBEDDING_COLS))

    test_df = map_cat_values(test_df, embedding_col_names)
    cat_value_cols = [col for col in test_df.columns if col.endswith("_val")]
    val_cols = cat_value_cols+ohe_col_names+continuous_cols

    random_weights = [np.random.random() for _ in val_cols]

    # Adding null values for test
    test_df.loc[test_df.index[:50], val_cols] = np.nan

    # Creating target (for regression and classification columns)
    test_df["target_reg"] = (test_df[val_cols]*random_weights).sum(axis=1)
    test_df["target_class"] = test_df["target_reg"]>test_df["target_reg"].mean()

    # Dropping values corresponding to embedding columns
    test_df.drop(columns=cat_value_cols, inplace=True)

    mle_regression = MLExecutor(
        test_df,
        target_col="target_reg",
        task="regression",
        metric="r2",
        exclude_cols=["target_class"]
    )

    mle_classification = MLExecutor(
        test_df,
        target_col="target_class",
        task="classification",
        metric="auc_roc",
        exclude_cols=["target_reg"]
    )

    best_classification_model = mle_classification.best_model_name
    best_regression_model = mle_regression.best_model_name

    # tests to see if models are predicting well enough
    assert mle_classification.results[best_classification_model]["auc_roc"] > 0.95
    assert mle_regression.results[best_regression_model]["r2"] > 0.7

    # tests to see if columns have been classified correctly
    assert set(mle_classification.high_cardinality_cols) == set(embedding_col_names)
    assert set(mle_regression.low_cardinality_cols) == set(ohe_col_names)
    assert set(mle_regression.continuous_cols) == set(continuous_cols)

if __name__=="__main__":
    test_dummy_df()
