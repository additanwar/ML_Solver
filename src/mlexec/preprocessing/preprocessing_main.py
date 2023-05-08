import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

class Preprocessor:
    def __init__(self, 
                df: pd.DataFrame,
                target_col:str,
                categorical_cols=[],
                continuous_cols=[],
                exclude_cols=[],
                categorical_threshold=0.05,
                cleanup=True,
                fix_col_names=False) -> None:
        self.df = df.drop(columns=exclude_cols)
        self.target_col = target_col
        self.categorical_cols = self._find_categorical_cols(categorical_cols,
                                                            categorical_threshold)
        if target_col in self.categorical_cols:
            self.categorical_cols.remove(target_col)
        
        self.continuous_cols = self._find_continuous_cols(continuous_cols)
        if target_col in self.continuous_cols:
            self.continuous_cols.remove(target_col)

        if cleanup:
            self.df.drop_duplicates(inplace=True)
            self.df = self.df[self.df[target_col].notnull()]
            self.clean_categorical_cols()
            self.clean_continuous_cols()
        
        if fix_col_names:
            self.standardize_col_names()
    
    def standardize_col_names(self):
        self.df.columns = [i.strip() for i in list(self.df.columns)]

    def _find_continuous_cols(self, continuous_cols):
        if continuous_cols:
            return continuous_cols
        else:
            return list(set(self.df.columns)-set(self.categorical_cols))

    def _find_categorical_cols(self, categorical_cols:list[str], threshold:float):
        if categorical_cols:
            return categorical_cols
        else:
            cut_off_length = threshold*len(self.df)
            for i in self.df.columns:
                if self.df[i].nunique()<cut_off_length or self.df[i].dtype=="object":
                    categorical_cols.append(i)
        return categorical_cols

    def report_categoricals(self):
        for i in self.categorical_cols:
            print(self.df[i].value_counts(True).head(1))
            print(self.df[i].isnull().sum() / len(self.df))
    
    def clean_continuous_cols(self):
        for col in self.continuous_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
    
    def clean_categorical_cols(self):
        for col in self.categorical_cols:
            if self.df[col].dtype=='object':
                self.df[col] = self.df[col].str.strip().str.lower().replace('', np.nan)

class DFImputer(Preprocessor):
    def __init__(self, df: pd.DataFrame,
                    target_col,
                    continuous_cols=[],
                    high_cardinality_cols=[],
                    low_cardinality_cols=[],
                    exclude_cols=[],
                    categorical_threshold=0.05,
                    cardinality_threshold=5,
                    continuous_impute_method="mean") -> None:
        categorical_cols = high_cardinality_cols+low_cardinality_cols
        super().__init__(df,
                        target_col,
                        categorical_cols,
                        continuous_cols,
                        exclude_cols,
                        categorical_threshold)
        
        self.high_cardinality_cols, self.low_cardinality_cols = self._classify_categorical_cols(high_cardinality_cols,
                                                                        low_cardinality_cols,
                                                                        cardinality_threshold)
        self._impute_categorical_cols()
        self._impute_continuous_cols(continuous_impute_method)

    def _classify_categorical_cols(self,
                    high_cardinality_cols,
                    low_cardinality_cols,
                    cardinality_threshold):
        if high_cardinality_cols:
            # both should be specified
            if not low_cardinality_cols:
                low_cardinality_cols = list(set(self.categorical_cols) 
                                        - set(high_cardinality_cols))
        else:
            high_cardinality_cols=[]
            low_cardinality_cols=[]
            for col, num_unique in self.df[self.categorical_cols].nunique().to_dict().items():
                if num_unique>cardinality_threshold:
                    high_cardinality_cols.append(col)
                else:
                    low_cardinality_cols.append(col)
        return high_cardinality_cols, low_cardinality_cols

    def _impute_categorical_cols(self):
        for col in self.low_cardinality_cols:
            col_mode = self.df[col].mode().values[0]
            self.df[col].fillna(col_mode, inplace=True)
        self.df[self.high_cardinality_cols].fillna(
            'MISSING', inplace=True)
    
    def _impute_continuous_cols(self, method):
        if method in ["bfill", "ffill"]:
            self.df[self.continuous_cols].fillna(method, inplace=True)
        elif isinstance(method, str):
            if method=="median":
                col_medians = self.df[self.continuous_cols].median()
                self.df[self.continuous_cols] = self.df[self.continuous_cols].fillna(col_medians)
            if method=="mean":
                col_means = self.df[self.continuous_cols].mean()
                self.df[self.continuous_cols] = self.df[self.continuous_cols].fillna(col_means)
            else:
                self.df[self.continuous_cols] = self.df[self.continuous_cols].interpolate(method)