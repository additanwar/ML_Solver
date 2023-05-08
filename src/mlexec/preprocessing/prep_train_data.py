import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from .embedding import DFEmbedder
from .preprocessing_main import DFImputer

class DFPreprocessor(DFImputer):
    def __init__(self,
                df: pd.DataFrame,
                target_col: str,
                task:str="classification",
                encode_opt:str="one-hot",
                model_save_path=".",
                **kwargs) -> None:
        """
        ## Perform data preprocessing
        1. Clean up
        2. Encoding 
        3. Imputation 
        4. Normalization 
        5. Generating Embeddings 
        6. Split
        ### Arguments:
        Kwargs are for the imputation:
        
        """
        super().__init__(df, 
                        target_col=target_col,
                        **kwargs)
        self._task = task
        self.model_save_path = model_save_path
        self.encode_categorical_cols(encode_opt)

    @staticmethod
    def normalize_cols(train, *args):
        """
        Uses train df to create normalizer object.
        The Normalizer object is applied to tranform all the other argument
        dataframes passed.
        """
        normalizer = StandardScaler()
        normalizer.fit(train)
        normalized_args = map(normalizer.transform, args)
        return train, *normalized_args

    def prepare_data(self, normalize=False):
        x_train, x_test, y_train, y_test = self.split_data()
        
        if normalize:
            x_train, x_test = DFPreprocessor.normalize_cols(x_train, x_test)
        
        if self.high_cardinality_cols:
            x_train, x_test = DFEmbedder.gen_multi_col_embeddings(
                                                train=x_train,
                                                encode_cols=self.high_cardinality_cols,
                                                test=x_test,
                                                root=self.model_save_path)
            x_train.drop(columns=self.high_cardinality_cols,
                        inplace=True)
            x_test.drop(columns=self.high_cardinality_cols,
                        inplace=True)
        return x_train, x_test, y_train, y_test

    def encode_categorical_cols(self, encode_opt):
        if encode_opt=="one-hot":
            self.df = pd.get_dummies(self.df,
                        columns=self.low_cardinality_cols,
                        prefix=self.low_cardinality_cols)
        elif encode_opt=="label":
            for i in self.low_cardinality_cols:
                self.df[i] = LabelEncoder().fit_transform(self.df[i])

    def split_data(self, test_size=0.2):
        x = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]
        if self._task=="classification":
            x_train, x_test, y_train, y_test = train_test_split(x,
                                                y,
                                                test_size=test_size,
                                                random_state=42,
                                                stratify=y)
        elif self._task=="regression":
            x_train, x_test, y_train, y_test = train_test_split(x,
                                                y,
                                                test_size=test_size,
                                                random_state=42)
        return x_train, x_test, y_train, y_test

    def pca(self):
        """
        To Do --> Convert basic continuous features to principal components
        """
        pass