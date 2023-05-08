import pathlib

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import tensorflow.python.keras as keras

from tensorflow.python.keras.models import Model
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Embedding, Input, Dense, Flatten, concatenate

tf.random.set_seed(173)

class DFEmbedder:
    @staticmethod
    def label_encode(series: pd.Series):
        if pd.api.types.infer_dtype(series)!="object":
            series = series.astype(str)
        return LabelEncoder().fit_transform(series)

    @staticmethod
    def gen_multi_col_embeddings(train: pd.DataFrame,
                        encode_cols: list[str],
                        test: pd.DataFrame = pd.DataFrame(),
                        root: str = "model",
                        max_output_size: int=10,
                        fraction_threshold: float=0.5):
        emb_c = {col: val.nunique() for col, val in train[encode_cols].items()}
        emb_sizes = {col: min(max_output_size, round(c*fraction_threshold)) 
                    for col, c in emb_c.items()}
        embeddings = []
        input_layers = {}
        x_train = DFEmbedder.adapt_df(train, encode_cols)
        for col in encode_cols:
            input_layer = Input(shape=(1,), name=col)
            input_layers[col] = input_layer
            vocab_size = emb_c[col] + 1
            embeddings.append(Flatten()(
                Embedding(vocab_size,
                        emb_sizes[col])(input_layer)
                ))
        output = concatenate(embeddings)
        model = Model(input_layers, output)
        loss_fn = keras.losses.SparseCategoricalCrossentropy()
        model.compile("rmsprop", loss=loss_fn)
        
        train = DFEmbedder.add_embedding_output(model, 
                    train,
                    x_train)
        model.save(pathlib.Path(root, "embeddings"))

        if test is not pd.DataFrame():
            x_test = DFEmbedder.adapt_df(test, encode_cols)
            test = DFEmbedder.add_embedding_output(model, 
                            test,
                            x_test)
            return train, test
        return train

    @staticmethod
    def gen_supervised_embeddings(train: pd.DataFrame,
                        encode_cols: list[str],
                        y_train: pd.Series,
                        test: pd.DataFrame = None,
                        embedding_output_dim: int = 10,
                        root: str = "model",
                        max_output_size: int=10,
                        fraction_threshold: float=0.1,
                        output_activation: str="swish"):
        """Fits, adds and saves embeddings for the categorical variables.

        Args:
            data (pd.DataFrame): Preprocessed DataFrame used for training
            encode_cols (list): The list of columns to be encoded (categorical)

        Returns:
            (pd.DataFrame): Dataframe with embedded columns
        """
        emb_c = {col: val.nunique() for col, val in train[encode_cols].items()}
        emb_sizes = {col: min(max_output_size, round(c*fraction_threshold)) 
                    for col, c in emb_c.items()}
        embeddings = []
        input_layers = {}
        x_train = DFEmbedder.adapt_df(train, encode_cols)
        for col in encode_cols:
            input_layer = Input(shape=(1,), name=col)
            input_layers[col] = input_layer
            vocab_size = emb_c[col] + 1
            embeddings.append(Flatten()(
                Embedding(vocab_size,
                        emb_sizes[col])(input_layer)
                ))
        x = concatenate(embeddings)
        outputs = Dense(embedding_output_dim, 
                        activation=output_activation)(x)
        model = Model(input_layers, outputs)
        loss_fn = keras.losses.SparseCategoricalCrossentropy()
        callback = keras.callbacks.EarlyStopping(monitor='loss',
                            patience=1,
                            min_delta=1e-6)
        model.compile("rmsprop", loss=loss_fn)
        model.fit(x_train,
                    y_train,
                    callbacks=[callback],
                    epochs=30)
        train = DFEmbedder.add_embedding_output(model, 
                        train,
                        x_train, 
                        embedding_output_dim)
        if not test is None:
            x_test = DFEmbedder.adapt_df(test, encode_cols)
            test = DFEmbedder.add_embedding_output(model, 
                            test,
                            x_test, 
                            embedding_output_dim)
        model.save(pathlib.Path(root, "embeddings"))
        return train, test
    
    @staticmethod
    def adapt_df(df: pd.DataFrame, col_list:list[str]=[]):
        """
        Adapt the dataframe to fit to the TF specified format
        """
        tf_train = {}
        if not col_list:
            col_list = list(df.columns)
        for col in col_list:
            tf_train[col] = DFEmbedder.label_encode(df[col])
        return tf_train

    @staticmethod
    def add_embedding_output(model, data, x, embedding_output_dim=None):
        if embedding_output_dim:
            output_array = model.predict(x).reshape(len(data), embedding_output_dim)
        else:
            output_array = model.predict(x)
            embedding_output_dim = output_array.shape[1]
        data[["embedding-" + str(i) for i in range(embedding_output_dim)]] = None
        data[["embedding-" + str(i) for i in range(embedding_output_dim)]] = output_array
        return data

    @staticmethod
    def gen_columnwise_embeddings(data: pd.DataFrame, encode_cols: list[str], root):
        """Fits, adds and saves embeddings for the categorical variables.

        Args:
            data (pd.DataFrame): Preprocessed DataFrame used for training
            encode_cols (list): The list of columns to be encoded (categorical)

        Returns:
            (pd.DataFrame): Dataframe with embedded columns
        """
        emb_c = {col: val.nunique() for col, val in data[encode_cols].items()}
        emb_sizes = {col: min(50, (c + 1) // 2) for col, c in emb_c.items()}
        embeddings = {}
        for col in encode_cols:
            vocab_size = max(data[col]) + 1
            output_size = emb_sizes[col]
            model = Sequential()
            model.add(Embedding(vocab_size, output_size, input_length=1))
            model.compile("rmsprop", "mse")
            embeddings[col] = model
            output_array = model.predict(data[col])
            data[[col + str(i) for i in range(output_size)]] = output_array.reshape(
                len(data), output_size
            )
        DFEmbedder.save_embeddings(embeddings, root)
        return data

    @staticmethod
    def save_embeddings(embeddings: dict[str, keras.Model], root):
        """Saving embeddings for each columns

        Args:
            embeddings (dict[str:tensorflow.keras.Layers])
        """
        for k, model in embeddings.items():
            model.save(pathlib.Path(root, "embeddings", f"{k}-embed"))