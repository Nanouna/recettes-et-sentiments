import numpy as np
import pandas as pd
import pickle
import logging

from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models import FastText


from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.common._apply_operation import apply_identity
from skl2onnx import update_registered_converter

from recettes_et_sentiments.api_model.preprocessing_pipeline import get_onnx_type

logger = logging.getLogger(__name__)


class FastVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, vector_size=100, window=5, min_count=1, workers=4):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None
        self.input_feature_name = None
        self.feature_names_out_ = None


    def fit(self, X, y=None):
        sentences = [text.split() for text in X]  # Utiliser les mots déjà prétraités
        self.model = FastText(sentences, vector_size=self.vector_size, window=self.window, min_count=self.min_count, workers=self.workers)
        self.input_feature_name = X.name if hasattr(X, 'name') else 'merged_text'

        original_columns = X.columns.tolist() if hasattr(X, 'columns') else []
        self.feature_names_out_ = original_columns + ['merged_text_vector']

        return self

    def transform(self, X):
        vectors = np.array([self._get_mean_vector(text) for text in X])

        X_transformed = X.copy()
        X_transformed['merged_text_vector'] = vectors

        return X_transformed

    def _get_mean_vector(self, text):
        words = text.split()  # Utiliser les mots déjà prétraités
        words = [word for word in words if word in self.model.wv.key_to_index]
        if len(words) >= 1:
            return np.mean(self.model.wv[words], axis=0)
        else:
            return np.zeros(self.vector_size)

    def get_feature_names_out(self, input_features=None):

        if self.feature_names_out_ is None:
            raise AttributeError("Transformer has not been fitted yet.")

        logger.info(f"FastVectorizer.get_feature_names_out(input_features={input_features}) -> {self.feature_names_out_}")

        return self.feature_names_out_


def fast_vectorizer_shape_calculator(operator):
    input = operator.inputs[0]
    N = input.type.shape[0]  # Batch size
    vector_size = operator.raw_operator.vector_size  # Number of features (vector size)

    # Directly set the output type to FloatTensorType with the vector size
    operator.outputs[0].type = FloatTensorType([N, vector_size])





def fast_vectorizer_converter(scope, operator, container):
    input_name = operator.inputs[0].full_name
    output_name = operator.outputs[0].full_name
    apply_identity(scope, input_name, output_name, container)

update_registered_converter(
    FastVectorizer, "FastVectorizer",
    fast_vectorizer_shape_calculator, fast_vectorizer_converter
)


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'FastVectorizer':
            from recettes_et_sentiments.api_model.fast_vectorizer import FastVectorizer
            return FastVectorizer
        return super().find_class(module, name)
