import numpy as np
import pandas as pd
import pickle
import logging

from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models import FastText

logger = logging.getLogger(__name__)


class FastVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, vector_size=100, window=5, min_count=1, workers=4):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None
        self.input_feature_name = None
        self.columns_out = None


    def fit(self, X, y=None):
        sentences = [text.split() for text in X]  # Utiliser les mots déjà prétraités
        self.model = FastText(sentences, vector_size=self.vector_size, window=self.window, min_count=self.min_count, workers=self.workers)
        self.input_feature_name = X.name if hasattr(X, 'name') else 'merged_text'

        self.columns_out = [f"{self.input_feature_name}_vec_{i}" for i in range(self.vector_size)]

        return self

    def transform(self, X):
        vectors = np.array([self._get_mean_vector(text) for text in X])

        # Create a DataFrame with proper column names
        return pd.DataFrame(vectors, columns=self.get_feature_names_out())

    def _get_mean_vector(self, text):
        words = text.split()  # Utiliser les mots déjà prétraités
        words = [word for word in words if word in self.model.wv.key_to_index]
        if len(words) >= 1:
            return np.mean(self.model.wv[words], axis=0)
        else:
            return np.zeros(self.vector_size)

    def get_feature_names_out(self, input_features=None):

        if self.columns_out is None:
            raise AttributeError("Transformer has not been fitted yet.")

        logger.info(f"FastVectorizer.get_feature_names_out(input_features={input_features}) -> {self.columns_out}")

        return self.columns_out



class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'FastVectorizer':
            from recettes_et_sentiments.api_model.fast_vectorizer import FastVectorizer
            return FastVectorizer
        return super().find_class(module, name)
