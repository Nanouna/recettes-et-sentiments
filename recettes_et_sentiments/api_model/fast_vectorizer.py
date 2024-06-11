import numpy as np
import pickle

from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models import FastText


class FastVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, vector_size=100, window=5, min_count=1, workers=4):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None

    def fit(self, X, y=None):
        sentences = [text.split() for text in X]  # Utiliser les mots déjà prétraités
        self.model = FastText(sentences, vector_size=self.vector_size, window=self.window, min_count=self.min_count, workers=self.workers)
        return self

    def transform(self, X):
        return np.array([self._get_mean_vector(text) for text in X])

    def _get_mean_vector(self, text):
        words = text.split()  # Utiliser les mots déjà prétraités
        words = [word for word in words if word in self.model.wv.key_to_index]
        if len(words) >= 1:
            return np.mean(self.model.wv[words], axis=0)
        else:
            return np.zeros(self.vector_size)


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'FastVectorizer':
            from recettes_et_sentiments.api_model.fast_vectorizer import FastVectorizer
            return FastVectorizer
        return super().find_class(module, name)
