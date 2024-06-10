import numpy as np
import pandas as pd
import typing

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity

from gensim.models import FastText

from recettes_et_sentiments.api_model.preprocessing_pipeline import CacheStep, BasicPreprocessing, ConcatColumns
from recettes_et_sentiments.api_model import rs_data, preprocessing, preprocessing_pipeline



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




def make_fast_preprocessor_pipeline(columns_to_merge_for_training:typing.List[str],
                                vector_size=100,
                                window=5,
                                min_count=1,
                                workers=4,
                                cache=True
                               ) -> Pipeline:

    vectorizer = FastVectorizer(vector_size=vector_size, window=window, min_count=min_count, workers=workers)

    basic_preproc = BasicPreprocessing()
    concat_columns = ConcatColumns(columns=columns_to_merge_for_training)

    folder =  "data/cache/"

    basic_preproc_filename = f"{folder}basic_preproc.parquet"
    concat_columns_filename = f"{folder}concat_columns_{'_'.join(columns_to_merge_for_training)}.parquet"

    if cache:
        basic_preproc = CacheStep(basic_preproc_filename, basic_preproc)
        concat_columns = CacheStep(concat_columns_filename, concat_columns)

    column_transformer = ColumnTransformer(
        transformers=[
            ('text', vectorizer, 'merged_text')
        ],
        remainder='passthrough',
        sparse_threshold=0,
        n_jobs=-1
    )

    preprocessing_pipeline = Pipeline(steps=[
        ('basic_preproc', basic_preproc),
        ('concat_columns', concat_columns),
        ('vectorize_and_combine', column_transformer)
    ])

    return preprocessing_pipeline


if __name__ == "__main__":

    from sklearn.model_selection import cross_validate
    from sklearn.linear_model import LinearRegression

    recipe_df_ori = rs_data.load_recipes("../../../batch-1672-recettes-et-sentiments-data/RAW_recipes.csv")

    preprocessor_pipeline = make_fast_preprocessor_pipeline(
        columns_to_merge_for_training=["name", "tags", "description", "merged_ingredients"],
        vector_size=2000,
        window=10,
        min_count=1,
        workers=6,
        cache=True
    )

    # train_size = int(recipe_df_ori.shape[0] * 0.8)
    # train_recipe_df = recipe_df_ori.iloc[0:train_size]
    # test_recipe_df  = recipe_df_ori.iloc[train_size+1:]

    recipe_processed = preprocessor_pipeline.fit_transform(recipe_df_ori)
    recipe_processed.to_parquet(f"../batch-1672-recettes-et-sentiments-data/preproc_recipes_fast_name-tag-desc-ingredients.parquet")


    available_ingredients = ['tomato', 'cheese', 'basil']

    #preprocessor_pipeline.transform()
