import numpy as np
import pandas as pd
import typing

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from gensim.models import Word2Vec

from recettes_et_sentiments.api_model.preprocessing_pipeline import CacheStep, BasicPreprocessing, ConcatColumns
from recettes_et_sentiments.api_model import rs_data, preprocessing, preprocessing_pipeline

class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    """
    custom class to integrate Word2Vec Vectorizer into a pipeline
    """
    def __init__(self, vector_size=100, window=5, min_count=1, workers=4):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None

    def fit(self, X, y=None):
        sentences = [text.split() for text in X]  # Utiliser les mots déjà prétraités
        self.model = Word2Vec(sentences, vector_size=self.vector_size, window=self.window, min_count=self.min_count, workers=self.workers)
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




def make_w2v_preprocessor_pipeline(columns_to_merge_for_training:typing.List[str],
                                vector_size=100,
                                window=5,
                                min_count=1,
                                workers=4,
                                cache=True
                               ) -> Pipeline:

    """
    build a pipeline to do basic preprpocessing, concat columns and text vectorization
    return the pipeline not fitted
    """
    vectorizer = Word2VecVectorizer(vector_size=vector_size, window=window, min_count=min_count, workers=workers)

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
    """
        Instanciate and train the pipeline
        then apply a cross validation with a linear regression to try to predict
        the ratings from users of food.com based on the recipe details

        This fails with a test score around -5.31 with vector size of 100
        2000 the model fails to traing with 364GB of memory available (64GB+300GB of Swap)
    """
    from sklearn.model_selection import cross_validate
    from sklearn.linear_model import LinearRegression

    recipe_df_ori = rs_data.load_recipes("../../../batch-1672-recettes-et-sentiments-data/RAW_recipes.csv")

    preprocessor_pipeline = make_w2v_preprocessor_pipeline(
        columns_to_merge_for_training=["name", "description", "merged_ingredients"],
        vector_size=2000,
        window=10,
        min_count=1,
        workers=6,
        cache=False
    )

    recipe_processed = preprocessor_pipeline.fit_transform(recipe_df_ori)
    recipe_processed.to_parquet(f"../batch-1672-recettes-et-sentiments-data/preproc_recipes_word2vec.parquet")

    reviews = pd.read_csv('../batch-1672-recettes-et-sentiments-data/RAW_interactions.csv')
    preproc_with_reviews = rs_data.get_y(recipe_processed, reviews)

    preproc_with_reviews.drop(columns=[2013,2002, 2003, 'count_rating'], axis=1, inplace=True)
    cv_nb = cross_validate(
        LinearRegression(n_jobs=-1),
        preproc_with_reviews.drop(columns='mean_rating'),
        preproc_with_reviews['mean_rating'],
        cv=5,
        n_jobs=-1,
        scoring='r2',
        verbose=1
    )

    print(cv_nb['test_score'].mean())
