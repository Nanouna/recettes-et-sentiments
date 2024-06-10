import os
import pandas as pd
import typing
import logging

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from recettes_et_sentiments.api_model import preprocessing

logger = logging.getLogger(__name__)


def save_to_parquet(df, filename):
    logger.info(f"Saving DataFrame to {filename}")
    df.to_parquet(filename, index=True)
    return df

def load_from_parquet(filename):
    if os.path.exists(filename):
        logger.info(f"Loading DataFrame from {filename}")
        return pd.read_parquet(filename)
    else:
        logger.info(f"{filename} does not exist. Skipping load.")
        return None



class BasicPreprocessing(BaseEstimator, TransformerMixin):
    """
    This class will preprocess all columns with basic transformations (text,numeric)
    see : preprocessing.full_basic_preproc_recipes for details
    """

    def __init__(self):
        pass


    def fit(self, X, y=None):
        logger.info("Fitting BasicPreprocessing")
        return self


    def transform(self, X):
        logger.info("Transforming data with BasicPreprocessing")
        X = X.copy()
        return preprocessing.full_basic_preproc_recipes(X)


    def get_feature_names_out(self):
        pass



class ConcatColumns(BaseEstimator, TransformerMixin):
    """
    This class will concat the columns specified by there name into a new column named 'merge_text'
    The text is concatenated with a space a separator
    the text is supposed to be already fully preprocessed (apart from Vectorizer)
    the merge_text column will later be replaced by the vectorizer
    """
    def __init__(self, columns, dropSourceColumn=True):
        self.columns = columns
        self.dropSourceColumn = dropSourceColumn

    def fit(self, X, y=None):
        logger.info("Fitting ConcatColumns")
        return self

    def transform(self, X):
        logger.info("Transforming data with ConcatColumns")
        X = X.copy()
        X['tags'].apply(lambda x: ' '.join(x))
        return preprocessing.concat_columns(X, self.columns, self.dropSourceColumn)

    def get_feature_names_out(self):
        pass





class CacheStep(BaseEstimator, TransformerMixin):
    def __init__(self, filename, step_func):
        self.filename = filename
        self.step_func = step_func
        self.cached_df = load_from_parquet(filename)

    def fit(self, X, y=None):
        if self.cached_df is not None:
            logger.info(f"FIT : Using cached data from {self.filename}")
        else:
            logger.info(f"FIT : No cached data available for {self.filename}, fitting step")
            self.step_func.fit(X, y)
        return self

    def transform(self, X):
        if self.cached_df is not None:
            logger.info(f"Transform : Using cached data from {self.filename}")
            return self.cached_df
        else:
            logger.info(f"Transforming data and saving to {self.filename}")
            transformed_X = self.step_func.transform(X)
            save_to_parquet(transformed_X, self.filename)
            return transformed_X



def make_preprocessor_pipeline( use_count_vectorizer:bool,
                                columns_to_merge_for_training:typing.List[str],
                                min_df:float=0.1,
                                max_df:float=0.95,
                                max_features:int=None,
                                ngram_range=(1,1),
                                cache=True
                               ) -> Pipeline:

    """

    Preprocessor pipeline that perform basic text and numerical preprocessing
    Concat into one colum a list of text column on which we want to train the model
    (see columns_to_merge_for_training parameter)
    And finaly apply a CountVectorizer or TF IDF Vectorizer (see use_count_vectorizer parameter)


    Args:
    ----------
        use_count_vectorizer : bool
            if True, a CountVectorizer is used, TfidfVectorizer otherwise

        columns_to_merge_for_training : List[str]
            List of column names that we'll merge into a single text and used by the vectorizer

        min_df : float
            see CountVectorizer/TfidfVectorizer documentation

        max_df : float
            see CountVectorizer/TfidfVectorizer documentation

        max_features : int
            see CountVectorizer/TfidfVectorizer documentation

        ngram_range : tuple
            see CountVectorizer/TfidfVectorizer documentation

        cache : bool
            Use cache for intermediary steps, result from BasicProcessing & ConcatColumn are stored as parquet.
            Those parket are retrieved if they exists.
            ConcatColum cache file name contains the number of column to concat (order of column is important)

    Returns:
    ----------
        Pipeline :  the pipeline ready to be fit and transform


    Example :
    ----------

```
min_df=6
max_df=0.98
max_features=None
ngram_range=(1,1)
countVectorizer=False
file_suffix = f"{'CountVectorizer' if countVectorizer else 'TfIdfVectorizer'}_{min_df}_{max_df}_{max_features}_{str(ngram_range[0])+'_'+str(ngram_range[1])}"

preprocessor_pipeline = preprocessing_pipeline.make_preprocessor_pipeline(
    use_count_vectorizer=True,
    columns_to_merge_for_training=["name", "description", "merged_ingredients"],
    min_df=min_df,
    max_df=max_df,
    max_features=max_features,
    ngram_range=ngram_range,
    cache=True
)

recipe_processed = preprocessor_pipeline.fit_transform(recipe_df_ori)

recipe_processed.to_parquet(f"../batch-1672-recettes-et-sentiments-data/preproc_recipes_{file_suffix}.parquet")
```
    """

    vectorizer_args = {"min_df":min_df, "max_df":max_df, "max_features":max_features, "ngram_range":ngram_range}
    vectorizer = CountVectorizer(**vectorizer_args) if use_count_vectorizer else TfidfVectorizer(**vectorizer_args)

    basic_preproc = BasicPreprocessing()
    concat_columns = ConcatColumns(columns=columns_to_merge_for_training)

    folder =  "data/cache/"


    basic_preproc_filename = f"{folder}basic_preproc.parquet"
    concat_columns_filename = f"{folder}concat_columns_{'_'.join(columns_to_merge_for_training)}.parquet"

    if cache:
        basic_preproc = CacheStep(basic_preproc_filename, basic_preproc)
        concat_columns = CacheStep(concat_columns_filename, concat_columns)


        # Créer le ColumnTransformer
    column_transformer = ColumnTransformer(
        transformers=[
            ('text', vectorizer, 'merged_text')
        ],
        remainder='passthrough',
        sparse_threshold=0,
        n_jobs=-1
    )

    # Configurer sparse_output pour retourner une matrice dense
    # column_transformer.set_output(transform="pandas")


    # Créer le pipeline principal avec ColumnTransformer
    preprocessing_pipeline = Pipeline(steps=[
        ('basic_preproc', basic_preproc),
        ('concat_columns', concat_columns),
        ('vectorize_and_combine', column_transformer)
    ])

    return preprocessing_pipeline
