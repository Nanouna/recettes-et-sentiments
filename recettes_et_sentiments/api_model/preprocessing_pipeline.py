import typing
import logging

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from recettes_et_sentiments.api_model import preprocessing

logger = logging.getLogger(__name__)


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
        logger.info("Transforming data with BasicPreprocessing")
        return preprocessing.concat_columns(X.copy(), self.columns, self.dropSourceColumn)

    def get_feature_names_out(self):
        pass



def make_preprocessor_pipeline( use_count_vectorizer:bool,
                                columns_to_merge_for_training:typing.List[str],
                                min_df:float=0.1,
                                max_df:float=0.95,
                                max_features:int=None,
                                ngram_range=(1,1)
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

    Returns:
    ----------
        Pipeline :  the pipeline ready to be fit and transform


    Example :
    ----------

```
    recipe_df_ori = data.load_recipes("../batch-1672-recettes-et-sentiments-data/RAW_recipes.csv")

    preprocessor_pipeline = preprocessing_pipeline.make_preprocessor_pipeline(
        use_count_vectorizer=True,
        columns_to_merge_for_training=["name", "description", "merged_ingredients"],
        )

    recipe_processed = preprocessor_pipeline.fit_transform(recipe_df)

    recipe_processed.to_parquet("../batch-1672-recettes-et-sentiments-data/basic_preproc_recipes.parquet")
```
    """

    vectorizer_args = {"min_df":min_df, "max_df":max_df, "max_features":max_features, "ngram_range":ngram_range}

    vectorizer = CountVectorizer(**vectorizer_args) if use_count_vectorizer else TfidfVectorizer(**vectorizer_args)


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
        ('basic_preproc', BasicPreprocessing()),
        ('concat_columns', ConcatColumns(columns=columns_to_merge_for_training)),
        ('vectorize_and_combine', column_transformer)
    ])

    return preprocessing_pipeline
