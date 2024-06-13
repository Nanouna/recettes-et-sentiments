import pandas as pd
import re
import string
import typing
import unidecode

import logging

logger = logging.getLogger(__name__)


from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import RobustScaler

from recettes_et_sentiments.api_model import parameters



def remove_punctuation(text:str) -> str:
    """
     remove punctuation (string.punctuation) from the intput 'text'
    """
    for punctuation in string.punctuation:
        text = str(text).replace(punctuation, ' ')
    return text

def lowercase(text:str) -> str:
    """
     return lower case text
    """
    lowercased = text.lower()
    return lowercased

def remove_numbers(text:str) -> str:
    """
     remove numbers from text
    """
    words_only = ''.join([i for i in text if not i.isdigit()])
    return words_only

def remove_stopwords(text:str) -> str:
    """
     tokenize and remove stopwords and parameters.RECIPE_STOPWORDS from text
    """
    tokenized = word_tokenize(text)
    without_stopwords = [word for word in tokenized if not word in parameters.STOP_WORDS]
    without_recipe_stopwords = [word for word in without_stopwords if not word in parameters.RECIPE_STOPWORDS]
    return ' '.join(without_recipe_stopwords)

def lemma(text:str)->str:
    """
    Return the lemmatize "text"

    Args:
    ----------
        text :str
            text to be processed. Text should have been preprocessed (lowercase, ponctuation removed, stopwords removed)

    Returns:
    ----------
        str: the input text without stop words
    """
    def get_wordnet_pos(treebank_tag):
        if treebank_tag.startswith('J'):
            return wn.ADJ
        elif treebank_tag.startswith('V'):
            return wn.VERB
        elif treebank_tag.startswith('N'):
            return wn.NOUN
        elif treebank_tag.startswith('R'):
            return wn.ADV
        else:
            return wn.NOUN  # Par défaut, retourne NOM

    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)

    lemmatizer = WordNetLemmatizer()

    lemmatized_tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(pos)) for token, pos in pos_tags]

    return ' '.join(lemmatized_tokens)



def basic_word_processing(text:str) -> str:
    """
    basic word processing :
    ponctuation, lowercase, numbers, stopwords, lemmatization
    """
    text = remove_punctuation(text)
    text = lowercase(text)
    text = remove_numbers(text)
    text = remove_stopwords(text)
    text = lemma(text)
    return text


# Recipe basic preprocessing
def basic_preprocess_tags(df: pd.DataFrame) -> pd.DataFrame:
    """
    remove tags considered as stop works (see parameters.py), and concat them as a string
    """
    removed_tags = parameters.TAGS_STOPWORDS
    df['tags'] = df['tags'].apply(lambda tag_list: [x for x in tag_list if x not in removed_tags])
    df['tags'] = df['tags'].apply(lambda tags : list(set([parameters.TAGS_REPLACEMENTS.get(tag, tag) for tag in tags])))
    # df['tags'] = df['tags'].apply(lambda tags: ' '.join(tags)) removed for word2vec
    return df

def clean_ingredient(ingredient):
    """
        remove all non [alphabetic characters & space]
        lower case
        unidecode the texte (the original dataset contains some special caracteres)
        lemmatize the result
        and return a strip version of the text
    """
    lemmatizer = WordNetLemmatizer()
    ingredient = re.sub(r'[^a-zA-Z\s]', '', ingredient)
    ingredient = ingredient.lower()
    ingredient = unidecode.unidecode(ingredient)
    ingredient = ' '.join([lemmatizer.lemmatize(word) for word in ingredient.split()])
    return ingredient.strip()

def basic_preprocess_ingredients(df: pd.DataFrame) -> pd.DataFrame:
    """
    remove ingredients from list and simplify them from a mapping
    """
    unique_ingredients = set([ingredient for row in df['ingredients'] for ingredient in row])
    cleaned_ingredients = [clean_ingredient(ing) for ing in unique_ingredients]
    dictionnary_ingredients = dict(zip(unique_ingredients, cleaned_ingredients))
    df['ingredients'] = df['ingredients'].apply(lambda ingredients : [dictionnary_ingredients[ingredient] for ingredient in ingredients])
    df['ingredients'] = df['ingredients'].apply(lambda ingredients : list(set([parameters.REPLACE_INGREDIENT[ing] for ing in ingredients if ing in parameters.REPLACE_INGREDIENT])))
    # df['tags'] = df['tags'].apply(lambda tags: ' '.join(tags)) removed for word2vec
    return df

def basic_preprocess_recipe(df: pd.DataFrame, columns_to_preproc: list) -> pd.DataFrame:
    """
    apply basic recipe preprocessing for each word of the recipe
    """
    for col in columns_to_preproc:
        df[col] = df[col].apply(basic_word_processing)
    return df

def numeric_preproc(data: pd.DataFrame, col_to_preproc: list) -> pd.DataFrame:
    """
    clip minutes & n_steps to custom values (see parameters.py)
    robust scale minuutes
    """
    data['minutes'] = data['minutes'].clip(
        lower = parameters.MINUTES_CLIP_LOWER,
        upper = parameters.MINUTES_CLIP_UPPER
        )
    data.drop(data[data.n_steps == 0].index, inplace=True)
    data['n_steps'] = data['n_steps'].clip(
        lower = parameters.N_STEPS_CLIP_LOWER,
        upper = parameters.N_STEPS_CLIP_UPPER
        )


    rb_scaler = RobustScaler()
    data[col_to_preproc] = rb_scaler.fit_transform(data[col_to_preproc])
    return data

def full_basic_preproc_recipes(data: pd.DataFrame) -> pd.DataFrame:
    """
    Function for a pipeline of all basic preprocessing:
    * basic_preprocess_tags
    * basic_preprocess_recipe
    * numeric_preproc
    """
    df = basic_preprocess_tags(data)
    df = basic_preprocess_ingredients(df)
    df = basic_preprocess_recipe(df, parameters.RECIPE_COLUMNS_FOR_TEXT_PREPROC)
    return numeric_preproc(df, parameters.RECIPE_COLUMNS_FOR_NUMERIC_PREPROC)

def count_vectorize(
    df:pd.DataFrame,
    column_name:str,
    min_df:float=0.1,
    max_df:float=0.95,
    max_features:int=None,
    ngram_range=(1,1)
    )->pd.DataFrame:
    """
    usage :
    CountVectorize the text in column column_name and add to the Data Frame the new columns
    df = count_vectorize(df, 'clean_txt')

    Note : defaults for min_df/max_df are not 1.0/1.0 as it fails right away

    Args:
        df (pd.DataFrame): DataFrame in which there is a preprocessed text column to be vectorized
        colum_name (str) : name of the column that contains the text to CountVectorize

    Returns:
        pd.DataFrame

    Raises :
        ValueError : column name is not in the dataFrame, the column has NaN,
                     min_df/max_df are outside of 0-1.0 range

    """

    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")

    if df[column_name].isnull().sum() > 0:
        raise ValueError(f"Column '{column_name}' contains NaN values")

    if 0 < min_df > 1.0:
        raise ValueError(f"min_df '{min_df}' is not between 0 & 1.0")

    if 0 < max_df > 1.0:
        raise ValueError(f"max_df '{max_df}' is not between 0 & 1.0")

    count_vectorizer = CountVectorizer(
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        ngram_range=ngram_range
        )

    X = count_vectorizer.fit_transform(df[column_name])
    X_df = pd.DataFrame(X.toarray(), columns = count_vectorizer.get_feature_names_out(), index=df.index)

    return pd.concat([df, X_df], axis = 1)

def tfidf_vectorize(
    df:pd.DataFrame,
    column_name:str,
    min_df:float=0.1,
    max_df:float=0.95,
    max_features:int=None,
    ngram_range=(1,1))->pd.DataFrame:
    """
    usage :
    TFIDF Vectorize the text in column column_name and add to the Data Frame the new columns
    df = tfidf_vectorize(df, 'clean_txt')

    Note : defaults for min_df/max_df are not 1.0/1.0 as it fails right away

    Args:
        df (pd.DataFrame): DataFrame in which there is a preprocessed text column to be vectorized
        colum_name (str) : name of the column that contains the text to Vectorize

    Returns:
        pd.DataFrame

    ValueError : column name is not in the dataFrame, the column has NaN,
                min_df/max_df are outside of 0-1.0 range

    """

    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")

    if df[column_name].isnull().sum() > 0:
        raise ValueError(f"Column '{column_name}' contains NaN values")

    if 0 < min_df > 1.0:
        raise ValueError(f"min_df '{min_df}' is not between 0 & 1.0")

    if 0 < max_df > 1.0:
        raise ValueError(f"max_df '{max_df}' is not between 0 & 1.0")

    tf_idf_vectorizer = TfidfVectorizer(
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        ngram_range=ngram_range
        )

    X = tf_idf_vectorizer.fit_transform(df[column_name])
    X_df = pd.DataFrame(X.toarray(), columns = tf_idf_vectorizer.get_feature_names_out(), index=df.index)

    return pd.concat([df, X_df], axis = 1)

def concat_columns(df:pd.DataFrame, columns:typing.List[str], dropSourceColumn:bool=True)->pd.DataFrame:
    """
    create a "merge_text" column in the dataframe and merges the listed columns into it, and drop the columns

    Args:
    ----------
        df : pd.DataFrame
            DataFrame
        columns : List[str]
            List of column names that we'll merge into "merge_text" column
        dropSourceColumn : bool
            if True, the column names passed (arg 'columns') are dropped

    Returns:
    ----------
        pd.DataFrame :  the updated dataframe

    """

    logger.info(df[columns].head())

    df['tags'] = df['tags'].apply(lambda x: ' '.join(x))
    df['merged_text'] = df[columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

    if dropSourceColumn:
        df.drop(columns=columns,axis=1, inplace=True)

    return df
