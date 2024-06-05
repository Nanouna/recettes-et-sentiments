
import pandas as pd
import string

from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import RobustScaler

from recettes_et_sentiments.api_model import parameters


# classic NLP preprocessing
def remove_punctuation(text:str) -> str:
    for punctuation in string.punctuation:
        text = str(text).replace(punctuation, ' ')
    return text

def lowercase(text:str) -> str:
    lowercased = text.lower()
    return lowercased

def remove_numbers(text:str) -> str:
    words_only = ''.join([i for i in text if not i.isdigit()])
    return words_only

def remove_stopwords(text:str) -> str:
    tokenized = word_tokenize(text)
    without_stopwords = [word for word in tokenized if not word in parameters.STOP_WORDS]
    without_recipe_stopwords = [word for word in without_stopwords if not word in parameters.RECIPE_STOPWORDS]
    return ' '.join(without_recipe_stopwords)

def lemma(text:str)->str:
    """
    Return the lemmatize "text"

    Args:
        text (str): text to be processed. Text should have been preprocessed (lowercase, ponctuation removed, stopwords removed)

    Returns:
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
    text = remove_punctuation(text)
    text = lowercase(text)
    text = remove_numbers(text)
    text = remove_stopwords(text)
    text = lemma(text)
    return text


# Recipe basic preprocessing
def basic_preprocess_tags(df: pd.DataFrame) -> pd.DataFrame:
    removed_tags = ['time-to-make']
    df['tags'] = df['tags'].apply(lambda tag_list: [x for x in tag_list if x not in removed_tags])
    df['tags'] = df['tags'].apply(lambda tags: ' '.join(tags))
    return df

def basic_preprocess_recipe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df =  basic_preprocess_tags(df)
    recipe_text_columns = [
        'name',
        'description',
        'merged_steps',
        'tags',
        'merged_ingredients'
    ]
    for col in recipe_text_columns:
        df[col] = df[col].apply(basic_word_processing)

    return df

def numeric_preproc(data: pd.DataFrame) -> pd.DataFrame:

    data['minutes'] = data['minutes'].clip(lower =5 ,upper=130)
    data.drop(data[data.n_steps == 0].index, inplace=True)
    data['n_steps'] = data['n_steps'].clip(upper=40)

    rb_scaler = RobustScaler()
    col_to_preproc = [
        'minutes',
        'calories',
        'total_fat_pdv',
        'sugar_pdv',
        'sodium_pdv',
        'protein_pdv',
        'saturated_fat_pdv',
        'carbohydrates_pdv']

    data[col_to_preproc] = rb_scaler.fit_transform(data[col_to_preproc])
    return data

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

def get_y(df_recipe: pd.DataFrame, df_reviews : pd.DataFrame) -> pd.DataFrame:
    df_reviews.rename(columns={"recipe_id": "id"}, inplace=True)
    ratings_rview_cnt = df_reviews.groupby("id")[['rating']].agg(
                                    mean_rating=('rating', 'mean')
                                    )
    return df_recipe.merge(ratings_rview_cnt, how="inner", on='id')
