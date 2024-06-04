import pandas as pd
import string
import typing

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from recettes_et_sentiments.api_model import parameters


# classic NLP preprocessing
def remove_punctuation(text:str) -> str:
    for punctuation in string.punctuation:
        text = text.replace(punctuation, ' ')
    return text

def lowercase(text:str) -> str:
    lowercased = text.lower()
    return lowercased

def remove_numbers(text:str) -> str:
    words_only = ''.join([i for i in text if not i.isdigit()])
    return words_only

def remove_stopwords(text:str) -> typing.List[str]:
    tokenized = word_tokenize(text)
    without_stopwords = [word for word in tokenized if not word in parameters.STOP_WORDS]
    return without_stopwords

def lemma(text: typing.List[str])-> str:
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word) for word in text]
    lemmatized_string = " ".join(lemmatized)
    return lemmatized_string

def basic_word_processing(text:str) -> str:
    text = remove_punctuation(text)
    text = lowercase(text)
    text = remove_numbers(text)
    text = remove_stopwords(text)
    text = lemma(text)
    return text


# Recipe basic preprocessing
<<<<<<< Updated upstream
=======
def basic_preprocess_tags(df: pd.DataFrame) -> pd.DataFrame:
    removed_tags = ['time-to-make']
    df['tags'] = df['tags'].apply(lambda tag_list: [x for x in tag_list if x not in removed_tags])
    df['tags'] = df['tags'].apply(lambda tags: ' '.join(tags))
    return df

>>>>>>> Stashed changes
def basic_preprocess_recipe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    recipe_text_columns = [
        'name',
        'description',
<<<<<<< Updated upstream
        'merged_steps'
=======
        'merged_steps',
        'tags',
        'ingredients'
>>>>>>> Stashed changes
    ]
    for col in recipe_text_columns:
        df[col] = df[col].apply(basic_word_processing)
    return df
