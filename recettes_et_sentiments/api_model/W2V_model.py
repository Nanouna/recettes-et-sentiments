from gensim.models import Word2Vec
import logging
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import typing

from recettes_et_sentiments.api_model import parameters


logger = logging.getLogger(__name__)

def train_word2vec(data: pd.Series) -> Word2Vec:
    """
    Instantiate the word2Vec model on the required text as list of all values
    """
    model =  Word2Vec(sentences=data,
                      vector_size=parameters.W2V_VECTOR_SIZE,
                      window=parameters.W2V_WINDOW,
                      min_count=parameters.W2V_MIN_COUNT
                      )
    logger.info("Model trained")
    return model

def embed_sentence(word2vec_model: Word2Vec, text_elements : list) -> list:
    '''
    Vectorization of one recipe text elements

    Then we apply the mean of each vector from each vectorization list of list elements
    to have a final average vector for each recipe

    return average vector as list for a recipe
    '''
    vectors = [word2vec_model.wv[word] for word in text_elements if word in word2vec_model.wv.key_to_index]
    return np.mean(np.vstack(vectors), axis=0) if vectors else np.zeros(parameters.W2V_VECTOR_SIZE)

def embedding(word2vec_model : Word2Vec, sentences:list) -> list:
    '''
    Vectorization of all recipes text elements as list

    As list comprehension to loop for each recipe

    return list of one vector by recipe
    '''
    logger.info("Vectorization and averaging started")
    list_vectors = [embed_sentence(word2vec_model, sentence) for sentence in sentences]
    logger.info("Vectorization and averaging completed")
    return list_vectors

def preprocess_data(data: pd.DataFrame, column_to_process: str)-> typing.Union[pd.DataFrame, Word2Vec]:
    '''
    Compilation function of training and vectorization
    '''
    word2vec_model = train_word2vec(data[column_to_process])
    data[column_to_process+'_vector'] = embedding(word2vec_model, data[column_to_process].tolist())
    return data, word2vec_model

def instantiate_model(data: pd.DataFrame, processed_col:str):
    '''
    Instantiation of KNN model on recipes
    '''
    neigh = NearestNeighbors(n_neighbors=2, radius=0.4)
    neigh.fit(np.array(data[processed_col+'_vector'].tolist()))
    return neigh

def recommend_recipe_from_another(model, data: pd.DataFrame, processed_col:str, entry_recipe_id: int) -> pd.DataFrame:
    """
    KNN kneighbors using an existing recipe_id as reference
    """
    recipe_index = data.index.get_loc(entry_recipe_id)
    distances, indices = model.kneighbors([data.iloc[recipe_index][processed_col+'_vector']])

    if len(indices[0]) < 2:
        return None
    return data.iloc[indices[0][1]]
