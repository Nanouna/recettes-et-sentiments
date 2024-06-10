import os
import joblib
from gensim.models import Word2Vec
import logging
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import typing

from recettes_et_sentiments.api_model import parameters
from recettes_et_sentiments.api_model.registry import load_model, save_model


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

    word2vec_model_cache_path = f"w2vec_model_{column_to_process}"

    word2vec_model = load_model(word2vec_model_cache_path)

    if word2vec_model is None:
        word2vec_model = train_word2vec(data[column_to_process])
        save_model(word2vec_model_cache_path)


    word2vec_df_cache_path = f"/tmp/data/w2vec_df_{column_to_process}.parquet"
    if os.path.exists(word2vec_df_cache_path):
        data = pd.read_parquet(word2vec_df_cache_path)
    else:
        data[column_to_process+'_vector'] = embedding(word2vec_model, data[column_to_process].tolist())
        data.to_parquet(word2vec_df_cache_path)

    return data, word2vec_model

def instantiate_model(data: pd.DataFrame, column_to_process:str):
    '''
    Instantiation of KNN model on recipes
    '''

    knn_model_cache_path = f"knn_model_{column_to_process}"

    knn_model = load_model(knn_model_cache_path)

    if knn_model is None:
        knn_model = NearestNeighbors(n_neighbors=2, radius=0.4)
        knn_model.fit(np.array(data[column_to_process+'_vector'].tolist()))
        save_model(knn_model_cache_path)

    return knn_model

def recommend_recipe_from_another(model, data: pd.DataFrame, processed_col:str, entry_recipe_id: int) -> pd.DataFrame:
    """
    KNN kneighbors using an existing recipe_id as reference
    """
    recipe_index = data.index.get_loc(entry_recipe_id)
    distances, indices = model.kneighbors([data.iloc[recipe_index][processed_col+'_vector']])

    if len(indices[0]) < 2:
        return None
    return data.iloc[indices[0][1]]

if __name__ == "__main__":



    data = pd.read_parquet('../batch-1672-recettes-et-sentiments-data/last_preproc_we_hope.parquet')

    recipes_with_vectors, word2vec_model = preprocess_data(data, 'tags')
    KNN_model = instantiate_model(recipes_with_vectors, 'tags')
    recommended_recipe = recommend_recipe_from_another(KNN_model, recipes_with_vectors, 'tags', entry_recipe_id=308080)


    print(recommended_recipe)
