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
    data = [list(row) for row in data]
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

def embedding(word2vec_model : Word2Vec, text_elements:list) -> list:
    '''
    Vectorization of all recipes text elements as list

    As list comprehension to loop for each recipe

    return list of one vector by recipe
    '''
    logger.info("Vectorization and averaging started")
    list_vectors = [embed_sentence(word2vec_model, text) for text in text_elements]
    logger.info("Vectorization and averaging completed")
    return list_vectors

def preprocess_data(data: pd.DataFrame, column_to_process: str)-> typing.Union[pd.DataFrame, Word2Vec]:
    '''
    Compilation function of training and vectorization
    '''

    word2vec_model_cache_path = f"w2vec_model_{column_to_process}"

    word2vec_model = load_model(word2vec_model_cache_path)

    if word2vec_model is None:
        logger.info("Initializing model...")
        word2vec_model = train_word2vec(data[column_to_process])
        save_model(word2vec_model, word2vec_model_cache_path)


    word2vec_df_cache_path = f"/tmp/data/w2vec_df_{column_to_process}.parquet"
    if os.path.exists(word2vec_df_cache_path):
        data = pd.read_parquet(word2vec_df_cache_path)
    else:
        data[column_to_process+'_vector'] = embedding(word2vec_model, data[column_to_process].tolist())
        #probleme de to_parquet sous windows, pas sous linux ou mac.
        data[column_to_process+'_vector'] = [np.array(liste, dtype=np.float32) for liste in data[column_to_process+'_vector']]
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
        save_model(knn_model, knn_model_cache_path)

    return knn_model

def recommend_recipe_from_another(model, data: pd.DataFrame, processed_col:str, entry_recipe_id: int) -> pd.DataFrame:
    """
    KNN kneighbors using an existing recipe_id as reference
    """
    logger.info(f"Looking for recommendation based on recipe based on recipe id {entry_recipe_id}")
    recipe_index = data.index.get_loc(entry_recipe_id)

    distances, indices = model.kneighbors(
        [data.iloc[recipe_index][processed_col+'_vector']],
        n_neighbors=parameters.KNN_N_NEIGHBORS
        )

    if len(indices[0]) < 2:
        return None

    return pd.DataFrame(data.iloc[indices[0][1:]])

def recommend_recipe_from_custom_input(W2V_model: Word2Vec,
                                       KNN_model : NearestNeighbors,
                                       data: pd.DataFrame,
                                       custom_input: list
                                       ) -> pd.DataFrame:
    vectors = pd.DataFrame(embed_sentence(W2V_model, custom_input)).T
    logger.info(f"Looking for recommendation based on curstom input {custom_input}")
    distances, indices = KNN_model.kneighbors(vectors,
                                              n_neighbors=parameters.KNN_N_NEIGHBORS - 1
                                              )

    return pd.DataFrame(data.iloc[indices[0]])


if __name__ == "__main__":

    data = pd.read_parquet('final_preproc_with_ingredients.parquet')
    column_to_train= 'col_concat'
    data['ingredients'] = [list(row) for row in data['ingredients']]
    data['tags'] = [list(row) for row in data['tags']]
    data[column_to_train] = data['ingredients'] + data['tags']

    print("from recipe id 308080")
    recipes_with_vectors, word2vec_model = preprocess_data(data, column_to_train)
    print('preprocess done')
    KNN_model = instantiate_model(recipes_with_vectors, column_to_train)
    recommended_recipe = recommend_recipe_from_another(KNN_model, recipes_with_vectors, column_to_train, entry_recipe_id=308081)

    print(recommended_recipe)

    print("from recipe tags 'christmas', 'gifts', 'chocolate', 'healthy'")
    recommended_recipe_custom = recommend_recipe_from_custom_input(word2vec_model, KNN_model, recipes_with_vectors, ['rosh hashanah', 'gifts', 'chocolate'])

    print(recommended_recipe_custom)
