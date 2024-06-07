import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from gensim.models import Word2Vec

def train_word2vec(ingredients, vector_size=100, window=5, min_count=1, workers=4):
    return Word2Vec(sentences=ingredients, vector_size=vector_size, window=window, min_count=min_count, workers=workers)

def embed_sentence(word2vec_model, sentence, vector_size=100):
    vectors = [word2vec_model.wv[word] for word in sentence if word in word2vec_model.wv.key_to_index]
    return np.mean(np.vstack(vectors), axis=0) if vectors else np.zeros(vector_size)

def embedding(word2vec_model, sentences, vector_size=100):
    return [embed_sentence(word2vec_model, sentence, vector_size) for sentence in sentences]

def preprocess_data(data, vector_size=100, window=5, min_count=1, workers=4):
    word2vec_model = train_word2vec(data['ingredients'], vector_size, window, min_count, workers)
    data['tag_vectors'] = embedding(word2vec_model, data['ingredients'].tolist(), vector_size)
    return data, word2vec_model

def recommend_recipe(recipes_with_vectors, recipe_id):
    neigh = NearestNeighbors(n_neighbors=2, radius=0.4)
    neigh.fit(np.array(recipes_with_vectors['ingredient_vectors'].tolist()))
    recipe_index = recipes_with_vectors.index.get_loc(recipe_id)
    distances, indices = neigh.kneighbors([recipes_with_vectors.iloc[recipe_index]['ingredient_vectors']])

    if len(indices[0]) < 2:
        return None
    return recipes_with_vectors.iloc[indices[0][1]]

data = pd.read_parquet('./data/last_preproc_we_hope.parquet')
recipes_with_vectors, word2vec_model = preprocess_data(data)
recommended_recipe = recommend_recipe(recipes_with_vectors, recipe_id=308080)
print(recommended_recipe)
