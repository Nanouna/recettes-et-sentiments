import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from gensim.models import Word2Vec

data = pd.read_parquet('../data/last_preproc_we_hope.parquet')

def train_word2vec(sentences, vector_size=700, window=5, min_count=1, workers=4):
    return Word2Vec(sentences=sentences, vector_size=vector_size, window=window, min_count=min_count, workers=workers)

def embed_sentence(word2vec_model, sentence, vector_size=700):
    vectors = [word2vec_model.wv[word] for word in sentence if word in word2vec_model.wv.key_to_index]
    return np.mean(np.vstack(vectors), axis=0) if vectors else np.zeros(vector_size)

def embedding(word2vec_model, sentences, vector_size):
    return [embed_sentence(word2vec_model, sentence, vector_size) for sentence in sentences]

def preprocess_data(data, vector_size=700, window=5, min_count=1, workers=4):
    word2vec_model_tags = train_word2vec(data['tags'], vector_size, window, min_count, workers)
    word2vec_model_ingr = train_word2vec(data['ingredients'], vector_size, window, min_count, workers)
    data['ingredients_vectors'] = embedding(word2vec_model_ingr, data['ingredients'].tolist(), vector_size=700)
    data['tag_vectors'] = embedding(word2vec_model_tags, data['tags'].tolist(), vector_size=700)
    data['combined_vectors'] = data.apply(lambda row: np.concatenate([row['tag_vectors'], row['ingredients_vectors']]), axis=1)

    return data, word2vec_model_ingr, word2vec_model_tags

def recommend_recipe(recipes_with_vectors, recipe_id):

    recipe_index = recipes_with_vectors.index.get_loc(recipe_id)
    distances, indices = neigh.kneighbors([recipes_with_vectors.iloc[recipe_index]['combined_vectors']])

    if len(indices[0]) < 2:
        return None
    return recipes_with_vectors.iloc[indices[0][1]]

recipes_with_vectors, word2vec_model_tags, word2vec_model_ingr = preprocess_data(data)

neigh = NearestNeighbors(n_neighbors=2, radius=0.9)
neigh.fit(np.array(data['combined_vectors'].tolist()))

recommended_recipe = recommend_recipe(recipes_with_vectors, recipe_id=33606)
print(recommended_recipe)
