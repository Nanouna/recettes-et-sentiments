import os
import pandas as pd
from recettes_et_sentiments.api_model import preprocessing, rs_data, W2V_model

if __name__ == "__main__":
    path = '/tmp/data/'
    if not os.path.exists(path):
        os.makedirs(path)

    local_raw_path = 'data/RAW_recipes.csv'
    raw = rs_data.load_recipes(local_raw_path)
    data = preprocessing.full_basic_preproc_recipes(raw)
    data.to_parquet('final_preproc_with_ingredients.parquet')

    data = pd.read_parquet('final_preproc_with_ingredients.parquet')
    column_to_train = ['col_concat', 'tags', 'ingredients']
    data['ingredients'] = [list(row) for row in data['ingredients']]
    data['tags'] = [list(row) for row in data['tags']]
    data['col_concat'] = data['ingredients'] + data['tags']

    for col in column_to_train:
        recipes_with_vectors, word2vec_model = W2V_model.preprocess_data(data, col)
        KNN_model = W2V_model.instantiate_model(recipes_with_vectors, col)
