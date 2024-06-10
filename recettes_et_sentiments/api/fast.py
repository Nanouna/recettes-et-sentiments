import pandas as pd
import logging

logger = logging.getLogger(__name__)


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from recettes_et_sentiments.api_model.registry import load_model
from recettes_et_sentiments.api_model.FAST_model_variant import find_recipie_with_similar_elements_model_fast
from recettes_et_sentiments.api_model import W2V_model



app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


#app.state.model_fast = load_model("model_fast")
#app.state.recipe_processed = pd.read_parquet("/tmp/data/preproc_recipes_fast_name-tag-desc-ingredients.parquet")

@app.get("/model_fast")
def model_fast(query:str):
    # in real world, we would check the input

    result = find_recipie_with_similar_elements_model_fast(query=query, model_fast=app.state.model_fast, recipe_processed=app.state.recipe_processed)
    return {
        'query': query,
        'result': result.to_json()
    }



# http://localhost:8000/model_w2vec_similar_to_recipe?recipe_id=20374
@app.get("/model_w2vec_similar_to_recipe")
def model_fast(recipe_id:int):
    # in real world, we would check the input


    recipes_with_vectors, word2vec_model = W2V_model.preprocess_data(pd.DataFrame(), 'tags')
    KNN_model = W2V_model.instantiate_model(recipes_with_vectors, 'tags')
    recommended_recipe_id, recommended_recipe = W2V_model.recommend_recipe_from_another(KNN_model, recipes_with_vectors, 'tags', entry_recipe_id=recipe_id)

    # recommended_recipe[['tags']] = recommended_recipe[['tags']].apply(lambda x:', '.join(x))


    return {
        'query': recipe_id,
        'recommended_recipe_id': int(recommended_recipe_id),
        'recommended_recipe_title': recommended_recipe.iloc[0],
        'recommended_recipe_url': f"https://www.food.com/recipe/*-{recommended_recipe_id}"
    }


@app.get("/model_w2vec_query_recipe")
def model_fast(query:str):
    # in real world, we would check the input
    recipes_with_vectors, word2vec_model = W2V_model.preprocess_data(pd.DataFrame(), 'tags')
    KNN_model = W2V_model.instantiate_model(recipes_with_vectors, 'tags')
    recommended_recipe_custom = W2V_model.recommend_recipe_from_custom_input(word2vec_model, KNN_model, recipes_with_vectors, ['christmas', 'gifts', 'chocolate', 'healthy'])

    return {
        'query': query,
        'recommended_recipe_id': recommended_recipe_custom.to_json(),
        'recommended_recipe_title': 0,
        'recommended_recipe_url': f"https://www.food.com/recipe/*-{recommended_recipe_id}"
    }


@app.get("/")
def root():
    return {
        'greeting': 'Hello'
    }
