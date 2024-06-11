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


app.state.recipes_with_vectors, app.state.word2vec_model = W2V_model.preprocess_data(pd.DataFrame(), 'tags')
app.state.knn_model = W2V_model.instantiate_model(app.state.recipes_with_vectors, 'tags')

# http://localhost:8000/model_w2vec_similar_to_recipe?recipe_id=20374
@app.get("/model_w2vec_similar_to_recipe")
def model_fast(recipe_id:int):

    logger.info(f"model_w2vec_similar_to_recipe(recipe_id={recipe_id})")
    recommended_recipes = W2V_model.recommend_recipe_from_another(app.state.knn_model, app.state.recipes_with_vectors, 'tags', entry_recipe_id=recipe_id)

    suggestions = []
    for index, row in recommended_recipes.iterrows():
        suggestions.append(
            [
                index,
                row['name'],
                f"https://www.food.com/recipe/*-{index}"
             ]
            )

    logger.info(f"model_w2vec_similar_to_recipe(recipe_id={recipe_id}) -> {suggestions}")
    return {
        'query': recipe_id,
        'suggestions':suggestions
    }



# http://localhost:8000/model_w2vec_query_recipe?query=christmas%20gifts%20chocolate%20healthy
@app.get("/model_w2vec_query_recipe")
def model_fast(query:str):
    # in real world, we would check the input
    logger.info(f"model_w2vec_similar_to_recipe(query={query})")

    recommended_recipe_custom = W2V_model.recommend_recipe_from_custom_input(
        app.state.word2vec_model,
        app.state.knn_model,
        app.state.recipes_with_vectors,
        query.split()
        )

    suggestions = []
    for index, row in recommended_recipe_custom.iterrows():
        suggestions.append(
            [
                index,
                row['name'],
                f"https://www.food.com/recipe/*-{index}"
             ]
            )

    logger.info(f"model_w2vec_similar_to_recipe(query={query}) -> {suggestions}")

    return {
        'query': query,
        'suggestions': suggestions
    }


@app.get("/")
def root():
    return {
        'greeting': 'Hello'
    }
