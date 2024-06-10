import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from recettes_et_sentiments.api_model.registry import load_model
from recettes_et_sentiments.api_model.FAST_model_variant import find_recipie_with_similar_elements_model_fast



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




@app.get("/")
def root():
    return {
        'greeting': 'Hello'
    }
