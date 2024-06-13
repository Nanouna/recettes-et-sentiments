import pandas as pd
import logging
from pydantic import BaseModel, Field
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from recettes_et_sentiments.api_model import registry
from recettes_et_sentiments.api_model.FAST_model_variant import find_recipie_with_similar_elements_model_fast, FastVectorizer
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

try:
    app.state.recipes_with_vectors_tags, app.state.word2vec_model_tags = W2V_model.preprocess_data(pd.DataFrame(), 'tags')
    app.state.knn_model_tags = W2V_model.instantiate_model(app.state.recipes_with_vectors_tags, 'tags')

    app.state.recipes_with_vectors_ingredients, app.state.word2vec_model_ingredients = W2V_model.preprocess_data(pd.DataFrame(), 'ingredients')
    app.state.knn_model_ingredients = W2V_model.instantiate_model(app.state.recipes_with_vectors_ingredients, 'ingredients')

    app.state.recipes_with_vectors_col_concat, app.state.word2vec_model_col_concat = W2V_model.preprocess_data(pd.DataFrame(), 'col_concat')
    app.state.knn_model_col_concat = W2V_model.instantiate_model(app.state.recipes_with_vectors_col_concat, 'col_concat')
except Exception as e:
    logger.error(f"An error occurred while loading the model: {e}", exc_info=True)



class RecipeSuggestion(BaseModel):
    index: int = Field(..., description="The unique identifier for the suggested recipe.")
    name: str = Field(..., description="The name of the suggested recipe.")
    url: str = Field(..., description="The URL to view the suggested recipe.")

class RecipeResponse(BaseModel):
    query: str = Field(..., description="The query that was used to find similar recipes.")
    suggestions: List[RecipeSuggestion] = Field(
        ...,
        description="A list of suggested recipes based on the query.",
        example=[
            {
                "index": 320948,
                "name": "GRUESOME MONSTER TOES",
                "url": "https://www.food.com/recipe/*-320948"
            }
        ]
    )


# http://localhost:8000/model_w2vec_similar_to_recipe?recipe_id=20374
# https://recettes-et-sentiments-api-p4x6pl7fiq-ew.a.run.app/model_w2vec_similar_to_recipe?recipe_id=331985
@app.get("/model_w2vec_similar_to_recipe")
def model_fast(recipe_id:int = Query(..., description="ID of an existing recipe from food.com"))->RecipeResponse:
    """
    Find recipes that are similar to an existing recipe of id recipe_id

    This method uses W2VEC model trained on tags of recipes (dataset kaggle built on food.com)

    - **query**: A string containing the ingredients separated by spaces.
    - **returns**: A list of suggested recipes with their names and URLs.

    maybe you can find a better soup than this one :
        <a href="https://www.food.com/recipe/cheeseburger-soup-44294" target="_blank">CHEESEBURGER SOUP</a>
    """
    logger.info(f"model_w2vec_similar_to_recipe(recipe_id={recipe_id})")
    try:
        suggestions = []

        recommended_recipes = W2V_model.recommend_recipe_from_another(
            app.state.knn_model_col_concat,
            app.state.recipes_with_vectors_col_concat,
            'col_concat',
            entry_recipe_id=recipe_id
            )

        for index, row in recommended_recipes.iterrows():
            suggestions.append(
                 RecipeSuggestion(
                    index=index,
                    name=row['name'],
                    url=f"https://www.food.com/recipe/*-{index}"
                )
            )
    except KeyError as key_error:
        error_message = f"The recipe_id={recipe_id} has not been found"
        logger.error(error_message, key_error)
        raise HTTPException(status_code=404, detail=error_message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


    logger.info(f"model_w2vec_similar_to_recipe(recipe_id={recipe_id}) -> {suggestions}")
    return RecipeResponse(
            query=str(recipe_id),
            suggestions=suggestions
        )



# http://localhost:8000/model_w2vec_query_recipe_with_tags?query=christmas%20gifts%20chocolate%20healthy
# https://recettes-et-sentiments-api-p4x6pl7fiq-ew.a.run.app/model_w2vec_query_recipe_with_tags?query=christmas%20gifts%20chocolate%20healthy
@app.get("/model_w2vec_query_recipe_with_tags")
def model_fast(query:str = Query(..., description="Tags query string",min_length=3, max_length=300))->RecipeResponse:
    """
    Find recipes based on the tags provided in the query string.

    This method uses W2VEC model trained on tags of recipes (dataset kaggle built on food.com)

    - **query**: A string containing the ingredients separated by spaces.
    - **returns**: A list of suggested recipes with their names and URLs.

    Stop playing with kids stuff, search healthy recipe to avoid this nightmare :

    <a href="https://www.food.com/recipe/koolaid-pie-106096" target="_blank">KOOLAID PIE</a>
    """
    # in real world, we would check the input
    logger.info(f"model_w2vec_query_recipe_with_tags(query={query})")

    try:

        recommended_recipe_custom = W2V_model.recommend_recipe_from_custom_input(
            app.state.word2vec_model_tags,
            app.state.knn_model_tags,
            app.state.recipes_with_vectors_tags,
            query.split()
            )



        suggestions = []
        for index, row in recommended_recipe_custom.iterrows():
            suggestions.append(
                 RecipeSuggestion(
                    index=index,
                    name=row['name'],
                    url=f"https://www.food.com/recipe/*-{index}"
                )
            )

        logger.info(f"model_w2vec_query_recipe_with_tags(query={query}) -> {suggestions}")

        return RecipeResponse(
            query=query,
            suggestions=suggestions
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# http://localhost:8000/model_w2vec_query_recipe_with_ingredients?query=christmas%20gifts%20chocolate%20healthy
# https://recettes-et-sentiments-api-p4x6pl7fiq-ew.a.run.app/model_w2vec_query_recipe_with_ingredients?query=chocolate%20mayonnaise
@app.get("/model_w2vec_query_recipe_with_ingredients")
def model_fast(query:str = Query(..., description="Ingredients query string",min_length=3, max_length=300))->RecipeResponse:
    """
    Find recipes based on the ingredients provided in the query string.

    This method uses W2VEC model trained on tags of recipes (dataset kaggle built on food.com)

    - **query**: A string containing the ingredients separated by spaces.
    - **returns**: A list of suggested recipes with their names and URLs.

    You surely can use Mayonnaise better than this recipe :

    <a href="https://www.food.com/recipe/moist-deep-chocolate-mayonnaise-cake-or-cupcakes-199954" target="_blank">MOIST DEEP CHOCOLATE MAYONNAISE CAKE OR CUPCAKES</a>
    """
    logger.info(f"model_w2vec_query_recipe_with_ingredients(query={query})")

    try:

        recommended_recipe_custom = W2V_model.recommend_recipe_from_custom_input(
            app.state.word2vec_model_ingredients,
            app.state.knn_model_ingredients,
            app.state.recipes_with_vectors_ingredients,
            query.split()
            )



        suggestions = []
        for index, row in recommended_recipe_custom.iterrows():
            suggestions.append(
                 RecipeSuggestion(
                    index=index,
                    name=row['name'],
                    url=f"https://www.food.com/recipe/*-{index}"
                )
            )

        logger.info(f"model_w2vec_query_recipe_with_ingredients(query={query}) -> {suggestions}")

        return RecipeResponse(
            query=query,
            suggestions=suggestions
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# http://localhost:8000/model_w2vec_query_recipe_with_tags_and_ingredients?query=christmas%20gifts%20chocolate%20healthy
# https://recettes-et-sentiments-api-p4x6pl7fiq-ew.a.run.app/model_w2vec_query_recipe_with_tags_and_ingredients?query=christmas%20gifts%20chocolate%20healthy
@app.get("/model_w2vec_query_recipe_with_tags_and_ingredients")
def model_fast(query:str = Query(..., description="Ingrédients or Tags query string",min_length=3, max_length=300))->RecipeResponse:
    """
    Find recipes based on the ingredients or tags provided in the query string.

    This method uses W2VEC model trained on tags of recipes (dataset kaggle built on food.com)

    - **query**: A string containing the ingredients separated by spaces.
    - **returns**: A list of suggested recipes with their names and URLs.

    You can do better with steak than that :
    <a href="https://www.food.com/recipe/cola-steak-crock-pot-92809" target="_blank">COLA STEAK CROCK POT</a>
    """
    logger.info(f"model_w2vec_query_recipe_with_tags_and_ingredients(query={query})")

    try:

        recommended_recipe_custom = W2V_model.recommend_recipe_from_custom_input(
            app.state.word2vec_model_col_concat,
            app.state.knn_model_col_concat,
            app.state.recipes_with_vectors_col_concat,
            query.split()
            )



        suggestions = []
        for index, row in recommended_recipe_custom.iterrows():
            suggestions.append(
                 RecipeSuggestion(
                    index=index,
                    name=row['name'],
                    url=f"https://www.food.com/recipe/*-{index}"
                )
            )

        logger.info(f"model_w2vec_query_recipe_with_tags_and_ingredients(query={query}) -> {suggestions}")

        return {
            'query': query,
            'suggestions': suggestions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {
        'greeting': 'Hello'
    }
