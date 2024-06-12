import os
import typing
import joblib
import logging

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.metrics.pairwise import cosine_similarity

from recettes_et_sentiments.api_model.preprocessing_pipeline import CacheStep, BasicPreprocessing, ConcatColumns
from recettes_et_sentiments.api_model import rs_data, registry
from recettes_et_sentiments.api_model.fast_vectorizer import FastVectorizer

logger = logging.getLogger(__name__)


def rename_columns(X, original_columns, vector_size, columns_to_merge_for_training):
    vector_columns = [f'vector_{i}' for i in range(vector_size-len(original_columns)-len(columns_to_merge_for_training)+1)]

    remaining_columns = [x for x in original_columns if x not in columns_to_merge_for_training]
    remaining_columns.append("merged_text")
    logger.info(f"rename_columns : {'----'*10}")
    logger.info("rename_columns : ", vector_columns + remaining_columns)
    return pd.DataFrame(X, columns=vector_columns + remaining_columns)


def make_fast_preprocessor_pipeline(columns_to_merge_for_training:typing.List[str],
                                    original_columns:typing.List[str],
                                vector_size=100,
                                window=5,
                                min_count=1,
                                workers=4,
                                cache=True
                               ) -> Pipeline:

    vectorizer = FastVectorizer(vector_size=vector_size, window=window, min_count=min_count, workers=workers)

    basic_preproc = BasicPreprocessing()
    concat_columns = ConcatColumns(columns=columns_to_merge_for_training)

    folder =  "/tmp/data/"
    if not os.path.exists(folder):
        os.mkdir(folder)

    basic_preproc_filename = f"{folder}basic_preproc.parquet"
    concat_columns_filename = f"{folder}concat_columns_{'_'.join(columns_to_merge_for_training)}.parquet"

    if cache:
        basic_preproc = CacheStep(basic_preproc_filename, basic_preproc)
        concat_columns = CacheStep(concat_columns_filename, concat_columns)

    column_transformer = ColumnTransformer(
        transformers=[
            ('text', vectorizer, 'merged_text')
        ],
        remainder='passthrough',
        sparse_threshold=0,
        n_jobs=-1
    )

    rename_transformer = FunctionTransformer(rename_columns, kw_args={
        'original_columns': original_columns,
        'vector_size': vector_size,
        'columns_to_merge_for_training':columns_to_merge_for_training
        })


    preprocessing_pipeline = Pipeline(steps=[
        ('basic_preproc', basic_preproc),
        ('concat_columns', concat_columns),
        ('vectorize_and_combine', column_transformer),
        ('rename_columns', rename_transformer)

    ])

    return preprocessing_pipeline


def find_recipie_with_similar_elements(query:str, model_fast=None, recipe_processed=None):

    if model_fast is None:
        model_fast = registry.load_fast_model(model_name="model_fast")
    if recipe_processed is None:
        recipe_processed = pd.read_parquet("/tmp/data/preproc_recipes_fast_name-tag-desc-ingredients.parquet")



    input_name = model_fast.get_inputs()[0].name


    ingredients_vector = model_fast.run(None, {input_name: query})

    # ingredients_vector = model_fast.named_steps['vectorize_and_combine'].named_transformers_['text']._get_mean_vector(query).reshape(1, -1)

    similarities = cosine_similarity(ingredients_vector, np.vstack(recipe_processed.iloc[:, :2000].values))

    top_indices = similarities.argsort()[0][-5:]  # Top 5 similar recipes

    recommended_recipes_id = recipe_processed.iloc[top_indices]['id']

    return recommended_recipes_id




if __name__ == "__main__":

    # train_size = int(recipe_df_ori.shape[0] * 0.8)
    # train_recipe_df = recipe_df_ori.iloc[0:train_size]
    # test_recipe_df  = recipe_df_ori.iloc[train_size+1:]

    recipe_df_ori = rs_data.load_recipes("../batch-1672-recettes-et-sentiments-data/RAW_recipes.csv")


    preprocessor_pipeline = registry.load_fast_model(model_name="model_fast")

    if preprocessor_pipeline is None:

        logger.info(f"creating FAST model")

        preprocessor_pipeline = make_fast_preprocessor_pipeline(
            columns_to_merge_for_training=["name", "tags", "description", "merged_ingredients"],
            original_columns=recipe_df_ori.columns,
            vector_size=10,
            window=10,
            min_count=1,
            workers=6,
            cache=True
        )
        preprocessor_pipeline.fit(recipe_df_ori)

        logger.info(f"preprocessor_pipeline.get_feature_names_out()={preprocessor_pipeline.get_feature_names_out()}")
        logger.info(f"preprocessor_pipeline.get_params()={preprocessor_pipeline.get_params()}")

        registry.save_fast_model(preprocessor_pipeline, recipe_df_ori.shape, model_name="model_fast")
        # reload from disk to have deal with the pipeline in the same way (just fitted :  we have a regular pipeline, loaded from disk we must comply with a specific syntax)
        preprocessor_pipeline = registry.load_fast_model(model_name="model_fast")
        logger.info(f"FAST model - DONE and saved")


    recipe_processed_cache_path = f"/tmp/data/preproc_recipes_fast_name-tag-desc-ingredients.parquet"

    if os.path.exists(recipe_processed_cache_path):
        logger.info(f"Loading Preprocessed DataFrame from {recipe_processed_cache_path}")
        recipe_processed = pd.read_parquet(recipe_processed_cache_path)
        logger.info(f"Loading Preprocessed DataFrame from {recipe_processed_cache_path} - DONE")
    else:
        logger.info(f"Creating Preprocessed DataFrame")
        recipe_processed = preprocessor_pipeline.transform(recipe_df_ori)
        print(recipe_processed.head())
        recipe_processed.to_parquet(recipe_processed_cache_path)
        logger.info(f"Storing Preprocessed DataFrame to {recipe_processed_cache_path}")


    available_ingredients = ['burrito', 'porc', 'feta']

    # Transform ingredients to vector
    ingredients_text = ' '.join(available_ingredients)
    input_name = preprocessor_pipeline.get_inputs()[0].name

    ingredients_vector = preprocessor_pipeline.run(None, {input_name: ingredients_text})
    # ingredients_vector = preprocessor_pipeline.named_steps['vectorize_and_combine'].named_transformers_['text']._get_mean_vector(ingredients_text).reshape(1, -1)


    similarities = cosine_similarity(ingredients_vector, np.vstack(recipe_processed.iloc[:, :2000].values))
    top_indices = similarities.argsort()[0][-5:]  # Top 5 similar recipes

    recommended_recipes = recipe_processed.iloc[top_indices]
    print(recommended_recipes)
