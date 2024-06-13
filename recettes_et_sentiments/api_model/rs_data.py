import os
import ast
import logging
import pandas as pd

logger = logging.getLogger(__name__)

def load_reviews(path: str) -> pd.DataFrame:
    """
    Load csv files from path, and parse dates columns
    """
    df = pd.read_csv(path,
                     parse_dates=['date'],
                     engine='python')
    return df


def convert_column_to_list(column):
    """
    Read csv as python interpretable data, used to convert a string reprenting
    a list of string ex : '"a","b","c"' to a list of string
    """
    return column.apply(ast.literal_eval)


def add_columns_and_merge_text(df:pd.DataFrame) -> pd.DataFrame:
    """
    Convert RAW_recipe_csv dataframe as python objects and process its data in
    relevant columns

    * set the recipe id (id) as index
    * convert each text columns, which all are list of phrases lists
    * steps & ingredients are converted to a single text with " " as concat separator
    * create a column per numérical values (calories & various Percent Daily Values)
    *

    return dataframe
    """
    logger.info("Starting add_columns_and_merge_text processing")

    # let's copy the df to be able to reload the function without missing column 'id' error
    df = df.copy()
    # set id as column
    df.set_index('id', inplace=True)

    # convert string list as list for future processing
    df['tags'] = convert_column_to_list(df['tags'])
    df['nutrition'] = convert_column_to_list(df['nutrition'])
    df['steps'] = convert_column_to_list(df['steps'])
    df['ingredients'] = convert_column_to_list(df['ingredients'])

    # extract nutrition facts as individual columns
    nutrition_columns = [
        'calories',
        'total_fat_pdv',
        'sugar_pdv',
        'sodium_pdv',
        'protein_pdv',
        'saturated_fat_pdv',
        'carbohydrates_pdv'
        ]
    df[nutrition_columns] = pd.DataFrame(df['nutrition'].tolist(), index=df.index)

    # merge list of steps as one string
    df['merged_steps'] = df['steps'].apply(lambda steps: " ".join(steps))
    df['merged_ingredients'] = df['ingredients'].apply(lambda steps: " ".join(steps))


    # drop redundant columns
    df.drop(columns={'nutrition', 'steps'}, inplace=True)

    logger.info("add_columns_and_merge_text done")

    return df

def load_recipes(path: str, store_parquet_path_prefix="/tmp/data/") -> pd.DataFrame:
    """
    load a parquet representing RAW_recipes.csv with some transformation if it exists
    or
    load the RAW_recipes.csv and apply the add_columns_and_merge_text() function
    and store the result as parquet file.
    """
    logger.info(f"loading '{path}'")

    # ex: "../batch-1672-recettes-et-sentiments-data/RAW_recipes.csv"
    base_name = os.path.splitext(os.path.basename(path))[0]  # Obtenir le nom de base sans l'extension
    new_file_name = base_name + ".parquet"

    parquet_path = f"{store_parquet_path_prefix}{new_file_name}"
    logger.info(f"checking if parquet with added columns and merged text exist in {parquet_path}")
    if os.path.exists(parquet_path):
        logger.info(f"{parquet_path} found - reading and returning the file")
        return pd.read_parquet(parquet_path)

    logger.info(f"{parquet_path} not found - generating parquet file")
    df = pd.read_csv(path,
                     parse_dates=['submitted'],
                     engine='python')
    logger.info(f"loading '{path}' done.")

    processed_df = add_columns_and_merge_text(df)
    processed_df.to_parquet(parquet_path, index=True)

    logger.info(f"{parquet_path} saved")

    return processed_df

def get_y(df_recipe: pd.DataFrame, df_reviews : pd.DataFrame) -> pd.DataFrame:
    """
    Compute the average recipe rating and the number of ratings per recipe.
    The number of recipe will be use to select recipe with enough rating to
    be relevant for our study
    """
    logger.info("Starting get_y processing")
    df_reviews.rename(columns={"recipe_id": "id"}, inplace=True)
    ratings_rview_cnt = df_reviews.groupby("id")[['rating']].agg(
                                    mean_rating=('rating', 'mean'),
                                    count_rating=('rating', 'count')
                                    )
    df =  df_recipe.merge(ratings_rview_cnt, how="inner", on='id')

    logger.info("get_y done")
    return df
