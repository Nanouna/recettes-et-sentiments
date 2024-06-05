import ast
import pandas as pd


def load_recipes(path: str) -> pd.DataFrame:
    '''
    load csv files
    '''
    df = pd.read_csv(path,
                     parse_dates=['submitted'],
                     engine='python')
    return df

def load_reviews(path: str) -> pd.DataFrame:
    '''
    load csv files
    '''
    df = pd.read_csv(path,
                     parse_dates=['date'],
                     engine='python')
    return df


def convert_column_to_list(column):
    '''
    Read csv as python interpretable data
    '''
    return column.apply(ast.literal_eval)


def add_columns_and_merge_text(df:pd.DataFrame) -> pd.DataFrame:
    '''
    Convert RAW_recipe_csv dataframe as python objects and process its data in
    relevant columns

    return dataframe
    '''

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
    df.drop(columns={'nutrition', 'steps', 'ingredients'}, inplace=True)

    return df

def get_y(df_recipe: pd.DataFrame, df_reviews : pd.DataFrame) -> pd.DataFrame:
    df_reviews.rename(columns={"recipe_id": "id"}, inplace=True)
    ratings_rview_cnt = df_reviews.groupby("id")[['rating']].agg(
                                    mean_rating=('rating', 'mean')
                                    )
    return df_recipe.merge(ratings_rview_cnt, how="inner", on='id')
