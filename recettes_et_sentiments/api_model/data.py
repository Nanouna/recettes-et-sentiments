import ast
import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    '''
    load csv files
    '''
    df = pd.read_csv(path,
                     parse_dates=['submitted'],
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
    df['merged_steps'] = df['steps'].apply(lambda steps: "\n".join(steps))
<<<<<<< Updated upstream
=======
    df['ingredients'] = df['steps'].apply(lambda steps: "\n".join(steps))
>>>>>>> Stashed changes
    # df['merged_steps_length'] = df['merged_steps'].apply(lambda x:len(x))

    # drop redundant columns
    df.drop(columns={'nutrition', 'steps'}, inplace=True)

    return df
