import pandas as pd
import ast


def ping():
    return "pong"


def convert_column_to_list(column):
    return column.apply(ast.literal_eval)


def add_columns_and_merge_text(df:pd.DataFrame) -> pd.DataFrame:

    # set id as column
    df.set_index('id', inplace=True)

    df['tags'] = convert_column_to_list(df['tags'])
    df['nutrition'] = convert_column_to_list(df['nutrition'])
    df['steps'] = convert_column_to_list(df['steps'])
    df['ingredients'] = convert_column_to_list(df['ingredients'])

    nutrition_columns = ['calories', 'total_fat_pdv', 'sugar_pdv', 'sodium_pdv', 'protein_pdv', 'saturated_fat_pdv', 'carbohydrates_pdv']
    df[nutrition_columns] = pd.DataFrame(df['nutrition'].tolist(), index=df.index)

    df['merged_steps'] = df['steps'].apply(lambda steps: "\n".join(steps))
    df['merged_steps_length'] = df['merged_steps'].apply(lambda x:len(x))

    # df['steps_merged_length'] =

    return df
