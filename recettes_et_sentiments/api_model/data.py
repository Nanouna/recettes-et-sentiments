import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv("../../batch-1672-recettes-et-sentiments-data/RAW_recipes.csv",
                     parse_dates=['submitted'],
                     engine='python')
    return df
