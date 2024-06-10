from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression
import pandas as pd

print("loading parquet")
preproc_with_reviews = pd.read_parquet(f"../../batch-1672-recettes-et-sentiments-data/preproc_recipes_tfidfvectorizer_defaults_with_y.parquet")
preproc_with_reviews.drop(columns='remainder__merged_steps', axis=1, inplace=True)
print("loading parquet - done ")
print("cross_validate - start")
cv_nb = cross_validate(
    LinearRegression(n_jobs=-1),
    preproc_with_reviews.drop(columns='mean_rating'),
    preproc_with_reviews['mean_rating'],
    cv=5,
    n_jobs=6,
    scoring='r2',
    verbose=2
)
print("cross_validate - end")
print("exporting results")
#round(cv_nb['test_score'].mean(),2)
cv_results_df = pd.DataFrame(cv_nb)
cv_results_df.index.name = 'Fold'
cv_results_df.reset_index(inplace=True)

cv_results_df.to_csv('cross_validate_results.csv', index=False)
print("exporting results - done")
