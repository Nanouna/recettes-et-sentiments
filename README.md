<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]


# recettes-et-sentiments

_Authors_
 - [Marie Anne](https://github.com/Nanouna)
 - [Théo](https://github.com/theorosen12)
 - [Thomas](https://github.com/mansonthomas-wagon)

## Description

This is the application project of the [Datascience & IA](https://www.lewagon.com/fr/remote/data-science-course) course, batch #1672 - April/June 2024

It uses the Kaggle [Food.com Recipes and Interactions](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions) dataset.

The purpose of the project is to apply our learnings in a full project from training of a model to deployment in GCP on Cloud Run with a web interface.

The first goal is to predict the the user review score from the recipe description, using a Machine Learning algorithm.

The second goal is to recommend some recipes, first based on a recipe proximity, then based on a user prompt.

The third goal is to generate a brand new receipes from a list of key ingredients.
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Setup

### requirements

 - python 3.10.6
 - Tested on Windows(11) & MacOSX (intel/sonoma)

 ### steps

- git clone git@github.com:Nanouna/recettes-et-sentiments.git
- cd recettes-et-sentiments
- pyenv virtualenv recettes-et-sentiments
- pyenv local recettes-et-sentiments
- pip install --upgrade pip
- cd recettes_et_sentiments
- pip install -r requirements.txt
- pip install .
- make all_tests
- cd docker
- docker build...
- setup GCP project
- deploy on cloud run

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Key Technologies used in this project

- Python
  - NLTK
  - scikitlearn
  - Pandas
- Google Cloud Platform (GCP)
  - Cloud Run
  - IAM
  -

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Project logic

### 1. Rating prediction

Proposed use case: find recipes that be highly rated if they would received reviews. It would help chefs to better their recipes and help promote 'new' content on the website.

Baseline model:

```
from sklearn.metrics import root_mean_squared_error, mean_squared_error

local_review_path = "data/RAW_interactions.csv"
df_reviews = data.load_reviews(local_review_path)
local_recipe_path = "data/RAW_recipes.csv"
df_recipes = data.load_recipes(local_recipe_path)
y_merged = data.get_y(df_recipes, df_reviews)

number_review_threshold = 9
review_filter = y_merged[y_merged['count_rating']> number_review_threshold]

mean_rating_overall = review_filter['mean_rating'].mean()
y_true = review_filter['mean_rating']
y_pred = np.repeat(mean_rating_overall, len(review_filter))

rmse = root_mean_squared_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)

rmse, mse
```


__Results :__ (0.43713, 0.19108)

#### Preprocessing

First issue we enconter is that the ratings highly skewed toward the maximum value. It seems that the Americans are either enthousiastic about reviewing a recipe or don't do it at all. It is significantly the same if we look at the average rate by recipe.

![alt text](readme_images/image-1.png)

Considering the objective, we need to have recipes that have enough reviews to be significant. We considered that having the normally distributed average rate by recipe is a good bases to have meanigful reviews we enough variation to have something to predict. Though we had to balance the cut to the dataset size with the significativity.

We selected a threshold of 10+ reviews by recipe.
We maintain 21399 recipes to train on with this following distribution. It's the smallest threshold that we still consider as a normally distributed distribution of average rating.

![alt text](readme_images/image.png)

Text preprocessing

- start with basic word preprocessing (removing uppercase, numerical characters, ponctuation, english stopwords, lemmatizing by word types)
- adding custom stopword list to be removed selected from highest recurring words in the corpus if we felt that the words do not differentiate one recipe from another
- tags are pre handled to the preprocessing to remove recurring tags that add no additional sense value

Numerical preprocessing

To handle excessive outliers while still maintaining some recipes exagerated features we mitigates the minutes and step by setting a minimum and a maximum
- minutes start at 5 and are capped at 130
- n_steps start at 1 and are capped at 40

We then applied a RobustScaler to the numerical beside the n_step and n_ingredients are they are already in a similar range to the rest. RobustSacler is ideal for our not always normal distribution with many outliers even if mitigated.

#### Model

### 2. Recipe recommendation

### 3. Recipe generation

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/Nanouna/recettes-et-sentiments.svg?style=for-the-badge
[contributors-url]: https://github.com/Nanouna/recettes-et-sentiments/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/Nanouna/recettes-et-sentiments.svg?style=for-the-badge
[forks-url]: https://github.com/Nanouna/recettes-et-sentiments/network/members
[stars-shield]: https://img.shields.io/github/stars/Nanouna/recettes-et-sentiments.svg?style=for-the-badge
[stars-url]: https://github.com/Nanouna/recettes-et-sentiments/stargazers
[issues-shield]: https://img.shields.io/github/issues/Nanouna/recettes-et-sentiments.svg?style=for-the-badge
[issues-url]: https://github.comNanouna/recettes-et-sentiments/issues
[license-shield]: https://img.shields.io/github/license/Nanouna/recettes-et-sentiments.svg?style=for-the-badge
[license-url]: https://github.com/Nanouna/recettes-et-sentiments/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/mansonthomas
