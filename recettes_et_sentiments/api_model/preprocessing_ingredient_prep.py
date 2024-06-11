# we then select the top ingredients that appears more than 115 time (0.5% of recipes)
# we end up with ~1600 ingredients that we regroup in ~800 unique ingredients.
# From the TFIDF clustering, we did a manual check to correct the mapping before saving it as a dictionnary in the parameters

import re
import unidecode
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk
import numpy as np

from recettes_et_sentiments.api_model import rs_data

# Télécharger les ressources NLTK nécessaires
nltk.download('wordnet')
nltk.download('omw-1.4')

# Exemple de liste d'ingrédients
rec_path = 'data/RAW_recipes.csv'
recipes = rs_data.load_recipes(rec_path)
ingredients = recipes['ingredients']

list_ingredients = []
for row in ingredients:
    for ingredient in row:
        list_ingredients.append(ingredient)

# Initialiser le lemmatizer
lemmatizer = WordNetLemmatizer()

# Étape 1 : Nettoyage des données
def nettoyer_ingredient(ingredient):
    # Retirer les caractères spéciaux
    ingredient = re.sub(r'[^a-zA-Z\s]', '', ingredient)
    # Convertir en minuscules
    ingredient = ingredient.lower()
    # Supprimer les accents
    ingredient = unidecode.unidecode(ingredient)
    # Lemmatisation des mots
    ingredient = ' '.join([lemmatizer.lemmatize(word) for word in ingredient.split()])
    return ingredient.strip()

ingredients_nettoyes = [nettoyer_ingredient(ing) for ing in list_ingredients]

# Étape 2 : Normalisation des noms (supprimer les doublons exacts)
ingredients_uniques = list(set(ingredients_nettoyes))

# Étape 3 : Clustering pour regrouper les ingrédients similaires
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(ingredients_uniques)

# Utilisation de KMeans pour le clustering
n_clusters = 1500  # Ajustez ce nombre pour avoir environ 1000 ingrédients uniques
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X)

# Récupérer les ingrédients regroupés
clusters = kmeans.labels_

# Créer un dictionnaire pour les clusters
clusters_dict = {}
for idx, label in enumerate(clusters):
    clusters_dict.setdefault(label, []).append(ingredients_uniques[idx])

# Sélectionner un représentant pour chaque cluster
representants = {}
for label, ing in clusters_dict.items():
    # Calculer le centroïde du cluster
    cluster_indices = [idx for idx, lbl in enumerate(clusters) if lbl == label]
    cluster_center = kmeans.cluster_centers_[label]
    closest_index = cluster_indices[np.argmin(np.linalg.norm(X[cluster_indices] - cluster_center, axis=1))]
    representants[label] = ingredients_uniques[closest_index]

ingredient_to_representant = {}
for ingredient in ingredients_uniques:
    for label, representant in representants.items():
        if ingredient in clusters_dict[label]:
            ingredient_to_representant[ingredient] = [representant, ingredients_nettoyes.count(ingredient)]

df = pd.DataFrame.from_dict([ingredient_to_representant]).T
df.to_csv("ingredient_tfidf.csv")
