from nltk.corpus import stopwords
from sklearn import set_config
import nltk

import logging
logging.basicConfig(level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)



# set output as pandas on sklearn
set_config(transform_output="pandas")

# download various dataset
# if it's already downloaded, it's instantaneously retrieve from a local cache
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')


STOP_WORDS = set(stopwords.words('english'))

RECIPE_STOPWORDS = ['minute', 'le',
                    'ingredient','preparation','dish','add',
                    'course','top','hour','combine',
                    'cup','pour','recipe','serving',
                    'taste','together','one','set',
                    'remaining','occasion','something','inch',
                    'let','time','half','degree',
                    "f",'cuisine','bring','use',
                    'food','cooking','put','aside',
                    'lightly','equipment','piece','spoon',
                    'using','cooked','tablespoon','turn',
                    '°f', '°c', '½re','½s'
                     ]

TAGS_STOPWORDS = ['60-minutes-or-less','30-minutes-or-less',
                  '15-minutes-or-less','time-to-make',
                  'preparation', 'course', 'main-ingredient',
                  'cuisine', '4-hours-or-less', '3-steps-or-less',
                  '5-ingredients-or-less','healthy-2','',
                  ]

RECIPE_COLUMNS_FOR_TEXT_PREPROC = [
        'name',
        'description',
        'merged_steps',
        'merged_ingredients'
    ]
RECIPE_COLUMNS_FOR_NUMERIC_PREPROC = [
        'minutes',
        'calories',
        'total_fat_pdv',
        'sugar_pdv',
        'sodium_pdv',
        'protein_pdv',
        'saturated_fat_pdv',
        'carbohydrates_pdv']


MINUTES_CLIP_LOWER=5
MINUTES_CLIP_UPPER=130

N_STEPS_CLIP_LOWER=0
N_STEPS_CLIP_UPPER=40
