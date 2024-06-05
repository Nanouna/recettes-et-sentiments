from nltk.corpus import stopwords

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

RECIPE_COLUMNS_FOR_TEXT_PREPROC = [
        'name',
        'description',
        'merged_steps',
        'tags',
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
