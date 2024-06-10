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
                  '5-ingredients-or-less','','baja',
                  'Throw the ultimate fiesta with this sopaipillas recipe from Food.com.',
                  'less_thansql:name_topics_of_recipegreater_than','main ingredient'
                  ]

TAGS_REPLACEMENTS = {'eggs breakfast':'breakfast eggs',
                    'for large groups holiday event':'for large groups',
                    'healthy 2':'healthy',
                    'heirloom historical recipes':'heirloom historical',
                    'irish st patricks day':'st patricks day',
                    'lasagne':'lasagna',
                    'pasta rice and grains elbow macaroni':'pasta rice and grains',
                    'pork loins':'pork loin',
                    'pork loins roast':'pork loin',
                    'rosh hashana':'rosh hashanah',
                    'simply potatoes2':'simply potatoes',
                    'steaks':'steak',
                    'super bowl':'superbowl',
                    'for-large-groups':'test'
                    }

TAGS_GROUPS = {
    'Cuisine ':['african',
        'american','amish mennonite','angolan','argentine',
        'asian','australian','austrian','beijing','belgian'
        'brazilian','british columbian','cajun','californian',
        'cambodian','canadian','cantonese','caribbean',
        'central american','chilean','chinese','colombian',
        'congolese','costa rican','creole','cuban',
        'czech','danish','dutch','ecuadorean',
        'egyptian','english','ethiopian','european',
        'filipino','finnish','french','georgian',
        'german','greek','guatemalan','hawaiian',
        'honduran','hunan','hungarian','icelandic',
        'indian','indonesian','iranian persian','iraqi',
        'irish','italian','japanese','jewish ashkenazi',
        'jewish sephardi','korean','laotian','lebanese',
        'libyan','malaysian','mexican','micro melanesia',
        'middle eastern','middle eastern main dish',
        'midwestern','mongolian','moroccan','namibian',
        'native american','nepalese','new zealand','nigerian',
        'north american','northeastern united states',
        'norwegian','oaxacan','ontario','pacific northwest',
        'pakistani','palestinian','pennsylvania dutch','peruvian',
        'polish','polynesian','portuguese','puerto rican',
        'quebec','russian','saudi arabian','scandinavian',
        'scottish','somalian','south african','south american',
        'south west pacific','southern united states',
        'southwestern united states','spanish','sudanese','swedish',
        'swiss','szechuan','tex mex','thai',
        'turkish','venezuelan','vietnamese','welsh'],
    'Diet type':['dairy free',
        'diabetic','dietary','egg free','eggs dairy',
        'free of something','gluten free','healthy','high calcium',
        'high fiber','high in something',
        'high in something diabetic friendly','high protein',
        'kosher','lactose','low calorie','low carb',
        'low cholesterol','low fat','low in something','low protein',
        'low saturated fat','low sodium','no shell fish','non alcoholic',
        'nut free','vegan','vegetarian','very low carbs'
        ],
    'Type':['appetizers','baking','beverages',
        'breakfast casseroles','breakfast eggs',
        'breakfast potatoes','brewing','broil','brownies',
        'burgers','cake fillings and frostings',
        'cakes','canning','casseroles','celebrity',
        'cheesecake','chutneys','clear soups','cobblers and crisps',
        'coffee cakes','college','comfort food','cookies and brownies',
        'cooking mixes','copycat','course','crock pot main dish',
        'crock pot slow cooker','crusts pastry dough 2',
        'cupcakes','curries','deep fry','desserts',
        'desserts easy','desserts fruit','dips','dips lunch snacks',
        'dips summer','drop cookies','fall','fillings and frostings chocolate',
        'finger food','flat shapes','from scratch','frozen desserts',
        'garnishes','granola and porridge','grilling','ham and bean soup',
        'hand formed cookies','heirloom historical','herb and spice mixes','ice cream',
        'inexpensive','infant baby friendly','jams and preserves','jellies',
        'kid friendly','lamb sheep main dish','lasagna','leftovers',
        'lunch','macaroni and cheese','main dish','main dish beef',
        'main dish chicken','main dish pasta','main dish pork','main dish seafood',
        'main ingredient','marinades and rubs','marinara sauce','mashed potatoes',
        'meatballs','meatloaf','muffins','mushroom soup',
        'no cook','novelty','number of servings','oamc freezer make ahead',
        'omelets and frittatas','one dish meal','pancakes and waffles','pasta rice and grains',
        'pasta salad','pies','pies and tarts','pitted fruit',
        'pizza','pot pie','pot roast','potluck',
        'preparation','prepared potatoes','presentation','puddings and mousses',
        'pumpkin bread','punch','quiche','quick breads',
        'ragu recipe contest','roast','roast beef comfort food','roast beef main dish',
        'rolled cookies','rolls biscuits','salad dressings','salsas',
        'saltwater fish','sandwiches','sauces','savory',
        'savory pies','savory sauces','scones','seafood',
        'seasonal','served cold','served hot','shakes',
        'shrimp main dish','side dishes','side dishes beans','simply potatoes',
        'smoothies','snacks','snacks kid friendly',
        'snacks sweet','sole and flounder','soul','soups stews',
        'spaghetti sauce','spicy','spreads','spring',
        'steam','stews','stews poultry','stir fry',
        'stocks','stuffings dressings','sugar cookies','summer',
        'sweet','sweet sauces','tarts','technique',
        'toddler friendly','tropical fruit','turkey burgers','vegetables',
        'veggie burgers','water bath','wings','winter'],
    'Ingredients':['a1 sauce',
        'apples','artichoke',
        'asparagus','avocado',
        'bacon','baked beans',
        'bananas','bar cookies',
        'bass','bean soup',
        'beans','beans side dishes',
        'bear','beef',
        'beef barley soup','beef crock pot',
        'beef kidney','beef liver',
        'beef organ meats','beef ribs',
        'beef sauces','beef sausage',
        'berries','biscotti',
        'bisques cream soups','black bean soup',
        'black beans','blueberries',
        'bok choys','bread pudding',
        'breads','broccoli',
        'brown rice','cabbage',
        'candy','carrots',
        'catfish','cauliflower',
        'chard','cheese',
        'cherries','chick peas garbanzos',
        'chicken','chicken breasts',
        'chicken crock pot','chicken livers',
        'chicken stew','chicken stews',
        'chicken thighs legs','chili',
        'chocolate','chocolate chip cookies',
        'chowders','citrus',
        'clams','coconut',
        'cod','collard greens',
        'condiments etc','corn',
        'crab','cranberry sauce',
        'crawfish','deer',
        'duck','duck breasts',
        'eggplant','eggs',
        'elbow macaroni','elk',
        'fish','freshwater fish',
        'fruit','fudge',
        'gelatin','goose',
        'grains','grapes',
        'green yellow beans','greens',
        'ground beef','gumbo',
        'halibut','ham',
        'hidden valley ranch','kiwifruit',
        'lamb sheep','lemon',
        'lentils','lettuces',
        'lime','lobster',
        'long grain rice','mahi mahi',
        'mango','manicotti',
        'meat','medium grain rice',
        'melons','moose',
        'mushrooms','mussels',
        'nuts','oatmeal',
        'octopus','onions',
        'orange roughy','oranges',
        'oysters','papaya',
        'pasta',
        'pasta elbow macaroni',
        'pasta shells',
        'peaches',
        'peanut butter',
        'pears',
        'penne',
        'peppers',
        'perch',
        'pheasant',
        'pickeral',
        'pineapple',
        'plums',
        'pork',
        'pork chops',
        'pork crock pot',
        'pork loin',
        'pork ribs',
        'pork sausage',
        'potatoes',
        'poultry',
        'pumpkin',
        'quail',
        'rabbit',
        'raspberries',
        'ravioli tortellini',
        'reynolds wrap',
        'rice',
        'roast beef',
        'salads',
        'salmon',
        'scallops',
        'shellfish',
        'short grain rice',
        'shrimp',
        'sourdough',
        'soy tofu',
        'spaghetti',
        'spinach',
        'squash',
        'squid',
        'steak',
        'strawberries',
        'tempeh',
        'tilapia',
        'tomatoes',
        'trout',
        'tuna',
        'turkey',
        'turkey breasts',
        'veal',
        'white rice',
        'whitefish',
        'whole chicken',
        'whole duck',
        'whole turkey',
        'wild game',
        'yams sweet potatoes',
        'yeast',
        'zucchini']
}

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

W2V_VECTOR_SIZE = 100
W2V_WINDOW = 5
W2V_MIN_COUNT = 1

KNN_N_NEIGHBORS = 3 # should not be inferior to 2. 1 would return itself
