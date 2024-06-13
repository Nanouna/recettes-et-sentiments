import os
import logging
import joblib
from sklearn.pipeline import Pipeline
import pickle

logger = logging.getLogger(__name__)

def load_model(model_name:str, prefix:str="/tmp/data/")-> Pipeline:
    """
    load the model from "{prefix}{model_name}.pkl" (prefix defaults to /tmp/data)
    return the model or None if the model do not exists
    """
    model_path = f"{prefix}{model_name}.pkl"

    if os.path.exists(model_path):
        logger.info(f"Loading model from {model_path}")
        model = joblib.load(f"{model_path}")
        logger.info(f"Loading model from {model_path} - DONE")
        return model
    else:
        logger.info(f"model not found at {model_path}")
        return None

def save_model(model, model_name:str, prefix:str="/tmp/data/"):
    """
    save the model to "{prefix}{model_name}.pkl" (prefix defaults to /tmp/data)
    """
    model_path = f"{prefix}{model_name}.pkl"

    logger.info(f"Saving model to {model_path}")
    joblib.dump(model, f"{model_path}")
    logger.info(f"Saving model to {model_path} - DONE")




class CustomUnpickler(pickle.Unpickler):
    """
    Custom loader for FastVecotrModel to avoid this error when loading the model from fast.py (from FAST_model_variant.py it works as expected)
    AttributeError: Can't get attribute 'FastVectorizer' on <module '__main__' from '/home/tom/.pyenv/versions/recettes-et-sentiments/bin/uvicorn'>

    unfortunalety it's not enough, although the stored model is correct and working loaded from FAST_model_variant,
    loading the model from fast.py with CustomUnpickler prevent the above error from occuring but return a numpy ndarray instead of a pipeline

    """
    def find_class(self, module, name):
        if name == 'FastVectorizer':
            from recettes_et_sentiments.api_model.FAST_model_variant import FastVectorizer
            return FastVectorizer
        return super().find_class(module, name)



def load_fast_model(model_name:str, prefix:str="/tmp/data/")-> Pipeline:
    """
    load the model from "{prefix}{model_name}.pkl" (prefix defaults to /tmp/data)
    return the model or None if the model do not exists
    """
    model_path = f"{prefix}{model_name}.pkl"

    if os.path.exists(model_path):
        logger.info(f"Loading model from {model_path}")
        with open(model_path, 'rb') as f:
            model = CustomUnpickler(f).load()
        logger.info(f"Loading model from {model_path} - DONE")
        return model
    else:
        logger.info(f"model not found at {model_path}")
        return None
