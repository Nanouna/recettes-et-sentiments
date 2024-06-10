import os
import logging
import joblib
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

def load_model(model_name:str, prefix:str="/tmp/data/")-> Pipeline:

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

    model_path = f"{prefix}{model_name}.pkl"

    logger.info(f"Saving model to {model_path}")
    joblib.dump(model, f"{model_path}")
    logger.info(f"Saving model to {model_path} - DONE")
