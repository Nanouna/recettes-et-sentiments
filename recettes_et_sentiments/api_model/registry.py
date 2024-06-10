import os
import logging
import joblib
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

def load_model(model_name:str, prefix:str="/tmp/data")-> Pipeline:

    model_path = f"{prefix}{model_name}.pkl"

    if os.path.exists(model_path):
        logger.info(f"Loading pipeline FAST from {model_path}")
        return joblib.load(f"{model_path}")
    else:
        logger.info(f"pipeline FAST not found at {model_path}")
        return None

def save_model(preprocessor_pipeline, model_name:str, prefix:str="/tmp/data"):

    model_path = f"{prefix}{model_name}.pkl"

    logger.info(f"Saving pipeline FAST to {model_path}")
    joblib.dump(preprocessor_pipeline, f"{model_path}")
    logger.info(f"Saving pipeline FAST to {model_path} - DONE")
