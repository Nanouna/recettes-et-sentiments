import os
import logging
import joblib
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

pipeline_FAST_pkl_path = 'data/preprocessor_pipeline.pkl'

def load_FAST_model(prefix:str="/tmp/")-> Pipeline:

    if os.path.exists(f"{prefix}{pipeline_FAST_pkl_path}"):
        logger.info(f"Loading pipeline FAST from {prefix}{pipeline_FAST_pkl_path}")
        return joblib.load(f"{prefix}{pipeline_FAST_pkl_path}")
    else:
        logger.info(f"pipeline FAST not found at {prefix}{pipeline_FAST_pkl_path}")
        return None


def save_FAST_model(preprocessor_pipeline, prefix:str="/tmp/"):
    logger.info(f"Saving pipeline FAST to {prefix}{pipeline_FAST_pkl_path}")
    joblib.dump(preprocessor_pipeline, f"{prefix}{pipeline_FAST_pkl_path}")
    logger.info(f"Saving pipeline FAST to {prefix}{pipeline_FAST_pkl_path} - DONE")
