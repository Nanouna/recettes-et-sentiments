import os
import logging
import joblib
from sklearn.pipeline import Pipeline
import pickle
from recettes_et_sentiments.api_model.fast_vectorizer import FastVectorizer, CustomUnpickler

import onnx
import skl2onnx
from skl2onnx.common.data_types import StringTensorType
from onnxruntime import InferenceSession
import onnxruntime as ort


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





def save_fast_model(model, input_shape, model_name:str, prefix:str="/tmp/data/"):
    initial_type = [('merged_text', StringTensorType([None, 1]))]
    onnx_model = skl2onnx.convert_sklearn(model, initial_types=initial_type)

    onnx_model_path = f"{prefix}{model_name}.onnx"

    with open(onnx_model_path, 'wb') as f:
        f.write(onnx_model.SerializeToString())


def load_fast_model(model_name:str, prefix:str="/tmp/data/")-> Pipeline:

    model_path = f"{prefix}{model_name}.onnx"

    if os.path.exists(model_path):
        logger.info(f"Loading model from {model_path}")

        session = ort.InferenceSession(model_path)

        logger.info(f"Loading model from {model_path} - DONE")
        return session
    else:
        logger.info(f"model not found at {model_path}")
        return None
