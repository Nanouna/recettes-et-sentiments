import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import pytz

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# app.state.model = registry.load_model()

# http://127.0.0.1:8000/predict?

@app.get("/predict_rating")
def predict(
        recipe_name: str,
        # pickup_longitude: float,
        # pickup_latitude: float,
        # dropoff_longitude: float,
        # dropoff_latitude: float,
        # passenger_count: int
    ):      # 1
    """
    Make a single course prediction.
    Assumes `pickup_datetime` is provided as a string by the user in "%Y-%m-%d %H:%M:%S" format
    Assumes `pickup_datetime` implicitly refers to the "US/Eastern" timezone (as any user in New York City would naturally write)
    """
    X = pd.DataFrame(data = {'recipe_name': recipe_name,
    #                   'pickup_longitude': float(pickup_longitude),
    #                   'pickup_latitude': float(pickup_latitude),
    #                   'dropoff_longitude': float(dropoff_longitude),
    #                   'dropoff_latitude': float(dropoff_latitude),
    #                   'passenger_count': int(passenger_count),
                      }, index=[0])
    X_processed = preprocessor.preprocess_features(X)
    y_pred = app.state.model.predict(X_processed)[0][0]
    return {'Average rating': float(y_pred)}


@app.get("/")
def root():
    return {'greeting':'Hello! We\'re here to make gourmet stuff'}
