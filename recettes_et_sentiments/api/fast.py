import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from taxifare.ml_logic import registry
from taxifare.ml_logic import preprocessor

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
model = registry.load_model()
assert model is not None
app.state.model = model





@app.get("/")
def root():
    return {
        'greeting': 'Hello'
    }
