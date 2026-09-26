from pathlib import Path

import joblib

from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = Path("models/model.joblib")

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Iris ML API",
    version="1.0"
)

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

species = [
    "setosa",
    "versicolor",
    "virginica"
]

@app.get("/health")
def health():
    return {
        "status": "ok"
    }

@app.post("/predict")
def predict(data: IrisInput):
    
    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    prediction = model.predict(features)[0]

    probabilities = model.predict_proba(features)[0]

    return {
        "prediction": species[prediction],
        "probability": float(max(probabilities))
    }