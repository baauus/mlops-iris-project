from pathlib import Path

import mlflow.sklearn
from mlflow import MlflowClient

client = MlflowClient()

from fastapi import FastAPI
from pydantic import BaseModel

model = mlflow.sklearn.load_model(
    "models:/IrisClassifier@champion"
)

model_version = client.get_model_version_by_alias(
    "IrisClassifier",
    "champion"
)

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

@app.get("/model")
def get_model():
    return {
        "model_name": model_version.name,
        "model_version": model_version.version
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