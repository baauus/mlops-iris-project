import os
import sys
import mlflow
from mlflow import MlflowClient

tracking_uri = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000"
)

mlflow.set_tracking_uri(tracking_uri)

MODEL_NAME = "IrisClassifier"
version = sys.argv[1]

client = MlflowClient(tracking_uri=tracking_uri)

client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="champion",
    version=version
)

print(f"Set model '{MODEL_NAME}' version '{version}' as champion.")