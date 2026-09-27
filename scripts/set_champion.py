import sys
from mlflow import MlflowClient

MODEL_NAME = "IrisClassifier"
version = sys.argv[1]

client = MlflowClient()

client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="champion",
    version=version
)

print(f"Set model '{MODEL_NAME}' version '{version}' as champion.")