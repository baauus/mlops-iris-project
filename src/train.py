import os
import joblib
import mlflow
import mlflow.sklearn

from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

MODEL_PATH = Path("models/model.joblib")

mlflow.set_tracking_uri(
    os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://localhost:5000"
    )
)

mlflow.set_experiment("iris-classification-v2")

def train():
    iris = load_iris()

    x = iris.data
    y = iris.target

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42
    )

    n_estimators=10
    random_state=42
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state
    )

    with mlflow.start_run():
        
        model.fit(x_train, y_train)

        predictions = model.predict(x_test)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_metric("accuracy", accuracy)

        mlflow.sklearn.log_model(
            model,
            name="model",
            registered_model_name="IrisClassifier",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

        MODEL_PATH.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        joblib.dump(model, MODEL_PATH)

        print(f"Training accuracy: {accuracy:.4f}")
        print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()