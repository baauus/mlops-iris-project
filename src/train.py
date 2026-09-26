from pathlib import Path

import joblib

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

MODEL_PATH = Path("models/model.joblib")

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

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(x_train, y_train)

    predictions = model.predict(x_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(f"Accuracy: {accuracy:.4f}")

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(model, MODEL_PATH)

    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()