# MLOps Iris Project

Small MLOps project built to learn the basic lifecycle of a Machine Learning model.

The project trains an Iris classifier, tracks experiments with MLflow, registers model versions, exposes the selected model through FastAPI, and runs the services with Docker Compose.

## Stack

- Python
- scikit-learn
- MLflow
- FastAPI
- pytest
- Docker
- Docker Compose
- GitHub Actions

## Architecture

```text
Training
   ↓
MLflow Tracking
   ↓
Model Registry
   ↓
IrisClassifier@champion
   ↓
FastAPI
   ↓
/predict
```

## Run

Start MLflow:

```bash
docker compose up -d mlflow
```

Train and register a model:

```bash
python src/train.py
```

Select the model version to use:

```bash
python scripts/set_champion.py <version>
```

Start the full stack:

```bash
docker compose up -d --build
```

## Services

MLflow:

```text
http://localhost:5000
```

FastAPI Swagger:

```text
http://localhost:8000/docs
```

## API endpoints

```text
GET  /health
GET  /model
POST /predict
```

Example prediction input:

```json
{
	"sepal_length": 5.1,
	"sepal_width": 3.5,
	"petal_length": 1.4,
	"petal_width": 0.2
}
```

## Tests

```bash
python -m pytest
```

## Goal

The goal of this project is to practice:

```text
Train → Track → Register → Version → Serve → Test → Containerize
```
