# ML FastAPI Docker

Practical Task 6 + SIS-3.

Iris classifier (RandomForest) served with FastAPI and Docker. Streamlit is used as a simple frontend, MLflow for experiment tracking and model registry.

## Files

- `train.py` - trains the model, saves `model.joblib`, logs to MLflow
- `main.py` - FastAPI app
- `app.py` - Streamlit frontend
- `Dockerfile`, `requirements.txt` - for the API container
- `screenshots/` - screenshots of all steps

## How to run

Install and train:

```
python -m venv venv
venv\Scripts\activate
pip install fastapi uvicorn streamlit mlflow scikit-learn joblib requests
python train.py
```

API:

```
uvicorn main:app --reload
```

Open http://localhost:8000 and http://localhost:8000/docs

Streamlit (the API must be running):

```
streamlit run app.py
```

MLflow UI:

```
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

Docker:

```
docker build -t ml-api .
docker run -p 8000:8000 ml-api
```

## API

- `GET /` returns `{"message": "ML API is running"}`
- `POST /predict` takes 4 features and returns the class

Example:

```
{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
```

Response: `{"prediction": 0, "class_name": "setosa"}`

## MLflow

Experiment: `iris-classification`. Logged params (n_estimators, max_depth, random_state), metrics (accuracy, f1_score) and the model file. Model is registered as `IrisClassifier`, version 1.
