from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="ML API")
model = joblib.load("model.joblib")
CLASSES = ["setosa", "versicolor", "virginica"]


class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/")
def root():
    return {"message": "ML API is running"}


@app.post("/predict")
def predict(d: IrisInput):
    features = [[d.sepal_length, d.sepal_width, d.petal_length, d.petal_width]]
    p = int(model.predict(features)[0])
    return {"prediction": p, "class_name": CLASSES[p]}