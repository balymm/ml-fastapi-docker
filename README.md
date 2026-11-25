# ML FastAPI Docker

Классификация ирисов (Iris, RandomForest), деплой через FastAPI и Docker,
фронтенд на Streamlit, трекинг экспериментов и реестр моделей в MLflow.

## Структура

- `train.py` - обучение модели, логирование в MLflow, сохранение `model.joblib`
- `main.py` - FastAPI (`GET /`, `POST /predict`)
- `app.py` - Streamlit-фронтенд
- `Dockerfile`, `requirements.txt` - контейнеризация API

## Запуск локально

```bash
python -m venv venv
venv\Scripts\activate
pip install fastapi uvicorn streamlit mlflow scikit-learn joblib requests
python train.py
uvicorn main:app --reload
```

API: http://localhost:8000, документация: http://localhost:8000/docs

## Streamlit

```bash
streamlit run app.py
```

## MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

Эксперимент `iris-classification`: логируются параметры, метрики
(accuracy, f1_score) и артефакт `model.joblib`. Модель зарегистрирована
как `IrisClassifier` (version 1).

## Docker

```bash
docker build -t ml-api .
docker run -p 8000:8000 ml-api
```

## Пример запроса

POST /predict

```json
{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
```

Ответ: `{"prediction": 0, "class_name": "setosa"}`

## Скриншоты

Папка `screenshots/`.