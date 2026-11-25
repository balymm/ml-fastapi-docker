import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("Iris Classifier")
sl = st.number_input("Sepal length", value=5.1)
sw = st.number_input("Sepal width", value=3.5)
pl = st.number_input("Petal length", value=1.4)
pw = st.number_input("Petal width", value=0.2)

if st.button("Predict"):
    r = requests.post(f"{API_URL}/predict", json={
        "sepal_length": sl, "sepal_width": sw,
        "petal_length": pl, "petal_width": pw})
    st.success(f"Result: {r.json()['class_name']}")