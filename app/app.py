from fastapi import FastAPI
import joblib
import os
import pandas as pd

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

model = joblib.load(MODEL_PATH)


@app.get("/")
def home():
    return {"message": "Employee Attrition AI Assistant Running"}


@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    df = pd.get_dummies(df)

    model_features = model.feature_names_in_

    df = df.reindex(columns=model_features, fill_value=0)

    prediction = model.predict(df)

    return {
    "attrition_prediction": str(prediction[0])
}