import sys
sys.path.append(".")
import joblib
import pandas as pd
from src.data_preprocessing import load_data, preprocess_data


def test_model_file_exists():
    model = joblib.load("src/model.pkl")
    assert model is not None


def test_model_can_predict():
    model = joblib.load("src/model.pkl")

    df = load_data()
    X, y = preprocess_data(df)

    predictions = model.predict(X.head(5))

    assert len(predictions) == 5