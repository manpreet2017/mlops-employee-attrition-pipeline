import sys
sys.path.append(".")

import joblib
from src.data_preprocessing import load_data


def test_model_file_exists():
    model = joblib.load("pipeline.pkl")
    assert model is not None


def test_model_can_predict():
    model = joblib.load("pipeline.pkl")

    df = load_data()

    X = df.drop("Attrition", axis=1)

    predictions = model.predict(X)

    assert len(predictions) == len(X)