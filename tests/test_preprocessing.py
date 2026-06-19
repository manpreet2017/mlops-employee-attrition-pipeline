import sys
sys.path.append(".")
import pandas as pd
from src.data_preprocessing import preprocess_data


def test_preprocess_returns_features_and_target():
    df = pd.DataFrame({
        "Attrition": ["Yes", "No"],
        "Gender": ["Male", "Female"],
        "Age": [30, 40]
    })

    X, y = preprocess_data(df)

    assert len(X) == 2
    assert len(y) == 2


def test_target_removed_from_features():
    df = pd.DataFrame({
        "Attrition": ["Yes", "No"],
        "Age": [25, 35]
    })

    X, y = preprocess_data(df)

    assert "Attrition" not in X.columns


def test_categorical_columns_encoded():
    df = pd.DataFrame({
        "Attrition": ["Yes", "No"],
        "Gender": ["Male", "Female"]
    })

    X, y = preprocess_data(df)

    assert len(X.columns) > 0