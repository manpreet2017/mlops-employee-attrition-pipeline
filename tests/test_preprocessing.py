import pandas as pd

from src.data_preprocessing import preprocess_data


def test_preprocess_returns_features_and_target():

    df = pd.DataFrame({
        "Attrition": ["Yes", "No"],
        "Gender": ["Male", "Female"],
        "Age": [30,40]
    })

    X, y, preprocessor = preprocess_data(df)

    assert X is not None
    assert y is not None
    assert preprocessor is not None



def test_target_removed_from_features():

    df = pd.DataFrame({
        "Attrition": ["Yes","No"],
        "Age":[25,35]
    })

    X, y, preprocessor = preprocess_data(df)

    assert "Attrition" not in X.columns



def test_categorical_columns_encoded():

    df = pd.DataFrame({
        "Attrition":["Yes","No"],
        "Gender":["Male","Female"]
    })

    X,y,preprocessor = preprocess_data(df)

    transformed = preprocessor.fit_transform(X)

    assert transformed.shape[0] == 2