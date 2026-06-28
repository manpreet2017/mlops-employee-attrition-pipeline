import pandas as pd
import os

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


def load_data():

    current_dir = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )

    path = os.path.join(
        current_dir,
        "data",
        "employee_attrition.csv"
    )

    return pd.read_csv(path)



def preprocess_data(df):

    target = "Attrition"

    X = df.drop(target, axis=1)

    y = df[target]


    # Convert target
    y = y.map(
        {
            "Yes": 1,
            "No": 0
        }
    )


    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns


    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns



    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            )
        ]
    )


    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )



    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numerical_pipeline,
                numerical_features
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_features
            )
        ]
    )


    return X, y, preprocessor