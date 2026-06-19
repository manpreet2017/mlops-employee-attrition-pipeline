import pandas as pd
import os


def load_data():

    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    path = os.path.join(
        current_dir,
        "data",
        "employee_attrition.csv"
    )

    df = pd.read_csv(path)

    return df


def preprocess_data(df):

    df = df.dropna()

    y = df["Attrition"]

    X = df.drop("Attrition", axis=1)

    X = pd.get_dummies(
        X,
        drop_first=True
    )

    return X, y