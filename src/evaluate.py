import joblib
import yaml

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from data_preprocessing import load_data, preprocess_data


with open("../configs/config.yaml") as file:
    config = yaml.safe_load(file)


def evaluate():

    df = load_data()

    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["test_size"],
        random_state=config["model"]["random_state"]
    )

    model = joblib.load("model.pkl")

    predictions = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Precision:", precision_score(
        y_test,
        predictions,
        pos_label="Yes"
    ))

    print("Recall:", recall_score(
        y_test,
        predictions,
        pos_label="Yes"
    ))

    print("F1:", f1_score(
        y_test,
        predictions,
        pos_label="Yes"
    ))

if __name__ == "__main__":
    evaluate()