import yaml
import joblib

import mlflow
import mlflow.sklearn

from data_preprocessing import load_data, preprocess_data

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# MLflow setup
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Employee Attrition")


# Load config
with open("configs/config.yaml") as file:
    config = yaml.safe_load(file)

print("Loaded config:", config)


def train_model():

    # Load data
    df = load_data()

    # Preprocess data
    X, y = preprocess_data(df)


    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["test_size"],
        random_state=config["model"]["random_state"]
    )


    # Create model
    model = RandomForestClassifier(
        n_estimators=config["model"]["n_estimators"],
        random_state=config["model"]["random_state"]
    )


    # Train model
    model.fit(X_train, y_train)


    # Prediction
    predictions = model.predict(X_test)


    # Accuracy
    acc = accuracy_score(y_test, predictions)


    print(f"Model Accuracy: {acc}")


    # MLflow tracking
    with mlflow.start_run(
        run_name=f"RandomForest_{config['model']['n_estimators']}"
    ):

        print("MLFLOW RUN STARTED")


        # Parameters
        mlflow.log_param(
            "model_type",
            config["model"]["type"]
        )

        mlflow.log_param(
            "n_estimators",
            config["model"]["n_estimators"]
        )


        # Metrics
        mlflow.log_metric(
            "accuracy",
            acc
        )


        # Save model in MLflow
        mlflow.sklearn.log_model(
            model,
            "model"
        )


    # Save local model
    joblib.dump(
        model,
        "model.pkl"
    )

    print("Model saved successfully")


if __name__ == "__main__":
    train_model()