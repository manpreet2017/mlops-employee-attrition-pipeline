import yaml
import joblib

import mlflow
import mlflow.sklearn

from data_preprocessing import load_data, preprocess_data

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# Use local MLflow database
mlflow.set_tracking_uri("sqlite:///mlflow.db")

mlflow.set_experiment("Employee Attrition")


# Load config
with open("configs/config.yaml") as file:
    config = yaml.safe_load(file)



def train_model():


    # Load data
    df = load_data()


    # Preprocess
    X, y, preprocessor = preprocess_data(df)



    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["test_size"],
        random_state=config["model"]["random_state"],
        stratify=y
    )



    # Model
    model = RandomForestClassifier(
        n_estimators=config["model"]["n_estimators"],
        random_state=config["model"]["random_state"]
    )



    # Full pipeline
    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )



    # Train
    pipeline.fit(
        X_train,
        y_train
    )



    # Predict
    predictions = pipeline.predict(X_test)

    probabilities = pipeline.predict_proba(
        X_test
    )[:,1]



    # Metrics

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions
    )

    recall = recall_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )



    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)
    print("ROC-AUC:", roc_auc)



    # MLflow

    with mlflow.start_run(
        run_name=f"RandomForest_{config['model']['n_estimators']}"
    ):


        mlflow.log_param(
            "model",
            "RandomForest"
        )


        mlflow.log_param(
            "n_estimators",
            config["model"]["n_estimators"]
        )


        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        mlflow.log_metric(
            "precision",
            precision
        )

        mlflow.log_metric(
            "recall",
            recall
        )

        mlflow.log_metric(
            "f1",
            f1
        )

        mlflow.log_metric(
            "roc_auc",
            roc_auc
        )


        mlflow.sklearn.log_model(
            pipeline,
            "pipeline"
        )



    # Save full pipeline
    joblib.dump(
        pipeline,
        "pipeline.pkl"
    )


    print("Pipeline saved successfully")



if __name__ == "__main__":
    train_model()