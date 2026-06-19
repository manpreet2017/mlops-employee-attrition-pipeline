import mlflow


mlflow.set_tracking_uri("sqlite:///mlflow.db")


experiment = mlflow.get_experiment_by_name(
    "Employee Attrition"
)


runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.accuracy DESC"]
)


print("Experiment Results:")
print(runs[[
    "run_id",
    "metrics.accuracy",
    "params.model_type",
    "params.n_estimators"
]])


best_run = runs.iloc[0]


print("\nBest Run:")
print("Run ID:", best_run["run_id"])
print("Accuracy:", best_run["metrics.accuracy"])