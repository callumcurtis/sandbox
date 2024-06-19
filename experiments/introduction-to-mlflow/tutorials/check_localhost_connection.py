import os

import mlflow


MLFLOW_SERVER_URL = os.environ.get("MLFLOW_SERVER_URL")
assert MLFLOW_SERVER_URL is not None, "MLFLOW_SERVER_URL environment variable must be provided"

mlflow.set_tracking_uri(MLFLOW_SERVER_URL)
mlflow.set_experiment("/check-localhost-connection")

with mlflow.start_run():
    mlflow.log_metric("foo", 1)
    mlflow.log_metric("bar", 2)

