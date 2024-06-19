import os

import mlflow
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor


# Connect to tracking server
MLFLOW_SERVER_URL = os.environ.get("MLFLOW_SERVER_URL")
assert MLFLOW_SERVER_URL is not None, "MLFLOW_SERVER_URL environment variable must be provided"

mlflow.set_tracking_uri(MLFLOW_SERVER_URL)
mlflow.set_experiment("/autologging-quickstart")

mlflow.autolog()

db = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
# MLflow triggers logging automatically upon model fitting
rf.fit(X_train, y_train)

