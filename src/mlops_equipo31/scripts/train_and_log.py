import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import os, random, numpy as np
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
os.environ["PYTHONHASHSEED"] = "42"
random.seed(42)
np.random.seed(42)

FEATURES = [
    "temp", "hum", "wind", "gen_diffuse_flows", "diffuse_flows",
    "z2_power_cons", "z3_power_cons", "hour", "day_of_week", "month", "day"
]
TARGET = "z1_power_cons"

def main(csv_path: str, experiment_name: str):
    csv = Path(csv_path)
    assert csv.exists(), f"CSV no encontrado: {csv}"

    # tracking local en carpeta ./mlruns (portátil y sin auth)
    mlflow.set_tracking_uri(f"file:{Path.cwd() / 'mlruns'}")
    mlflow.set_experiment(experiment_name)

    df = pd.read_csv(csv)
    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run() as run:
        model = RandomForestRegressor(
            n_estimators=300, max_depth=None, random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        r2 = r2_score(y_test, preds)

        mlflow.log_params({
            "model_type": "RandomForestRegressor",
            "n_estimators": 300,
            "max_depth": None,
            "random_state": 42
        })
        mlflow.log_metrics({"mae": mae, "rmse": rmse, "r2": r2})

        # firma e input_example para que el serving sepa el esquema
        signature = infer_signature(X_train, model.predict(X_train[:5]))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_train.head(2)
        )

        print("\n==============================")
        print(f"Experiment: {experiment_name}")
        print(f"Run ID    : {run.info.run_id}")
        print("Artifact  : model/")
        print("==============================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Ruta al CSV de entrenamiento")
    parser.add_argument("--experiment", default="equipo31-remote",
                        help="Nombre del experimento en MLflow")
    args = parser.parse_args()
    main(args.csv, args.experiment)
