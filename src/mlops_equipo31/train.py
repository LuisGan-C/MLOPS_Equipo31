import os
from pathlib import Path
import yaml
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

CFG_DEFAULT = "configs/train.yaml"

def load_config(cfg_path: str):
    if not Path(cfg_path).exists():
        raise FileNotFoundError(f"No existe el archivo de config: {cfg_path}")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

def main(cfg_path: str = CFG_DEFAULT):
    cfg = load_config(cfg_path)

    dataset_path = cfg["dataset_path"]
    target = cfg["target"]

    if not Path(dataset_path).exists():
        raise FileNotFoundError(
            f"No encuentro el dataset en {dataset_path}. "
            "Ajusta configs/train.yaml o corre `dvc pull`."
        )

    # Carga de datos
    if dataset_path.endswith(".parquet"):
        df = pd.read_parquet(dataset_path)
    else:
        df = pd.read_csv(dataset_path)

    if 'Zone 1 Power Consumption' in df.columns and target == 'Target':
        df = df.rename(columns={'Zone 1 Power Consumption': 'Target'})

    if target not in df.columns:
        raise ValueError(f"La columna objetivo {target} no está en el dataset. Columnas: {list(df.columns)}")

    # Preprocesamiento de columnas no numéricas (manejo de fechas)
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                # Intentar convertir a datetime
                df[col] = pd.to_datetime(df[col])
                # Convertir a timestamp numérico (Unix timestamp)
                df[col] = df[col].apply(lambda x: x.timestamp() if pd.notna(x) else np.nan)
            except (ValueError, TypeError):
                # Si no se puede convertir a datetime, podría ser una columna categórica
                # Por ahora, la descartamos ya que el modelo espera solo numéricos.
                # En un escenario real, se debería aplicar One-Hot Encoding o similar.
                print(f"Advertencia: la columna '{col}' no se pudo convertir a numérico y será descartada.")
                df = df.drop(columns=[col])

    y = df[target]
    X = df.drop(columns=[target])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.get("test_size", 0.2), random_state=cfg.get("random_state", 42)
    )

    # Config MLflow
    tracking_uri = cfg["mlflow"]["tracking_uri"]
    experiment_name = cfg["mlflow"]["experiment_name"]

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        # Log de parámetros
        mlflow.log_params({
            "dataset_path": dataset_path,
            "target": target,
            "test_size": cfg.get("test_size", 0.2),
            "random_state": cfg.get("random_state", 42),
            "model_type": cfg["model"]["type"],
            "n_estimators": cfg["model"]["n_estimators"],
            "max_depth": cfg["model"]["max_depth"],
        })

        # Modelo (RandomForest de ejemplo)
        model = RandomForestRegressor(
            n_estimators=cfg["model"]["n_estimators"],
            max_depth=cfg["model"]["max_depth"],
            random_state=cfg.get("random_state", 42)
        )
        model.fit(X_train, y_train)

        # Métrica
        y_pred = model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)

        # Artefactos
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Guarda la config usada como artefacto
        artifacts_dir = Path("artifacts"); artifacts_dir.mkdir(exist_ok=True)
        used_cfg = artifacts_dir / "train_used.yaml"
        with open(used_cfg, "w") as f:
            yaml.safe_dump(cfg, f)
        mlflow.log_artifact(str(used_cfg))

        # Adjunta archivos de DVC para trazabilidad (si existen)
        if Path("dvc.yaml").exists(): mlflow.log_artifact("dvc.yaml")
        if Path("dvc.lock").exists(): mlflow.log_artifact("dvc.lock")

    print(f"Run completado. Tracking URI: {tracking_uri}  |  Experimento: {experiment_name}")

if __name__ == "__main__":
    main()
