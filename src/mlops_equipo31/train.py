import pandas as pd
import numpy as np
import os
import yaml
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


MODELS = {
    "Ridge": Ridge,
    "Lasso": Lasso,
    "LinearRegression": LinearRegression,
    "RandomForestRegressor": RandomForestRegressor
}


def load_config(config_path="train.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_data(path):
    return pd.read_csv(path)


def clean_data(df):
    df = df.dropna()
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    df = df[~df.isin(['?', 'error', 'NAN', 'invalid', 'null']).any(axis=1)]
    if 'DateTime' in df.columns:
        df['DateTime'] = pd.to_datetime(df['DateTime'].str.strip())
    return df


def preprocess(df, target):
    X = df.drop(columns=["DateTime", target], errors="ignore")
    numeric_features = selector(dtype_include="number")(X)
    categorical_features = selector(dtype_include="object")(X)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )
    return preprocessor


def train_pipeline(X_train, y_train, preprocessor, model):
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])
    pipe.fit(X_train, y_train)
    return pipe


def evaluate_model(pipe, X_test, y_test):
    y_pred = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return mae, rmse, r2


def main():
    config = load_config("train.yaml")

    data_path = config["dataset_path"]
    target = config["target"]
    test_size = config.get("test_size", 0.2)
    random_state = config.get("random_state", 42)

    model_type = config["model"]["type"]
    model_params = {k: v for k, v in config["model"].items() if k != "type"}
    model_cls = MODELS[model_type]
    model = model_cls(**model_params)

    experiment_name = config["mlflow"].get("experiment_name", "default")
    tracking_uri = config["mlflow"].get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        df = load_data(data_path)
        df = clean_data(df)

        y = df[target]
        X = df.drop(columns=[target, "DateTime"], errors="ignore")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        preprocessor = preprocess(df, target)
        pipe = train_pipeline(X_train, y_train, preprocessor, model)

        mae, rmse, r2 = evaluate_model(pipe, X_test, y_test)

        mlflow.log_param("model", model_type)
        for param, val in model_params.items():
            mlflow.log_param(param, val)

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        mlflow.sklearn.log_model(pipe, "model")

        print(f"Model: {model_type} | MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.2f}")


if __name__ == "__main__":
    main()
