import os
from functools import lru_cache

class Settings:
    app_name: str = "Equipo31 Serving"
    app_version: str = "0.2.0"
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "")
    model_uri: str = os.getenv("MODEL_URI", "")

@lru_cache
def get_settings() -> Settings:
    return Settings()
