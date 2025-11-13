import time
from typing import Any, List

import pandas as pd
import mlflow
import mlflow.pyfunc
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .settings import get_settings
from .schemas import PredictRequest, PredictResponse

app = FastAPI(
    title="Equipo31 Model Serving",
    description="API para servir el modelo con MLflow + FastAPI",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ajusta si quieres restringir
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carga perezosa del modelo
_model = None
_model_details = {"version": None, "run_id": None}

def _load_model(settings):
    global _model, _model_details
    if _model is None:
        if not settings.model_uri:
            raise RuntimeError("MODEL_URI no está configurado")
        if settings.mlflow_tracking_uri:
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

        _model = mlflow.pyfunc.load_model(settings.model_uri)

        # Intenta derivar version desde el URI (models:/name/<version>)
        try:
            parts = settings.model_uri.split("/")
            if len(parts) >= 3 and parts[-1].isdigit():
                _model_details["version"] = parts[-1]
        except Exception:
            pass
        # Intenta obtener run_id si existe metadata
        try:
            _model_details["run_id"] = getattr(_model.metadata, "run_id", None)
        except Exception:
            pass
    return _model

@app.get("/health")
def health(settings = Depends(get_settings)):
    # No forzamos carga aquí para que /health sea barato
    return {"status": "ok", "app": settings.app_name, "version": settings.app_version}

@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, settings = Depends(get_settings)):
    """
    Realiza inferencia sobre una o varias filas.
    """
    t0 = time.time()
    try:
        model = _load_model(settings)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"No se pudo cargar el modelo: {e}")

    # JSON -> DataFrame
    try:
        rows = [row.model_dump() for row in payload.inputs]
        df = pd.DataFrame(rows)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Entrada inválida: {e}")

    # Predicción
    try:
        preds = model.predict(df)
        if hasattr(preds, "tolist"):
            preds = preds.tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante la predicción: {e}")

    latency_ms = (time.time() - t0) * 1000.0
    return PredictResponse(
        predictions=preds,
        model_uri=settings.model_uri,
        model_version=_model_details.get("version"),
        run_id=_model_details.get("run_id"),
        latency_ms=round(latency_ms, 2),
    )
