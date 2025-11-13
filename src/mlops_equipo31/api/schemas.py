from typing import List, Optional, Any
from pydantic import BaseModel, Field

class PredictItem(BaseModel):
    temp: float = Field(..., description="Temperatura (°C)")
    hum: float = Field(..., ge=0, le=100, description="Humedad relativa (%)")
    wind: float = Field(..., ge=0, description="Velocidad del viento (m/s)")
    gen_diffuse_flows: float = Field(..., ge=0, description="Radiación difusa generada")
    diffuse_flows: float = Field(..., ge=0, description="Radiación difusa")
    z2_power_cons: float = Field(..., ge=0, description="Consumo zona 2")
    z3_power_cons: float = Field(..., ge=0, description="Consumo zona 3")
    hour: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6, description="0=Lunes ... 6=Domingo")
    month: int = Field(..., ge=1, le=12)
    day: int = Field(..., ge=1, le=31)

class PredictRequest(BaseModel):
    # v2: usa min_length para listas
    inputs: List[PredictItem] = Field(..., min_length=1)

class PredictResponse(BaseModel):
    predictions: List[Any]
    model_uri: str
    model_version: Optional[str] = None
    run_id: Optional[str] = None
    latency_ms: float
