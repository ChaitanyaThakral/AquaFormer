from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class PredictionRequest(BaseModel):
    lat: float = Field(..., description="Latitude of the location", ge=-90.0, le=90.0)
    lon: float = Field(..., description="Longitude of the location", ge=-180.0, le=180.0)
    time: datetime = Field(..., description="Time of the prediction query")

class PredictionResponse(BaseModel):
    lat: float
    lon: float
    time: datetime
    prediction: float = Field(..., description="Predicted rainfall/risk value")
    cached: bool = Field(False, description="True if the response was served from cache")
    latency_ms: Optional[float] = Field(None, description="Latency of the prediction in milliseconds")
