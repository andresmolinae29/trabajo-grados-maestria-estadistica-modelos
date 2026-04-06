from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class PredictionRow(BaseModel):
    timestamp: datetime
    predicted_volatility: float
    lower_ci: float | None = None
    upper_ci: float | None = None


class PredictionResult(BaseModel):
    model_name: str
    asset: str
    horizon: int
    rows: list[PredictionRow]
