from __future__ import annotations

from datetime import datetime

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class PredictionRow(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    timestamp: datetime
    predicted_value: float = Field(
        validation_alias=AliasChoices("predicted_value", "predicted_volatility")
    )
    real_value: float | None = Field(
        default=None,
        validation_alias=AliasChoices("real_value", "actual_volatility")
    )


class PredictionResult(BaseModel):
    model_name: str
    asset: str
    horizon: int
    rows: list[PredictionRow]
