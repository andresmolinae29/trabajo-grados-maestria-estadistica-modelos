from __future__ import annotations

from pydantic import BaseModel


class EvaluationResult(BaseModel):
    model_name: str
    asset: str
    rmse: float
    mae: float
    versus_baseline: bool = True
    dm_statistic: float | None = None
    dm_p_value: float | None = None


class ComparisonResult(BaseModel):
    baseline_model: str
    challenger_model: str
    asset: str
    metrics: list[EvaluationResult]
