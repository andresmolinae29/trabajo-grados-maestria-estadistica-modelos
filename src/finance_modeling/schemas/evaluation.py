from __future__ import annotations

from pydantic import BaseModel


class EvaluationResult(BaseModel):
    model_name: str
    asset: str
    rmse: float
    mae: float
    versus_baseline: bool = True


class ComparisonResult(BaseModel):
    baseline_model: str
    challenger_model: str
    asset: str
    rmse_baseline: float | None = None
    rmse_challenger: float | None = None
    rmse_diff: float | None = None
    mae_baseline: float | None = None
    mae_challenger: float | None = None
    mae_diff: float | None = None
    dm_statistic: float | None = None
    dm_p_value: float | None = None
