from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    name: str
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    random_seed: int = 42


class DataConfig(BaseModel):
    asset: str
    source: str = Field(description="e.g. 'yahoo', 'binance', 'investing'")
    interval: str = Field(default="15m", description="e.g. '15m', '1h', '1d'")
    start_date: date
    end_date: date


class ExperimentConfig(BaseModel):
    models: list[ModelConfig]
    data: list[DataConfig]
    output_dir: Path = Field(default=Path("outputs"))
