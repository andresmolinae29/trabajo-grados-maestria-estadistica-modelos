from __future__ import annotations

import os

from typing import Any

from pydantic import BaseModel, Field

from ..utils import get_main_root


class ModelConfig(BaseModel):
    name: str
    hyperparameters_list: list[dict[str, Any]] = Field(default_factory=list)
    random_seed: int = 42


class ExperimentConfig(BaseModel):
    models: list[ModelConfig]
    train_ratio: float = Field(default=0.8, ge=0.5, le=0.95)
    output_dir: str = Field(default_factory=lambda: os.path.join(get_main_root(), "results", 'models'))
