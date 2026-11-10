from __future__ import annotations

import os

from typing import Any

from pydantic import BaseModel, Field

from finance_modeling.utils.common import get_main_root


class ModelConfig(BaseModel):
    name: str
    hyperparameters_list: list[dict[str, Any]] = Field(default_factory=list)
    random_seed: int = 42
    seeds: list[int] = Field(default_factory=lambda: [42])
    causal: bool = False
    train_context: bool = False
    search_space: dict[str, list[Any]] = Field(default_factory=dict)
    n_trials: int = Field(default=0, ge=0)


class GarchCandidate(BaseModel):
    hyperparameters: dict[str, Any]
    justification: str = Field(min_length=1)


class GarchSelection(BaseModel):
    symbol: str
    candidates: list[GarchCandidate] = Field(min_length=2)


class ExperimentConfig(BaseModel):
    models: list[ModelConfig]
    experiment_name: str
    train_ratio: float = Field(default=0.8, ge=0.5, le=0.95)
    output_dir: str = Field(default_factory=lambda: os.path.join(get_main_root(), "results", 'models'))
