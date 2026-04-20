from __future__ import annotations

from abc import ABC, abstractmethod

from ..schemas import (
    PredictionResult,
    AssetMetadata,
    ModelConfig
)

from torch import nn

import pandas as pd
import pickle
import json
import os
import torch


class BaseVolatilityModel(ABC):

    name: str
    config: ModelConfig
    is_fitted: bool = False

    def __init__(self, config: ModelConfig, asset_metadata: AssetMetadata):

        self.config = config
        self.asset_metadata = asset_metadata
        self.best_hyperparameters: dict = {}

    @abstractmethod
    def fit(self, X: pd.Series, y: pd.Series | None = None) -> None:
        raise NotImplementedError("The fit method must be implemented by the subclass.")

    @abstractmethod
    def predict(self, X: pd.Series, y: pd.Series) -> PredictionResult:
        raise NotImplementedError("The predict method must be implemented by the subclass.")

    def get_params(self) -> ModelConfig:

        return self.config

    def save_model(self, experiment_path: str) -> None:

        model_file = os.path.join(experiment_path, f"{self.name}_{self.asset_metadata.symbol}.pkl")
        with open(model_file, "wb") as f:
            pickle.dump(self, f)

    def save_results(self, experiment_path: str, results: PredictionResult) -> None:

        file_path = os.path.join(experiment_path, f"{self.name}_{self.asset_metadata.symbol}_results.csv")
        results_df = pd.DataFrame(
            [
                {
                    "model_name": results.model_name,
                    "asset": results.asset,
                    "horizon": results.horizon,
                    **row.model_dump(),
                }
                for row in results.rows
            ]
        )
        results_df.to_csv(file_path, index=False)

    def save_model_best_hyperparameters(self, experiment_path: str) -> None:

        params_file = os.path.join(experiment_path, f"{self.name}_{self.asset_metadata.symbol}_best_hyperparameters.json")
        with open(params_file, "w") as f:
            json.dump(self.best_hyperparameters, f, indent=4)


class _Regressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float, output_size: int):
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=effective_dropout,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        sequence_output, _ = self.lstm(features)
        return self.output(sequence_output[:, -1, :])