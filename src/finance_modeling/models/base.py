from __future__ import annotations

from abc import ABC, abstractmethod

from ..schemas import (
    PredictionResult,
    AssetMetadata,
    ModelConfig
)

from ..utils import (
    get_main_root,
    create_experiment_directory
)

from datetime import datetime

import pandas as pd
import pickle
import json
import os


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
        results_df = pd.DataFrame([vars(results)])
        results_df.to_csv(file_path, index=False)

    def save_model_best_hyperparameters(self, experiment_path: str) -> None:

        params_file = os.path.join(experiment_path, f"{self.name}_{self.asset_metadata.symbol}_best_hyperparameters.json")
        with open(params_file, "w") as f:
            json.dump(self.best_hyperparameters, f, indent=4)
