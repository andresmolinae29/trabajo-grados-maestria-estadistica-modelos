from __future__ import annotations

from abc import ABC, abstractmethod

from ..schemas import (
    PredictionResult,
    EvaluationResult,
    ModelConfig,
    PredictionRow
)

import pandas as pd


class BaseVolatilityModel(ABC):

    name: str
    config: ModelConfig
    is_fitted: bool = False

    @abstractmethod
    def fit(self, X: pd.Series, y: pd.Series | None = None) -> None:
        raise NotImplementedError("The fit method must be implemented by the subclass.")

    @abstractmethod
    def predict(self, X: pd.Series, y: pd.Series) -> PredictionResult:
        raise NotImplementedError("The predict method must be implemented by the subclass.")

    @abstractmethod
    def evaluate(self, y_true: pd.Series, y_pred: PredictionResult) -> EvaluationResult:
        raise NotImplementedError("The evaluate method must be implemented by the subclass.")

    def get_params(self) -> ModelConfig:
        raise NotImplementedError("The get_params method must be implemented by the subclass.")

    def save(self, path: str) -> None:
        raise NotImplementedError("The save method must be implemented by the subclass.")