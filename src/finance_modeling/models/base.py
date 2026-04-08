from __future__ import annotations

from abc import ABC, abstractmethod

from ..schemas import (
    PredictionResult,
    EvaluationResult,
    ModelConfig
)


class BaseVolatilityModel(ABC):

    name: str
    config: ModelConfig
    is_fitted: bool = False

    def __init__(self, model_type) -> None:
        self.model_type = model_type

    @abstractmethod
    def fit(self, X, y):
        raise NotImplementedError("The fit method must be implemented by the subclass.")

    @abstractmethod
    def predict(self, X) -> PredictionResult:
        raise NotImplementedError("The predict method must be implemented by the subclass.")

    @abstractmethod
    def evaluate(self, y_true, y_pred) -> EvaluationResult:
        raise NotImplementedError("The evaluate method must be implemented by the subclass.")

    def get_params(self) -> ModelConfig:
        raise NotImplementedError("The get_params method must be implemented by the subclass.")

    def save(self, path: str) -> None:
        raise NotImplementedError("The save method must be implemented by the subclass.")