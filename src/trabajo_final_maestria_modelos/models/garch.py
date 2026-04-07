import pandas as pd

from ..schemas import (
    EvaluationResult,
    ModelConfig,
    PredictionResult,
    TimeSeriesInput
)

from .base import BaseVolatilityModel

from ..evaluation import Metrics


class GARCHModel(BaseVolatilityModel):

    name = "GARCH"

    def __init__(self, data: TimeSeriesInput, config: ModelConfig):
        self.data = data
        self.config = config

    def fit(self, X, y):
        raise NotImplementedError("The fit method must be implemented by the subclass.")

    def predict(self, X) -> PredictionResult:
        raise NotImplementedError("The predict method must be implemented by the subclass.")

    def evaluate(self, y_true, y_pred) -> EvaluationResult:
        
        rmse = Metrics.root_mean_squared_error(y_true, y_pred)
        mae = Metrics.mean_absolute_error(y_true, y_pred)
        return EvaluationResult(
            model_name=GARCHModel.name,
            asset=self.data.metadata.symbol,
            rmse=rmse,
            mae=mae
        )

    def get_params(self) -> ModelConfig:
        
        return self.config
    
    def save(self, path: str) -> None:
        raise NotImplementedError("The save method must be implemented by the subclass.")