import pandas as pd

from ..schemas import (
    EvaluationResult,
    ModelConfig,
    PredictionResult,
    AssetMetadata
)

from .base import BaseVolatilityModel

from ..evaluation import Metrics

import arch


class GARCHModel(BaseVolatilityModel):

    name = "GARCH"

    def __init__(self, model, config: ModelConfig, asset_metadata: AssetMetadata):
        self.model = model
        self.config = config
        self.asset_metadata = asset_metadata

    def fit(self, X: pd.Series, y: pd.Series):

        arch.arch_model(X, vol='GARCH', p=1, q=1).fit(disp="off")

    def predict(self, X: pd.Series) -> PredictionResult:
        raise NotImplementedError("The predict method must be implemented by the subclass.")

    def evaluate(self, y_true, y_pred) -> EvaluationResult:

        rmse = Metrics.root_mean_squared_error(y_true, y_pred)
        mae = Metrics.mean_absolute_error(y_true, y_pred)
        return EvaluationResult(
            model_name=GARCHModel.name,
            asset=self.asset_metadata.symbol,
            rmse=rmse,
            mae=mae
        )

    def get_params(self) -> ModelConfig:

        return self.config

    def save(self, path: str) -> None:
        raise NotImplementedError("The save method must be implemented by the subclass.")