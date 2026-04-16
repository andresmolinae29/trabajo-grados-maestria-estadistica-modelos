import pandas as pd

from ..schemas import (
    EvaluationResult,
    ModelConfig,
    PredictionResult,
    AssetMetadata,
    PredictionRow
)

from ..utils import generate_future_timestamps, logger

from .base import BaseVolatilityModel

from ..evaluation import Metrics

import arch

import pmdarima as pm

from arch.univariate.mean import HARX


class GARCHModel(BaseVolatilityModel):

    name = "GARCH"

    def __init__(self, config: ModelConfig, asset_metadata: AssetMetadata):

        self.config = config
        self.asset_metadata = asset_metadata

    def set_model(self, X: pd.Series, hyperparameters: dict) -> HARX:

        arima_model = pm.auto_arima(
            X,
            seasonal=False,
            information_criterion='aic',
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore'
        )

        residuals = pd.Series(arima_model.resid())

        model = arch.arch_model(
            residuals,
            vol='GARCH',
            **hyperparameters
        )

        return model

    def select_hyperparameters(self, X: pd.Series) -> dict:

        best_aic = float('inf')
        best_hyperparameters = {}

        for hyperparameters in self.config.hyperparameters_list:
            try:
                model = self.set_model(X, hyperparameters)
                fitted_model = model.fit(disp='off')
                aic = fitted_model.aic
                if aic < best_aic:
                    best_aic = aic
                    best_hyperparameters = hyperparameters
            except Exception as e:
                logger.warning(f"Error fitting GARCH model with hyperparameters {hyperparameters}: {e}")
                continue

        return best_hyperparameters

    def fit(self, X: pd.Series, y: pd.Series | None = None) -> None:

        best_hyperparameters = self.select_hyperparameters(X)
        model = self.set_model(X, best_hyperparameters)
        self.model = model.fit(disp='off')
        self.is_fitted = True

    def predict(self, X: pd.Series, y: pd.Series) -> PredictionResult:

        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")

        forecast = self.model.forecast(start=X.index[-1], horizon=len(y))
        values = forecast.variance.values.flatten()   # o forecast.variance, según lo que quieras predecir
        timestamps = y.index

        rows = [
            PredictionRow(
                timestamp=ts,
                predicted_volatility=float(val),
                lower_ci=None,
                upper_ci=None,
            )
            for ts, val in zip(timestamps, values)
        ]

        return PredictionResult(
            model_name=self.name,
            asset=self.asset_metadata.symbol,
            horizon=len(y),
            rows=rows,
        )

    def evaluate(self, y_true: pd.Series, y_pred: PredictionResult) -> EvaluationResult:

        y_pred_series = pd.Series(
            [row.predicted_volatility for row in y_pred.rows],
            index=[row.timestamp for row in y_pred.rows]
        )

        rmse = Metrics.root_mean_squared_error(y_true, y_pred_series)
        mae = Metrics.mean_absolute_error(y_true, y_pred_series)
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