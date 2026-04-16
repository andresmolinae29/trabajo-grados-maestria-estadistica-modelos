import arch
import pmdarima as pm
import pandas as pd

from arch.univariate.mean import HARX
from .base import BaseVolatilityModel
from ..schemas import ModelConfig, PredictionResult, AssetMetadata, PredictionRow
from ..utils import logger


class GARCHModel(BaseVolatilityModel):

    name = "GARCH"

    def __init__(self, config: ModelConfig, asset_metadata: AssetMetadata):

        super().__init__(config, asset_metadata)

    def __set_model(self, X: pd.Series, hyperparameters: dict) -> HARX:

        arima_model = pm.auto_arima(
            X,
            seasonal=False,
            information_criterion="aic",
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
        )

        residuals = pd.Series(arima_model.resid())

        model = arch.arch_model(residuals, vol="GARCH", **hyperparameters)

        return model

    def __select_hyperparameters(self, X: pd.Series) -> dict:

        best_aic = float("inf")
        best_hyperparameters = {}

        for hyperparameters in self.config.hyperparameters_list:
            try:
                model = self.__set_model(X, hyperparameters)
                fitted_model = model.fit(disp="off")
                aic = fitted_model.aic
                if aic < best_aic:
                    best_aic = aic
                    best_hyperparameters = hyperparameters
            except Exception as e:
                logger.warning(
                    f"Error fitting GARCH model with hyperparameters {hyperparameters}: {e}"
                )
                continue

        return best_hyperparameters

    def fit(self, X: pd.Series, y: pd.Series | None = None) -> None:

        best_hyperparameters = self.__select_hyperparameters(X)
        model = self.__set_model(X, best_hyperparameters)
        self.model = model.fit(disp="off")
        self.is_fitted = True

    def predict(self, X: pd.Series, y: pd.Series) -> PredictionResult:

        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")

        forecast = self.model.forecast(start=X.index[-1], horizon=len(y))
        values = (
            forecast.variance.values.flatten()
        )  # o forecast.variance, según lo que quieras predecir
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

    def get_params(self) -> ModelConfig:

        return self.config

    def save(self, path: str) -> None:
        raise NotImplementedError(
            "The save method must be implemented by the subclass."
        )
