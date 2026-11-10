import numpy as np
import pandas as pd

from .base import BaseVolatilityModel

from ..schemas import ModelConfig


class NaiveModel(BaseVolatilityModel):

    name = "NAIVE"

    def __init__(self, config: ModelConfig, symbol: str, **kwargs):
        super().__init__(config, symbol, **kwargs)

        self.volatility: pd.Series = kwargs.get("volatility", pd.Series())
        self.split = int(len(self.volatility) * 0.8)
        self.y_test: np.ndarray = self.volatility.iloc[self.split:].to_numpy()

    def fit(self) -> BaseVolatilityModel:
        self.is_fitted = True
        return self

    def predict(self) -> np.ndarray:

        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")

        return self.volatility.shift(1).iloc[self.split:].to_numpy()
