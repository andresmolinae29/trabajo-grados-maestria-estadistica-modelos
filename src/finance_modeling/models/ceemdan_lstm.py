import pandas as pd

from .base import BaseVolatilityModel


class CEEMDANLSTMModel(BaseVolatilityModel):

    name = "CEEMDAN-LSTM"

    def fit(self, X, y):
        pass

    def predict(self, X) -> pd.DataFrame:
        pass

    @staticmethod
    def evaluate(y_true, y_pred) -> dict[str, float]:
        pass

    def get_params(self) -> dict:
        pass

    def save(self, path: str) -> None:
        pass
