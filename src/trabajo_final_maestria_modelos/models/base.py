import pandas as pd
from abc import ABC, abstractmethod


class BaseVolatilityModel(ABC):

    name: str
    config: dict
    is_fitted: bool = False

    @abstractmethod
    def fit(self, X, y):
        raise NotImplementedError("The fit method must be implemented by the subclass.")

    @abstractmethod
    def predict(self, X) -> pd.DataFrame:
        raise NotImplementedError("The predict method must be implemented by the subclass.")

    @abstractmethod
    @staticmethod
    def evaluate(y_true, y_pred) -> dict[str, float]: # esto deberia ir con pydantic
        raise NotImplementedError("The evaluate method must be implemented by the subclass.")

    def get_params(self) -> dict:
        raise NotImplementedError("The get_params method must be implemented by the subclass.")
    
    def save(self, path: str) -> None:
        raise NotImplementedError("The save method must be implemented by the subclass.")