from __future__ import annotations

from abc import ABC, abstractmethod

from ..schemas import (
    ModelConfig
)

from torch import nn

import pandas as pd
import pickle
import os
import numpy as np


class BaseVolatilityModel(ABC):

    name: str
    config: ModelConfig
    is_fitted: bool = False

    def __init__(self, config: ModelConfig, symbol: str, **kwargs):

        self.config = config
        self.symbol = symbol

    @abstractmethod
    def fit(self) -> BaseVolatilityModel:
        raise NotImplementedError("The fit method must be implemented by the subclass.")

    @abstractmethod
    def predict(self) -> np.ndarray:
        raise NotImplementedError("The predict method must be implemented by the subclass.")

    def save_model(self, experiment_path: str) -> None:

        model_file = os.path.join(experiment_path, f"{self.name}_{self.symbol}.pkl")
        with open(model_file, "wb") as f:
            pickle.dump(self, f)
