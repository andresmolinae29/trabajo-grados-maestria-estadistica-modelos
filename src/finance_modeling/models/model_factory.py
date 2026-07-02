from .base import BaseVolatilityModel
from .garch import GARCHModel
from .psoqrnn import PSOQRNNModel
from .ceemdan_lstm import CEEMDANLSTMModel

from ..schemas import ModelConfig


class ModelFactory:

    MODEL_REGISTRY = {
        "garch": GARCHModel,
        "psoqrnn": PSOQRNNModel,
        "ceemdan_lstm": CEEMDANLSTMModel,
    }

    @classmethod
    def create_model(cls, model_name: str, model_config: ModelConfig, symbol: str, **kwargs) -> BaseVolatilityModel:
        model_class = cls.MODEL_REGISTRY.get(model_name.lower().strip())
        if not model_class:
            raise ValueError(f"Model '{model_name}' is not registered. Available models: {list(cls.MODEL_REGISTRY.keys())}")
        return model_class(config=model_config, symbol=symbol, **kwargs)
