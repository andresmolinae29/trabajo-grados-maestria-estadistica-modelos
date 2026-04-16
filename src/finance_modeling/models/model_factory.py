from .base import BaseVolatilityModel
from .garch import GARCHModel
from .psoqrnn import PSOQRNNModel
from .ceemdan_lstm import CEEMDANLSTMModel

from ..schemas import ModelConfig, AssetMetadata


class ModelFactory:

    MODEL_REGISTRY = {
        "garch": GARCHModel,
        "psoqrnn": PSOQRNNModel,
        "ceemdan_lstm": CEEMDANLSTMModel,
    }

    @classmethod
    def create_model(cls, model_name: str, model_config: ModelConfig, asset_metadata: AssetMetadata) -> BaseVolatilityModel:
        model_class = cls.MODEL_REGISTRY.get(model_name.lower().strip())
        if not model_class:
            raise ValueError(f"Model '{model_name}' is not registered. Available models: {list(cls.MODEL_REGISTRY.keys())}")
        return model_class(config=model_config, asset_metadata=asset_metadata)
