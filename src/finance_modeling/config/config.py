import yaml
import os
from functools import cached_property, lru_cache

from ..schemas import AssetMetadata, ExperimentConfig, ModelConfig, ListOfAssets
from ..utils import get_main_root, validate_file_exists


DATA_CONFIG = "data_loading_config.yml"
MODEL_CONFIG = "experiment_config.yml"


class ConfigLoader:
    @staticmethod
    @lru_cache(maxsize=None)
    def load_config(config_path: str) -> dict:

        validate_file_exists(config_path)

        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    # @cached_property
    def load_model_config(self) -> ExperimentConfig:
        file_path = os.path.join(get_main_root(), "config", MODEL_CONFIG)
        config = self.load_config(file_path)
        return ExperimentConfig(models=[ModelConfig(**model) for model in config])

    # @cached_property
    def load_data_config(self) -> ListOfAssets:
        file_path = os.path.join(get_main_root(), "config", DATA_CONFIG)
        config = self.load_config(file_path)
        return ListOfAssets(assets=[AssetMetadata(**asset) for asset in config])


if __name__ == "__main__":
    pass
