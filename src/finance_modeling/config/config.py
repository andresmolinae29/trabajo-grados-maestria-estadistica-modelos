import yaml
import os
from functools import cached_property, lru_cache

from ..schemas import AssetMetadata, ExperimentConfig, GarchSelection, ModelConfig, ListOfAssets
from ..utils import get_main_root, validate_file_exists


DATA_CONFIG = "data_loading_config.yml"
MODEL_CONFIG = "experiment_config.yml"
GARCH_SELECTION_CONFIG = "garch_selection.yml"


class ConfigLoader:
    @staticmethod
    @lru_cache(maxsize=None)
    def load_config(config_path: str) -> dict:

        validate_file_exists(config_path)

        with open(config_path, 'r', encoding="utf-8") as file:
            config = yaml.safe_load(file)
        return config

    def load_model_config(self, file_name: str = MODEL_CONFIG) -> ExperimentConfig:
        file_path = os.path.join(get_main_root(), "config", file_name)
        config = self.load_config(file_path)
        return ExperimentConfig(
            experiment_name=config[0].get("experiment_name", "default_experiment"),
            models=[ModelConfig(**model) for model in config[1:]])

    def load_data_config(self) -> ListOfAssets:
        file_path = os.path.join(get_main_root(), "config", DATA_CONFIG)
        config = self.load_config(file_path)
        return ListOfAssets(assets=[AssetMetadata(**asset) for asset in config])

    def load_garch_selection(self) -> dict[str, GarchSelection]:
        file_path = os.path.join(get_main_root(), "config", GARCH_SELECTION_CONFIG)
        config = self.load_config(file_path) or []
        selections = [GarchSelection(**selection) for selection in config]
        return {selection.symbol: selection for selection in selections}


if __name__ == "__main__":
    pass
