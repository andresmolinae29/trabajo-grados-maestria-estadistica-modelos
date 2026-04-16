import pandas as pd
import os

from ..schemas import AssetMetadata, TimeSeriesInput
from ..utils import get_main_root, validate_file_exists


class RawDataLoader:
    def __init__(self, data_config: AssetMetadata):
        self.data_config = data_config

    def _set_data_path(self) -> str:
        return os.path.join(
            get_main_root(),
            self.data_config.data_path,
            self.data_config.data_folder,
            f"{self.data_config.symbol}_curated.csv",
        )

    def load_data(self) -> TimeSeriesInput:

        path = self._set_data_path()
        validate_file_exists(path)

        df = pd.read_csv(
            path,
            sep=";",
            encoding="utf-8",
            parse_dates=["timestamp"],
            usecols=["timestamp", self.data_config.column_to_use],
            dtype={self.data_config.column_to_use: float},
        )
        df.set_index("timestamp", inplace=True)

        return TimeSeriesInput(
            metadata=self.data_config,
            series=df[self.data_config.column_to_use],
            frequency="15min",
            log_returns=pd.Series(),
            train=pd.Series(),
            test=pd.Series(),
            split_index=0
        )
