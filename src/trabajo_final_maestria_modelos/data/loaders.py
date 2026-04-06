import pandas as pd

from ..schemas import AssetMetadata, TimeSeriesInput

class RawDataLoader:
    def __init__(self, data_config: AssetMetadata):
        self.data_config = data_config

    def load_data(self) -> TimeSeriesInput:
        
        df = pd.read_csv(self.data_config.data_path, parse_dates=['timestamp'])
        df.set_index('timestamp', inplace=True)

        return TimeSeriesInput(
            metadata=self.data_config,
            series=df[self.data_config.column_to_use],
            frequency="15min",
        )