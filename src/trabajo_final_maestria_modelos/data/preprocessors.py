import numpy as np

from ..schemas import (
    TimeSeriesInput
)


class DataPreprocessor:
    def __init__(self, config):
        self.config = config

    def preprocess(self, data: TimeSeriesInput, handle_missing: bool = True) -> TimeSeriesInput:

        if handle_missing:
            data.series.ffill(inplace=True)
            data.series.bfill(inplace=True)

        data.series = data.series.astype(float)

        return data
    
    def compute_log_returns(self, data: TimeSeriesInput) -> TimeSeriesInput:
        data.log_returns = data.series.pct_change().apply(lambda x: np.log(1 + x)).dropna()
        return data
    
    def normalize(self, data: TimeSeriesInput) -> TimeSeriesInput:

        if data.log_returns is None:
            raise ValueError("Log returns must be computed before normalization.")
        data.log_returns = (data.log_returns - data.log_returns.mean()) / data.log_returns.std()
        return data