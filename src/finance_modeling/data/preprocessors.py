import numpy as np

from ..schemas import (
    TimeSeriesInput
)


class DataPreprocessor:
    def __init__(self, data: TimeSeriesInput):
        self.data = data

    def _preprocess(self, handle_missing: bool = True) -> None:

        if handle_missing:
            self.data.series.ffill(inplace=True)
            self.data.series.bfill(inplace=True)

        self.data.series = self.data.series.astype(float)

    def _returns_series(self) -> None:
        self.data.returns = (
            (self.data.series - self.data.series.shift(1)) / self.data.series).dropna()

    def _innovations_series(self,) -> None:
        if self.data.returns is None:
            raise ValueError("Returns series must be computed before computing innovations.")
        self.data.innovations = self.data.returns - self.data.returns.mean()

    def _volatility_series(self, omega: int = 252) -> None:
        if self.data.returns is None:
            raise ValueError("Returns series must be computed before computing volatility.")
        self.data.volatility_series = (
            self.data.returns.rolling(window=omega).std(ddof=1) * np.sqrt(252)
            ).iloc[omega - 1 : -1]

    def pipeline(self) -> TimeSeriesInput:
        self._preprocess()
        self._returns_series()
        self._innovations_series()
        self._volatility_series()

        return self.data