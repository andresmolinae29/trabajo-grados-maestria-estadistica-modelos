from __future__ import annotations

import pandas as pd


def make_series(values: list[float]) -> pd.Series:
    index = pd.date_range("2026-01-01", periods=len(values), freq="15min")
    return pd.Series(values, index=index, dtype=float)
