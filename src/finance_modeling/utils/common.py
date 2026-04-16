import os
import pandas as pd
from functools import lru_cache


@lru_cache
def get_main_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def validate_file_exists(file_path: str) -> None:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at: {file_path}")


def generate_future_timestamps(last_index, steps, freq):
    """
    last_index: último timestamp de la serie (DatetimeIndex o datetime)
    steps: número de pasos a predecir
    freq: frecuencia como string pandas ('D', '15min', etc)
    """
    return pd.date_range(start=last_index, periods=steps+1, freq=freq)[1:]


def convert_list_to_series(index, y_pred: list) -> pd.Series:
        return pd.Series(
            data=y_pred,
            index=index
        )