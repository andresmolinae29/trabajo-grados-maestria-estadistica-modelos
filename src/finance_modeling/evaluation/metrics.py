import pandas as pd

from dieboldmariano import dm_test

from ..utils import logger


class Metrics:

    @staticmethod
    def mean_absolute_error(
        y_true: pd.Series | list[float], y_pred: pd.Series | list[float]
    ):
        return sum(abs(t - p) for t, p in zip(y_true, y_pred)) / len(y_true)

    @staticmethod
    def mean_squared_error(
        y_true: pd.Series | list[float], y_pred: pd.Series | list[float]
    ):
        return sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / len(y_true)

    @staticmethod
    def root_mean_squared_error(
        y_true: pd.Series | list[float], y_pred: pd.Series | list[float]
    ):
        mse = Metrics.mean_squared_error(y_true, y_pred)
        return mse**0.5

    @staticmethod
    def diebold_mariano_test(
        y_true: pd.Series | list[float],
        y_pred1: pd.Series | list[float],
        y_pred2: pd.Series | list[float],
    ):
        y_true_secuence = y_true.tolist() if isinstance(y_true, pd.Series) else y_true
        y_pred1_secuence = y_pred1.tolist() if isinstance(y_pred1, pd.Series) else y_pred1
        y_pred2_secuence = y_pred2.tolist() if isinstance(y_pred2, pd.Series) else y_pred2
        dm_stat, p_value = dm_test(
            y_true_secuence, y_pred1_secuence, y_pred2_secuence, h=1, variance_estimator="acf"
        )
        return dm_stat, p_value
