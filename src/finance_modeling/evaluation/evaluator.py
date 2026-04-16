import pandas as pd

from .metrics import Metrics
from ..schemas import EvaluationResult, PredictionResult, TimeSeriesInput
from ..utils import convert_list_to_series


class Evaluator:

    def evaluate(
        self, model_name: str, asset: str, y_true: pd.Series, y_pred: pd.Series
    ) -> EvaluationResult:

        rmse = Metrics.root_mean_squared_error(y_true, y_pred)
        mae = Metrics.mean_absolute_error(y_true, y_pred)

        return EvaluationResult(model_name=model_name, asset=asset, rmse=rmse, mae=mae)

    @classmethod
    def from_timeinput_and_prediction_result(
        cls, y_true: TimeSeriesInput, y_pred: PredictionResult
    ) -> EvaluationResult:

        y_pred_series = convert_list_to_series(
            [row.timestamp for row in y_pred.rows],
            [row.predicted_volatility for row in y_pred.rows],
        )

        return cls().evaluate(
            model_name=y_pred.model_name,
            asset=y_pred.asset,
            y_true=y_true.test,
            y_pred=y_pred_series,
        )
