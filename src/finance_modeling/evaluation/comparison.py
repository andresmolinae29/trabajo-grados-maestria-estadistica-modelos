import pandas as pd

from .metrics import Metrics
from ..schemas import EvaluationResult, ComparisonResult, PredictionResult
from ..utils import convert_list_to_series


class ModelComparator:

    def __get_dm_statistic_and_p_value(self, y_true: pd.Series, pred_baseline: pd.Series, pred_challenger: pd.Series) -> tuple[float, float]:

        dm_statistic, dm_p_value = Metrics.diebold_mariano_test(
            y_true,
            pred_baseline,
            pred_challenger,
        )

        return dm_statistic, dm_p_value

    def compare(self, baseline: EvaluationResult, challenger: EvaluationResult, y_pred_baseline: pd.Series, y_pred_challenger: pd.Series, y_true: pd.Series) -> ComparisonResult:

        dm_statistic, dm_p_value = self.__get_dm_statistic_and_p_value(
            y_true=y_true,
            pred_baseline=y_pred_baseline,
            pred_challenger=y_pred_challenger
        )

        return ComparisonResult(
            baseline_model=baseline.model_name,
            challenger_model=challenger.model_name,
            asset=baseline.asset,
            rmse_baseline=baseline.rmse,
            rmse_challenger=challenger.rmse,
            rmse_diff=challenger.rmse - baseline.rmse,
            mae_baseline=baseline.mae,
            mae_challenger=challenger.mae,
            mae_diff=challenger.mae - baseline.mae,
            dm_statistic=dm_statistic,
            dm_p_value=dm_p_value
        )

    @classmethod
    def compare_from_timeinput_and_prediction_results(
        cls,
        baseline: EvaluationResult,
        challenger: EvaluationResult,
        y_true: pd.Series,
        pred_baseline: PredictionResult,
        pred_challenger: PredictionResult
    ) -> ComparisonResult:
        comparator = cls()

        converted_pred_baseline = convert_list_to_series(
            [row.timestamp for row in pred_baseline.rows],
            [row.predicted_volatility for row in pred_baseline.rows],
        )
        converted_pred_challenger = convert_list_to_series(
            [row.timestamp for row in pred_challenger.rows],
            [row.predicted_volatility for row in pred_challenger.rows],
        )

        return comparator.compare(
            baseline=baseline,
            challenger=challenger,
            y_true=y_true,
            y_pred_baseline=converted_pred_baseline,
            y_pred_challenger=converted_pred_challenger
        )