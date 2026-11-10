import json
import os

import pandas as pd

from .metrics import Metrics
from ..schemas import EvaluationResult, ComparisonResult


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

    @staticmethod
    def save_comparison_results(experiment_path: str, comparison_result: ComparisonResult) -> None:

        file_name = (
            f"{comparison_result.baseline_model}_vs_"
            f"{comparison_result.challenger_model}_"
            f"{comparison_result.asset}_comparison.json"
        )
        with open(os.path.join(experiment_path, file_name), "w") as file_pointer:
            json.dump(comparison_result.model_dump(), file_pointer, indent=4)
