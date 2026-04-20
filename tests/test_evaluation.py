from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from finance_modeling.evaluation import Evaluator, Metrics
from finance_modeling.schemas import EvaluationResult, PredictionResult, PredictionRow, TimeSeriesInput
from finance_modeling.schemas.data import AssetMetadata, AssetType


def make_asset_metadata() -> AssetMetadata:
    return AssetMetadata(
        symbol="BTC-USD",
        asset_type=AssetType.CRYPTO,
        description="Bitcoin",
        data_folder="bitcoin",
    )


def make_time_series_input(test_values: list[float]) -> TimeSeriesInput:
    index = pd.date_range("2026-01-01", periods=len(test_values), freq="15min")
    return TimeSeriesInput(
        metadata=make_asset_metadata(),
        series=pd.Series(dtype=float),
        log_returns=pd.Series(dtype=float),
        test=pd.Series(test_values, index=index, dtype=float),
    )


def make_prediction_result(values: list[float]) -> PredictionResult:
    index = pd.date_range("2026-01-01", periods=len(values), freq="15min")
    rows = [
        PredictionRow(timestamp=timestamp, predicted_volatility=value)
        for timestamp, value in zip(index, values)
    ]
    return PredictionResult(
        model_name="PSO-QRNN",
        asset="BTC-USD",
        horizon=len(values),
        rows=rows,
    )


def test_mean_absolute_error_returns_expected_value() -> None:
    mae = Metrics.mean_absolute_error([1.0, 2.0, 3.0], [1.5, 1.0, 2.5])

    assert mae == pytest.approx((0.5 + 1.0 + 0.5) / 3)


def test_mean_squared_error_returns_expected_value() -> None:
    mse = Metrics.mean_squared_error([1.0, 2.0], [3.0, 4.0])

    assert mse == pytest.approx(4.0)


def test_root_mean_squared_error_returns_expected_value() -> None:
    rmse = Metrics.root_mean_squared_error([1.0, 2.0], [3.0, 4.0])

    assert rmse == pytest.approx(2.0)


def test_diebold_mariano_test_delegates_to_library(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_dm_test(y_true, y_pred1, y_pred2, h, variance_estimator):
        captured["args"] = (y_true, y_pred1, y_pred2, h, variance_estimator)
        return 1.23, 0.04

    monkeypatch.setattr("finance_modeling.evaluation.metrics.dm_test", fake_dm_test)

    statistic, p_value = Metrics.diebold_mariano_test(
        pd.Series([1.0, 2.0]),
        pd.Series([1.1, 1.9]),
        pd.Series([0.9, 2.2]),
    )

    assert (statistic, p_value) == (1.23, 0.04)
    assert captured["args"] == ([1.0, 2.0], [1.1, 1.9], [0.9, 2.2], 1, "acf")


def test_evaluate_returns_evaluation_result() -> None:
    evaluation = Evaluator().evaluate(
        model_name="GARCH",
        asset="BTC-USD",
        y_true=pd.Series([1.0, 2.0]),
        y_pred=pd.Series([1.5, 2.5]),
    )

    assert isinstance(evaluation, EvaluationResult)
    assert evaluation.model_name == "GARCH"
    assert evaluation.asset == "BTC-USD"
    assert evaluation.mae == pytest.approx(0.5)
    assert evaluation.rmse == pytest.approx(0.5)


def test_from_timeinput_and_prediction_result_uses_prediction_rows() -> None:
    time_input = make_time_series_input([1.0, 2.0, 3.0])
    prediction_result = make_prediction_result([1.0, 1.5, 3.5])

    evaluation = Evaluator.from_timeinput_and_prediction_result(time_input, prediction_result)

    assert evaluation.model_name == "PSO-QRNN"
    assert evaluation.asset == "BTC-USD"
    assert evaluation.mae == pytest.approx((0.0 + 0.5 + 0.5) / 3)
    assert evaluation.rmse == pytest.approx(((0.0**2 + 0.5**2 + 0.5**2) / 3) ** 0.5)


def test_save_evaluation_results_writes_json_file(tmp_path: Path) -> None:
    evaluation = EvaluationResult(model_name="GARCH", asset="BTC-USD", rmse=0.3, mae=0.2)

    Evaluator.save_evaluation_results(str(tmp_path), evaluation)

    saved_file = tmp_path / "GARCH_BTC-USD_evaluation.json"
    assert saved_file.exists()
    saved_payload = json.loads(saved_file.read_text(encoding="utf-8"))
    assert saved_payload == {
        "model_name": "GARCH",
        "asset": "BTC-USD",
        "rmse": 0.3,
        "mae": 0.2,
        "versus_baseline": True,
    }