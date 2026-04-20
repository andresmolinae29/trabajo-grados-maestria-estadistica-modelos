from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from finance_modeling.evaluation.comparison import ModelComparator
from finance_modeling.schemas import ComparisonResult, EvaluationResult, PredictionResult, PredictionRow


def make_prediction_result(model_name: str, values: list[float]) -> PredictionResult:
    index = pd.date_range("2026-01-01", periods=len(values), freq="15min")
    return PredictionResult(
        model_name=model_name,
        asset="BTC-USD",
        horizon=len(values),
        rows=[
            PredictionRow(timestamp=timestamp, predicted_volatility=value)
            for timestamp, value in zip(index, values)
        ],
    )


def test_compare_returns_expected_differences(monkeypatch: pytest.MonkeyPatch) -> None:
    baseline = EvaluationResult(model_name="GARCH", asset="BTC-USD", rmse=1.0, mae=0.8)
    challenger = EvaluationResult(model_name="PSO-QRNN", asset="BTC-USD", rmse=0.7, mae=0.5)

    def fake_dm_test(y_true, pred_baseline, pred_challenger):
        assert list(y_true) == [1.0, 2.0, 3.0]
        assert list(pred_baseline) == [1.1, 2.1, 3.1]
        assert list(pred_challenger) == [0.9, 1.9, 2.9]
        return 2.5, 0.03

    monkeypatch.setattr(ModelComparator, "_ModelComparator__get_dm_statistic_and_p_value", staticmethod(fake_dm_test))

    result = ModelComparator().compare(
        baseline=baseline,
        challenger=challenger,
        y_true=pd.Series([1.0, 2.0, 3.0]),
        y_pred_baseline=pd.Series([1.1, 2.1, 3.1]),
        y_pred_challenger=pd.Series([0.9, 1.9, 2.9]),
    )

    assert isinstance(result, ComparisonResult)
    assert result.baseline_model == "GARCH"
    assert result.challenger_model == "PSO-QRNN"
    assert result.asset == "BTC-USD"
    assert result.rmse_baseline == 1.0
    assert result.rmse_challenger == 0.7
    assert result.rmse_diff == pytest.approx(-0.3)
    assert result.mae_baseline == 0.8
    assert result.mae_challenger == 0.5
    assert result.mae_diff == pytest.approx(-0.3)
    assert result.dm_statistic == 2.5
    assert result.dm_p_value == 0.03


def test_compare_from_timeinput_and_prediction_results_converts_prediction_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    baseline = EvaluationResult(model_name="GARCH", asset="BTC-USD", rmse=1.0, mae=0.8)
    challenger = EvaluationResult(model_name="PSO-QRNN", asset="BTC-USD", rmse=0.7, mae=0.5)
    y_true = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2026-01-01", periods=3, freq="15min"))
    baseline_prediction = make_prediction_result("GARCH", [1.1, 2.1, 3.1])
    challenger_prediction = make_prediction_result("PSO-QRNN", [0.9, 1.9, 2.9])
    captured = {}

    def fake_compare(self, baseline, challenger, y_pred_baseline, y_pred_challenger, y_true):
        captured["baseline"] = baseline
        captured["challenger"] = challenger
        captured["y_pred_baseline"] = y_pred_baseline
        captured["y_pred_challenger"] = y_pred_challenger
        captured["y_true"] = y_true
        return ComparisonResult(
            baseline_model=baseline.model_name,
            challenger_model=challenger.model_name,
            asset=baseline.asset,
        )

    monkeypatch.setattr(ModelComparator, "compare", fake_compare)

    result = ModelComparator.compare_from_timeinput_and_prediction_results(
        baseline=baseline,
        challenger=challenger,
        y_true=y_true,
        pred_baseline=baseline_prediction,
        pred_challenger=challenger_prediction,
    )

    assert result.baseline_model == "GARCH"
    assert result.challenger_model == "PSO-QRNN"
    expected_baseline = pd.Series([1.1, 2.1, 3.1], index=y_true.index)
    expected_challenger = pd.Series([0.9, 1.9, 2.9], index=y_true.index)
    pd.testing.assert_series_equal(captured["y_pred_baseline"], expected_baseline, check_freq=False)
    pd.testing.assert_series_equal(captured["y_pred_challenger"], expected_challenger, check_freq=False)
    pd.testing.assert_series_equal(captured["y_true"], y_true)


def test_save_comparison_results_writes_json_file(tmp_path: Path) -> None:
    comparison = ComparisonResult(
        baseline_model="GARCH",
        challenger_model="PSO-QRNN",
        asset="BTC-USD",
        rmse_baseline=1.0,
        rmse_challenger=0.7,
        rmse_diff=-0.3,
        mae_baseline=0.8,
        mae_challenger=0.5,
        mae_diff=-0.3,
        dm_statistic=2.5,
        dm_p_value=0.03,
    )

    ModelComparator.save_comparison_results(str(tmp_path), comparison)

    saved_file = tmp_path / "GARCH_vs_PSO-QRNN_BTC-USD_comparison.json"
    assert saved_file.exists()
    payload = json.loads(saved_file.read_text(encoding="utf-8"))
    assert payload == comparison.model_dump()