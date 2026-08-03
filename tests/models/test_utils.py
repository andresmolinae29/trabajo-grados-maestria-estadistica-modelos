from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from finance_modeling.models.utils import (
    build_prediction_result,
    save_forecast_plot,
    save_prediction_results_to_csv,
)
from finance_modeling.schemas import PredictionResult, PredictionRow


def make_vol_index(n: int = 5) -> pd.Index:
    return pd.date_range("2026-01-01", periods=n, freq="15min")


def test_build_prediction_result_aligns_predictions_with_tail_of_vol_index() -> None:
    vol_index = make_vol_index(5)
    predictions = np.array([0.1, 0.2, 0.3])
    y_test = np.array([0.15, 0.25, 0.35])

    result = build_prediction_result(
        model_name="GARCH",
        asset_symbol="BTC-USD",
        predictions=predictions,
        y_test=y_test,
        vol_index=vol_index,
    )

    assert isinstance(result, PredictionResult)
    assert result.model_name == "GARCH"
    assert result.asset == "BTC-USD"
    assert result.horizon == 3
    assert [row.timestamp for row in result.rows] == list(vol_index[-3:])
    assert [row.predicted_value for row in result.rows] == pytest.approx([0.1, 0.2, 0.3])
    assert [row.real_value for row in result.rows] == pytest.approx([0.15, 0.25, 0.35])


def test_build_prediction_result_leaves_real_value_none_when_y_test_is_none() -> None:
    vol_index = make_vol_index(3)
    predictions = np.array([0.1, 0.2])

    result = build_prediction_result(
        model_name="GARCH",
        asset_symbol="BTC-USD",
        predictions=predictions,
        y_test=None,
        vol_index=vol_index,
    )

    assert all(row.real_value is None for row in result.rows)


def test_save_prediction_results_to_csv_writes_expected_columns(tmp_path: Path) -> None:
    vol_index = make_vol_index(3)
    predictions = np.array([0.1, 0.2, 0.3])
    y_test = np.array([0.15, 0.25, 0.35])

    save_prediction_results_to_csv(
        experiment_path=str(tmp_path),
        model_name="GARCH",
        asset_symbol="BTC-USD",
        predictions=predictions,
        y_test=y_test,
        vol_index=vol_index,
    )

    saved_file = tmp_path / "GARCH_BTC-USD_predictions.csv"
    assert saved_file.exists()
    frame = pd.read_csv(saved_file)
    assert frame.columns.tolist() == [
        "model_name",
        "asset_symbol",
        "timestamp",
        "predicted_value",
        "real_value",
    ]
    assert frame["model_name"].tolist() == ["GARCH", "GARCH", "GARCH"]
    assert frame["predicted_value"].tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert frame["real_value"].tolist() == pytest.approx([0.15, 0.25, 0.35])


def test_save_prediction_results_to_csv_writes_empty_real_value_when_y_test_is_none(tmp_path: Path) -> None:
    vol_index = make_vol_index(2)
    predictions = np.array([0.1, 0.2])

    save_prediction_results_to_csv(
        experiment_path=str(tmp_path),
        model_name="GARCH",
        asset_symbol="BTC-USD",
        predictions=predictions,
        y_test=None,
        vol_index=vol_index,
    )

    saved_file = tmp_path / "GARCH_BTC-USD_predictions.csv"
    frame = pd.read_csv(saved_file)
    assert frame["real_value"].isna().all()


def test_save_forecast_plot_writes_png_file(tmp_path: Path) -> None:
    index = make_vol_index(4)
    prediction_result = PredictionResult(
        model_name="GARCH",
        asset="BTC-USD",
        horizon=4,
        rows=[
            PredictionRow(timestamp=index[0], predicted_value=0.1, real_value=None),
            PredictionRow(timestamp=index[1], predicted_value=0.2, real_value=0.25),
            PredictionRow(timestamp=index[2], predicted_value=0.3, real_value=0.28),
            PredictionRow(timestamp=index[3], predicted_value=0.4, real_value=None),
        ],
    )

    save_forecast_plot(prediction_result, str(tmp_path))

    saved_file = tmp_path / "GARCH_BTC-USD_forecast.png"
    assert saved_file.exists()
    assert saved_file.stat().st_size > 0
