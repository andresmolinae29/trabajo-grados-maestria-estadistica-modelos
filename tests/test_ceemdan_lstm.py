from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

import pytest

from finance_modeling.models.ceemdan_lstm import CEEMDANLSTMModel
from finance_modeling.schemas import ModelConfig, PredictionResult
from finance_modeling.schemas.data import AssetMetadata, AssetType


def make_model(hyperparameters_list: list[dict] | None = None) -> CEEMDANLSTMModel:
    return CEEMDANLSTMModel(
        config=ModelConfig(name="CEEMDAN_LSTM", hyperparameters_list=hyperparameters_list or [{}]),
        asset_metadata=AssetMetadata(
            symbol="BTC-USD",
            asset_type=AssetType.CRYPTO,
            description="Bitcoin",
            data_folder="bitcoin",
        ),
    )


def make_series(values: list[float]) -> pd.Series:
    index = pd.date_range("2026-01-01", periods=len(values), freq="15min")
    return pd.Series(values, index=index, dtype=float)


def test_fit_updates_state_from_selected_hyperparameters(monkeypatch: pytest.MonkeyPatch) -> None:
    model = make_model([
        {
            "window_size": 4,
            "hidden_size": 8,
            "num_layers": 1,
        }
    ])
    train_series = make_series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    selected_hyperparameters = {
        "window_size": 4,
        "hidden_size": 8,
        "num_layers": 1,
    }
    train_imfs = [
        np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        np.asarray([0.05, 0.1, 0.15, 0.2], dtype=np.float32),
    ]
    trained_models = [SimpleNamespace(name="imf_1"), SimpleNamespace(name="imf_2")]
    call_log: list[str] = []

    monkeypatch.setattr(model, "_CEEMDANLSTMModel__set_random_seed", lambda: call_log.append("seed"))
    monkeypatch.setattr(model, "_CEEMDANLSTMModel__select_hyperparameters", lambda X: selected_hyperparameters)
    monkeypatch.setattr(model, "_CEEMDANLSTMModel__resolve_window_size", lambda series_length, hyperparameters: 4)
    monkeypatch.setattr(model, "_CEEMDANLSTMModel__decompose_series", lambda X: train_imfs)

    train_calls: list[tuple[int, int, int]] = []

    def fake_train_single_imf_model(imf, hyperparameters, imf_index=None, total_imfs=None):
        train_calls.append((len(imf), imf_index, total_imfs))
        return trained_models[imf_index - 1]  # type: ignore[index]

    monkeypatch.setattr(model, "_CEEMDANLSTMModel__train_single_imf_model", fake_train_single_imf_model)

    model.fit(train_series)

    assert call_log == ["seed"]
    assert model.best_hyperparameters == selected_hyperparameters
    assert model.window_size == 4
    assert model.train_imfs == train_imfs
    assert model.models == trained_models
    assert train_calls == [(4, 1, 2), (4, 2, 2)]
    assert model.is_fitted is True


def test_predict_raises_before_fit() -> None:
    model = make_model()

    with pytest.raises(ValueError, match="Model must be fitted before prediction"):
        model.predict(make_series([0.1, 0.2]), make_series([0.3]))


def test_predict_aggregates_component_forecasts_into_prediction_result(monkeypatch: pytest.MonkeyPatch) -> None:
    model = make_model()
    train = make_series([0.1, 0.2, 0.3, 0.4])
    test = make_series([0.5, 0.6])
    model.is_fitted = True
    model.window_size = 4
    model.models = [SimpleNamespace(name="imf_1"), SimpleNamespace(name="imf_2")]
    model.train_imfs = [
        np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        np.asarray([0.05, 0.1, 0.15, 0.2], dtype=np.float32),
    ]

    component_outputs = [
        np.asarray([0.10, 0.20], dtype=np.float32),
        np.asarray([0.02, 0.03], dtype=np.float32),
    ]
    forecast_calls: list[tuple[int, int, int]] = []

    def fake_forecast_single_imf(model_instance, imf, horizon, window_size):
        forecast_calls.append((len(imf), horizon, window_size))
        return component_outputs[len(forecast_calls) - 1]

    monkeypatch.setattr(model, "_CEEMDANLSTMModel__forecast_single_imf", fake_forecast_single_imf)

    prediction = model.predict(train, test)

    assert isinstance(prediction, PredictionResult)
    assert prediction.model_name == "CEEMDAN-LSTM"
    assert prediction.asset == "BTC-USD"
    assert prediction.horizon == 2
    assert forecast_calls == [(4, 2, 4), (4, 2, 4)]
    assert [row.timestamp for row in prediction.rows] == list(test.index)
    assert [row.predicted_volatility for row in prediction.rows] == pytest.approx([0.12, 0.23])
    assert all(row.lower_ci is None for row in prediction.rows)
    assert all(row.upper_ci is None for row in prediction.rows)