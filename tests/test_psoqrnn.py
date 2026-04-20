from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

import pytest

from finance_modeling.models.psoqrnn import PSOQRNNModel
from finance_modeling.schemas import ModelConfig, PredictionResult
from finance_modeling.schemas.data import AssetMetadata, AssetType


def make_model(hyperparameters_list: list[dict] | None = None) -> PSOQRNNModel:
    return PSOQRNNModel(
        config=ModelConfig(name="PSOQRNN", hyperparameters_list=hyperparameters_list or [{}]),
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
            "quantiles": [0.1, 0.5, 0.9],
        }
    ])
    train_series = make_series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    selected_hyperparameters = {
        "window_size": 4,
        "hidden_size": 8,
        "num_layers": 1,
        "quantiles": [0.1, 0.5, 0.9],
    }
    training_dataset = TensorDataset(
        torch.zeros((2, 4, 1), dtype=torch.float32),
        torch.zeros((2, 1), dtype=torch.float32),
    )
    fake_trained_model = SimpleNamespace(name="trained")
    call_log: list[str] = []

    monkeypatch.setattr(model, "_PSOQRNNModel__set_random_seed", lambda: call_log.append("seed"))
    monkeypatch.setattr(model, "_PSOQRNNModel__select_hyperparameters", lambda X: selected_hyperparameters)
    monkeypatch.setattr(model, "_PSOQRNNModel__resolve_window_size", lambda series_length, hyperparameters: 4)
    monkeypatch.setattr(model, "_PSOQRNNModel__resolve_quantiles", lambda hyperparameters: [0.1, 0.5, 0.9])
    monkeypatch.setattr(model, "_PSOQRNNModel__build_training_dataset", lambda X, hyperparameters: training_dataset)
    monkeypatch.setattr(model, "_PSOQRNNModel__train_model", lambda dataset, hyperparameters: fake_trained_model)

    model.fit(train_series)

    assert call_log == ["seed"]
    assert model.best_hyperparameters == selected_hyperparameters
    assert model.window_size == 4
    assert model.quantiles == [0.1, 0.5, 0.9]
    pd.testing.assert_series_equal(model.train_series, train_series) # type: ignore
    assert model.training_dataset is training_dataset
    assert model.model is fake_trained_model
    assert model.model_state == {
        "window_size": 4,
        "quantiles": [0.1, 0.5, 0.9],
        "asset": "BTC-USD",
        "device": model.device,
        "training_windows": 2,
        "hidden_size": 8,
        "num_layers": 1,
    }
    assert model.is_fitted is True


def test_predict_raises_before_fit() -> None:
    model = make_model()

    with pytest.raises(ValueError, match="Model must be fitted before prediction"):
        model.predict(make_series([0.1, 0.2]), make_series([0.3]))


def test_predict_maps_forecast_matrix_to_prediction_result(monkeypatch: pytest.MonkeyPatch) -> None:
    model = make_model()
    train = make_series([0.1, 0.2, 0.3, 0.4])
    test = make_series([0.5, 0.6])
    model.is_fitted = True
    setattr(model, "model", SimpleNamespace(name="trained"))
    model.quantiles = [0.1, 0.5, 0.9]
    monkeypatch.setattr(
        model,
        "_PSOQRNNModel__forecast_quantiles",
        lambda X, horizon: np.asarray(
            [
                [0.05, 0.10, 0.15],
                [0.15, 0.20, 0.25],
            ],
            dtype=np.float32,
        ),
    )

    prediction = model.predict(train, test)

    assert isinstance(prediction, PredictionResult)
    assert prediction.model_name == "PSO-QRNN"
    assert prediction.asset == "BTC-USD"
    assert prediction.horizon == 2
    assert [row.timestamp for row in prediction.rows] == list(test.index)
    assert [row.predicted_volatility for row in prediction.rows] == pytest.approx([0.10, 0.20])
    assert [row.lower_ci for row in prediction.rows] == pytest.approx([0.05, 0.15])
    assert [row.upper_ci for row in prediction.rows] == pytest.approx([0.15, 0.25])