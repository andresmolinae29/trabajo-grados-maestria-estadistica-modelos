from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from finance_modeling.models.ceemdan_lstm import CEEMDANLSTM, CEEMDANLSTMModel, create_sequences
from finance_modeling.schemas import ModelConfig


def fake_ceemdan_features(series: pd.Series) -> np.ndarray:
    """Sustituye la descomposición CEEMDAN real (costosa) por 3 pseudo-IMFs deterministas."""
    values = series.to_numpy(dtype=np.float32)
    return np.column_stack([values, values * 0.5, values * -0.5])


def patch_lightweight_hyperparameters(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(CEEMDANLSTMModel, "WINDOW_SIZE", 5)
    monkeypatch.setattr(CEEMDANLSTMModel, "BATCH_SIZE", 4)
    monkeypatch.setattr(CEEMDANLSTMModel, "EPOCHS", 1)
    monkeypatch.setattr("finance_modeling.models.ceemdan_lstm.ceemdan_features", fake_ceemdan_features)


def make_returns(n: int) -> pd.Series:
    index = pd.date_range("2026-01-01", periods=n, freq="15min")
    rng = np.random.default_rng(2)
    return pd.Series(rng.normal(0, 0.01, n), index=index, dtype=float)


def make_volatility(n: int) -> pd.Series:
    index = pd.date_range("2026-01-01", periods=n, freq="15min")
    rng = np.random.default_rng(3)
    return pd.Series(rng.uniform(0.1, 0.5, n), index=index, dtype=float)


def make_model(monkeypatch: pytest.MonkeyPatch, n: int = 40) -> CEEMDANLSTMModel:
    patch_lightweight_hyperparameters(monkeypatch)
    return CEEMDANLSTMModel(
        config=ModelConfig(name="CEEMDAN_LSTM", hyperparameters_list=[{}]),
        symbol="BTC-USD",
        returns=make_returns(n),
        volatility=make_volatility(n),
    )


def test_create_sequences_builds_sliding_windows() -> None:
    X = np.arange(10).reshape(10, 1).astype(np.float32)
    y = np.arange(10).astype(np.float32)

    X_seq, y_seq = create_sequences(X, y, seq_length=3)

    assert X_seq.shape == (7, 3, 1)
    assert y_seq.shape == (7,)
    np.testing.assert_array_equal(X_seq[0].ravel(), [0, 1, 2])
    assert y_seq[0] == 3


def test_ceemdan_lstm_forward_returns_expected_output_shape() -> None:
    net = CEEMDANLSTM(n_features=4)
    features = torch.randn(3, 5, 4)

    output = net(features)

    assert tuple(output.shape) == (3, 1)


def test_init_builds_train_and_test_loaders_with_expected_feature_count(monkeypatch: pytest.MonkeyPatch) -> None:
    model = make_model(monkeypatch, n=40)

    assert isinstance(model.model, CEEMDANLSTM)
    assert model.is_fitted is False

    sample_X, sample_y = next(iter(model.train_loader))
    assert sample_X.shape[1] == CEEMDANLSTMModel.WINDOW_SIZE
    assert sample_X.shape[2] == 4  # 3 pseudo-IMFs + 1 valor de volatilidad
    assert sample_y.shape[-1] == 1
    assert model.y_test.shape[1] == 1


def test_predict_raises_before_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    model = make_model(monkeypatch)

    with pytest.raises(ValueError, match="Model must be fitted before prediction"):
        model.predict()


def test_fit_trains_model_and_sets_is_fitted(monkeypatch: pytest.MonkeyPatch) -> None:
    model = make_model(monkeypatch)

    result = model.fit()

    assert result is model
    assert model.is_fitted is True


def test_predict_returns_array_matching_y_test_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    model = make_model(monkeypatch)
    model.is_fitted = True

    predictions = model.predict()

    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == model.y_test.shape
