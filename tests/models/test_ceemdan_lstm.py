from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from finance_modeling.models.ceemdan_lstm import (
    CEEMDANLSTM,
    CEEMDANLSTMModel,
    causal_ceemdan_features,
    ceemdan_features,
    create_sequences,
)
from finance_modeling.schemas import ModelConfig


def fake_ceemdan_features(series: pd.Series, seed: int = 42) -> np.ndarray:
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


def test_inputs_of_each_window_end_one_bar_before_target(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_lightweight_hyperparameters(monkeypatch)
    n_volatility = 40
    n_returns = n_volatility + 252
    bars = np.arange(n_returns, dtype=float)
    volatility_bars = 251 + np.arange(n_volatility, dtype=float)
    index = pd.date_range("2026-01-01", periods=n_returns, freq="15min")

    model = CEEMDANLSTMModel(
        config=ModelConfig(name="CEEMDAN_LSTM"),
        symbol="BTC-USD",
        returns=pd.Series(bars, index=index),
        volatility=pd.Series(volatility_bars, index=index[251:291]),
        imfs=bars.reshape(-1, 1),
    )

    X_raw, y_raw = model._build_raw_dataset(bars.reshape(-1, 1))
    X_windows, y_windows = create_sequences(X_raw, y_raw, seq_length=5)

    np.testing.assert_array_equal(X_raw[:, 0], X_raw[:, 1])
    np.testing.assert_array_equal(X_windows[:, -1, 0], y_windows - 1)
    np.testing.assert_array_equal(X_windows[:, -1, 1], y_windows - 1)


def test_hyperparameters_and_seed_can_be_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_lightweight_hyperparameters(monkeypatch)
    seeds: list[int] = []

    def recording_features(series: pd.Series, seed: int = 42) -> np.ndarray:
        seeds.append(seed)
        return fake_ceemdan_features(series, seed)

    monkeypatch.setattr("finance_modeling.models.ceemdan_lstm.ceemdan_features", recording_features)
    model = CEEMDANLSTMModel(
        config=ModelConfig(name="CEEMDAN_LSTM"),
        symbol="BTC-USD",
        returns=make_returns(40),
        volatility=make_volatility(40),
        window_size=3,
        epochs=2,
        seed=7,
    )

    assert seeds == [7]
    assert model.epochs == 2
    assert next(iter(model.train_loader))[0].shape[1] == 3


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


def test_ceemdan_features_puts_components_in_columns_and_uses_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    received: dict = {}

    class FakeCEEMDAN:
        def __init__(self, seed: int):
            received["seed"] = seed

        def __call__(self, values: np.ndarray) -> np.ndarray:
            received["values"] = values
            return np.vstack([values, values * 2, values * 3])

    monkeypatch.setattr("finance_modeling.models.ceemdan_lstm.CEEMDAN", FakeCEEMDAN)
    series = pd.Series([1.0, 2.0, 3.0, 4.0])

    features = ceemdan_features(series, seed=7)

    assert received["seed"] == 7
    np.testing.assert_array_equal(received["values"], series.values)
    assert features.shape == (4, 3)
    np.testing.assert_array_equal(features[:, 2], series.values * 3)


def test_fit_logs_loss_every_ten_epochs(monkeypatch: pytest.MonkeyPatch) -> None:
    model = make_model(monkeypatch)
    model.epochs = 10
    messages: list[str] = []
    monkeypatch.setattr("finance_modeling.models.ceemdan_lstm.logger.info", messages.append)

    model.fit()

    assert len(messages) == 1
    assert messages[0].startswith("Epoch 10/10")


def test_architecture_and_training_settings_can_be_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_lightweight_hyperparameters(monkeypatch)

    model = CEEMDANLSTMModel(
        config=ModelConfig(name="CEEMDAN_LSTM"),
        symbol="BTC-USD",
        returns=make_returns(40),
        volatility=make_volatility(40),
        hidden=16,
        layers=1,
        dropout=0.3,
        dense=8,
        lr=0.01,
        weight_decay=0.0001,
        batch_size=2,
        shuffle=True,
    )

    assert (model.model.lstm.hidden_size, model.model.lstm.num_layers, model.model.lstm.dropout) == (16, 1, 0.0)
    assert model.model.fc[0].out_features == 8
    assert (model.lr, model.weight_decay) == (0.01, 0.0001)
    assert model.train_loader.batch_size == 2
    assert isinstance(model.train_loader.sampler, torch.utils.data.RandomSampler)


def test_causal_features_use_only_the_window_that_ends_at_each_bar(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    calls: list[int] = []

    def fake_components(window: np.ndarray, max_imf: int, seed: int) -> np.ndarray:
        calls.append(len(window))
        return np.array([window.sum(), window[-1]])

    monkeypatch.setattr("finance_modeling.models.ceemdan_lstm.last_components", fake_components)
    returns = make_returns(30)
    values = returns.to_numpy()

    features = causal_ceemdan_features(returns, "BTC-USD", window=4, max_imf=1, processes=1, cache_dir=str(tmp_path))

    assert np.isnan(features[:3]).all()
    np.testing.assert_allclose(features[3:, 0], [values[t - 3:t + 1].sum() for t in range(3, 30)])
    np.testing.assert_allclose(features[3:, 1], values[3:])
    assert calls == [4] * 27

    prefix = causal_ceemdan_features(returns.iloc[:20], "BTC-USD", window=4, max_imf=1, processes=1, cache_dir=str(tmp_path))

    np.testing.assert_array_equal(prefix, features[:20])
    assert len(calls) == 27


def test_train_context_forecasts_every_test_row(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_lightweight_hyperparameters(monkeypatch)
    model = CEEMDANLSTMModel(
        config=ModelConfig(name="CEEMDAN_LSTM", train_context=True),
        symbol="BTC-USD",
        returns=make_returns(40),
        volatility=make_volatility(40),
    )

    X_raw, y_raw = model._build_raw_dataset()
    split = int(len(X_raw) * 0.8)
    train_rows = model.x_scaler.transform(X_raw[:split])

    assert len(model.test_loader.dataset) == len(X_raw) - split
    np.testing.assert_allclose(model.y_test.ravel(), y_raw[split:], rtol=1e-5)
    np.testing.assert_allclose(model.test_loader.dataset.X[0].numpy(), train_rows[-5:], rtol=1e-5, atol=1e-6)


def test_causal_model_trains_only_on_rows_with_complete_decomposition(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_lightweight_hyperparameters(monkeypatch)
    n_volatility = 60
    returns = make_returns(n_volatility + 252)
    features = np.ones((len(returns), 3))
    features[:262] = np.nan
    requested: list[str] = []

    def fake_causal(series: pd.Series, symbol: str) -> np.ndarray:
        requested.append(symbol)
        return features[:len(series)]

    monkeypatch.setattr("finance_modeling.models.ceemdan_lstm.causal_ceemdan_features", fake_causal)
    model = CEEMDANLSTMModel(
        config=ModelConfig(name="CEEMDAN_LSTM", causal=True),
        symbol="ETH-USD",
        returns=returns,
        volatility=make_volatility(n_volatility),
    )

    split = int((n_volatility - 1) * 0.8)
    assert requested == ["ETH-USD"]
    assert len(model.train_loader.dataset) == split - 10 - CEEMDANLSTMModel.WINDOW_SIZE
    assert not torch.isnan(model.train_loader.dataset.X).any()
