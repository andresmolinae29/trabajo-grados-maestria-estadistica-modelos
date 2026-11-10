from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from finance_modeling.models.psoqrnn import PSO, PSOQRNNModel, QRNN, quantile_loss
from finance_modeling.schemas import ModelConfig


def make_volatility(n: int) -> pd.Series:
    index = pd.date_range("2026-01-01", periods=n, freq="15min")
    rng = np.random.default_rng(0)
    return pd.Series(rng.uniform(0.1, 0.5, n), index=index, dtype=float)


def make_innovation(n: int) -> pd.Series:
    index = pd.date_range("2020-01-01", periods=n, freq="15min")
    rng = np.random.default_rng(1)
    return pd.Series(rng.normal(0, 1, n), index=index, dtype=float)


def make_model(volatility: pd.Series | None = None, innovation: pd.Series | None = None) -> PSOQRNNModel:
    return PSOQRNNModel(
        config=ModelConfig(name="PSOQRNN", hyperparameters_list=[{}]),
        symbol="BTC-USD",
        innovation=innovation if innovation is not None else make_innovation(300),
        volatility=volatility if volatility is not None else make_volatility(12),
    )


def test_quantile_loss_computes_pinball_loss() -> None:
    y_true = torch.tensor([1.0, 2.0, 3.0])
    y_pred = torch.tensor([0.5, 2.5, 2.0])

    loss = quantile_loss(y_true, y_pred, tau=0.5)

    assert loss.item() == pytest.approx(1 / 3)


def test_qrnn_forward_returns_tanh_bounded_output_of_expected_shape() -> None:
    qrnn = QRNN(n_hidden=3)
    features = torch.randn(4, 2)

    output = qrnn(features)

    assert tuple(output.shape) == (4, 1)
    assert torch.all(output >= -1) and torch.all(output <= 1)


def test_qrnn_get_weights_and_set_weights_roundtrip() -> None:
    qrnn = QRNN(n_hidden=3)
    original_weights = qrnn.get_weights()
    new_weights = np.arange(len(original_weights), dtype=np.float32) * 0.01

    qrnn.set_weights(new_weights)

    assert qrnn.n_params() == len(original_weights)
    np.testing.assert_allclose(qrnn.get_weights(), new_weights, atol=1e-6)


def test_pso_fitness_matches_manual_quantile_loss() -> None:
    pso = PSO(tau=0.5)
    qrnn = QRNN(n_hidden=2)
    weights = qrnn.get_weights()
    X = torch.randn(5, 2)
    y = torch.randn(5, 1)

    fitness = pso._fitness(weights, qrnn, X, y)

    with torch.no_grad():
        expected = quantile_loss(y, qrnn(X), pso.tau).item()
    assert fitness == pytest.approx(expected)


def test_pso_optimize_returns_best_particle_position() -> None:
    pso = PSO(n_particles=4, max_iter=3, verbose_every=1)
    qrnn = QRNN(n_hidden=2)
    rng = np.random.default_rng(3)
    X_train = rng.normal(size=(6, 2)).astype(np.float32)
    y_train = rng.normal(size=(6,)).astype(np.float32)

    best_weights = pso.optimize(qrnn, X_train, y_train)

    assert isinstance(best_weights, np.ndarray)
    assert best_weights.shape == (qrnn.n_params(),)


def test_init_builds_train_and_test_datasets_from_volatility_and_innovation() -> None:
    model = make_model(volatility=make_volatility(12), innovation=make_innovation(300))

    assert model.X_train.shape == (8, 2)
    assert model.y_train.shape == (8,)
    assert model.X_test.shape == (3, 2)
    assert model.y_test.shape == (3,)
    assert model.X_train.dtype == np.float32
    assert model.is_fitted is False


def test_predict_raises_before_fit() -> None:
    model = make_model()

    with pytest.raises(ValueError, match="Model must be fitted before prediction"):
        model.predict()


def test_fit_uses_pso_optimize_and_sets_resulting_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    model = make_model()
    known_weights = np.linspace(0.0, 1.0, model.qrnn.n_params(), dtype=np.float32)

    def fake_optimize(self, qrnn, X_train, y_train):
        assert qrnn is model.qrnn
        np.testing.assert_array_equal(X_train, model.X_train)
        np.testing.assert_array_equal(y_train, model.y_train)
        return known_weights

    monkeypatch.setattr(PSO, "optimize", fake_optimize)

    result = model.fit()

    assert result is model
    assert model.is_fitted is True
    np.testing.assert_allclose(model.qrnn.get_weights(), known_weights, atol=1e-6)


def test_predict_returns_array_matching_test_set_length() -> None:
    model = make_model()
    model.is_fitted = True

    predictions = model.predict()

    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (len(model.X_test),)
    assert np.all(predictions >= -1) and np.all(predictions <= 1)


def test_swarm_fitness_matches_fitness_of_each_particle() -> None:
    pso = PSO(tau=0.5)
    qrnn = QRNN(n_hidden=3)
    rng = np.random.default_rng(4)
    swarm = rng.uniform(-1, 1, (5, qrnn.n_params()))
    X = torch.as_tensor(rng.normal(size=(20, 2)).astype(np.float32))
    y = torch.as_tensor(rng.normal(size=20).astype(np.float32))

    batched = pso._swarm_fitness(swarm, X, y, n_hidden=3)
    one_by_one = [pso._fitness(particle.astype(np.float32), qrnn, X, y.unsqueeze(1)) for particle in swarm]

    np.testing.assert_allclose(batched, one_by_one, rtol=1e-5)


def test_pso_hyperparameters_can_be_configured() -> None:
    model = PSOQRNNModel(
        config=ModelConfig(name="PSOQRNN"),
        symbol="BTC-USD",
        innovation=make_innovation(300),
        volatility=make_volatility(12),
        n_particles=7,
        max_iter=9,
        inertia=0.5,
        c1=1.0,
        c2=1.5,
        v_max=0.1,
        init="symmetric",
    )

    assert (model.pso.n_particles, model.pso.max_iter, model.pso.inertia) == (7, 9, 0.5)
    assert (model.pso.c1, model.pso.c2, model.pso.v_min, model.pso.v_max) == (1.0, 1.5, -0.1, 0.1)
    assert model.pso.init == "symmetric"


def test_causal_dataset_uses_last_known_return_and_training_mean() -> None:
    n_volatility = 12
    returns = make_innovation(n_volatility + 252)
    volatility = make_volatility(n_volatility)

    model = PSOQRNNModel(
        config=ModelConfig(name="PSOQRNN", causal=True),
        symbol="BTC-USD",
        returns=returns,
        volatility=volatility,
    )

    r = returns.to_numpy()
    split = int((n_volatility - 1) * 0.8)
    expected_eps = r[251:251 + n_volatility - 1] - r[:split + 252].mean()
    np.testing.assert_allclose(model.X_train[:, 1], expected_eps[:split], rtol=1e-5)
    np.testing.assert_allclose(model.X_test[:, 1], expected_eps[split:], rtol=1e-5)
    np.testing.assert_allclose(model.X_train[:, 0], volatility.to_numpy()[:split], rtol=1e-6)
    np.testing.assert_allclose(model.y_test, volatility.to_numpy()[split + 1:], rtol=1e-6)
