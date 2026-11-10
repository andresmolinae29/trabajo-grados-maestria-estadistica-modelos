from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finance_modeling.experiments import network_tuning
from finance_modeling.experiments.network_tuning import (
    candidates,
    is_tunable,
    seed_variability,
    tune_model,
    validation_kwargs,
)
from finance_modeling.schemas import ModelConfig


def make_kwargs(n_volatility: int = 100) -> dict:
    index = pd.date_range("2026-01-01", periods=n_volatility + 252, freq="15min")
    returns = pd.Series(np.arange(len(index), dtype=float), index=index)
    return {
        "returns": returns,
        "innovation": returns,
        "volatility": pd.Series(np.ones(n_volatility), index=index[251:251 + n_volatility]),
    }


class FakeNetwork:

    def __init__(self, errors: dict, n_hidden: int, seed: int, received: list):
        self.y_test = np.zeros(4)
        self.error = errors[(n_hidden, seed)]
        received.append((n_hidden, seed))

    def fit(self):
        return self

    def predict(self):
        return np.full(4, self.error)


def patch_factory(monkeypatch: pytest.MonkeyPatch, errors: dict, received: list) -> None:
    monkeypatch.setattr(
        network_tuning.ModelFactory,
        "create_model",
        lambda model_name, model_config, symbol, **kwargs: FakeNetwork(errors, kwargs["n_hidden"], kwargs["seed"], received),
    )


def test_is_tunable_only_for_networks_with_a_grid() -> None:
    assert is_tunable(ModelConfig(name="PSOQRNN", hyperparameters_list=[{"n_hidden": 2}, {"n_hidden": 3}]))
    assert not is_tunable(ModelConfig(name="PSOQRNN", hyperparameters_list=[{}]))
    assert not is_tunable(ModelConfig(name="GARCH", hyperparameters_list=[{"p": 1}, {"p": 2}]))


def test_validation_kwargs_never_include_test_bars() -> None:
    kwargs = make_kwargs(100)

    inputs = validation_kwargs(kwargs)

    assert len(inputs["volatility"]) == 80
    assert len(inputs["returns"]) == 80 + 252
    assert inputs["volatility"].index[-1] == kwargs["volatility"].index[79]
    assert inputs["returns"].index[-1] < kwargs["volatility"].index[80 + 1]
    assert len(kwargs["volatility"]) == 100


def test_tune_model_selects_lowest_mean_error_not_luckiest_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    errors = {
        (2, 1): 0.01, (2, 2): 0.99,
        (3, 1): 0.30, (3, 2): 0.20,
    }
    received: list = []
    patch_factory(monkeypatch, errors, received)
    config = ModelConfig(name="PSOQRNN", seeds=[1, 2], hyperparameters_list=[{"n_hidden": 2}, {"n_hidden": 3}])

    result = tune_model(config, "BTC-USD", make_kwargs())

    assert result.hyperparameters == {"n_hidden": 3}
    assert result.seed == 2
    assert len(result.table) == 4
    assert result.table["chosen"].sum() == 1
    assert len(received) == 4


def test_tune_model_reuses_decomposition_for_each_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    errors = {(h, s): 0.1 for h in (2, 3) for s in (1, 2)}
    patch_factory(monkeypatch, errors, [])
    decompositions: list[tuple[int, int]] = []
    monkeypatch.setattr(
        network_tuning,
        "ceemdan_features",
        lambda series, seed: decompositions.append((len(series), seed)) or np.zeros((len(series), 2)),
    )
    config = ModelConfig(name="CEEMDAN_LSTM", seeds=[1, 2], hyperparameters_list=[{"n_hidden": 2}, {"n_hidden": 3}])

    tune_model(config, "BTC-USD", make_kwargs(100))

    assert decompositions == [(332, 1), (332, 2)]


def test_seed_variability_uses_only_common_dates() -> None:
    index = pd.date_range("2026-01-01", periods=4, freq="15min")
    y_true = pd.Series([1.0, 1.0], index=index[2:])
    predictions = {
        1: pd.Series([9.0, 9.0, 1.0, 1.0], index=index),
        2: pd.Series([9.0, 9.0, 2.0, 2.0], index=index),
    }

    table = seed_variability(y_true, predictions, chosen_seed=1)

    assert table["rmse"].tolist() == pytest.approx([0.0, 1.0])
    assert table["chosen"].tolist() == [True, False]


def test_candidates_keep_the_grid_and_add_distinct_sampled_configurations() -> None:
    config = ModelConfig(
        name="PSOQRNN",
        hyperparameters_list=[{"n_hidden": 2}, {"n_hidden": 3}],
        search_space={"n_hidden": [1, 2, 3, 4], "v_max": [0.1, 5.0]},
        n_trials=5,
    )

    options = candidates(config)

    assert options[:2] == [{"n_hidden": 2}, {"n_hidden": 3}]
    assert len(options) == 7
    assert all(option["n_hidden"] in [1, 2, 3, 4] and option["v_max"] in [0.1, 5.0] for option in options[2:])
    assert len({tuple(option.items()) for option in options[2:]}) == 5
    assert candidates(config) == options
    assert is_tunable(ModelConfig(name="PSOQRNN", hyperparameters_list=[{}], search_space={"n_hidden": [1, 2]}, n_trials=1))


def test_tune_model_causal_decomposes_full_series_once(monkeypatch: pytest.MonkeyPatch) -> None:
    received: list[int] = []
    monkeypatch.setattr(
        network_tuning.ModelFactory,
        "create_model",
        lambda model_name, model_config, symbol, **kwargs: received.append(len(kwargs["imfs"])) or FakeNetwork(
            {(h, s): 0.1 for h in (2, 3) for s in (1, 2)}, kwargs["n_hidden"], kwargs["seed"], []),
    )
    decompositions: list[int] = []
    monkeypatch.setattr(
        network_tuning,
        "causal_ceemdan_features",
        lambda series, symbol: decompositions.append(len(series)) or np.zeros((len(series), 2)),
    )
    monkeypatch.setattr(network_tuning, "ceemdan_features", lambda series, seed: pytest.fail("non-causal decomposition"))
    config = ModelConfig(name="CEEMDAN_LSTM", causal=True, seeds=[1, 2], hyperparameters_list=[{"n_hidden": 2}, {"n_hidden": 3}])

    tune_model(config, "BTC-USD", make_kwargs(100))

    assert decompositions == [352]
    assert received == [332] * 4
