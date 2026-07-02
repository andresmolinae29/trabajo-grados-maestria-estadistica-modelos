from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from finance_modeling.models.garch import GARCHModel
from finance_modeling.schemas import ModelConfig


def make_prices(n: int = 400) -> pd.Series:
    """Serie de precios suficiente para la ventana rolling de 252 periodos."""
    rng = np.random.default_rng(42)
    log_r = rng.normal(0, 0.005, n)
    prices = 100.0 * np.exp(np.cumsum(log_r))
    return pd.Series(prices, index=pd.date_range("2024-01-01", periods=n, freq="15min"), dtype=float)


def make_model(hyperparameters_list: list[dict]) -> GARCHModel:
    return GARCHModel(
        config=ModelConfig(name="GARCH", hyperparameters_list=hyperparameters_list),
        symbol="BTC-USD",
        prices=make_prices(),
    )


class _FakeArchResult:
    """Resultado falso de arch_model compatible con fit() y fix()."""
    def __init__(self, aic: float, n: int = 10):
        self.aic = aic
        self.params = pd.Series({"omega": 0.01, "alpha[1]": 0.1, "beta[1]": 0.8})
        self.resid = np.zeros(n, dtype=float)
        self.conditional_volatility = np.ones(n, dtype=float) * 0.5

    def fit(self, disp="off", options=None):
        return self

    def fix(self, params):
        return self



def test_fit_selects_hyperparameters_with_lowest_aic(monkeypatch: pytest.MonkeyPatch) -> None:
    candidate_aic: dict[frozenset, float] = {
        frozenset({"p": 1, "q": 1}.items()): 15.0,
        frozenset({"p": 1, "q": 2}.items()): 9.0,
        frozenset({"p": 2, "q": 1}.items()): 12.0,
    }

    def fake_arch_model(residuals, vol="GARCH", rescale=False, mean="Zero", **hp):
        aic = candidate_aic[frozenset(hp.items())]
        return _FakeArchResult(aic)

    model = make_model([{"p": 1, "q": 1}, {"p": 1, "q": 2}, {"p": 2, "q": 1}])
    monkeypatch.setattr("finance_modeling.models.garch.arch.arch_model", fake_arch_model)
    monkeypatch.setattr(
        GARCHModel,
        "_GARCHModel__fit_mean_model",
        lambda self, X: SimpleNamespace(resid=lambda: np.zeros(10, dtype=float)),
    )

    model.fit()

    assert model.best_hyperparameters == {"p": 1, "q": 2}
    assert model.is_fitted is True
    assert model.model.aic == 9.0


def test_fit_skips_failing_hyperparameter_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_arch_model(residuals, vol="GARCH", rescale=False, mean="Zero", **hp):
        if hp == {"p": 1, "q": 1}:
            raise RuntimeError("bad candidate")
        return _FakeArchResult(7.0)

    model = make_model([{"p": 1, "q": 1}, {"p": 1, "q": 2}])
    monkeypatch.setattr("finance_modeling.models.garch.arch.arch_model", fake_arch_model)
    monkeypatch.setattr(
        GARCHModel,
        "_GARCHModel__fit_mean_model",
        lambda self, X: SimpleNamespace(resid=lambda: np.zeros(10, dtype=float)),
    )

    model.fit()

    assert model.best_hyperparameters == {"p": 1, "q": 2}
    assert model.is_fitted is True


def test_predict_raises_before_fit() -> None:
    model = make_model([{"p": 1, "q": 1}])

    with pytest.raises(ValueError, match="Model must be fitted before prediction"):
        model.predict()


def test_predict_returns_ndarray_matching_test_period_length(monkeypatch: pytest.MonkeyPatch) -> None:
    model = make_model([{"p": 1, "q": 1}])
    n_test = len(model.test_returns)
    n_train_resid = 10  # residuos simulados de entrenamiento

    model.is_fitted = True
    model.best_hyperparameters = {"p": 1, "q": 1}
    model.model = SimpleNamespace(
        params=pd.Series({"omega": 0.01, "alpha[1]": 0.1, "beta[1]": 0.8}),
        resid=np.zeros(n_train_resid, dtype=float),
    )

    fake_mean_model = SimpleNamespace(
        predict=lambda n_periods=1: np.asarray([0.001], dtype=float),
        update=lambda v: None,
    )
    monkeypatch.setattr("finance_modeling.models.garch.deepcopy", lambda obj: fake_mean_model)

    total_len = n_train_resid + n_test
    fake_cond_vol = np.abs(np.random.default_rng(0).normal(1.0, 0.1, total_len))
    monkeypatch.setattr(
        "finance_modeling.models.garch.arch.arch_model",
        lambda residuals, vol="GARCH", rescale=False, mean="Zero", **hp: SimpleNamespace(
            fix=lambda params: SimpleNamespace(conditional_volatility=fake_cond_vol[: len(residuals)])
        ),
    )

    result = model.predict()

    assert isinstance(result, np.ndarray)
    assert len(result) == n_test
    assert np.all(result > 0)