from __future__ import annotations

import warnings

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from arch.utility.exceptions import ConvergenceWarning
from pydantic import ValidationError

from finance_modeling.data import DataPreprocessor
from finance_modeling.models.garch import GARCHModel
from finance_modeling.schemas import AssetMetadata, AssetType, GarchSelection, ModelConfig, TimeSeriesInput


def _asset() -> AssetMetadata:
    return AssetMetadata(symbol="BTC-USD", asset_type=AssetType.CRYPTO, description="Bitcoin", data_folder="bitcoin")


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
        lambda self, X: SimpleNamespace(resid=lambda: np.arange(10, dtype=float)),
    )

    model.fit()

    assert model.best_hyperparameters == {"p": 1, "q": 2}
    assert model.is_fitted is True
    assert model.model.aic == 9.0


def _patch_fit_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "finance_modeling.models.garch.arch.arch_model",
        lambda residuals, vol="GARCH", rescale=False, mean="Zero", **hp: _FakeArchResult(10.0 * hp["q"]),
    )
    monkeypatch.setattr(
        GARCHModel,
        "_GARCHModel__fit_mean_model",
        lambda self, X: SimpleNamespace(resid=lambda: np.arange(10, dtype=float)),
    )


def _selection(*hyperparameters: dict) -> GarchSelection:
    return GarchSelection(
        symbol="BTC-USD",
        candidates=[{"hyperparameters": hp, "justification": "Razón."} for hp in hyperparameters],
    )


def test_fit_lets_aic_decide_only_among_preselected_models(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_fit_dependencies(monkeypatch)
    candidates = [{"p": 1, "q": 1}, {"p": 1, "q": 2}, {"p": 1, "q": 3}]
    model = GARCHModel(
        config=ModelConfig(name="GARCH", hyperparameters_list=candidates),
        symbol="BTC-USD",
        prices=make_prices(),
        selection=_selection({"p": 1, "q": 2}, {"p": 1, "q": 3}),
    )

    model.fit()

    assert model.best_hyperparameters == {"p": 1, "q": 2}


def test_fit_rejects_preselected_model_outside_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_fit_dependencies(monkeypatch)
    model = GARCHModel(
        config=ModelConfig(name="GARCH", hyperparameters_list=[{"p": 1, "q": 1}]),
        symbol="BTC-USD",
        prices=make_prices(),
        selection=_selection({"p": 1, "q": 1}, {"p": 3, "q": 1}),
    )

    with pytest.raises(ValueError, match="is not a candidate"):
        model.fit()


def test_selection_requires_two_models_with_justification() -> None:
    with pytest.raises(ValidationError):
        _selection({"p": 1, "q": 1})
    with pytest.raises(ValidationError):
        GarchSelection(
            symbol="BTC-USD",
            candidates=[
                {"hyperparameters": {"p": 1, "q": 1}, "justification": ""},
                {"hyperparameters": {"p": 1, "q": 2}, "justification": "Razón."},
            ],
        )


def test_fit_scales_residuals_to_unit_variance(monkeypatch: pytest.MonkeyPatch) -> None:
    received: list[float] = []

    def fake_arch_model(residuals, vol="GARCH", rescale=False, mean="Zero", **hp):
        received.append(float(np.std(residuals, ddof=1)))
        return _FakeArchResult(1.0)

    monkeypatch.setattr("finance_modeling.models.garch.arch.arch_model", fake_arch_model)
    monkeypatch.setattr(
        GARCHModel,
        "_GARCHModel__fit_mean_model",
        lambda self, X: SimpleNamespace(resid=lambda: np.random.default_rng(0).normal(0, 0.002, 100)),
    )
    model = make_model([{"p": 1, "q": 1}])

    model.fit()

    assert received == pytest.approx([1.0, 1.0])
    assert model.scale == pytest.approx(500, rel=0.2)


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
        lambda self, X: SimpleNamespace(resid=lambda: np.arange(10, dtype=float)),
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
    n_train_resid = GARCHModel.OMEGA  # residuos simulados de entrenamiento

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


def test_predict_averages_conditional_variance_over_target_window(monkeypatch: pytest.MonkeyPatch) -> None:
    model = make_model([{"p": 1, "q": 1}])
    n_test = len(model.test_returns)
    n_train_resid = GARCHModel.OMEGA

    model.is_fitted = True
    model.best_hyperparameters = {"p": 1, "q": 1}
    model.model = SimpleNamespace(params=pd.Series(dtype=float), resid=np.zeros(n_train_resid, dtype=float))
    monkeypatch.setattr(
        "finance_modeling.models.garch.deepcopy",
        lambda obj: SimpleNamespace(predict=lambda n_periods=1: np.zeros(1), update=lambda v: None),
    )

    # Varianza 1 en la escala del modelo, con un salto a 253 en la última barra.
    cond_vol = np.full(n_train_resid + n_test, model.scale)
    cond_vol[-1] *= np.sqrt(GARCHModel.OMEGA + 1)
    monkeypatch.setattr(
        "finance_modeling.models.garch.arch.arch_model",
        lambda residuals, vol="GARCH", rescale=False, mean="Zero", **hp: SimpleNamespace(
            fix=lambda params: SimpleNamespace(conditional_volatility=cond_vol)
        ),
    )

    result = model.predict()

    np.testing.assert_allclose(result[:-1], np.sqrt(252))
    np.testing.assert_allclose(result[-1], np.sqrt(2.0) * np.sqrt(252))
    np.testing.assert_allclose(model.point_predictions[-1], np.sqrt(253.0) * np.sqrt(252))


def test_build_arch_model_uses_garch_by_default_and_configured_process() -> None:
    residuals = np.random.default_rng(0).normal(0, 1, 500)

    default = GARCHModel.build_arch_model(residuals, {"p": 1, "q": 1})
    figarch = GARCHModel.build_arch_model(residuals, {"vol": "FIGARCH", "p": 1, "q": 1})

    assert "FIGARCH" not in default.volatility.name
    assert figarch.volatility.name == "FIGARCH"


def test_garch_uses_same_returns_as_preprocessor() -> None:
    prices = make_prices()
    data = DataPreprocessor(TimeSeriesInput(metadata=_asset(), series=prices.copy())).pipeline()

    model = make_model([{"p": 1, "q": 1}])

    pd.testing.assert_series_equal(model.returns, data.returns, check_names=False, check_freq=False)


def test_garch_is_evaluated_against_received_volatility() -> None:
    prices = make_prices()
    index = prices.index[260:-1]
    volatility = pd.Series(np.linspace(0.01, 0.02, len(index)), index=index)

    model = GARCHModel(
        config=ModelConfig(name="GARCH", hyperparameters_list=[{"p": 1, "q": 1}]),
        symbol="BTC-USD",
        prices=prices,
        volatility=volatility,
    )

    split = int(len(volatility) * 0.8)
    np.testing.assert_array_equal(model.y_test, volatility.iloc[split:].to_numpy())
    assert model.test_returns.index[-1] == volatility.index[-1]
    assert model.train_returns.index[-1] == volatility.index[split - 1]


class _NonConvergingArchResult(_FakeArchResult):
    def fit(self, disp="off", options=None):
        warnings.warn("optimizer did not converge", ConvergenceWarning)
        return self


def test_garch_requires_prices() -> None:
    with pytest.raises(ValueError, match="prices"):
        GARCHModel(config=ModelConfig(name="GARCH", hyperparameters_list=[{}]), symbol="BTC-USD")


def test_fit_discards_candidates_that_do_not_converge(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_arch_model(residuals, vol="GARCH", rescale=False, mean="Zero", **hp):
        if hp == {"p": 1, "q": 1}:
            return _NonConvergingArchResult(1.0)
        return _FakeArchResult(5.0)

    model = make_model([{"p": 1, "q": 1}, {"p": 1, "q": 2}])
    monkeypatch.setattr("finance_modeling.models.garch.arch.arch_model", fake_arch_model)
    monkeypatch.setattr(
        GARCHModel,
        "_GARCHModel__fit_mean_model",
        lambda self, X: SimpleNamespace(resid=lambda: np.arange(10, dtype=float)),
    )

    model.fit()

    assert model.best_hyperparameters == {"p": 1, "q": 2}


def test_fit_raises_when_no_candidate_can_be_fitted(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing_arch_model(residuals, vol="GARCH", rescale=False, mean="Zero", **hp):
        raise RuntimeError("bad candidate")

    model = make_model([{"p": 1, "q": 1}, {"p": 1, "q": 2}])
    monkeypatch.setattr("finance_modeling.models.garch.arch.arch_model", failing_arch_model)
    monkeypatch.setattr(
        GARCHModel,
        "_GARCHModel__fit_mean_model",
        lambda self, X: SimpleNamespace(resid=lambda: np.arange(10, dtype=float)),
    )

    with pytest.raises(RuntimeError, match="No se pudo ajustar"):
        model.fit()
