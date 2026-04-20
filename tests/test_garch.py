from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from finance_modeling.models.garch import GARCHModel
from finance_modeling.schemas import ModelConfig, PredictionResult
from finance_modeling.schemas.data import AssetMetadata, AssetType


def make_model(hyperparameters_list: list[dict]) -> GARCHModel:
    return GARCHModel(
        config=ModelConfig(name="GARCH", hyperparameters_list=hyperparameters_list),
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


class FakeSetModelResult:
    def __init__(self, aic: float):
        self.aic = aic
        self.fit_calls = 0

    def fit(self, disp: str = "off"):
        self.fit_calls += 1
        return SimpleNamespace(aic=self.aic)


def test_fit_selects_hyperparameters_with_lowest_aic(monkeypatch: pytest.MonkeyPatch) -> None:
    candidate_aic = {
        "{'p': 1, 'q': 1}": 15.0,
        "{'p': 1, 'q': 2}": 9.0,
        "{'p': 2, 'q': 1}": 12.0,
    }
    set_model_calls: list[dict] = []

    def fake_set_model(self, X: pd.Series, hyperparameters: dict):
        set_model_calls.append(hyperparameters)
        return FakeSetModelResult(candidate_aic[str(hyperparameters)])

    model = make_model([
        {"p": 1, "q": 1},
        {"p": 1, "q": 2},
        {"p": 2, "q": 1},
    ])
    monkeypatch.setattr(GARCHModel, "_GARCHModel__set_model", fake_set_model)

    model.fit(make_series([0.1, 0.2, 0.3, 0.4]))

    assert set_model_calls == [
        {"p": 1, "q": 1},
        {"p": 1, "q": 2},
        {"p": 2, "q": 1},
        {"p": 1, "q": 2},
    ]
    assert model.best_hyperparameters == {"p": 1, "q": 2}
    assert model.is_fitted is True
    assert model.model.aic == 9.0


def test_fit_skips_failing_hyperparameter_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_set_model(self, X: pd.Series, hyperparameters: dict):
        if hyperparameters == {"p": 1, "q": 1}:
            raise RuntimeError("bad candidate")
        return FakeSetModelResult(7.0)

    model = make_model([
        {"p": 1, "q": 1},
        {"p": 1, "q": 2},
    ])
    monkeypatch.setattr(GARCHModel, "_GARCHModel__set_model", fake_set_model)

    model.fit(make_series([0.1, 0.2, 0.3, 0.4]))

    assert model.best_hyperparameters == {"p": 1, "q": 2}
    assert model.is_fitted is True


def test_predict_raises_before_fit() -> None:
    model = make_model([{"p": 1, "q": 1}])

    with pytest.raises(ValueError, match="Model must be fitted before prediction"):
        model.predict(make_series([0.1, 0.2]), make_series([0.3]))


def test_predict_maps_forecast_output_to_prediction_result() -> None:
    model = make_model([{"p": 1, "q": 1}])
    train = make_series([0.1, 0.2, 0.3])
    test = make_series([0.4, 0.5])
    model.is_fitted = True
    fake_model = SimpleNamespace(
        forecast=lambda start, horizon: SimpleNamespace(
            variance=SimpleNamespace(values=np.asarray([[1.5, 2.5]], dtype=float))
        )
    )
    setattr(model, "model", fake_model)

    prediction = model.predict(train, test)

    assert isinstance(prediction, PredictionResult)
    assert prediction.model_name == "GARCH"
    assert prediction.asset == "BTC-USD"
    assert prediction.horizon == 2
    assert [row.timestamp for row in prediction.rows] == list(test.index)
    assert [row.predicted_volatility for row in prediction.rows] == [1.5, 2.5]
    assert all(row.lower_ci is None for row in prediction.rows)
    assert all(row.upper_ci is None for row in prediction.rows)