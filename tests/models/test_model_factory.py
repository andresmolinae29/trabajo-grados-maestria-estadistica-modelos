from __future__ import annotations

import pytest

from finance_modeling.models.base import BaseVolatilityModel
from finance_modeling.models.ceemdan_lstm import CEEMDANLSTMModel
from finance_modeling.models.garch import GARCHModel
from finance_modeling.models.model_factory import ModelFactory
from finance_modeling.models.psoqrnn import PSOQRNNModel
from finance_modeling.schemas import ModelConfig

from tests.models.test_garch import make_prices


class _FakeModel(BaseVolatilityModel):
    name = "FAKE"

    def __init__(self, config: ModelConfig, symbol: str, **kwargs):
        super().__init__(config, symbol, **kwargs)
        self.received_kwargs = kwargs

    def fit(self) -> "_FakeModel":
        self.is_fitted = True
        return self

    def predict(self):
        raise NotImplementedError


def test_registry_maps_expected_model_names_to_classes() -> None:
    assert ModelFactory.MODEL_REGISTRY == {
        "garch": GARCHModel,
        "psoqrnn": PSOQRNNModel,
        "ceemdan_lstm": CEEMDANLSTMModel,
    }


def test_create_model_raises_for_unknown_model_name() -> None:
    config = ModelConfig(name="UNKNOWN", hyperparameters_list=[{}])

    with pytest.raises(ValueError, match="not registered"):
        ModelFactory.create_model(model_name="unknown", model_config=config, symbol="BTC-USD")


@pytest.mark.parametrize("raw_name", ["fake", "FAKE", "  fake  ", "Fake"])
def test_create_model_normalizes_name_case_and_whitespace(monkeypatch: pytest.MonkeyPatch, raw_name: str) -> None:
    monkeypatch.setitem(ModelFactory.MODEL_REGISTRY, "fake", _FakeModel)
    config = ModelConfig(name="FAKE", hyperparameters_list=[{}])

    model = ModelFactory.create_model(
        model_name=raw_name,
        model_config=config,
        symbol="BTC-USD",
        extra_kwarg="value",
    )

    assert isinstance(model, _FakeModel)
    assert model.symbol == "BTC-USD"
    assert model.config is config
    assert model.received_kwargs == {"extra_kwarg": "value"}


def test_create_model_returns_registered_class_instance() -> None:
    config = ModelConfig(name="GARCH", hyperparameters_list=[{"p": 1, "q": 1}])

    model = ModelFactory.create_model(
        model_name="garch",
        model_config=config,
        symbol="BTC-USD",
        prices=make_prices(),
    )

    assert isinstance(model, GARCHModel)
    assert model.symbol == "BTC-USD"
    assert model.is_fitted is False
