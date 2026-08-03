from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from finance_modeling.models.base import BaseVolatilityModel
from finance_modeling.schemas import ModelConfig


class DummyModel(BaseVolatilityModel):
    name = "DUMMY"

    def fit(self) -> "DummyModel":
        self.is_fitted = True
        return self

    def predict(self) -> np.ndarray:
        return np.array([0.1, 0.2, 0.3])


def make_model(**kwargs) -> DummyModel:
    return DummyModel(
        config=ModelConfig(name="DUMMY", hyperparameters_list=[{"alpha": 1}]),
        symbol="BTC-USD",
        **kwargs,
    )


def test_init_stores_config_and_symbol() -> None:
    config = ModelConfig(name="DUMMY", hyperparameters_list=[{"alpha": 1}])

    model = DummyModel(config=config, symbol="BTC-USD", unused_kwarg="ignored")

    assert model.config is config
    assert model.symbol == "BTC-USD"
    assert model.is_fitted is False


def test_fit_updates_is_fitted() -> None:
    model = make_model()

    result = model.fit()

    assert result is model
    assert model.is_fitted is True


def test_subclass_missing_abstract_methods_cannot_be_instantiated() -> None:
    class IncompleteModel(BaseVolatilityModel):
        name = "INCOMPLETE"

    with pytest.raises(TypeError):
        IncompleteModel(config=ModelConfig(name="INCOMPLETE"), symbol="BTC-USD")  # type: ignore[abstract]


def test_base_fit_and_predict_raise_not_implemented_when_called_directly() -> None:
    class SuperCallingModel(BaseVolatilityModel):
        name = "SUPER"

        def fit(self):
            return super().fit()

        def predict(self):
            return super().predict()

    model = SuperCallingModel(config=ModelConfig(name="SUPER"), symbol="BTC-USD")

    with pytest.raises(NotImplementedError):
        model.fit()

    with pytest.raises(NotImplementedError):
        model.predict()


def test_save_model_persists_pickled_model(tmp_path: Path) -> None:
    model = make_model()

    model.save_model(str(tmp_path))

    saved_file = tmp_path / "DUMMY_BTC-USD.pkl"
    assert saved_file.exists()
    with saved_file.open("rb") as file_pointer:
        restored = pickle.load(file_pointer)
    assert isinstance(restored, DummyModel)
    assert restored.symbol == "BTC-USD"
    assert restored.config.name == "DUMMY"
