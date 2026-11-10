from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finance_modeling.models.naive import NaiveModel
from finance_modeling.schemas import ModelConfig


def make_model() -> NaiveModel:
    volatility = pd.Series(
        np.arange(1.0, 11.0),
        index=pd.date_range("2024-01-01", periods=10, freq="15min"),
    )
    return NaiveModel(config=ModelConfig(name="NAIVE"), symbol="BTC-USD", volatility=volatility)


def test_predict_raises_before_fit() -> None:
    with pytest.raises(ValueError, match="Model must be fitted before prediction"):
        make_model().predict()


def test_predict_returns_previous_value_of_target() -> None:
    model = make_model()

    model.fit()

    np.testing.assert_array_equal(model.y_test, [9.0, 10.0])
    np.testing.assert_array_equal(model.predict(), [8.0, 9.0])
