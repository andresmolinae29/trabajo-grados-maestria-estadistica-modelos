from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from finance_modeling.experiments import run_causal_ceemdan
from finance_modeling.schemas import ListOfAssets

from tests.factories import make_asset_metadata


def test_main_caches_requested_active_assets_only(monkeypatch: pytest.MonkeyPatch) -> None:
    assets = ListOfAssets(assets=[
        make_asset_metadata(),
        make_asset_metadata(symbol="ETH-USD", data_folder="ethereum"),
        make_asset_metadata(symbol="GC=F", data_folder="gold", active=False),
    ])
    returns = pd.Series([0.01, -0.02, 0.03])
    decomposed: list[tuple[str, int]] = []

    class FakeConfigLoader:
        def load_data_config(self) -> ListOfAssets:
            return assets

    class FakeRawDataLoader:
        def __init__(self, asset):
            self.symbol = asset.symbol

        def load_data(self):
            return self.symbol

    class FakePreprocessor:
        def __init__(self, data):
            self.data = data

        def pipeline(self):
            return SimpleNamespace(returns=returns)

    def fake_features(series: pd.Series, symbol: str) -> np.ndarray:
        decomposed.append((symbol, len(series)))
        return np.zeros((len(series), 7))

    monkeypatch.setattr(run_causal_ceemdan, "ConfigLoader", FakeConfigLoader)
    monkeypatch.setattr(run_causal_ceemdan, "RawDataLoader", FakeRawDataLoader)
    monkeypatch.setattr(run_causal_ceemdan, "DataPreprocessor", FakePreprocessor)
    monkeypatch.setattr(run_causal_ceemdan, "causal_ceemdan_features", fake_features)

    run_causal_ceemdan.main()
    assert decomposed == [("BTC-USD", 3), ("ETH-USD", 3)]

    decomposed.clear()
    run_causal_ceemdan.main(["ETH-USD", "GC=F"])
    assert decomposed == [("ETH-USD", 3)]
