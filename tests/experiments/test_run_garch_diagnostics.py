from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from finance_modeling.experiments import run_garch_diagnostics
from finance_modeling.schemas import ExperimentConfig, ListOfAssets, ModelConfig

from tests.factories import make_asset_metadata


def test_main_fits_garch_for_requested_symbol_and_saves_diagnostics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assets = ListOfAssets(assets=[make_asset_metadata(), make_asset_metadata(symbol="ETH-USD", data_folder="ethereum")])
    experiment = ExperimentConfig(
        experiment_name="exp",
        output_dir=str(tmp_path),
        models=[ModelConfig(name="NAIVE"), ModelConfig(name="GARCH", hyperparameters_list=[{"p": 1, "q": 1}])],
    )
    preprocessed = SimpleNamespace(series=pd.Series([100.0, 101.0]), volatility_series=pd.Series([0.1]))
    loaded: list[str] = []
    saved: list[tuple[object, str]] = []

    class FakeConfigLoader:
        def load_data_config(self) -> ListOfAssets:
            return assets

        def load_model_config(self) -> ExperimentConfig:
            return experiment

        def load_garch_selection(self) -> dict:
            return {"ETH-USD": "eth-selection"}

    class FakeRawDataLoader:
        def __init__(self, asset):
            loaded.append(asset.symbol)

        def load_data(self):
            return "raw"

    class FakePreprocessor:
        def __init__(self, data):
            assert data == "raw"

        def pipeline(self):
            return preprocessed

    class FakeGARCHModel:
        def __init__(self, config, symbol, **kwargs):
            self.config = config
            self.symbol = symbol
            self.kwargs = kwargs
            self.is_fitted = False

        def fit(self):
            self.is_fitted = True
            return self

    monkeypatch.setattr(run_garch_diagnostics, "ConfigLoader", FakeConfigLoader)
    monkeypatch.setattr(run_garch_diagnostics, "RawDataLoader", FakeRawDataLoader)
    monkeypatch.setattr(run_garch_diagnostics, "DataPreprocessor", FakePreprocessor)
    monkeypatch.setattr(run_garch_diagnostics, "GARCHModel", FakeGARCHModel)
    monkeypatch.setattr(run_garch_diagnostics, "save_garch_diagnostics", lambda model, path: saved.append((model, path)))
    monkeypatch.setattr(run_garch_diagnostics, "get_main_root", lambda: str(tmp_path))

    run_garch_diagnostics.main("ETH-USD")

    assert loaded == ["ETH-USD"]
    model, path = saved[0]
    assert path == os.path.join(str(tmp_path), "results")
    assert model.is_fitted is True
    assert model.config.name == "GARCH"
    assert model.symbol == "ETH-USD"
    assert model.kwargs["prices"] is preprocessed.series
    assert model.kwargs["volatility"] is preprocessed.volatility_series
    assert model.kwargs["selection"] == "eth-selection"
