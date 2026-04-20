from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
import torch

from finance_modeling.models.base import BaseVolatilityModel, _Regressor
from finance_modeling.schemas import ModelConfig, PredictionResult, PredictionRow
from finance_modeling.schemas.data import AssetMetadata, AssetType


def make_asset_metadata() -> AssetMetadata:
    return AssetMetadata(
        symbol="BTC-USD",
        asset_type=AssetType.CRYPTO,
        description="Bitcoin",
        data_folder="bitcoin",
    )


class DummyModel(BaseVolatilityModel):
    name = "DUMMY"

    def fit(self, X: pd.Series, y: pd.Series | None = None) -> None:
        self.is_fitted = True

    def predict(self, X: pd.Series, y: pd.Series) -> PredictionResult:
        return PredictionResult(
            model_name=self.name,
            asset=self.asset_metadata.symbol,
            horizon=len(y),
            rows=[
                PredictionRow(timestamp=timestamp, predicted_volatility=float(value))
                for timestamp, value in zip(y.index, y.values)
            ],
        )


def make_model() -> DummyModel:
    return DummyModel(
        config=ModelConfig(name="DUMMY", hyperparameters_list=[{"alpha": 1}]),
        asset_metadata=make_asset_metadata(),
    )


def test_get_params_returns_original_model_config() -> None:
    model = make_model()

    assert model.get_params() == model.config


def test_save_model_persists_pickled_model(tmp_path: Path) -> None:
    model = make_model()

    model.save_model(str(tmp_path))

    saved_file = tmp_path / "DUMMY_BTC-USD.pkl"
    assert saved_file.exists()
    with saved_file.open("rb") as file_pointer:
        restored = pickle.load(file_pointer)
    assert isinstance(restored, DummyModel)
    assert restored.asset_metadata.symbol == "BTC-USD"
    assert restored.config.name == "DUMMY"


def test_save_results_writes_tabular_prediction_rows(tmp_path: Path) -> None:
    model = make_model()
    prediction_result = PredictionResult(
        model_name="DUMMY",
        asset="BTC-USD",
        horizon=2,
        rows=[
            PredictionRow(
                timestamp=pd.Timestamp("2026-01-01 00:00:00"),
                predicted_volatility=0.1,
                lower_ci=0.05,
                upper_ci=0.15,
            ),
            PredictionRow(
                timestamp=pd.Timestamp("2026-01-01 00:15:00"),
                predicted_volatility=0.2,
                lower_ci=0.1,
                upper_ci=0.3,
            ),
        ],
    )

    model.save_results(str(tmp_path), prediction_result)

    saved_file = tmp_path / "DUMMY_BTC-USD_results.csv"
    frame = pd.read_csv(saved_file)
    assert frame.columns.tolist() == [
        "model_name",
        "asset",
        "horizon",
        "timestamp",
        "predicted_volatility",
        "lower_ci",
        "upper_ci",
    ]
    assert frame.shape == (2, 7)
    assert frame["model_name"].tolist() == ["DUMMY", "DUMMY"]
    assert frame["predicted_volatility"].tolist() == [0.1, 0.2]


def test_save_model_best_hyperparameters_writes_json_file(tmp_path: Path) -> None:
    model = make_model()
    model.best_hyperparameters = {"window_size": 10, "hidden_size": 32}

    model.save_model_best_hyperparameters(str(tmp_path))

    saved_file = tmp_path / "DUMMY_BTC-USD_best_hyperparameters.json"
    assert saved_file.exists()
    payload = json.loads(saved_file.read_text(encoding="utf-8"))
    assert payload == {"window_size": 10, "hidden_size": 32}


def test_regressor_forward_returns_expected_output_shape() -> None:
    regressor = _Regressor(
        input_size=1,
        hidden_size=4,
        num_layers=1,
        dropout=0.5,
        output_size=3,
    )
    features = torch.randn(2, 5, 1)

    output = regressor(features)

    assert tuple(output.shape) == (2, 3)
    assert regressor.lstm.dropout == 0.0