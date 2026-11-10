from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from finance_modeling.experiments import runner
from finance_modeling.schemas import ComparisonResult, ExperimentConfig, ListOfAssets, ModelConfig, TimeSeriesInput
from finance_modeling.schemas.data import AssetMetadata, AssetType
from finance_modeling.utils import DataLoaderException

from tests.models.test_garch import make_prices


def make_asset(active: bool = True) -> AssetMetadata:
    return AssetMetadata(
        symbol="BTC-USD",
        asset_type=AssetType.CRYPTO,
        description="Bitcoin",
        data_folder="bitcoin",
        active=active,
    )


class FakeConfigLoader:
    def __init__(self, model_config: ExperimentConfig, data_config: ListOfAssets):
        self._model_config = model_config
        self._data_config = data_config

    def load_model_config(self, file_name: str = "experiment_config.yml") -> ExperimentConfig:
        return self._model_config

    def load_data_config(self) -> ListOfAssets:
        return self._data_config

    def load_garch_selection(self) -> dict:
        return {}


class FakeRawDataLoader:
    def __init__(self, asset: AssetMetadata, processed_data: TimeSeriesInput, should_raise: bool = False):
        self.asset = asset
        self.processed_data = processed_data
        self.should_raise = should_raise

    def load_data(self) -> TimeSeriesInput:
        if self.should_raise:
            raise DataLoaderException("broken source")
        return self.processed_data


class FakeVolatilityModel:
    """Doble de BaseVolatilityModel para la API actual: fit()/predict() sin argumentos."""

    def __init__(self, name: str, predictions: np.ndarray, y_test: np.ndarray, call_log: list[str]):
        self.name = name
        self._predictions = predictions
        self.y_test = y_test
        self.call_log = call_log
        self.is_fitted = False

    def fit(self) -> "FakeVolatilityModel":
        self.is_fitted = True
        self.call_log.append(f"fit:{self.name}")
        return self

    def predict(self) -> np.ndarray:
        self.call_log.append(f"predict:{self.name}")
        return self._predictions

    def save_model(self, experiment_path: str) -> None:
        self.call_log.append(f"save_model:{self.name}")


def test_runner_main_orchestrates_models_evaluation_and_comparison(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    asset = make_asset()
    experiment_path = str(tmp_path / "exp")
    os.makedirs(experiment_path, exist_ok=True)

    raw_data = TimeSeriesInput(metadata=asset, series=make_prices())
    experiment_config = ExperimentConfig(
        experiment_name="exp",
        output_dir=str(tmp_path),
        models=[
            ModelConfig(name="GARCH", hyperparameters_list=[{}]),
            ModelConfig(name="PSOQRNN", hyperparameters_list=[{}]),
        ],
    )
    data_config = ListOfAssets(assets=[asset])
    call_log: list[str] = []
    saved_comparisons: list[tuple[str, str]] = []

    y_test_common = np.array([0.10, 0.11, 0.12, 0.13, 0.14])
    fake_models = {
        "GARCH": FakeVolatilityModel("GARCH", np.array([0.11, 0.12, 0.13, 0.14, 0.15]), y_test_common, call_log),
        "PSOQRNN": FakeVolatilityModel("PSOQRNN", np.array([0.09, 0.10, 0.11, 0.12, 0.13]), y_test_common, call_log),
    }

    monkeypatch.setattr(runner, "ConfigLoader", lambda: FakeConfigLoader(experiment_config, data_config))
    monkeypatch.setattr(runner, "create_experiment_directory", lambda output_dir, experiment_name: experiment_path)
    monkeypatch.setattr(runner, "RawDataLoader", lambda asset: FakeRawDataLoader(asset, raw_data))
    monkeypatch.setattr(
        runner.ModelFactory,
        "create_model",
        lambda model_name, model_config, symbol, **kwargs: fake_models[model_name],
    )
    monkeypatch.setattr(
        runner.ModelComparator,
        "compare",
        lambda self, baseline, challenger, y_pred_baseline, y_pred_challenger, y_true: ComparisonResult(
            baseline_model=baseline.model_name,
            challenger_model=challenger.model_name,
            asset=baseline.asset,
            dm_statistic=1.0,
            dm_p_value=0.05,
        ),
    )
    monkeypatch.setattr(
        runner.ModelComparator,
        "save_comparison_results",
        lambda experiment_path, comparison_result: saved_comparisons.append(
            (comparison_result.baseline_model, comparison_result.challenger_model)
        ),
    )

    runner.main()

    assert call_log == [
        "fit:GARCH", "save_model:GARCH", "predict:GARCH",
        "fit:PSOQRNN", "save_model:PSOQRNN", "predict:PSOQRNN",
    ]
    assert saved_comparisons == [("GARCH", "PSOQRNN")]
    assert (Path(experiment_path) / "GARCH_BTC-USD_predictions.csv").exists()
    assert (Path(experiment_path) / "PSOQRNN_BTC-USD_predictions.csv").exists()
    assert (Path(experiment_path) / "GARCH_BTC-USD_evaluation.json").exists()
    assert (Path(experiment_path) / "PSOQRNN_BTC-USD_evaluation.json").exists()
    assert (Path(experiment_path) / "GARCH_BTC-USD_forecast.png").exists()


def test_runner_main_evaluates_all_models_on_common_dates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    asset = make_asset()
    experiment_path = str(tmp_path / "exp")
    os.makedirs(experiment_path, exist_ok=True)

    raw_data = TimeSeriesInput(metadata=asset, series=make_prices())
    experiment_config = ExperimentConfig(
        experiment_name="exp",
        output_dir=str(tmp_path),
        models=[ModelConfig(name=name) for name in ("GARCH", "PSOQRNN", "NAIVE")],
    )
    call_log: list[str] = []
    captured: list[tuple[str, str, int]] = []

    # PSOQRNN cubre menos barras; un error enorme fuera de esas fechas no debe contar.
    fake_models = {
        "GARCH": FakeVolatilityModel(
            "GARCH", np.array([9.0, 0.11, 0.12, 0.13, 0.14]), np.array([0.10, 0.10, 0.10, 0.10, 0.10]), call_log
        ),
        "PSOQRNN": FakeVolatilityModel("PSOQRNN", np.array([0.10, 0.10]), np.array([0.10, 0.10]), call_log),
        "NAIVE": FakeVolatilityModel(
            "NAIVE", np.array([9.0, 0.10, 0.10, 0.10, 0.10]), np.array([0.10, 0.10, 0.10, 0.10, 0.10]), call_log
        ),
    }

    monkeypatch.setattr(
        runner, "ConfigLoader", lambda: FakeConfigLoader(experiment_config, ListOfAssets(assets=[asset]))
    )
    monkeypatch.setattr(runner, "create_experiment_directory", lambda output_dir, experiment_name: experiment_path)
    monkeypatch.setattr(runner, "RawDataLoader", lambda asset: FakeRawDataLoader(asset, raw_data))
    monkeypatch.setattr(
        runner.ModelFactory,
        "create_model",
        lambda model_name, model_config, symbol, **kwargs: fake_models[model_name],
    )

    def fake_compare(self, baseline, challenger, y_pred_baseline, y_pred_challenger, y_true):
        captured.append((baseline.model_name, challenger.model_name, len(y_true)))
        return ComparisonResult(
            baseline_model=baseline.model_name,
            challenger_model=challenger.model_name,
            asset=baseline.asset,
            dm_statistic=1.0,
            dm_p_value=0.05,
        )

    monkeypatch.setattr(runner.ModelComparator, "compare", fake_compare)
    monkeypatch.setattr(runner.ModelComparator, "save_comparison_results", lambda experiment_path, comparison_result: None)

    runner.main()

    assert captured == [
        ("GARCH", "PSOQRNN", 2),
        ("GARCH", "NAIVE", 2),
        ("NAIVE", "GARCH", 2),
        ("NAIVE", "PSOQRNN", 2),
    ]
    naive_evaluation = json.loads((Path(experiment_path) / "NAIVE_BTC-USD_evaluation.json").read_text())
    assert naive_evaluation["rmse"] == pytest.approx(0.0)


def test_runner_main_tunes_networks_and_saves_seed_variability(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    asset = make_asset()
    experiment_path = str(tmp_path / "exp")
    os.makedirs(experiment_path, exist_ok=True)

    raw_data = TimeSeriesInput(metadata=asset, series=make_prices())
    experiment_config = ExperimentConfig(
        experiment_name="exp",
        output_dir=str(tmp_path),
        models=[
            ModelConfig(
                name="PSOQRNN",
                seeds=[1, 2],
                hyperparameters_list=[{"n_hidden": 2}, {"n_hidden": 3}],
            ),
            ModelConfig(name="NAIVE"),
        ],
    )

    def create_model(model_name, model_config, symbol, **kwargs):
        y_test = np.full(5, 0.10)
        if model_name == "NAIVE":
            return FakeVolatilityModel(model_name, np.full(5, 0.10), y_test, [])
        error = 0.01 * kwargs["n_hidden"] * kwargs["seed"]
        return FakeVolatilityModel(model_name, np.full(5, 0.10 + error), y_test, [])

    monkeypatch.setattr(
        runner, "ConfigLoader", lambda: FakeConfigLoader(experiment_config, ListOfAssets(assets=[asset]))
    )
    monkeypatch.setattr(runner, "create_experiment_directory", lambda output_dir, experiment_name: experiment_path)
    monkeypatch.setattr(runner, "RawDataLoader", lambda asset: FakeRawDataLoader(asset, raw_data))
    monkeypatch.setattr(runner.ModelFactory, "create_model", create_model)
    monkeypatch.setattr(
        runner.ModelComparator,
        "compare",
        lambda self, baseline, challenger, y_pred_baseline, y_pred_challenger, y_true: ComparisonResult(
            baseline_model=baseline.model_name,
            challenger_model=challenger.model_name,
            asset=baseline.asset,
            dm_statistic=1.0,
            dm_p_value=0.05,
        ),
    )
    monkeypatch.setattr(runner.ModelComparator, "save_comparison_results", lambda experiment_path, comparison_result: None)

    runner.main()

    tuning = pd.read_csv(Path(experiment_path) / "PSOQRNN_BTC-USD_tuning.csv")
    assert len(tuning) == 4
    chosen = tuning[tuning["chosen"]]
    assert chosen[["n_hidden", "seed"]].values.tolist() == [[2, 1]]

    seeds = pd.read_csv(Path(experiment_path) / "PSOQRNN_BTC-USD_seeds.csv")
    assert seeds["seed"].tolist() == [1, 2]
    assert seeds["chosen"].tolist() == [True, False]
    assert seeds["rmse"].tolist() == pytest.approx([0.02, 0.04])

    evaluation = json.loads((Path(experiment_path) / "PSOQRNN_BTC-USD_evaluation.json").read_text())
    assert evaluation["rmse"] == pytest.approx(0.02)


def test_runner_main_skips_inactive_assets(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    inactive_asset = make_asset(active=False)
    experiment_config = ExperimentConfig(
        experiment_name="exp",
        output_dir=str(tmp_path),
        models=[ModelConfig(name="GARCH", hyperparameters_list=[{}])],
    )
    data_config = ListOfAssets(assets=[inactive_asset])
    create_model_calls: list[str] = []

    monkeypatch.setattr(runner, "ConfigLoader", lambda: FakeConfigLoader(experiment_config, data_config))
    monkeypatch.setattr(runner, "create_experiment_directory", lambda output_dir, experiment_name: str(tmp_path / "exp"))
    monkeypatch.setattr(
        runner.ModelFactory,
        "create_model",
        lambda model_name, model_config, symbol, **kwargs: create_model_calls.append(model_name),
    )

    runner.main()

    assert create_model_calls == []


def test_runner_main_runs_only_requested_symbols(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    experiment_config = ExperimentConfig(
        experiment_name="exp",
        output_dir=str(tmp_path),
        models=[ModelConfig(name="GARCH", hyperparameters_list=[{}])],
    )
    data_config = ListOfAssets(assets=[make_asset()])
    create_model_calls: list[str] = []

    monkeypatch.setattr(runner, "ConfigLoader", lambda: FakeConfigLoader(experiment_config, data_config))
    monkeypatch.setattr(runner, "create_experiment_directory", lambda output_dir, experiment_name: str(tmp_path / "exp"))
    monkeypatch.setattr(
        runner.ModelFactory,
        "create_model",
        lambda model_name, model_config, symbol, **kwargs: create_model_calls.append(model_name),
    )

    runner.main(symbols=["ETH-USD"])

    assert create_model_calls == []


def test_runner_main_skips_asset_when_data_loader_fails(monkeypatch, tmp_path) -> None:
    asset = make_asset()
    model_config = ExperimentConfig(
        experiment_name="exp",
        output_dir=str(tmp_path),
        models=[ModelConfig(name="GARCH", hyperparameters_list=[{}])],
    )
    data_config = ListOfAssets(assets=[asset])
    create_model_calls: list[str] = []

    monkeypatch.setattr(runner, "ConfigLoader", lambda: FakeConfigLoader(model_config, data_config))
    monkeypatch.setattr(runner, "create_experiment_directory", lambda output_dir, experiment_name: str(tmp_path / "exp"))
    monkeypatch.setattr(runner, "RawDataLoader", lambda asset: FakeRawDataLoader(asset, TimeSeriesInput(metadata=asset, series=make_prices()), should_raise=True))
    monkeypatch.setattr(runner.ModelFactory, "create_model", lambda model_name, model_config, asset: create_model_calls.append(model_name))

    runner.main()

    assert create_model_calls == []


class FakeGarchModel(FakeVolatilityModel):
    """Se registra como GARCHModel del runner para activar el diagnóstico."""


def fake_comparison(self, baseline, challenger, y_pred_baseline, y_pred_challenger, y_true) -> ComparisonResult:
    return ComparisonResult(
        baseline_model=baseline.model_name,
        challenger_model=challenger.model_name,
        asset=baseline.asset,
        dm_statistic=1.0,
        dm_p_value=0.05,
    )


def test_runner_main_saves_diagnostics_only_for_garch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    asset = make_asset()
    experiment_path = str(tmp_path / "exp")
    os.makedirs(experiment_path, exist_ok=True)
    experiment_config = ExperimentConfig(
        experiment_name="exp",
        output_dir=str(tmp_path),
        models=[ModelConfig(name="GARCH"), ModelConfig(name="NAIVE")],
    )
    y_test = np.full(5, 0.10)
    fake_models = {
        "GARCH": FakeGarchModel("GARCH", np.full(5, 0.11), y_test, []),
        "NAIVE": FakeVolatilityModel("NAIVE", np.full(5, 0.10), y_test, []),
    }
    diagnosed: list[tuple[str, str]] = []

    monkeypatch.setattr(
        runner, "ConfigLoader", lambda: FakeConfigLoader(experiment_config, ListOfAssets(assets=[asset]))
    )
    monkeypatch.setattr(runner, "create_experiment_directory", lambda output_dir, experiment_name: experiment_path)
    monkeypatch.setattr(
        runner, "RawDataLoader", lambda asset: FakeRawDataLoader(asset, TimeSeriesInput(metadata=asset, series=make_prices()))
    )
    monkeypatch.setattr(
        runner.ModelFactory,
        "create_model",
        lambda model_name, model_config, symbol, **kwargs: fake_models[model_name],
    )
    monkeypatch.setattr(runner, "GARCHModel", FakeGarchModel)
    monkeypatch.setattr(runner, "save_garch_diagnostics", lambda model, path: diagnosed.append((model.name, path)))
    monkeypatch.setattr(runner.ModelComparator, "compare", fake_comparison)
    monkeypatch.setattr(runner.ModelComparator, "save_comparison_results", lambda experiment_path, comparison_result: None)

    runner.main()

    assert diagnosed == [("GARCH", experiment_path)]


def test_runner_main_continues_with_next_asset_after_unexpected_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    failing_asset = make_asset()
    healthy_asset = AssetMetadata(
        symbol="ETH-USD", asset_type=AssetType.CRYPTO, description="Ethereum", data_folder="ethereum"
    )
    experiment_path = str(tmp_path / "exp")
    os.makedirs(experiment_path, exist_ok=True)
    experiment_config = ExperimentConfig(
        experiment_name="exp", output_dir=str(tmp_path), models=[ModelConfig(name="NAIVE")]
    )

    def create_model(model_name, model_config, symbol, **kwargs):
        if symbol == failing_asset.symbol:
            raise RuntimeError("model failure")
        return FakeVolatilityModel(model_name, np.full(5, 0.10), np.full(5, 0.10), [])

    monkeypatch.setattr(
        runner,
        "ConfigLoader",
        lambda: FakeConfigLoader(experiment_config, ListOfAssets(assets=[failing_asset, healthy_asset])),
    )
    monkeypatch.setattr(runner, "create_experiment_directory", lambda output_dir, experiment_name: experiment_path)
    monkeypatch.setattr(
        runner, "RawDataLoader", lambda asset: FakeRawDataLoader(asset, TimeSeriesInput(metadata=asset, series=make_prices()))
    )
    monkeypatch.setattr(runner.ModelFactory, "create_model", create_model)

    runner.main()

    assert not (Path(experiment_path) / "NAIVE_BTC-USD_predictions.csv").exists()
    assert (Path(experiment_path) / "NAIVE_ETH-USD_predictions.csv").exists()
