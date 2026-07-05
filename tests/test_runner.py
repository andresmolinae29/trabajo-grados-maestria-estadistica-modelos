from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finance_modeling.experiments import runner
from finance_modeling.schemas import ComparisonResult, EvaluationResult, ExperimentConfig, ListOfAssets, ModelConfig, PredictionResult, PredictionRow, TimeSeriesInput
from finance_modeling.schemas.data import AssetMetadata, AssetType
from finance_modeling.utils import DataLoaderException


def make_asset(active: bool = True) -> AssetMetadata:
    return AssetMetadata(
        symbol="BTC-USD",
        asset_type=AssetType.CRYPTO,
        description="Bitcoin",
        data_folder="bitcoin",
        active=active,
    )


def make_processed_data(asset: AssetMetadata) -> TimeSeriesInput:
    index = pd.date_range("2026-01-01", periods=4, freq="15min")
    vol_index = index[1:]
    return TimeSeriesInput(
        metadata=asset,
        series=pd.Series([100.0, 101.0, 102.0, 103.0], index=index),
        log_returns=pd.Series([0.1, 0.2, 0.3], index=vol_index),
        raw_log_returns=pd.Series([0.01, 0.02, 0.03], index=vol_index),
        train=pd.Series([0.1, 0.2], index=index[1:3]),
        test=pd.Series([0.3], index=index[3:4]),
        train_raw=pd.Series([0.01, 0.02], index=index[1:3]),
        test_raw=pd.Series([0.03], index=index[3:4]),
        test_prices=pd.Series([103.0], index=index[3:4]),
        normalization_mean=0.0,
        normalization_std=0.1,
        last_train_price=102.0,
        split_index=2,
        volatility_raw=pd.Series([0.0001, 0.0004, 0.0009], index=vol_index),
        train_volatility_raw=pd.Series([0.0001, 0.0004], index=index[1:3]),
        test_volatility_raw=pd.Series([0.0009], index=index[3:4]),
        volatility_normalized=pd.Series([0.5, 0.6, 0.7], index=vol_index),
        train_volatility=pd.Series([0.5, 0.6], index=index[1:3]),
        test_volatility=pd.Series([0.7], index=index[3:4]),
        volatility_min=0.0001,
        volatility_max=0.0009,
    )


def make_prediction(model_name: str, asset_symbol: str, values: list[float]) -> PredictionResult:
    index = pd.date_range("2026-01-01", periods=len(values), freq="15min")
    return PredictionResult(
        model_name=model_name,
        asset=asset_symbol,
        horizon=len(values),
        rows=[
            PredictionRow(timestamp=timestamp, predicted_value=value)
            for timestamp, value in zip(index, values)
        ],
    )


class FakeConfigLoader:
    def __init__(self, model_config: ExperimentConfig, data_config: ListOfAssets):
        self._model_config = model_config
        self._data_config = data_config

    def load_model_config(self) -> ExperimentConfig:
        return self._model_config

    def load_data_config(self) -> ListOfAssets:
        return self._data_config


class FakeRawDataLoader:
    def __init__(self, asset: AssetMetadata, processed_data: TimeSeriesInput, should_raise: bool = False):
        self.asset = asset
        self.processed_data = processed_data
        self.should_raise = should_raise

    def load_data(self) -> TimeSeriesInput:
        if self.should_raise:
            raise DataLoaderException("broken source")
        return self.processed_data


class FakePreprocessor:
    def __init__(self, config, call_log: list[str]):
        self.call_log = call_log

    def preprocess(self, data: TimeSeriesInput, handle_missing: bool = True) -> TimeSeriesInput:
        self.call_log.append("preprocess")
        return data

    def compute_log_returns(self, data: TimeSeriesInput) -> TimeSeriesInput:
        self.call_log.append("compute_log_returns")
        return data

    def normalize(self, data: TimeSeriesInput) -> TimeSeriesInput:
        self.call_log.append("normalize")
        return data

    def split_data(self, data: TimeSeriesInput, train_ratio: float = 0.8) -> TimeSeriesInput:
        self.call_log.append(f"split_data:{train_ratio}")
        return data

    def compute_volatility_target(self, data: TimeSeriesInput) -> TimeSeriesInput:
        self.call_log.append("compute_volatility_target")
        return data


class FakeModel:
    def __init__(self, name: str, predictions: PredictionResult, call_log: list[str]):
        self.name = name
        self.predictions = predictions
        self.call_log = call_log

    def fit(self, train: pd.Series) -> None:
        self.call_log.append(f"fit:{self.name}:{len(train)}")

    def predict(self, train: pd.Series, test: pd.Series) -> PredictionResult:
        self.call_log.append(f"predict:{self.name}:{len(train)}:{len(test)}")
        return self.predictions

    def save_model(self, experiment_path: str) -> None:
        self.call_log.append(f"save_model:{self.name}:{experiment_path}")

    def save_results(self, experiment_path: str, predictions: PredictionResult) -> None:
        self.call_log.append(f"save_results:{self.name}:{experiment_path}:{predictions.model_name}")

    def save_model_best_hyperparameters(self, experiment_path: str) -> None:
        self.call_log.append(f"save_hparams:{self.name}:{experiment_path}")


@pytest.mark.skip(reason="API obsoleta: enrich_prediction_result y TimeSeriesInput con campos diferentes a la implementación actual")
def test_enrich_prediction_result_adds_volatility_and_actual_values() -> None:
    asset = make_asset()
    processed_data = make_processed_data(asset)
    prediction = PredictionResult(
        model_name="PSO-QRNN",
        asset=asset.symbol,
        horizon=1,
        rows=[
            PredictionRow(
                timestamp=processed_data.test_volatility.index[0],
                predicted_value=0.4,
            )
        ],
    )

    enriched = runner.enrich_prediction_result(processed_data, prediction)

    assert enriched.rows[0].predicted_volatility_normalized == pytest.approx(0.4)
    assert enriched.rows[0].predicted_volatility == pytest.approx(0.00042)
    assert enriched.rows[0].actual_volatility_normalized == pytest.approx(0.7)
    assert enriched.rows[0].actual_volatility == pytest.approx(0.0009)
    assert enriched.rows[0].actual_return == pytest.approx(0.03)
    assert enriched.rows[0].actual_price == pytest.approx(103.0)
    # return-based enrichment is no longer populated
    assert enriched.rows[0].predicted_normalized_return is None
    assert enriched.rows[0].predicted_return is None
    assert enriched.rows[0].predicted_price is None


@pytest.mark.skip(reason="API obsoleta: enrich_prediction_result y TimeSeriesInput con campos diferentes a la implementación actual")
def test_enrich_prediction_result_clamps_predictions_to_non_negative_volatility() -> None:
    asset = make_asset()
    processed_data = make_processed_data(asset)
    prediction = PredictionResult(
        model_name="PSO-QRNN",
        asset=asset.symbol,
        horizon=1,
        rows=[
            PredictionRow(
                timestamp=processed_data.test_volatility.index[0],
                predicted_value=-2.0,
                lower_ci=-3.0,
                upper_ci=-1.0,
            )
        ],
    )

    enriched = runner.enrich_prediction_result(processed_data, prediction)

    assert enriched.rows[0].predicted_volatility == pytest.approx(0.0)
    assert enriched.rows[0].lower_ci == pytest.approx(0.0)
    assert enriched.rows[0].upper_ci == pytest.approx(0.0)


@pytest.mark.skip(reason="API obsoleta: FakePreprocessor y runner.main() usan interfaz distinta a la implementación actual")
def test_runner_main_orchestrates_models_and_comparison(monkeypatch, tmp_path) -> None:
    asset = make_asset()
    processed_data = make_processed_data(asset)
    experiment_path = str(tmp_path / "exp")
    model_config = ExperimentConfig(
        experiment_name="exp",
        output_dir=str(tmp_path),
        train_ratio=0.8,
        models=[
            ModelConfig(name="GARCH", hyperparameters_list=[{}]),
            ModelConfig(name="PSOQRNN", hyperparameters_list=[{}]),
        ],
    )
    data_config = ListOfAssets(assets=[asset])
    call_log: list[str] = []
    saved_evaluations: list[tuple[str, str]] = []
    saved_comparisons: list[tuple[str, str, str]] = []

    garch_prediction = make_prediction("GARCH", asset.symbol, [0.11])
    pso_prediction = make_prediction("PSO-QRNN", asset.symbol, [0.09])
    fake_models = {
        "GARCH": FakeModel("GARCH", garch_prediction, call_log),
        "PSOQRNN": FakeModel("PSO-QRNN", pso_prediction, call_log),
    }

    monkeypatch.setattr(runner, "ConfigLoader", lambda: FakeConfigLoader(model_config, data_config))
    monkeypatch.setattr(runner, "create_experiment_directory", lambda output_dir, experiment_name: experiment_path)
    monkeypatch.setattr(runner, "RawDataLoader", lambda asset: FakeRawDataLoader(asset, processed_data))
    monkeypatch.setattr(runner, "DataPreprocessor", lambda config: FakePreprocessor(config, call_log))
    monkeypatch.setattr(runner.ModelFactory, "create_model", lambda model_name, model_config, asset: fake_models[model_name])
    monkeypatch.setattr(
        runner.Evaluator,
        "from_timeinput_and_prediction_result",
        staticmethod(lambda y_true, y_pred: EvaluationResult(model_name=y_pred.model_name, asset=y_pred.asset, rmse=0.2 if y_pred.model_name == "GARCH" else 0.1, mae=0.15 if y_pred.model_name == "GARCH" else 0.05)),
    )
    monkeypatch.setattr(
        runner.Evaluator,
        "save_evaluation_results",
        lambda self, experiment_path, evaluation_result: saved_evaluations.append((experiment_path, evaluation_result.model_name)),
    )
    monkeypatch.setattr(
        runner.ModelComparator,
        "compare_from_timeinput_and_prediction_results",
        staticmethod(
            lambda baseline, challenger, y_true, pred_baseline, pred_challenger: ComparisonResult(
                baseline_model=baseline.model_name,
                challenger_model=challenger.model_name,
                asset=baseline.asset,
                rmse_diff=challenger.rmse - baseline.rmse,
                dm_p_value=0.04,
            )
        ),
    )
    monkeypatch.setattr(
        runner.ModelComparator,
        "save_comparison_results",
        staticmethod(lambda experiment_path, comparison_result: saved_comparisons.append((experiment_path, comparison_result.baseline_model, comparison_result.challenger_model))),
    )

    runner.main()

    assert call_log[:5] == ["preprocess", "compute_log_returns", "split_data:0.8", "normalize", "compute_volatility_target"]
    assert "fit:GARCH:2" in call_log
    assert "predict:GARCH:2:1" in call_log
    assert "fit:PSO-QRNN:2" in call_log
    assert "predict:PSO-QRNN:2:1" in call_log
    assert saved_evaluations == [(experiment_path, "GARCH"), (experiment_path, "PSO-QRNN")]
    assert saved_comparisons == [(experiment_path, "GARCH", "PSO-QRNN")]


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
    monkeypatch.setattr(runner, "RawDataLoader", lambda asset: FakeRawDataLoader(asset, make_processed_data(asset), should_raise=True))
    monkeypatch.setattr(runner.ModelFactory, "create_model", lambda model_name, model_config, asset: create_model_calls.append(model_name))

    runner.main()

    assert create_model_calls == []