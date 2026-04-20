from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from finance_modeling.data import DataPreprocessor, RawDataLoader
from finance_modeling.schemas.data import AssetMetadata, AssetType, TimeSeriesInput
from finance_modeling.utils import DataLoaderException


def make_asset_metadata() -> AssetMetadata:
    return AssetMetadata(
        symbol="BTC-USD",
        asset_type=AssetType.CRYPTO,
        description="Bitcoin",
        data_folder="bitcoin",
        column_to_use="close",
    )


def make_time_series_input(series: pd.Series) -> TimeSeriesInput:
    return TimeSeriesInput(
        metadata=make_asset_metadata(),
        series=series,
        log_returns=pd.Series(dtype=float),
    )


def test_raw_data_loader_builds_expected_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    asset = make_asset_metadata()
    monkeypatch.setattr("finance_modeling.data.loaders.get_main_root", lambda: str(tmp_path))

    loader = RawDataLoader(asset)

    expected_path = tmp_path / "data" / "files" / "bitcoin" / "BTC-USD_curated.csv"
    assert Path(loader._set_data_path()) == expected_path


def test_raw_data_loader_raises_domain_error_when_file_is_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("finance_modeling.data.loaders.get_main_root", lambda: str(tmp_path))
    loader = RawDataLoader(make_asset_metadata())

    with pytest.raises(DataLoaderException, match="Error loading data"):
        loader.load_data()


def test_raw_data_loader_reads_curated_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    asset = make_asset_metadata()
    data_dir = tmp_path / "data" / "files" / "bitcoin"
    data_dir.mkdir(parents=True)
    csv_path = data_dir / "BTC-USD_curated.csv"
    csv_path.write_text(
        "timestamp;close\n"
        "2026-01-01 00:00:00;100.0\n"
        "2026-01-01 00:15:00;101.5\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("finance_modeling.data.loaders.get_main_root", lambda: str(tmp_path))

    loaded = RawDataLoader(asset).load_data()

    assert loaded.frequency == "15min"
    assert list(loaded.series.index) == [
        pd.Timestamp("2026-01-01 00:00:00"),
        pd.Timestamp("2026-01-01 00:15:00"),
    ]
    assert list(loaded.series.values) == [100.0, 101.5]
    assert loaded.metadata.symbol == "BTC-USD"


def test_preprocess_fills_missing_values_and_casts_to_float() -> None:
    series = pd.Series(
        [100, np.nan, 102],
        index=pd.date_range("2026-01-01", periods=3, freq="15min"),
        dtype=float,
    )
    data = make_time_series_input(series)

    processed = DataPreprocessor(make_asset_metadata()).preprocess(data)

    assert processed.series.dtype == float
    assert processed.series.isna().sum() == 0
    assert list(processed.series.values) == [100.0, 100.0, 102.0]


def test_compute_log_returns_matches_expected_formula() -> None:
    series = pd.Series(
        [100.0, 110.0, 121.0],
        index=pd.date_range("2026-01-01", periods=3, freq="15min"),
    )
    data = make_time_series_input(series)

    processed = DataPreprocessor(make_asset_metadata()).compute_log_returns(data)

    expected = np.log(pd.Series([1.1, 1.1], index=series.index[1:]))
    pd.testing.assert_series_equal(processed.log_returns, expected, check_names=False) # type: ignore


def test_normalize_standardizes_log_returns() -> None:
    data = make_time_series_input(pd.Series(dtype=float))
    data.log_returns = pd.Series(
        [1.0, 2.0, 3.0],
        index=pd.date_range("2026-01-01", periods=3, freq="15min"),
    )

    normalized = DataPreprocessor(make_asset_metadata()).normalize(data)

    assert pytest.approx(float(normalized.log_returns.mean()), abs=1e-9) == 0.0
    assert pytest.approx(float(normalized.log_returns.std()), abs=1e-9) == 1.0


def test_split_data_uses_train_ratio_and_records_split_index() -> None:
    data = make_time_series_input(pd.Series(dtype=float))
    data.log_returns = pd.Series(
        [0.1, 0.2, 0.3, 0.4, 0.5],
        index=pd.date_range("2026-01-01", periods=5, freq="15min"),
    )

    split = DataPreprocessor(make_asset_metadata()).split_data(data, train_ratio=0.6)

    assert split.split_index == 3
    assert list(split.train.values) == [0.1, 0.2, 0.3]
    assert list(split.test.values) == [0.4, 0.5]