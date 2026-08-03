from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from finance_modeling.data import DataPreprocessor, RawDataLoader
from finance_modeling.schemas.data import TimeSeriesInput
from finance_modeling.utils import DataLoaderException

from tests.factories import make_asset_metadata


def make_time_series_input(series: pd.Series) -> TimeSeriesInput:
    return TimeSeriesInput(
        metadata=make_asset_metadata(),
        series=series,
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
    preprocessor = DataPreprocessor(data)
    preprocessor._preprocess()

    assert preprocessor.data.series.dtype == float
    assert preprocessor.data.series.isna().sum() == 0
    assert list(preprocessor.data.series.values) == [100.0, 100.0, 102.0]


def test_returns_series_computes_simple_returns() -> None:
    series = pd.Series(
        [100.0, 110.0, 121.0],
        index=pd.date_range("2026-01-01", periods=3, freq="15min"),
    )
    data = make_time_series_input(series)
    preprocessor = DataPreprocessor(data)
    preprocessor._returns_series()

    # retorno: (P_t - P_{t-1}) / P_t
    expected = pd.Series(
        [10.0 / 110.0, 11.0 / 121.0],
        index=series.index[1:],
        dtype=float,
    )
    pd.testing.assert_series_equal(preprocessor.data.returns, expected, check_names=False)  # type: ignore


def test_returns_series_drops_leading_nan() -> None:
    series = pd.Series(
        [50.0, 55.0, 60.5],
        index=pd.date_range("2026-01-01", periods=3, freq="15min"),
    )
    data = make_time_series_input(series)
    preprocessor = DataPreprocessor(data)
    preprocessor._returns_series()

    assert preprocessor.data.returns is not None
    assert preprocessor.data.returns.isna().sum() == 0
    assert len(preprocessor.data.returns) == 2


def test_innovations_series_is_centered_returns() -> None:
    series = pd.Series(
        [100.0, 104.0, 108.16, 112.486],
        index=pd.date_range("2026-01-01", periods=4, freq="15min"),
    )
    data = make_time_series_input(series)
    preprocessor = DataPreprocessor(data)
    preprocessor._returns_series()
    preprocessor._innovations_series()

    returns = preprocessor.data.returns
    innovations = preprocessor.data.innovations
    assert innovations is not None
    assert pytest.approx(float(innovations.mean()), abs=1e-12) == 0.0
    pd.testing.assert_series_equal(
        innovations, returns - returns.mean(), check_names=False  # type: ignore
    )


def test_volatility_series_computes_annualized_rolling_std() -> None:
    rng = np.random.default_rng(0)
    prices = pd.Series(
        100.0 * np.exp(np.cumsum(rng.normal(0, 0.005, 300))),
        index=pd.date_range("2020-01-01", periods=300, freq="15min"),
        dtype=float,
    )
    data = make_time_series_input(prices)
    preprocessor = DataPreprocessor(data)
    preprocessor._returns_series()
    preprocessor._volatility_series(omega=20)

    vol = preprocessor.data.volatility_series
    assert vol is not None
    assert vol.isna().sum() == 0
    # 300 precios -> 299 retornos; rolling(20) con iloc[19:-1] -> 299 - 20 = 279 valores
    assert len(vol) == 279


def test_volatility_series_matches_manual_calculation() -> None:
    rng = np.random.default_rng(7)
    prices = pd.Series(
        100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 30))),
        index=pd.date_range("2026-01-01", periods=30, freq="15min"),
    )
    data = make_time_series_input(prices)
    preprocessor = DataPreprocessor(data)
    preprocessor._returns_series()
    preprocessor._volatility_series(omega=5)

    returns = preprocessor.data.returns
    # replica el mismo slice que usa _volatility_series: iloc[omega-1:-1]
    expected = (
        returns.rolling(window=5).std(ddof=1) * np.sqrt(252)
    ).iloc[4:-1]
    pd.testing.assert_series_equal(
        preprocessor.data.volatility_series, expected, check_names=False  # type: ignore
    )


def test_pipeline_populates_returns_innovations_and_volatility() -> None:
    rng = np.random.default_rng(1)
    prices = pd.Series(
        100.0 * np.exp(np.cumsum(rng.normal(0, 0.005, 300))),
        index=pd.date_range("2025-01-01", periods=300, freq="15min"),
        dtype=float,
    )
    data = make_time_series_input(prices)

    result = DataPreprocessor(data).pipeline()

    assert result.returns is not None and len(result.returns) > 0
    assert result.innovations is not None and len(result.innovations) > 0
    assert result.volatility_series is not None and len(result.volatility_series) > 0
    assert result.returns.isna().sum() == 0
    assert result.volatility_series.isna().sum() == 0


def test_pipeline_volatility_has_no_leading_nans_from_rolling_window() -> None:
    rng = np.random.default_rng(2)
    prices = pd.Series(
        100.0 * np.exp(np.cumsum(rng.normal(0, 0.005, 300))),
        index=pd.date_range("2025-01-01", periods=300, freq="15min"),
        dtype=float,
    )
    result = DataPreprocessor(make_time_series_input(prices)).pipeline()

    assert result.volatility_series is not None
    assert result.volatility_series.isna().sum() == 0