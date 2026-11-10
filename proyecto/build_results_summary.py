"""Summarizes an experiment folder into the CSV files behind the thesis tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "src" / "finance_modeling" / "results" / "models"
SYMBOLS = ["BTC-USD", "ETH-USD", "EURUSD=X", "GC=F", "^GSPC", "^IXIC"]
MODELS = ["NAIVE", "GARCH", "CEEMDAN_LSTM", "PSOQRNN"]
BASELINES = ["NAIVE", "GARCH"]
TARGET_WINDOW = 252


def hac_dm(differential: np.ndarray, lags: int) -> tuple[float, float]:
    """Diebold-Mariano statistic with a Bartlett long-run variance."""
    n = len(differential)
    centered = differential - differential.mean()
    variance = centered @ centered / n
    for lag in range(1, lags + 1):
        variance += 2 * (1 - lag / (lags + 1)) * (centered[lag:] @ centered[:-lag]) / n
    statistic = differential.mean() / np.sqrt(variance / n)
    return float(statistic), float(2 * stats.norm.sf(abs(statistic)))


def load_common(folder: Path, symbol: str) -> pd.DataFrame:
    frames = {
        model: pd.read_csv(folder / f"{model}_{symbol}_predictions.csv", parse_dates=["timestamp"]).set_index("timestamp")
        for model in MODELS
    }
    common = pd.DataFrame({model: frame["predicted_value"] for model, frame in frames.items()}).dropna()
    common["real"] = frames["NAIVE"]["real_value"].loc[common.index]
    return common


def metrics_rows(symbol: str, common: pd.DataFrame) -> list[dict]:
    naive_rmse = np.sqrt(((common["NAIVE"] - common["real"]) ** 2).mean())
    rows = []
    for model in MODELS:
        error = common[model] - common["real"]
        half = len(common) // 2
        rmse = np.sqrt((error ** 2).mean())
        rows.append({
            "symbol": symbol,
            "model": model,
            "n": len(common),
            "first_date": common.index.min(),
            "last_date": common.index.max(),
            "mae": error.abs().mean(),
            "rmse": rmse,
            "rmse_over_naive": rmse / naive_rmse,
            "bias": error.mean(),
            "bias_share_mse": error.mean() ** 2 / (error ** 2).mean(),
            "correlation": common[model].corr(common["real"]),
            "std_ratio": common[model].std() / common["real"].std(),
            "rmse_first_half": np.sqrt((error.iloc[:half] ** 2).mean()),
            "rmse_second_half": np.sqrt((error.iloc[half:] ** 2).mean()),
            "prediction_min": common[model].min(),
            "prediction_max": common[model].max(),
            "real_min": common["real"].min(),
            "real_max": common["real"].max(),
            "real_mean": common["real"].mean(),
        })
    return rows


def comparison_rows(folder: Path, symbol: str, common: pd.DataFrame) -> list[dict]:
    lags = min(TARGET_WINDOW, len(common) // 4)
    rows = []
    for baseline in BASELINES:
        for model in MODELS:
            if model == baseline:
                continue
            loss = (common[baseline] - common["real"]) ** 2 - (common[model] - common["real"]) ** 2
            statistic, p_value = hac_dm(loss.to_numpy(), lags)
            saved = json.loads((folder / f"{baseline}_vs_{model}_{symbol}_comparison.json").read_text())
            rows.append({
                "symbol": symbol,
                "baseline": baseline,
                "model": model,
                "lags": lags,
                "loss_autocorrelation": loss.autocorr(1),
                "dm_hac": statistic,
                "dm_hac_p_value": p_value,
                "dm_simple": saved["dm_statistic"],
                "dm_simple_p_value": saved["dm_p_value"],
            })
    return rows


def network_rows(folder: Path, symbol: str) -> list[dict]:
    rows = []
    for model in ["CEEMDAN_LSTM", "PSOQRNN"]:
        seeds = pd.read_csv(folder / f"{model}_{symbol}_seeds.csv")
        tuning = pd.read_csv(folder / f"{model}_{symbol}_tuning.csv")
        chosen = tuning[tuning["chosen"]].iloc[0]
        keys = [c for c in tuning.columns if c not in ("configuration", "seed", "validation_rmse", "chosen")]
        rows.append({
            "symbol": symbol,
            "model": model,
            **{key: chosen[key] for key in keys},
            "chosen_seed": int(chosen["seed"]),
            "validation_rmse": chosen["validation_rmse"],
            "test_rmse_chosen": seeds.loc[seeds["chosen"], "rmse"].iloc[0],
            "test_rmse_mean": seeds["rmse"].mean(),
            "test_rmse_std": seeds["rmse"].std(ddof=1),
            "test_rmse_min": seeds["rmse"].min(),
            "test_rmse_max": seeds["rmse"].max(),
            "chosen_rank_in_test": int(seeds["rmse"].rank().loc[seeds["chosen"]].iloc[0]),
            **{f"rmse_seed_{int(row.seed)}": row.rmse for row in seeds.itertuples()},
        })
    return rows


def main(experiment: str) -> None:
    folder = RESULTS / experiment
    metrics, comparisons, networks = [], [], []
    for symbol in SYMBOLS:
        common = load_common(folder, symbol)
        metrics += metrics_rows(symbol, common)
        comparisons += comparison_rows(folder, symbol, common)
        networks += network_rows(folder, symbol)
    pd.DataFrame(metrics).to_csv(folder / "summary_metrics.csv", index=False)
    pd.DataFrame(comparisons).to_csv(folder / "summary_comparisons.csv", index=False)
    pd.DataFrame(networks).to_csv(folder / "summary_networks.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="20261031")
    main(parser.parse_args().experiment)
