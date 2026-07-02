import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from ..schemas import PredictionResult, PredictionRow
from ..utils import logger


def save_forecast_plot(
    prediction_result: PredictionResult,
    experiment_path: str,
) -> None:
    """Guarda una grafica PNG con la comparacion de valores reales vs predichos."""
    timestamps = [row.timestamp for row in prediction_result.rows]
    y_pred = [row.predicted_value for row in prediction_result.rows]
    timestamps_real = [row.timestamp for row in prediction_result.rows if row.real_value is not None]
    y_real = [row.real_value for row in prediction_result.rows if row.real_value is not None]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(timestamps_real, y_real, label="Real", color="steelblue", linewidth=1.2) # type: ignore
    ax.plot(timestamps, y_pred, label="Predicho", color="tomato", linewidth=1.2, linestyle="--") # type: ignore
    ax.set_title(f"{prediction_result.model_name} — {prediction_result.asset}: Volatilidad real vs predicha")
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Volatilidad")
    ax.legend()
    fig.tight_layout()

    file_name = f"{prediction_result.model_name}_{prediction_result.asset}_forecast.png"
    fig.savefig(os.path.join(experiment_path, file_name), dpi=150)
    plt.close(fig)
    logger.info(f"Grafica guardada: {file_name}")


def build_prediction_result(
    model_name: str,
    asset_symbol: str,
    predictions: np.ndarray,
    y_test: np.ndarray,
    vol_index: pd.Index,
) -> PredictionResult:
    """Une predicciones y valores reales en un PredictionResult con timestamps."""
    n = len(predictions)
    timestamps = vol_index[-n:]
    flat_preds = predictions.ravel()
    flat_real = y_test.ravel() if y_test is not None else [None] * n

    rows = [
        PredictionRow(
            timestamp=ts,
            predicted_value=flat_preds[i],
            real_value=flat_real[i],
        )
        for i, ts in enumerate(timestamps)
    ]

    return PredictionResult(
        model_name=model_name,
        asset=asset_symbol,
        horizon=n,
        rows=rows,
    )


def save_prediction_results_to_csv(
    experiment_path: str,
    model_name: str,
    asset_symbol: str,
    predictions: np.ndarray,
    y_test: np.ndarray,
    vol_index: pd.Index) -> None:

    """Guarda los resultados de predicción en un CSV."""

    df = pd.DataFrame({
        "model_name": model_name,
        "asset_symbol": asset_symbol,
        "timestamp": vol_index[-len(predictions):],
        "predicted_value": predictions.ravel(),
        "real_value": y_test.ravel() if y_test is not None else [None] * len(predictions),
    })

    file_name = f"{model_name}_{asset_symbol}_predictions.csv"
    df.to_csv(os.path.join(experiment_path, file_name), index=False)
    logger.info(f"Resultados de predicción guardados: {file_name}")