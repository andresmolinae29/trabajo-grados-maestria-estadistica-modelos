import numpy as np
import pandas as pd
import torch

from PyEMD import CEEMDAN
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .base import BaseVolatilityModel

from ..schemas import (
    ModelConfig,
    PredictionResult,
    AssetMetadata,
    PredictionRow,
)
from ..utils import logger


class _LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=effective_dropout,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        sequence_output, _ = self.lstm(features)
        return self.output(sequence_output[:, -1, :])


class CEEMDANLSTMModel(BaseVolatilityModel):

    name = "CEEMDAN-LSTM"

    def __init__(self, config: ModelConfig, asset_metadata: AssetMetadata):

        super().__init__(config, asset_metadata)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(
            "Using device for CEEMDAN-LSTM: %s (torch=%s, cuda_available=%s)",
            self.device,
            torch.__version__,
            torch.cuda.is_available(),
        )
        self.models: list[_LSTMRegressor] = []
        self.train_imfs: list[np.ndarray] = []
        self.best_hyperparameters: dict = {}
        self.window_size: int = 0

    def __set_random_seed(self) -> None:

        torch.manual_seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.random_seed)

    def __decompose_series(self, X: pd.Series | np.ndarray) -> list[np.ndarray]:

        values = np.asarray(X, dtype=np.float32)
        logger.info("Starting CEEMDAN decomposition for %s observations.", len(values))
        ceemdan = CEEMDAN()
        imfs = ceemdan(values)
        logger.info("Completed CEEMDAN decomposition with %s IMFs.", len(imfs))
        return [np.asarray(imf, dtype=np.float32) for imf in imfs]

    def __resolve_window_size(self, series_length: int, hyperparameters: dict) -> int:

        if series_length < 4:
            raise ValueError("CEEMDAN-LSTM requires at least 4 observations.")

        requested_window = int(hyperparameters.get("window_size", 20))
        return max(2, min(requested_window, series_length - 2))

    def __make_windows(self, series: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:

        if len(series) <= window_size:
            raise ValueError(
                f"Series length {len(series)} is not enough for window size {window_size}."
            )

        features = []
        targets = []

        for index in range(len(series) - window_size):
            features.append(series[index : index + window_size])
            targets.append(series[index + window_size])

        return np.asarray(features, dtype=np.float32), np.asarray(targets, dtype=np.float32)

    def __build_model(self, hyperparameters: dict) -> _LSTMRegressor:

        return _LSTMRegressor(
            input_size=1,
            hidden_size=int(hyperparameters.get("hidden_size", 32)),
            num_layers=int(hyperparameters.get("num_layers", 1)),
            dropout=float(hyperparameters.get("dropout", 0.0)),
        ).to(self.device)

    def __train_single_imf_model(
        self,
        imf: np.ndarray,
        hyperparameters: dict,
        imf_index: int | None = None,
        total_imfs: int | None = None,
    ) -> _LSTMRegressor:

        window_size = self.__resolve_window_size(len(imf), hyperparameters)
        features, targets = self.__make_windows(imf, window_size)

        dataset = TensorDataset(
            torch.from_numpy(features).unsqueeze(-1),
            torch.from_numpy(targets).unsqueeze(-1),
        )

        batch_size = min(int(hyperparameters.get("batch_size", 32)), len(dataset))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = self.__build_model(hyperparameters)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(hyperparameters.get("learning_rate", 1e-3)),
        )
        loss_fn = nn.MSELoss()
        epochs = int(hyperparameters.get("epochs", 25))
        progress_label = (
            f"IMF {imf_index}/{total_imfs}"
            if imf_index is not None and total_imfs is not None
            else "IMF"
        )

        logger.info(
            "Training %s with %s windows, batch_size=%s, epochs=%s, device=%s.",
            progress_label,
            len(dataset),
            batch_size,
            epochs,
            self.device,
        )

        model.train()
        log_every = max(1, epochs // 5)
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_features, batch_targets in loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)

                optimizer.zero_grad()
                predictions = model(batch_features)
                loss = loss_fn(predictions, batch_targets)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())

            should_log_epoch = epoch == 0 or epoch == epochs - 1 or (epoch + 1) % log_every == 0
            if should_log_epoch:
                average_loss = epoch_loss / max(1, len(loader))
                logger.info(
                    "%s training progress: epoch %s/%s, avg_loss=%.6f",
                    progress_label,
                    epoch + 1,
                    epochs,
                    average_loss,
                )

        logger.info("Finished training %s.", progress_label)

        return model

    def __forecast_single_imf(self, model: _LSTMRegressor, imf: np.ndarray, horizon: int, window_size: int) -> np.ndarray:

        history = list(np.asarray(imf, dtype=np.float32))
        forecasts = []

        logger.info("Forecasting %s steps from IMF history of length %s.", horizon, len(history))

        model.eval()
        with torch.no_grad():
            for _ in range(horizon):
                window = np.asarray(history[-window_size:], dtype=np.float32).reshape(1, window_size, 1)
                features = torch.from_numpy(window).to(self.device)
                prediction = float(model(features).item())
                forecasts.append(prediction)
                history.append(prediction)

        logger.info("Finished forecasting %s steps.", horizon)

        return np.asarray(forecasts, dtype=np.float32)

    def __score_hyperparameters(self, X: pd.Series, hyperparameters: dict) -> float:

        window_size = self.__resolve_window_size(len(X), hyperparameters)
        max_validation_size = len(X) - window_size - 1

        if max_validation_size < 1:
            raise ValueError("Not enough observations to create a validation split.")

        validation_size = int(hyperparameters.get("validation_size", max(1, int(len(X) * 0.2))))
        validation_size = max(1, min(validation_size, max_validation_size))

        train_series = X.iloc[:-validation_size]
        validation_series = X.iloc[-validation_size:]
        train_imfs = self.__decompose_series(train_series)

        component_forecasts = []
        total_imfs = len(train_imfs)
        for imf_index, imf in enumerate(train_imfs, start=1):
            model = self.__train_single_imf_model(
                imf,
                hyperparameters,
                imf_index=imf_index,
                total_imfs=total_imfs,
            )
            component_forecasts.append(
                self.__forecast_single_imf(model, imf, horizon=validation_size, window_size=window_size)
            )

        aggregated_forecast = np.sum(np.vstack(component_forecasts), axis=0)
        validation_values = validation_series.to_numpy(dtype=np.float32)

        return float(np.sqrt(np.mean((aggregated_forecast - validation_values) ** 2)))

    def __select_hyperparameters(self, X: pd.Series) -> dict:

        hyperparameter_candidates = self.config.hyperparameters_list or [{}]
        best_score = float("inf")
        best_hyperparameters = None

        for hyperparameters in hyperparameter_candidates:
            try:
                logger.info(f"Evaluating CEEMDAN-LSTM hyperparameters: {hyperparameters}")
                score = self.__score_hyperparameters(X, hyperparameters)
                if score < best_score:
                    best_score = score
                    best_hyperparameters = hyperparameters
            except Exception as error:
                logger.warning(
                    f"Error scoring CEEMDAN-LSTM hyperparameters {hyperparameters}: {error}"
                )

        if best_hyperparameters is None:
            raise ValueError("No valid CEEMDAN-LSTM hyperparameter configuration could be fitted.")

        return best_hyperparameters

    def fit(self, X: pd.Series, y: pd.Series | None = None) -> None:

        self.__set_random_seed()
        self.best_hyperparameters = self.__select_hyperparameters(X)
        logger.info(f"Selected CEEMDAN-LSTM hyperparameters: {self.best_hyperparameters}")
        self.window_size = self.__resolve_window_size(len(X), self.best_hyperparameters)
        self.train_imfs = self.__decompose_series(X)
        total_imfs = len(self.train_imfs)
        self.models = []
        for imf_index, imf in enumerate(self.train_imfs, start=1):
            self.models.append(
                self.__train_single_imf_model(
                    imf,
                    self.best_hyperparameters,
                    imf_index=imf_index,
                    total_imfs=total_imfs,
                )
            )
        self.is_fitted = True

    def predict(self, X: pd.Series, y: pd.Series) -> PredictionResult:

        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")

        horizon = len(y)
        component_forecasts = [
            self.__forecast_single_imf(model, imf, horizon, self.window_size)
            for model, imf in zip(self.models, self.train_imfs)
        ]
        values = np.sum(np.vstack(component_forecasts), axis=0)
        timestamps = y.index

        rows = [
            PredictionRow(
                timestamp=ts,
                predicted_volatility=float(val),
                lower_ci=None,
                upper_ci=None,
            )
            for ts, val in zip(timestamps, values)
        ]

        return PredictionResult(
            model_name=self.name,
            asset=self.asset_metadata.symbol,
            horizon=horizon,
            rows=rows,
        )


if __name__ == "__main__":
    pass
