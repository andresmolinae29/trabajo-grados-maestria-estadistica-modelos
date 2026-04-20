from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .base import BaseVolatilityModel, _Regressor

from ..schemas import AssetMetadata, ModelConfig, PredictionResult, PredictionRow
from ..utils import logger


class PSOQRNNModel(BaseVolatilityModel):

    INTEGER_HYPERPARAMETERS = {
        "window_size",
        "hidden_size",
        "num_layers",
        "batch_size",
        "epochs",
        "validation_size",
    }
    PSO_PARAMETER_DEFAULTS = {
        "pso_particles": 6,
        "pso_iterations": 5,
        "pso_inertia": 0.7,
        "pso_cognitive": 1.5,
        "pso_social": 1.5,
    }
    RESERVED_PSO_KEYS = set(PSO_PARAMETER_DEFAULTS)

    name = "PSO-QRNN"

    def __init__(self, config: ModelConfig, asset_metadata: AssetMetadata):

        super().__init__(config, asset_metadata)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.window_size: int = 0
        self.quantiles: list[float] = []
        self.model_state: dict = {}
        self.training_dataset: TensorDataset | None = None
        self.model: _Regressor | None = None
        self.train_series: pd.Series | None = None

        logger.info(
            "Using device for PSO-QRNN: %s (torch=%s, cuda_available=%s)",
            self.device,
            torch.__version__,
            torch.cuda.is_available(),
        )

    def __set_random_seed(self) -> None:

        torch.manual_seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.random_seed)

    def __resolve_window_size(self, series_length: int, hyperparameters: dict) -> int:

        if series_length < 4:
            raise ValueError("PSO-QRNN requires at least 4 observations.")

        requested_window = int(hyperparameters.get("window_size", 20))
        return max(2, min(requested_window, series_length - 2))

    def __resolve_quantiles(self, hyperparameters: dict) -> list[float]:

        quantiles = hyperparameters.get("quantiles", [0.1, 0.5, 0.9])
        resolved_quantiles = sorted({float(quantile) for quantile in quantiles})

        if not resolved_quantiles or any(not 0 < quantile < 1 for quantile in resolved_quantiles):
            raise ValueError("PSO-QRNN quantiles must be between 0 and 1.")

        return resolved_quantiles

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

    def __build_training_dataset(self, X: pd.Series, hyperparameters: dict) -> TensorDataset:

        window_size = self.__resolve_window_size(len(X), hyperparameters)
        features, targets = self.__make_windows(X.to_numpy(dtype=np.float32), window_size)

        return self.__build_tensor_dataset(features, targets, window_size)

    def __build_tensor_dataset(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        window_size: int,
    ) -> TensorDataset:

        logger.info(
            "Prepared PSO-QRNN training dataset with %s windows, window_size=%s, quantiles=%s.",
            len(features),
            window_size,
            self.quantiles,
        )

        return TensorDataset(
            torch.from_numpy(features).unsqueeze(-1),
            torch.from_numpy(targets).unsqueeze(-1),
        )

    def __build_model(self, hyperparameters: dict) -> _Regressor:

        return _Regressor(
            input_size=1,
            hidden_size=int(hyperparameters.get("hidden_size", 32)),
            num_layers=int(hyperparameters.get("num_layers", 1)),
            dropout=float(hyperparameters.get("dropout", 0.0)),
            output_size=len(self.quantiles),
        ).to(self.device)

    def __quantile_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        quantile_tensor = torch.tensor(
            self.quantiles,
            dtype=predictions.dtype,
            device=predictions.device,
        ).view(1, -1)
        expanded_targets = targets.expand(-1, len(self.quantiles))
        errors = expanded_targets - predictions
        return torch.maximum(quantile_tensor * errors, (quantile_tensor - 1) * errors).mean()

    def __train_model(self, dataset: TensorDataset, hyperparameters: dict) -> _Regressor:

        model = self.__build_model(hyperparameters)
        batch_size = min(int(hyperparameters.get("batch_size", 32)), len(dataset))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(hyperparameters.get("learning_rate", 1e-3)),
        )
        epochs = int(hyperparameters.get("epochs", 25))
        log_every = max(1, epochs // 5)

        logger.info(
            "Training PSO-QRNN surrogate with %s windows, batch_size=%s, epochs=%s, device=%s.",
            len(dataset),
            batch_size,
            epochs,
            self.device,
        )

        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_features, batch_targets in loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)

                optimizer.zero_grad()
                predictions = model(batch_features)
                loss = self.__quantile_loss(predictions, batch_targets)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())

            should_log_epoch = epoch == 0 or epoch == epochs - 1 or (epoch + 1) % log_every == 0
            if should_log_epoch:
                average_loss = epoch_loss / max(1, len(loader))
                logger.info(
                    "PSO-QRNN training progress: epoch %s/%s, avg_quantile_loss=%.6f",
                    epoch + 1,
                    epochs,
                    average_loss,
                )

        logger.info("Finished PSO-QRNN surrogate training.")
        return model

    def __select_prediction_indices(self) -> tuple[int, int, int]:

        median_index = min(
            range(len(self.quantiles)),
            key=lambda index: abs(self.quantiles[index] - 0.5),
        )
        lower_index = 0
        upper_index = len(self.quantiles) - 1
        return lower_index, median_index, upper_index

    def __score_forecast(self, predictions: np.ndarray, targets: np.ndarray) -> float:

        _, median_index, _ = self.__select_prediction_indices()
        median_predictions = predictions[:, median_index]
        return float(np.sqrt(np.mean((median_predictions - targets) ** 2)))

    def __forecast_quantiles_with_model(
        self,
        model: _Regressor,
        history_series: pd.Series,
        horizon: int,
        window_size: int,
    ) -> np.ndarray:

        history = list(history_series.to_numpy(dtype=np.float32))
        forecast_rows = []
        _, median_index, _ = self.__select_prediction_indices()

        model.eval()
        with torch.no_grad():
            for _ in range(horizon):
                window = np.asarray(history[-window_size:], dtype=np.float32).reshape(1, window_size, 1)
                features = torch.from_numpy(window).to(self.device)
                quantile_predictions = model(features).squeeze(0).detach().cpu().numpy()
                quantile_predictions = np.asarray(quantile_predictions, dtype=np.float32)
                quantile_predictions = np.maximum.accumulate(quantile_predictions)
                forecast_rows.append(quantile_predictions)
                history.append(float(quantile_predictions[median_index]))

        return np.asarray(forecast_rows, dtype=np.float32)

    def __forecast_quantiles(self, X: pd.Series, horizon: int) -> np.ndarray:

        if self.model is None:
            raise ValueError("Model must be fitted before forecasting.")

        forecast_rows = self.__forecast_quantiles_with_model(
            self.model,
            X,
            horizon,
            self.window_size,
        )

        logger.info("Finished PSO-QRNN forecasting for %s steps.", horizon)
        return forecast_rows

    def __score_hyperparameters(self, X: pd.Series, hyperparameters: dict) -> float:

        window_size = self.__resolve_window_size(len(X), hyperparameters)
        max_validation_size = len(X) - window_size - 1

        if max_validation_size < 1:
            raise ValueError("Not enough observations to create a validation split.")

        validation_size = int(hyperparameters.get("validation_size", max(1, int(len(X) * 0.2))))
        validation_size = max(1, min(validation_size, max_validation_size))

        train_series = X.iloc[:-validation_size]
        validation_series = X.iloc[-validation_size:]
        features, targets = self.__make_windows(
            train_series.to_numpy(dtype=np.float32),
            window_size,
        )
        dataset = self.__build_tensor_dataset(features, targets, window_size)
        model = self.__train_model(dataset, hyperparameters)
        forecast_matrix = self.__forecast_quantiles_with_model(
            model,
            train_series,
            validation_size,
            window_size,
        )
        score = self.__score_forecast(
            forecast_matrix,
            validation_series.to_numpy(dtype=np.float32),
        )
        logger.info(
            "PSO-QRNN validation RMSE for hyperparameters %s: %.6f",
            hyperparameters,
            score,
        )
        return score

    def __sanitize_hyperparameters(self, hyperparameters: dict) -> dict:

        return {
            key: value
            for key, value in hyperparameters.items()
            if key not in self.RESERVED_PSO_KEYS
        }

    def __resolve_pso_settings(self, hyperparameter_candidates: list[dict]) -> dict:

        settings = dict(self.PSO_PARAMETER_DEFAULTS)
        for hyperparameters in hyperparameter_candidates:
            for key in self.RESERVED_PSO_KEYS:
                if key in hyperparameters:
                    settings[key] = hyperparameters[key]

        settings["pso_particles"] = max(1, int(settings["pso_particles"]))
        settings["pso_iterations"] = max(1, int(settings["pso_iterations"]))
        settings["pso_inertia"] = float(settings["pso_inertia"])
        settings["pso_cognitive"] = float(settings["pso_cognitive"])
        settings["pso_social"] = float(settings["pso_social"])
        return settings

    def __extract_search_space(self, hyperparameter_candidates: list[dict]) -> tuple[dict, list[str]]:

        numeric_keys = set()
        for hyperparameters in hyperparameter_candidates:
            numeric_keys.update(
                key
                for key, value in hyperparameters.items()
                if key not in self.RESERVED_PSO_KEYS and key != "quantiles" and isinstance(value, (int, float))
            )

        variable_keys = []
        search_space = {}

        for key in sorted(numeric_keys):
            values = [float(hyperparameters[key]) for hyperparameters in hyperparameter_candidates if key in hyperparameters]
            lower_bound = min(values)
            upper_bound = max(values)
            if np.isclose(lower_bound, upper_bound):
                search_space[key] = {"type": "fixed", "value": lower_bound}
            else:
                variable_keys.append(key)
                search_space[key] = {"type": "variable", "lower": lower_bound, "upper": upper_bound}

        return search_space, variable_keys

    def __select_base_hyperparameters(self, hyperparameter_candidates: list[dict]) -> dict:

        base_hyperparameters = dict(self.__sanitize_hyperparameters(hyperparameter_candidates[0]))

        if "quantiles" not in base_hyperparameters:
            for hyperparameters in hyperparameter_candidates:
                if "quantiles" in hyperparameters:
                    base_hyperparameters["quantiles"] = hyperparameters["quantiles"]
                    break

        return base_hyperparameters

    def __materialize_hyperparameters(
        self,
        position: np.ndarray,
        variable_keys: list[str],
        search_space: dict,
        base_hyperparameters: dict,
    ) -> dict:

        resolved_hyperparameters = dict(base_hyperparameters)
        for key, spec in search_space.items():
            if spec["type"] == "fixed":
                value = spec["value"]
                if key in self.INTEGER_HYPERPARAMETERS:
                    value = int(round(value))
                resolved_hyperparameters[key] = value

        for index, key in enumerate(variable_keys):
            spec = search_space[key]
            raw_value = float(np.clip(position[index], spec["lower"], spec["upper"]))
            if key in self.INTEGER_HYPERPARAMETERS:
                resolved_hyperparameters[key] = int(round(raw_value))
            else:
                resolved_hyperparameters[key] = raw_value

        return resolved_hyperparameters

    def __candidate_to_position(self, hyperparameters: dict, variable_keys: list[str]) -> np.ndarray:

        return np.asarray([float(hyperparameters[key]) for key in variable_keys], dtype=np.float32)

    def __evaluate_particle(
        self,
        X: pd.Series,
        position: np.ndarray,
        variable_keys: list[str],
        search_space: dict,
        base_hyperparameters: dict,
        score_cache: dict,
    ) -> tuple[float, dict]:

        resolved_hyperparameters = self.__materialize_hyperparameters(
            position,
            variable_keys,
            search_space,
            base_hyperparameters,
        )
        cache_key = tuple(sorted((key, repr(value)) for key, value in resolved_hyperparameters.items()))
        if cache_key in score_cache:
            return score_cache[cache_key], resolved_hyperparameters

        previous_quantiles = list(self.quantiles)
        try:
            self.quantiles = self.__resolve_quantiles(resolved_hyperparameters)
            score = self.__score_hyperparameters(X, resolved_hyperparameters)
        finally:
            self.quantiles = previous_quantiles

        score_cache[cache_key] = score
        return score, resolved_hyperparameters

    def __run_particle_swarm_optimization(
        self,
        X: pd.Series,
        hyperparameter_candidates: list[dict],
    ) -> dict:

        sanitized_candidates = [self.__sanitize_hyperparameters(candidate) for candidate in hyperparameter_candidates]
        pso_settings = self.__resolve_pso_settings(hyperparameter_candidates)
        base_hyperparameters = self.__select_base_hyperparameters(sanitized_candidates)
        search_space, variable_keys = self.__extract_search_space(sanitized_candidates)

        if not variable_keys:
            logger.info("PSO-QRNN search space has no variable numeric hyperparameters; using fixed configuration.")
            return base_hyperparameters

        particle_count = max(int(pso_settings["pso_particles"]), len(sanitized_candidates))
        iteration_count = int(pso_settings["pso_iterations"])
        inertia = float(pso_settings["pso_inertia"])
        cognitive = float(pso_settings["pso_cognitive"])
        social = float(pso_settings["pso_social"])
        bounds = np.asarray(
            [[search_space[key]["lower"], search_space[key]["upper"]] for key in variable_keys],
            dtype=np.float32,
        )

        logger.info(
            "Running PSO-QRNN hyperparameter search with particles=%s, iterations=%s, dimensions=%s.",
            particle_count,
            iteration_count,
            len(variable_keys),
        )

        positions = np.zeros((particle_count, len(variable_keys)), dtype=np.float32)
        velocities = np.zeros_like(positions)
        candidate_positions = [self.__candidate_to_position(candidate, variable_keys) for candidate in sanitized_candidates]
        for index in range(particle_count):
            if index < len(candidate_positions):
                positions[index] = candidate_positions[index]
            else:
                random_unit = np.random.rand(len(variable_keys)).astype(np.float32)
                positions[index] = bounds[:, 0] + random_unit * (bounds[:, 1] - bounds[:, 0])

        personal_best_positions = positions.copy()
        personal_best_scores = np.full(particle_count, np.inf, dtype=np.float32)
        global_best_position: np.ndarray | None = None
        global_best_score = float("inf")
        global_best_hyperparameters: dict | None = None
        score_cache: dict = {}

        for particle_index in range(particle_count):
            score, resolved_hyperparameters = self.__evaluate_particle(
                X,
                positions[particle_index],
                variable_keys,
                search_space,
                base_hyperparameters,
                score_cache,
            )
            personal_best_scores[particle_index] = score
            if score < global_best_score:
                global_best_score = score
                global_best_position = positions[particle_index].copy()
                global_best_hyperparameters = resolved_hyperparameters

        if global_best_position is None or global_best_hyperparameters is None:
            raise ValueError("PSO-QRNN could not initialize a valid swarm state.")

        for iteration_index in range(iteration_count):
            for particle_index in range(particle_count):
                random_cognitive = np.random.rand(len(variable_keys)).astype(np.float32)
                random_social = np.random.rand(len(variable_keys)).astype(np.float32)
                velocities[particle_index] = (
                    inertia * velocities[particle_index]
                    + cognitive * random_cognitive * (personal_best_positions[particle_index] - positions[particle_index])
                    + social * random_social * (global_best_position - positions[particle_index])
                )
                positions[particle_index] = positions[particle_index] + velocities[particle_index]
                positions[particle_index] = np.clip(positions[particle_index], bounds[:, 0], bounds[:, 1])

                score, resolved_hyperparameters = self.__evaluate_particle(
                    X,
                    positions[particle_index],
                    variable_keys,
                    search_space,
                    base_hyperparameters,
                    score_cache,
                )

                if score < personal_best_scores[particle_index]:
                    personal_best_scores[particle_index] = score
                    personal_best_positions[particle_index] = positions[particle_index].copy()

                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[particle_index].copy()
                    global_best_hyperparameters = resolved_hyperparameters

            logger.info(
                "PSO-QRNN swarm iteration %s/%s, best_validation_rmse=%.6f, best_hyperparameters=%s",
                iteration_index + 1,
                iteration_count,
                global_best_score,
                global_best_hyperparameters,
            )

        return global_best_hyperparameters

    def __select_hyperparameters(self, X: pd.Series) -> dict:

        hyperparameter_candidates = self.config.hyperparameters_list or [{}]
        best_hyperparameters = self.__run_particle_swarm_optimization(X, hyperparameter_candidates)

        if best_hyperparameters is None:
            raise ValueError("No valid PSO-QRNN hyperparameter configuration could be fitted.")

        logger.info(
            "Selected PSO-QRNN hyperparameters by PSO validation search: %s",
            best_hyperparameters,
        )

        self.__resolve_window_size(len(X), best_hyperparameters)
        self.__resolve_quantiles(best_hyperparameters)
        return best_hyperparameters

    def fit(self, X: pd.Series, y: pd.Series | None = None) -> None:

        self.__set_random_seed()
        self.best_hyperparameters = self.__select_hyperparameters(X)
        self.window_size = self.__resolve_window_size(len(X), self.best_hyperparameters)
        self.quantiles = self.__resolve_quantiles(self.best_hyperparameters)
        self.train_series = X.copy()
        self.training_dataset = self.__build_training_dataset(X, self.best_hyperparameters)
        self.model = self.__train_model(self.training_dataset, self.best_hyperparameters)
        self.model_state = {
            "window_size": self.window_size,
            "quantiles": self.quantiles,
            "asset": self.asset_metadata.symbol,
            "device": self.device,
            "training_windows": len(self.training_dataset),
            "hidden_size": int(self.best_hyperparameters.get("hidden_size", 32)),
            "num_layers": int(self.best_hyperparameters.get("num_layers", 1)),
        }
        self.is_fitted = True

        logger.info("PSO-QRNN fit completed without PSO optimization.")

    def predict(self, X: pd.Series, y: pd.Series) -> PredictionResult:

        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction.")

        horizon = len(y)
        forecast_matrix = self.__forecast_quantiles(X, horizon)
        lower_index, median_index, upper_index = self.__select_prediction_indices()

        rows = [
            PredictionRow(
                timestamp=timestamp,
                predicted_volatility=float(forecast_row[median_index]),
                lower_ci=float(forecast_row[lower_index]),
                upper_ci=float(forecast_row[upper_index]),
            )
            for timestamp, forecast_row in zip(y.index, forecast_matrix)
        ]

        return PredictionResult(
            model_name=self.name,
            asset=self.asset_metadata.symbol,
            horizon=horizon,
            rows=rows,
        )


if __name__ == "__main__":
    pass
