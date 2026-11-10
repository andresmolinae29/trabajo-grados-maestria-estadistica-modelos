from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from finance_modeling.evaluation import Metrics
from finance_modeling.models import ModelFactory
from finance_modeling.models.ceemdan_lstm import causal_ceemdan_features, ceemdan_features
from finance_modeling.schemas import ModelConfig
from finance_modeling.utils import logger


TUNED_MODELS = {"PSOQRNN", "CEEMDAN_LSTM"}
TRAIN_RATIO = 0.8
OMEGA = 252


@dataclass
class TuningResult:
    hyperparameters: dict
    seed: int
    table: pd.DataFrame


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def candidates(model_config: ModelConfig) -> list[dict]:
    # La grilla fija va primero; la búsqueda aleatoria agrega n_trials combinaciones distintas del espacio.
    options = [dict(hyperparameters) for hyperparameters in model_config.hyperparameters_list]
    if not model_config.search_space:
        return options

    rng = np.random.default_rng(model_config.random_seed)
    sampled: list[dict] = []
    for _ in range(100 * model_config.n_trials):
        if len(sampled) == model_config.n_trials:
            break
        option = {key: values[rng.integers(len(values))] for key, values in model_config.search_space.items()}
        if option not in options and option not in sampled:
            sampled.append(option)
    return options + sampled


def is_tunable(model_config: ModelConfig) -> bool:
    return model_config.name in TUNED_MODELS and len(candidates(model_config)) > 1


def validation_kwargs(model_kwargs: dict) -> dict:
    # Con el primer 80 %, la partición 80/20 de cada modelo deja 64 % de entrenamiento y 16 % de validación.
    volatility = model_kwargs["volatility"]
    train_volatility = volatility.iloc[:int(len(volatility) * TRAIN_RATIO)]
    return {
        **model_kwargs,
        "volatility": train_volatility,
        "returns": model_kwargs["returns"].iloc[:len(train_volatility) + OMEGA],
    }


def fit_and_predict(model_config: ModelConfig, symbol: str, model_kwargs: dict, seed: int):
    set_seed(seed)
    model = ModelFactory.create_model(
        model_name=model_config.name,
        model_config=model_config,
        symbol=symbol,
        **{**model_kwargs, "seed": seed},
    )
    model.fit()
    return model, model.predict()


def tune_model(model_config: ModelConfig, symbol: str, model_kwargs: dict) -> TuningResult:
    inputs = validation_kwargs(model_kwargs)
    options = candidates(model_config)
    rows = []
    decompose = model_config.name == "CEEMDAN_LSTM"

    if decompose and model_config.causal:
        # Es causal: los primeros valores de la serie completa coinciden con los de la serie de validación.
        inputs["imfs"] = causal_ceemdan_features(model_kwargs["returns"], symbol)[:len(inputs["returns"])]

    for seed in model_config.seeds:
        if decompose and not model_config.causal:
            inputs["imfs"] = ceemdan_features(inputs["returns"], seed)

        for position, hyperparameters in enumerate(options):
            model, predictions = fit_and_predict(model_config, symbol, {**inputs, **hyperparameters}, seed)
            rmse = Metrics.root_mean_squared_error(
                np.asarray(model.y_test, dtype=float).ravel().tolist(),
                np.asarray(predictions, dtype=float).ravel().tolist(),
            )
            logger.info(f"[{model_config.name}] {hyperparameters} seed={seed} validation RMSE={rmse:.6f}")
            rows.append({"configuration": position, **hyperparameters, "seed": seed, "validation_rmse": rmse})

    table = pd.DataFrame(rows)
    best_position = int(table.groupby("configuration")["validation_rmse"].mean().idxmin())
    best_rows = table[table["configuration"] == best_position]
    best_seed = int(best_rows.loc[best_rows["validation_rmse"].idxmin(), "seed"])
    table["chosen"] = (table["configuration"] == best_position) & (table["seed"] == best_seed)

    return TuningResult(options[best_position], best_seed, table)


def seed_variability(
    y_true: pd.Series,
    predictions_by_seed: dict[int, pd.Series],
    chosen_seed: int,
) -> pd.DataFrame:
    rows = []
    for seed, predictions in predictions_by_seed.items():
        aligned = predictions.loc[y_true.index]
        rows.append({
            "seed": seed,
            "rmse": Metrics.root_mean_squared_error(y_true.tolist(), aligned.tolist()),
            "mae": Metrics.mean_absolute_error(y_true.tolist(), aligned.tolist()),
            "chosen": seed == chosen_seed,
        })
    return pd.DataFrame(rows)
