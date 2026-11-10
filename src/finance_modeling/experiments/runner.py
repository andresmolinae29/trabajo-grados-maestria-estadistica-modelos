import argparse
import os

from functools import reduce

import pandas as pd

from finance_modeling.experiments.network_tuning import (
    fit_and_predict,
    is_tunable,
    seed_variability,
    set_seed,
    tune_model,
)

from finance_modeling.utils import (
    DataLoaderException,
    create_experiment_directory,
    logger,
)
from finance_modeling.config import ConfigLoader
from finance_modeling.config.config import MODEL_CONFIG

from finance_modeling.data import RawDataLoader, DataPreprocessor
from finance_modeling.models import (
    GARCHModel,
    ModelFactory,
    build_prediction_result,
    save_forecast_plot,
    save_prediction_results_to_csv
)
from finance_modeling.evaluation import Evaluator, ModelComparator, save_garch_diagnostics


def main(config_file: str = MODEL_CONFIG, symbols: list[str] | None = None):
    logger.info("Starting the experiment runner...")

    models_config = ConfigLoader().load_model_config(config_file)
    data_config = ConfigLoader().load_data_config()
    garch_selections = ConfigLoader().load_garch_selection()

    experiment_path = create_experiment_directory(models_config.output_dir, models_config.experiment_name)
    baseline_model_names = ["GARCH", "NAIVE"]

    for asset in data_config.assets:
        if not asset.active:
            logger.info(f"Skipping inactive asset: {asset.symbol}")
            continue
        if symbols and asset.symbol not in symbols:
            continue

        logger.info(f"Processing asset: {asset.symbol}")

        try:
            data_loader = RawDataLoader(asset)
            time_series_input = data_loader.load_data()
            time_series_output = DataPreprocessor(time_series_input).pipeline()

            logger.info(f"Finished preprocessing asset: {asset.symbol}")

            evaluation_results_by_model: dict = {}
            y_true_by_model: dict[str, pd.Series] = {}
            y_pred_by_model: dict[str, pd.Series] = {}
            predictions_by_seed: dict[str, dict[int, pd.Series]] = {}
            chosen_seeds: dict[str, int] = {}
            vol_index = time_series_output.volatility_series.index # type: ignore

            for model_config in models_config.models:
                logger.info(f"Running model: {model_config.name} on asset: {asset.symbol}")

                model_kwargs: dict = {
                    "returns": time_series_output.returns,
                    "innovation": time_series_output.innovations,
                    "volatility": time_series_output.volatility_series,
                    "prices": time_series_output.series,
                    "selection": garch_selections.get(asset.symbol),
                }

                tunable = is_tunable(model_config)
                if tunable:
                    tuning = tune_model(model_config, asset.symbol, model_kwargs)
                    tuning.table.to_csv(
                        os.path.join(experiment_path, f"{model_config.name}_{asset.symbol}_tuning.csv"),
                        index=False,
                    )
                    logger.info(f"[{model_config.name}] chosen: {tuning.hyperparameters} seed={tuning.seed}")
                    model_kwargs = {**model_kwargs, **tuning.hyperparameters, "seed": tuning.seed}
                    chosen_seeds[model_config.name] = tuning.seed
                    set_seed(tuning.seed)

                model = ModelFactory.create_model(
                    model_name=model_config.name,
                    model_config=model_config,
                    symbol=asset.symbol,
                    **model_kwargs,
                )
                model.fit()

                if isinstance(model, GARCHModel):
                    save_garch_diagnostics(model, experiment_path)

                model.save_model(experiment_path)

                predictions = model.predict()

                prediction_result = build_prediction_result(
                    model_name=model_config.name,
                    asset_symbol=asset.symbol,
                    predictions=predictions,
                    y_test=model.y_test, # type: ignore
                    vol_index=vol_index,
                )

                save_prediction_results_to_csv(
                    experiment_path=experiment_path,
                    model_name=model_config.name,
                    asset_symbol=asset.symbol,
                    predictions=predictions,
                    y_test=model.y_test,  # type: ignore
                    vol_index=vol_index,
                )

                y_true_by_model[model_config.name] = pd.Series(
                    data=[row.real_value for row in prediction_result.rows],
                    index=[row.timestamp for row in prediction_result.rows],
                    dtype=float,
                )
                y_pred_by_model[model_config.name] = pd.Series(
                    data=[row.predicted_value for row in prediction_result.rows],
                    index=[row.timestamp for row in prediction_result.rows],
                    dtype=float,
                )

                save_forecast_plot(prediction_result, experiment_path)

                if tunable:
                    by_seed = {tuning.seed: y_pred_by_model[model_config.name]}
                    for seed in model_config.seeds:
                        if seed == tuning.seed:
                            continue
                        _, other_predictions = fit_and_predict(model_config, asset.symbol, model_kwargs, seed)
                        flat_predictions = other_predictions.ravel()
                        by_seed[seed] = pd.Series(flat_predictions, index=vol_index[-len(flat_predictions):], dtype=float)
                    predictions_by_seed[model_config.name] = by_seed

            # Todos los modelos se evalúan en las mismas fechas: las que todos cubren
            common_index = reduce(
                lambda left, right: left.intersection(right),
                [y_true.index for y_true in y_true_by_model.values()],
            )

            for model_name in y_true_by_model:
                y_true_by_model[model_name] = y_true_by_model[model_name].loc[common_index]
                y_pred_by_model[model_name] = y_pred_by_model[model_name].loc[common_index]

                eval_result = Evaluator().evaluate(
                    model_name=model_name,
                    asset=asset.symbol,
                    y_true=y_true_by_model[model_name],
                    y_pred=y_pred_by_model[model_name],
                )
                Evaluator.save_evaluation_results(experiment_path, eval_result)
                evaluation_results_by_model[model_name] = eval_result

                logger.info(
                    f"[{model_name}] RMSE={eval_result.rmse:.6f}  MAE={eval_result.mae:.6f}"
                )

            for model_name, by_seed in predictions_by_seed.items():
                seed_variability(y_true_by_model[model_name], by_seed, chosen_seeds[model_name]).to_csv(
                    os.path.join(experiment_path, f"{model_name}_{asset.symbol}_seeds.csv"),
                    index=False,
                )

            # Comparar cada challenger contra cada modelo de referencia
            for baseline_model_name in baseline_model_names:
                if baseline_model_name not in evaluation_results_by_model:
                    continue

                baseline_eval = evaluation_results_by_model[baseline_model_name]

                for model_name, challenger_eval in evaluation_results_by_model.items():
                    if model_name == baseline_model_name:
                        continue

                    comparison = ModelComparator().compare(
                        baseline=baseline_eval,
                        challenger=challenger_eval,
                        y_pred_baseline=y_pred_by_model[baseline_model_name],
                        y_pred_challenger=y_pred_by_model[model_name],
                        y_true=y_true_by_model[baseline_model_name],
                    )
                    ModelComparator.save_comparison_results(experiment_path, comparison)
                    logger.info(
                        f"Comparacion {baseline_model_name} vs {model_name}: "
                        f"DM={comparison.dm_statistic:.4f}  p={comparison.dm_p_value:.4f}"
                    )

        except DataLoaderException as e:
            logger.error(f"Data loading error for asset {asset.symbol}: {e}")
        except Exception:
            logger.exception(f"Experiment failed for asset {asset.symbol}")


    logger.info("Experiment runner finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=MODEL_CONFIG)
    parser.add_argument("--symbol", action="append")
    args = parser.parse_args()
    main(args.config, args.symbol)
