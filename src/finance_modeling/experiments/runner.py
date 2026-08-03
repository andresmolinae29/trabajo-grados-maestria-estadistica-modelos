import pandas as pd

from finance_modeling.utils import (
    DataLoaderException,
    create_experiment_directory,
    logger,
)
from finance_modeling.config import ConfigLoader

from finance_modeling.data import RawDataLoader, DataPreprocessor
from finance_modeling.models import (
    ModelFactory,
    build_prediction_result,
    save_forecast_plot,
    save_prediction_results_to_csv
)
from finance_modeling.evaluation import Evaluator, ModelComparator


def main():
    logger.info("Starting the experiment runner...")

    models_config = ConfigLoader().load_model_config()
    data_config = ConfigLoader().load_data_config()

    experiment_path = create_experiment_directory(models_config.output_dir, models_config.experiment_name)
    baseline_model_name = "GARCH"

    for asset in data_config.assets:
        if not asset.active:
            logger.info(f"Skipping inactive asset: {asset.symbol}")
            continue

        logger.info(f"Processing asset: {asset.symbol}")

        try:
            data_loader = RawDataLoader(asset)
            time_series_input = data_loader.load_data()
            time_series_output = DataPreprocessor(time_series_input).pipeline()

            logger.info(f"Finished preprocessing asset: {asset.symbol}")

            evaluation_results_by_model: dict = {}
            prediction_results_by_model: dict = {}
            vol_index = time_series_output.volatility_series.index # type: ignore

            for model_config in models_config.models:
                logger.info(f"Running model: {model_config.name} on asset: {asset.symbol}")

                model = ModelFactory.create_model(
                    model_name=model_config.name,
                    model_config=model_config,
                    symbol=asset.symbol,
                    returns=time_series_output.returns,
                    innovation=time_series_output.innovations,
                    volatility=time_series_output.volatility_series,
                    prices=time_series_output.series,
                )
                model.fit()

                # model.save_model(experiment_path)

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

                prediction_results_by_model[model_config.name] = prediction_result

                y_true_series = pd.Series(
                    data=[row.real_value for row in prediction_result.rows],
                    index=[row.timestamp for row in prediction_result.rows],
                    dtype=float,
                )
                y_pred_series = pd.Series(
                    data=[row.predicted_value for row in prediction_result.rows],
                    index=[row.timestamp for row in prediction_result.rows],
                    dtype=float,
                )

                eval_result = Evaluator().evaluate(
                    model_name=model_config.name,
                    asset=asset.symbol,
                    y_true=y_true_series,
                    y_pred=y_pred_series,
                )
                Evaluator.save_evaluation_results(experiment_path, eval_result)
                evaluation_results_by_model[model_config.name] = eval_result

                save_forecast_plot(prediction_result, experiment_path)

                logger.info(
                    f"[{model_config.name}] RMSE={eval_result.rmse:.6f}  MAE={eval_result.mae:.6f}"
                )

            # Comparar cada challenger contra el baseline GARCH
            if baseline_model_name in evaluation_results_by_model:
                baseline_eval = evaluation_results_by_model[baseline_model_name]
                baseline_pred = prediction_results_by_model[baseline_model_name]
                baseline_y_true = pd.Series(
                    data=[row.real_value for row in baseline_pred.rows],
                    index=[row.timestamp for row in baseline_pred.rows],
                    dtype=float,
                )

                for model_name, challenger_eval in evaluation_results_by_model.items():
                    if model_name == baseline_model_name:
                        continue

                    challenger_pred = prediction_results_by_model[model_name]

                    # Alinear indices para el test de Diebold-Mariano
                    common_index = baseline_y_true.index.intersection(
                        pd.Index([row.timestamp for row in challenger_pred.rows])
                    )
                    y_true_aligned = baseline_y_true.loc[common_index]
                    pred_baseline_aligned = pd.Series(
                        data=[row.predicted_value for row in baseline_pred.rows],
                        index=[row.timestamp for row in baseline_pred.rows],
                        dtype=float,
                    ).loc[common_index]
                    pred_challenger_aligned = pd.Series(
                        data=[row.predicted_value for row in challenger_pred.rows],
                        index=[row.timestamp for row in challenger_pred.rows],
                        dtype=float,
                    ).reindex(common_index)

                    comparison = ModelComparator().compare(
                        baseline=baseline_eval,
                        challenger=challenger_eval,
                        y_pred_baseline=pred_baseline_aligned,
                        y_pred_challenger=pred_challenger_aligned,
                        y_true=y_true_aligned,
                    )
                    ModelComparator.save_comparison_results(experiment_path, comparison)
                    logger.info(
                        f"Comparacion {baseline_model_name} vs {model_name}: "
                        f"DM={comparison.dm_statistic:.4f}  p={comparison.dm_p_value:.4f}"
                    )

        except DataLoaderException as e:
            logger.error(f"Data loading error for asset {asset.symbol}: {e}")


    logger.info("Experiment runner finished.")


if __name__ == "__main__":
    main()
