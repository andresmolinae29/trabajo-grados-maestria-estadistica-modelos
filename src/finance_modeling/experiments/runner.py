from finance_modeling.utils import (
    DataLoaderException,
    create_experiment_directory,
    logger,
)
from finance_modeling.config import ConfigLoader

from finance_modeling.data import RawDataLoader, DataPreprocessor
from finance_modeling.models import ModelFactory
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

            preprocessor = DataPreprocessor(asset)
            processed_data = preprocessor.preprocess(time_series_input)
            processed_data = preprocessor.compute_log_returns(processed_data)
            processed_data = preprocessor.normalize(processed_data)
            processed_data = preprocessor.split_data(processed_data, train_ratio=models_config.train_ratio)

            logger.info(f"Finished pre processing asset: {asset.symbol}")

            evaluation_results_by_model = {}
            prediction_results_by_model = {}

            for model_config in models_config.models:

                logger.info(f"Running model: {model_config.name} on asset: {asset.symbol}")

                model = ModelFactory.create_model(model_config.name, model_config, asset)
                model.fit(processed_data.train)
                predictions = model.predict(processed_data.train, processed_data.test)
                evaluation_result = Evaluator.from_timeinput_and_prediction_result(
                    y_true=processed_data,
                    y_pred=predictions
                )

                model.save_model(experiment_path)
                model.save_results(experiment_path, predictions)
                model.save_model_best_hyperparameters(experiment_path)

                Evaluator().save_evaluation_results(experiment_path, evaluation_result)
                evaluation_results_by_model[model.name] = evaluation_result
                prediction_results_by_model[model.name] = predictions

                logger.info(f"Evaluation results for {model.name} on {asset.symbol}: RMSE={evaluation_result.rmse}, MAE={evaluation_result.mae}")

            baseline_evaluation = evaluation_results_by_model.get(baseline_model_name)
            baseline_predictions = prediction_results_by_model.get(baseline_model_name)

            if baseline_evaluation is None or baseline_predictions is None:
                logger.warning(
                    f"Skipping model comparison for {asset.symbol}: baseline {baseline_model_name} was not produced."
                )
                continue

            for model_name, challenger_evaluation in evaluation_results_by_model.items():
                if model_name == baseline_model_name:
                    continue

                challenger_predictions = prediction_results_by_model[model_name]
                comparison_result = ModelComparator.compare_from_timeinput_and_prediction_results(
                    baseline=baseline_evaluation,
                    challenger=challenger_evaluation,
                    y_true=processed_data.test,
                    pred_baseline=baseline_predictions,
                    pred_challenger=challenger_predictions,
                )
                ModelComparator.save_comparison_results(experiment_path, comparison_result)

                logger.info(
                    f"Comparison for {asset.symbol}: {baseline_model_name} vs {model_name}, "
                    f"RMSE diff={comparison_result.rmse_diff}, "
                    f"DM p-value={comparison_result.dm_p_value}"
                )

        except DataLoaderException as e:
            logger.error(f"Data loading error for asset {asset.symbol}: {e}")

    logger.info("Experiment runner finished.")


if __name__ == "__main__":
    main()
