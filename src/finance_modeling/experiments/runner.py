from finance_modeling.utils import logger
from finance_modeling.config import ConfigLoader

from finance_modeling.data import RawDataLoader, DataPreprocessor

from finance_modeling.models import ModelFactory

from finance_modeling.evaluation import Metrics, Evaluator


def main():
    logger.info("Starting the experiment runner...")

    models_config = ConfigLoader().load_model_config()
    data_config = ConfigLoader().load_data_config()

    for asset in data_config.assets:
        if not asset.active:
            logger.info(f"Skipping inactive asset: {asset.symbol}")
            continue

        logger.info(f"Processing asset: {asset.symbol}")

        data_loader = RawDataLoader(asset)
        time_series_input = data_loader.load_data()

        preprocessor = DataPreprocessor(asset)
        processed_data = preprocessor.preprocess(time_series_input)
        processed_data = preprocessor.compute_log_returns(processed_data)
        processed_data = preprocessor.normalize(processed_data)
        processed_data = preprocessor.split_data(processed_data, train_ratio=models_config.train_ratio)

        logger.info(f"Finished pre processing asset: {asset.symbol}")

        for model_config in models_config.models:

            logger.info(f"Running model: {model_config.name} on asset: {asset.symbol}")

            model = ModelFactory.create_model(model_config.name, model_config, asset)
            model.fit(processed_data.train)
            predictions = model.predict(processed_data.train, processed_data.test)
            evaluation_result = Evaluator.from_timeinput_and_prediction_result(
                y_true=processed_data,
                y_pred=predictions
            )

            logger.info(f"Evaluation results for {model.name} on {asset.symbol}: RMSE={evaluation_result.rmse}, MAE={evaluation_result.mae}")

    logger.info("Experiment runner finished.")


if __name__ == "__main__":
    main()
