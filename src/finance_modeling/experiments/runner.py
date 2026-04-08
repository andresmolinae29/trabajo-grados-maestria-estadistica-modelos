from finance_modeling.utils import logger
from finance_modeling.config import ConfigLoader

from finance_modeling.data import RawDataLoader, DataPreprocessor


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

        for model in models_config.models:

            logger.info(f"Running model: {model.name} on asset: {asset.symbol}")





    logger.info("Experiment runner finished.")


if __name__ == "__main__":
    main()
