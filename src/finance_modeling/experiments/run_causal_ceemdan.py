from __future__ import annotations

import argparse

from finance_modeling.config import ConfigLoader
from finance_modeling.data import DataPreprocessor, RawDataLoader
from finance_modeling.models.ceemdan_lstm import causal_ceemdan_features
from finance_modeling.utils import logger


def main(symbols: list[str] | None = None) -> None:
    # Deja en caché la descomposición causal que usa el CEEMDAN-LSTM con causal: true.
    for asset in ConfigLoader().load_data_config().assets:
        if not asset.active or (symbols and asset.symbol not in symbols):
            continue
        data = DataPreprocessor(RawDataLoader(asset).load_data()).pipeline()
        features = causal_ceemdan_features(data.returns, asset.symbol)
        logger.info(f"[{asset.symbol}] causal CEEMDAN ready: {features.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", action="append")
    main(parser.parse_args().symbol)
