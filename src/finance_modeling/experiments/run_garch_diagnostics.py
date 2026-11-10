from __future__ import annotations

import argparse
import os

from finance_modeling.config import ConfigLoader
from finance_modeling.data import DataPreprocessor, RawDataLoader
from finance_modeling.evaluation import save_garch_diagnostics
from finance_modeling.models import GARCHModel
from finance_modeling.utils import get_main_root


def main(symbol: str) -> None:
    asset = next(a for a in ConfigLoader().load_data_config().assets if a.symbol == symbol)
    config = next(m for m in ConfigLoader().load_model_config().models if m.name == "GARCH")
    data = DataPreprocessor(RawDataLoader(asset).load_data()).pipeline()

    model = GARCHModel(
        config,
        symbol,
        prices=data.series,
        volatility=data.volatility_series,
        selection=ConfigLoader().load_garch_selection().get(symbol),
    )
    model.fit()
    save_garch_diagnostics(model, os.path.join(get_main_root(), "results"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTC-USD")
    main(parser.parse_args().symbol)
