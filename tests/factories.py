from __future__ import annotations

from finance_modeling.schemas.data import AssetMetadata, AssetType


def make_asset_metadata(**overrides: object) -> AssetMetadata:
    defaults = dict(
        symbol="BTC-USD",
        asset_type=AssetType.CRYPTO,
        description="Bitcoin",
        data_folder="bitcoin",
    )
    defaults.update(overrides)
    return AssetMetadata(**defaults)
