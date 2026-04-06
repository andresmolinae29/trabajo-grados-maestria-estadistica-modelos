from __future__ import annotations

from enum import StrEnum

import pandas as pd
from pydantic import BaseModel, ConfigDict


class AssetType(StrEnum):
    FOREX = "forex"
    INDEX = "index"
    CRYPTO = "crypto"


class AssetMetadata(BaseModel):
    symbol: str
    asset_type: AssetType
    description: str = ""


class TimeSeriesInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    metadata: AssetMetadata
    series: pd.Series
    frequency: str = "15min"
